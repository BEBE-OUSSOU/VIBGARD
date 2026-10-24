"""
VibeGuard — AI predictive maintenance for SMEs, using any phone (or a $5 MEMS sensor) as a vibration sensor.

Run:   uvicorn app:app --host 0.0.0.0 --port 8000
Open:  http://localhost:8000          (dashboard)
       https://<tunnel>/phone          (on the phone attached to the machine)
"""
import asyncio
import json
import math
import os
import random
import re
import time
from collections import deque

import httpx
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.ensemble import IsolationForest

# ---------------------------------------------------------------- config
WINDOW = 128          # samples per analysis window
HOP = 64              # 50% overlap
CALIB_WINDOWS = 20    # windows used to learn the "healthy" baseline
FEATURE_KEYS = ["rms", "peak", "crest", "kurtosis", "dom_freq",
                "band_1", "band_2", "band_3", "band_4"]
FEATURE_LABELS = {
    "rms": "RMS vibration (m/s²)", "peak": "Peak (m/s²)", "crest": "Crest factor",
    "kurtosis": "Kurtosis (impulsiveness)", "dom_freq": "Dominant frequency (Hz)",
    "band_1": "Energy low band", "band_2": "Energy mid-low band",
    "band_3": "Energy mid-high band", "band_4": "Energy high band",
}

BASE = os.path.dirname(os.path.abspath(__file__))


def load_dotenv():
    """Read KEY=value lines from config.txt (or .env) next to app.py (no extra package needed)."""
    for name in ("config.txt", ".env"):
        path = os.path.join(BASE, name)
        if os.path.exists(path):
            break
    else:
        return
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

# Free option 1 (recommended): OpenRouter — free key, no credit card, Google's open Gemma model.
# Works with any OpenAI-compatible provider (Groq, Mistral...) by changing LLM_BASE_URL / LLM_MODEL.
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
if "paste-your" in LLM_API_KEY:          # placeholder from config.txt not replaced yet
    LLM_API_KEY = ""
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-31b-it:free")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openrouter/free")
# Optional paid/other options
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")
# URL the phone must open (your https tunnel). Empty = auto-detect.
PUBLIC_URL = os.getenv("PUBLIC_URL", "").rstrip("/")

app = FastAPI(title="VibeGuard")
app.mount("/static", StaticFiles(directory=os.path.join(BASE, "static")), name="static")

dashboards: set = set()


# ---------------------------------------------------------------- signal processing
def compute_features(ts, xs):
    x = np.asarray(xs, dtype=float)
    x = x - x.mean()                      # removes gravity / DC offset
    t = np.asarray(ts, dtype=float)
    dt = np.diff(t)
    dt = dt[dt > 0]
    fs = 1.0 / float(np.median(dt)) if len(dt) else 60.0

    m2 = float(np.mean(x ** 2))
    rms = math.sqrt(m2)
    peak = float(np.max(np.abs(x)))
    kurt = float(np.mean(x ** 4) / (m2 ** 2 + 1e-12))

    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1.0 / fs)
    s, f = spec[1:], freqs[1:]
    dom = float(f[int(np.argmax(s))])
    energy = s ** 2
    total = float(energy.sum()) + 1e-12
    bands = [float(c.sum() / total) for c in np.array_split(energy, 4)]

    feats = {"rms": rms, "peak": peak, "crest": peak / (rms + 1e-9), "kurtosis": kurt,
             "dom_freq": dom, "band_1": bands[0], "band_2": bands[1],
             "band_3": bands[2], "band_4": bands[3]}
    return feats, fs, [round(v, 2) for v in freqs.tolist()], [round(v, 4) for v in spec.tolist()]


class Machine:
    def __init__(self, mid, name="Water pump P-101", kind="Centrifugal pump, 3-phase induction motor 7.5 kW"):
        self.id, self.name, self.kind = mid, name, kind
        self.reset()

    def reset(self):
        self.t = deque(maxlen=WINDOW)
        self.x = deque(maxlen=WINDOW)
        self.since = 0
        self.mode = "idle"
        self.calib = []
        self.mu = self.sigma = self.model = None
        self.health = 100.0
        self.status = "idle"
        self.last_feats = None
        self.last_z = None
        self.events = deque(maxlen=40)

    def log(self, level, text):
        ev = {"ts": time.time(), "level": level, "text": text}
        self.events.appendleft(ev)
        return ev

    def start_calibration(self):
        self.calib = []
        self.mode = "calibrating"
        self.status = "calibrating"
        self.health = 100.0
        return self.log("info", "Calibration started — learning the healthy vibration signature")

    def add(self, t, v):
        self.t.append(t)
        self.x.append(v)
        self.since += 1
        if len(self.x) == WINDOW and self.since >= HOP:
            self.since = 0
            return self.process()
        return None

    def process(self):
        feats, fs, freqs, spec = compute_features(self.t, self.x)
        vec = np.array([feats[k] for k in FEATURE_KEYS])
        self.last_feats = feats
        msg = {"type": "metrics", "machine": self.id, "name": self.name, "mode": self.mode,
               "fs": round(fs, 1), "features": {k: round(v, 4) for k, v in feats.items()},
               "freqs": freqs, "spectrum": spec, "ts": time.time()}
        new_events = []

        if self.mode == "calibrating":
            self.calib.append(vec)
            msg["calib_progress"] = len(self.calib) / CALIB_WINDOWS
            if len(self.calib) >= CALIB_WINDOWS:
                X = np.array(self.calib)
                self.mu = X.mean(axis=0)
                self.sigma = np.maximum(X.std(axis=0), np.maximum(0.10 * np.abs(self.mu), 1e-6))
                self.model = IsolationForest(n_estimators=150, contamination="auto",
                                             random_state=0).fit(X)
                self.mode = "monitoring"
                self.status = "healthy"
                new_events.append(self.log("ok", f"Baseline learned from {len(X)} windows at {fs:.0f} Hz — monitoring"))

        elif self.mode == "monitoring":
            z = np.abs((vec - self.mu) / self.sigma)
            self.last_z = {k: round(float(v), 2) for k, v in zip(FEATURE_KEYS, z)}
            maxz = float(z.max())
            raw = float(np.clip(100 - max(0.0, maxz - 2.5) * 12, 0, 100))
            iso_anomaly = bool(self.model.predict(vec.reshape(1, -1))[0] == -1)
            if iso_anomaly:
                raw = min(raw, 85)
            self.health = 0.7 * self.health + 0.3 * raw
            status = "healthy" if self.health >= 75 else ("warning" if self.health >= 45 else "critical")
            top = sorted(self.last_z.items(), key=lambda kv: -kv[1])[:3]
            if status != self.status:
                lvl = {"healthy": "ok", "warning": "warn", "critical": "crit"}[status]
                drivers = ", ".join(f"{FEATURE_LABELS[k]} {v:.1f}σ" for k, v in top)
                new_events.append(self.log(lvl, f"Status → {status.upper()} (drivers: {drivers})"))
                self.status = status
            msg.update({"health": round(self.health, 1), "status": self.status, "z": self.last_z,
                        "top": top, "iso_anomaly": iso_anomaly,
                        "baseline": {k: round(float(m), 4) for k, m in zip(FEATURE_KEYS, self.mu)}})

        msg["status"] = self.status
        msg["health"] = round(self.health, 1)
        msg["events"] = new_events
        return msg


machines = {"M1": Machine("M1")}


def get_machine(mid):
    if mid not in machines:
        machines[mid] = Machine(mid, name=f"Machine {mid}", kind="Rotating equipment")
    return machines[mid]


async def broadcast(msg):
    data = json.dumps(msg)
    dead = []
    for ws in list(dashboards):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        dashboards.discard(ws)


async def ingest(mid, samples):
    m = get_machine(mid)
    for t, v in samples:
        msg = m.add(t, v)
        if msg:
            await broadcast(msg)


# ---------------------------------------------------------------- simulator (demo safety net)
sim = {"on": False, "fault": False, "task": None}


async def sim_loop():
    fs, n, t0 = 100.0, 0, time.time()
    while sim["on"]:
        batch = []
        for _ in range(10):
            t = n / fs
            n += 1
            v = 9.81 + 0.30 * math.sin(2 * math.pi * 12 * t) + random.gauss(0, 0.05)
            if sim["fault"]:
                v += 0.45 * math.sin(2 * math.pi * 24 * t) + 0.25 * math.sin(2 * math.pi * 36 * t)
                if random.random() < 0.03:
                    v += random.choice([-1, 1]) * 2.0
            batch.append((t0 + t, v))
        await ingest("M1", batch)
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------- routes
@app.get("/")
async def dashboard():
    return FileResponse(os.path.join(BASE, "static", "dashboard.html"))


@app.get("/phone")
async def phone():
    return FileResponse(os.path.join(BASE, "static", "phone.html"))


@app.websocket("/ws/sensor")
async def ws_sensor(ws: WebSocket):
    await ws.accept()
    mid = ws.query_params.get("machine", "M1")
    await broadcast({"type": "sensor", "machine": mid, "connected": True})
    try:
        while True:
            data = json.loads(await ws.receive_text())
            samples = [(s[0] / 1000.0, math.sqrt(s[1] ** 2 + s[2] ** 2 + s[3] ** 2))
                       for s in data.get("samples", []) if s[1] is not None]
            await ingest(mid, samples)
    except WebSocketDisconnect:
        await broadcast({"type": "sensor", "machine": mid, "connected": False})


@app.websocket("/ws/dashboard")
async def ws_dashboard(ws: WebSocket):
    await ws.accept()
    dashboards.add(ws)
    m = get_machine("M1")
    await ws.send_text(json.dumps({"type": "hello", "machine": m.id, "name": m.name, "kind": m.kind,
                                   "events": list(m.events), "status": m.status, "sim": sim["on"],
                                   "fault": sim["fault"]}))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        dashboards.discard(ws)


class MachineReq(BaseModel):
    machine: str = "M1"


class SimReq(BaseModel):
    on: bool = False
    fault: bool = False


class DiagReq(BaseModel):
    machine: str = "M1"
    lang: str = "English"


@app.post("/api/calibrate")
async def calibrate(req: MachineReq):
    ev = get_machine(req.machine).start_calibration()
    await broadcast({"type": "event", "events": [ev]})
    return {"ok": True}


@app.post("/api/sim")
async def set_sim(req: SimReq):
    sim["fault"] = req.fault
    if req.on and not sim["on"]:
        sim["on"] = True
        sim["task"] = asyncio.create_task(sim_loop())
    elif not req.on:
        sim["on"] = False
    return {"on": sim["on"], "fault": sim["fault"]}


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


@app.get("/api/info")
async def info(request: Request):
    """Tell the dashboard which link the phone must open (used for the QR code)."""
    host = request.headers.get("host", "")
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if PUBLIC_URL:
        url = PUBLIC_URL
    elif host.split(":")[0] in ("localhost", "127.0.0.1"):
        url = f"http://{lan_ip()}:{request.url.port or 8000}"   # same Wi-Fi only, no motion sensor on most phones
    else:
        url = f"{proto}://{host}"
    ai = (LLM_MODEL if LLM_API_KEY else None) or (GEMINI_MODEL if GEMINI_KEY else None) or f"Ollama {OLLAMA_MODEL} / rules"
    return {"phone_url": url + "/phone", "https": url.startswith("https://"), "ai": ai}


@app.get("/api/state")
async def state():
    m = get_machine("M1")
    return {"mode": m.mode, "status": m.status, "health": m.health, "features": m.last_feats, "z": m.last_z}


# ---------------------------------------------------------------- AI technician
def build_prompt(m, lang):
    rows = []
    for i, k in enumerate(FEATURE_KEYS):
        base = float(m.mu[i]) if m.mu is not None else float("nan")
        cur = m.last_feats.get(k, float("nan")) if m.last_feats else float("nan")
        z = m.last_z.get(k, 0) if m.last_z else 0
        rows.append(f"- {FEATURE_LABELS[k]}: baseline={base:.4f}, now={cur:.4f}, deviation={z:.1f} sigma")
    events = "\n".join(f"- {e['level']}: {e['text']}" for e in list(m.events)[:8])
    return f"""You are a senior reliability / maintenance engineer.
Machine: {m.name} ({m.kind}). Health score now: {m.health:.0f}/100, status: {m.status}.
Vibration features from an accelerometer on the motor housing (healthy baseline vs now):
{chr(10).join(rows)}
Recent events:
{events}

Diagnose the most likely mechanical/electrical fault (imbalance, misalignment, bearing wear,
mechanical looseness, cavitation, electrical issue...). Be concrete and practical for an SME technician.
Answer ONLY with JSON, written in {lang}, with this schema:
{{"diagnosis": str, "confidence": "low|medium|high", "likely_causes": [str],
  "severity": "low|medium|high|critical", "recommended_actions": [str],
  "work_order": {{"title": str, "priority": "P1|P2|P3", "deadline": str,
                  "estimated_downtime": str, "spare_parts": [str], "safety_notes": [str]}},
  "risk_if_ignored": str}}"""


def parse_json(text):
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


async def ask_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3}}
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.post(url, params={"key": GEMINI_KEY}, json=body)
        r.raise_for_status()
        return parse_json(r.json()["candidates"][0]["content"]["parts"][0]["text"])


async def ask_openai_compatible(prompt, model):
    headers = {"Authorization": f"Bearer {LLM_API_KEY}",
               "HTTP-Referer": "http://localhost:8000", "X-Title": "VibeGuard"}
    body = {"model": model, "temperature": 0.3,
            "messages": [{"role": "system", "content": "You answer only with valid JSON."},
                         {"role": "user", "content": prompt}]}
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code} {r.text[:200]}")
        return parse_json(r.json()["choices"][0]["message"]["content"])


async def ask_ollama(prompt):
    async with httpx.AsyncClient(timeout=90) as c:
        r = await c.post(f"{OLLAMA_URL}/api/generate",
                         json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"})
        r.raise_for_status()
        return parse_json(r.json()["response"])


def rule_diagnosis(m):
    z = m.last_z or {k: 0 for k in FEATURE_KEYS}
    causes, actions = [], []
    if z["kurtosis"] > 3 or z["crest"] > 3:
        causes.append("Impulsive shocks: early bearing defect or mechanical looseness")
        actions += ["Inspect bearings (noise, temperature, grease)", "Check mounting bolts and base plate"]
    if z["band_2"] > 3 or z["band_3"] > 3 or z["band_4"] > 3:
        causes.append("Energy moved to harmonics: shaft misalignment or coupling wear")
        actions += ["Check shaft alignment (laser/dial)", "Inspect coupling elements"]
    if z["rms"] > 3 and not causes:
        causes.append("Overall vibration increase at running speed: rotor imbalance")
        actions += ["Clean impeller/fan, check for build-up or missing balance weights"]
    if not causes:
        causes, actions = ["No significant deviation from baseline"], ["Continue monitoring"]
    sev = {"healthy": "low", "warning": "medium", "critical": "high"}.get(m.status, "low")
    return {"diagnosis": causes[0], "confidence": "medium", "likely_causes": causes, "severity": sev,
            "recommended_actions": actions,
            "work_order": {"title": f"Inspect {m.name}", "priority": "P1" if sev == "high" else "P2",
                           "deadline": "24 h" if sev == "high" else "7 days", "estimated_downtime": "1-2 h",
                           "spare_parts": ["Bearing set", "Coupling insert"],
                           "safety_notes": ["Lock-out / tag-out before intervention"]},
            "risk_if_ignored": "Fault progression leading to unplanned stop and secondary damage."}


@app.post("/api/diagnose")
async def diagnose(req: DiagReq):
    m = get_machine(req.machine)
    if m.mode != "monitoring":
        return {"source": "none", "report": None, "error": "Calibrate the machine first."}
    prompt = build_prompt(m, req.lang)
    errors = []
    if LLM_API_KEY:
        for model in dict.fromkeys([LLM_MODEL, LLM_FALLBACK_MODEL]):
            try:
                return {"source": model, "report": await ask_openai_compatible(prompt, model)}
            except Exception as e:
                errors.append(f"{model}: {e}")
    if GEMINI_KEY:
        try:
            return {"source": f"Gemini ({GEMINI_MODEL})", "report": await ask_gemini(prompt)}
        except Exception as e:
            errors.append(f"gemini: {e}")
    try:
        return {"source": f"Ollama ({OLLAMA_MODEL})", "report": await ask_ollama(prompt)}
    except Exception as e:
        errors.append(f"ollama: {e}")
    return {"source": "rule engine (offline fallback)", "report": rule_diagnosis(m), "errors": errors}

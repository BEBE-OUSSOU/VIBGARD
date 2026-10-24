# VibeGuard — step by step

## 1. Stop the old version
Close the terminal where the old VibeGuard is running (or press Ctrl+C in it).

## 2. Unzip
Unzip `vibeguard.zip` somewhere easy (for example the Desktop). Use this new folder from now on.

## 3. Get a free AI key (OpenRouter, no credit card)
1. Go to https://openrouter.ai and sign up.
2. Open **Keys** → **Create Key** → copy the key (starts with `sk-or-v1-`).

## 4. Paste the key
1. Open `config.txt` (in the vibeguard folder) with Notepad / TextEdit.
2. Replace `sk-or-v1-paste-your-key-here` with your real key.
3. Save.

## 5. Start VibeGuard
- Windows: double-click `start_windows.bat`
- Mac/Linux: in a terminal, `sh start_mac_linux.sh`

Open http://localhost:8000 . Below the QR code you should read **AI model: google/gemma-4-31b-it:free**.

## 6. Test without the phone
1. Click **Simulator: OFF** (it turns ON).
2. Click **1 · Learn healthy baseline** and wait ~15 s → score 100, green.
3. Click **Inject fault (sim)** → score drops, turns red.
4. Click **2 · Diagnose & create work order** → AI report. "Source" should show the Gemma model.
5. Click the simulator button again to turn it OFF before using the phone.

## 7. Connect the phone (needs HTTPS)
1. Install cloudflared once:
   - Windows: `winget install --id Cloudflare.cloudflared` (then close and reopen the terminal)
   - Mac: `brew install cloudflared`
2. Open a **second** terminal (keep VibeGuard running in the first) and run:
   `cloudflared tunnel --url http://localhost:8000`
3. Copy the `https://....trycloudflare.com` link it prints and open it **on the laptop**.
4. Scan the QR code with the phone → **Start streaming** → allow motion access.
5. Tape the phone on a fan → **Learn healthy baseline** → disturb the fan → **Diagnose**.

## If something goes wrong
- `python` not found (Windows): install Python from python.org and tick "Add Python to PATH".
- Report says "Fallback used": the key is wrong or the free model is busy; the app still answers with the offline rules.
- Orange "Not HTTPS" warning under the QR code: you opened localhost instead of the trycloudflare link.
- Phone page says "Disconnected": the tunnel terminal was closed; start it again and use the new link.

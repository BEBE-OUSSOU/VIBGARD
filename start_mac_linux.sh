#!/bin/sh
cd "$(dirname "$0")"
echo "Installing packages (first time only takes a minute)..."
python3 -m pip install -r requirements.txt
echo
echo "VibeGuard is starting. Open http://localhost:8000 in your browser."
echo "Press Ctrl+C to stop it."
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000

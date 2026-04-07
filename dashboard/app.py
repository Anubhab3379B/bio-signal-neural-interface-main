"""
Real-Time Bio-Signal Monitoring Dashboard — Backend
=====================================================

FastAPI + WebSocket server that:
  1. Runs a synthetic bio-signal pipeline in a background thread
  2. Streams live data (signal, intent, anomalies, latency) to the browser
     via WebSocket as JSON messages at ~10 Hz

Start the server:
    cd "d:\\Completed Projects\\bio-signal-neural-interface-main"
    python dashboard/app.py

Then open: http://localhost:8000/
"""

import asyncio
import json
import math
import os
import sys
import time
import threading
import numpy as np
from pathlib import Path

# ── Add project root to path so imports work ─────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# Attempt FastAPI import — graceful degradation if not installed
# ─────────────────────────────────────────────────────────────────────────────
try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print('[Dashboard] FastAPI/uvicorn not installed.')
    print('[Dashboard] Install with: pip install fastapi uvicorn websockets')


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic Signal Engine (runs in background thread)
# ─────────────────────────────────────────────────────────────────────────────

INTENT_NAMES = ['REST', 'GRASP', 'RELEASE', 'POINT', 'PINCH']
BAND_NAMES   = ['delta', 'theta', 'alpha', 'beta', 'gamma']

class SignalEngine:
    """
    Generates synthetic EMG / EEG data and simulated inference results
    at 10 Hz for dashboard streaming.
    """

    def __init__(self, fs: float = 1000.0):
        self.fs = fs
        self._t = 0.0
        self._dt = 0.1    # 100 ms per update tick (10 Hz dashboard)
        self._intent_phase = 0
        self._intent_timer = 0.0
        self._current_intent = 0
        self._session_start = time.time()

    def tick(self) -> dict:
        """Return one dashboard data frame."""
        t = self._t
        self._t += self._dt

        # ── EMG waveform (last 200 ms = 200 samples at 1kHz) ────────────
        n = 200
        ts = np.linspace(t, t + self._dt, n)
        freq = 15.0 + self._current_intent * 5.0
        amp  = 0.6 + 0.3 * math.sin(2 * math.pi * 0.2 * t)
        emg_window = (amp * np.sin(2 * math.pi * freq * ts)
                      + 0.1 * np.random.randn(n))

        # ── EEG band powers ───────────────────────────────────────────────
        alpha_base = 0.5 + 0.2 * math.sin(2 * math.pi * 0.1 * t)
        bands = {
            'delta': max(0.05, 0.1  + 0.05 * np.random.randn()),
            'theta': max(0.05, 0.15 + 0.05 * np.random.randn()),
            'alpha': max(0.05, alpha_base),
            'beta':  max(0.05, 0.25 + 0.05 * np.random.randn()),
            'gamma': max(0.05, 0.05 + 0.02 * np.random.randn()),
        }
        total = sum(bands.values())
        bands = {k: round(v / total, 4) for k, v in bands.items()}

        # ── Intent probabilities ──────────────────────────────────────────
        self._intent_timer += self._dt
        if self._intent_timer > (2.0 + np.random.rand() * 3.0):
            self._current_intent = np.random.randint(0, 5)
            self._intent_timer = 0.0

        raw_probs = np.random.dirichlet(
            [4.0 if i == self._current_intent else 0.3 for i in range(5)]
        )
        intent_probs = {INTENT_NAMES[i]: round(float(raw_probs[i]), 4)
                        for i in range(5)}

        # ── Simulated latency ─────────────────────────────────────────────
        preprocess_ms = max(0.5, 2.1 + np.random.randn() * 0.5)
        inference_ms  = max(0.5, 4.3 + np.random.randn() * 1.0)
        total_ms = preprocess_ms + inference_ms

        # ── Anomaly (occasional) ──────────────────────────────────────────
        anomaly = None
        if np.random.rand() < 0.04:
            anomaly_types = ['MOTION_ARTIFACT', 'NOISE_BURST', 'POWER_LINE']
            anomaly = np.random.choice(anomaly_types)

        # ── User state ────────────────────────────────────────────────────
        stress = min(1.0, 0.3 + 0.4 * math.sin(2 * math.pi * 0.05 * t) ** 2)
        if   stress > 0.75: user_state = 'OVERLOADED'
        elif stress > 0.55: user_state = 'FATIGUED'
        elif stress > 0.30: user_state = 'MILD_LOAD'
        else:               user_state = 'OPTIMAL'

        return {
            'ts':           round(time.time() - self._session_start, 2),
            'emg_raw':      emg_window[:50].tolist(),    # 50 pts for plot
            'eeg_bands':    bands,
            'intent_probs': intent_probs,
            'top_intent':   INTENT_NAMES[self._current_intent],
            'confidence':   round(float(raw_probs[self._current_intent]), 3),
            'latency': {
                'preprocess_ms': round(preprocess_ms, 2),
                'inference_ms':  round(inference_ms, 2),
                'total_ms':      round(total_ms, 2),
            },
            'anomaly':      anomaly,
            'user_state':   user_state,
            'stress':       round(stress, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HTML content (embedded so no static files needed)
# ─────────────────────────────────────────────────────────────────────────────

def read_html():
    html_path = Path(__file__).parent / 'index.html'
    if html_path.exists():
        return html_path.read_text(encoding='utf-8')
    return '<h1>Dashboard HTML not found — run from project root.</h1>'


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    app = FastAPI(title='Bio-Signal Dashboard', version='1.0')
    engine = SignalEngine()

    @app.get('/', response_class=HTMLResponse)
    async def get_dashboard():
        return HTMLResponse(read_html())

    @app.websocket('/ws')
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                frame = engine.tick()
                await ws.send_text(json.dumps(frame))
                await asyncio.sleep(0.1)   # 10 Hz
        except (WebSocketDisconnect, Exception):
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if not FASTAPI_AVAILABLE:
        print('Install dependencies: pip install fastapi uvicorn websockets')
        sys.exit(1)

    print('=' * 60)
    print('Bio-Signal Real-Time Dashboard')
    print('Open http://localhost:8000 in your browser')
    print('Press Ctrl+C to stop')
    print('=' * 60)
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level='warning')

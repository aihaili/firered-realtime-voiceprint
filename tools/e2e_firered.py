# -*- coding: utf-8 -*-
"""e2e：把 wav 按 30ms 帧灌进 FireRedASR2S 服务器 WebSocket，模拟浏览器录音。

用法: python e2e_firered.py [wav路径] [倍速]
"""
import asyncio
import json
import os
import sys

import numpy as np
import soundfile as sf
import websockets

BASE = os.path.dirname(os.path.abspath(__file__))
WAV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "mic_debug.wav")
SPEED = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8766
URL = f"ws://127.0.0.1:{PORT}/ws"

finals: list = []
lives: list = []


async def recv_loop(ws):
    while True:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
        except asyncio.TimeoutError:
            continue
        except Exception:
            return
        if isinstance(msg, bytes):
            continue
        d = json.loads(msg)
        t = d.get("type")
        if t == "final":
            finals.append(d)
            print(f"[WS final] spk={d.get('spk')} text={d.get('text')}", flush=True)
        elif t == "live":
            lives.append(d.get("text", ""))
            print(f"[WS live ] {d.get('text')}", flush=True)
        elif t == "relabel":
            print(f"[WS relabel] 气泡「{d.get('text')}」→ {d.get('spk')}", flush=True)
        elif t == "revision":
            print(f"\n[WS revision] 定版改正 {len(d.get('bubbles') or [])} 条：", flush=True)
            for b in (d.get("bubbles") or []):
                print(f"    {b.get('spk')}: {b.get('text')}", flush=True)
        elif t in ("started", "stopped", "status"):
            print(f"[WS {t}] {d}", flush=True)


async def main():
    wav, sr = sf.read(WAV, dtype="float32")
    if sr != 16000:
        raise SystemExit(f"需要 16kHz 音频，实际 {sr}")
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16).tobytes()
    print(f"pumping {len(wav)/sr:.1f}s audio at {SPEED}x speed", flush=True)

    async with websockets.connect(URL, max_size=None, ping_interval=None) as ws:
        await ws.recv()  # status
        await ws.send(json.dumps({"cmd": "start"}))
        task = asyncio.create_task(recv_loop(ws))

        chunk_sec = 1.0
        chunk = int(chunk_sec * 16000) * 2
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i:i + chunk])
            await asyncio.sleep(chunk_sec / SPEED)
        print("--- audio done, waiting for finals ---", flush=True)
        await asyncio.sleep(3.0)
        await ws.send(json.dumps({"cmd": "stop"}))
        await asyncio.sleep(8.0)
        task.cancel()

    print(f"\n=== {len(finals)} finals, {len(lives)} live updates ===", flush=True)
    for f in finals:
        print(f"  spk={f.get('spk')}: {f.get('text')}", flush=True)
    spks = {f.get("spk") for f in finals}
    print(f"distinct speakers: {sorted(s for s in spks if s)}", flush=True)


asyncio.run(main())

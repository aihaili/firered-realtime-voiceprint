# -*- coding: utf-8 -*-
"""验证 CUDA Graph 解码：与官方输出是否一致 + 提速幅度（含首次捕获开销/稳态）。"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
sys.path.insert(0, BASE)
os.environ.setdefault("USERNAME", "00")

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402
import firede_graph                                                   # noqa: E402

OUT = os.path.join(BASE, "verify_graph_result.txt")
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))

wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav = wav * 32768.0
cases = [(f"test_call {a}-{b}s", wav[a * sr:b * sr]) for a, b in
         [(20, 30), (20, 40), (35, 65), (0, 12), (40, 60)]]
for nm in ["assets/hello_zh.wav", "assets/hello_en.wav"]:
    w2, _ = sf.read(os.path.join(REPO, nm), dtype="float32")
    cases.append((os.path.basename(nm), w2 * 32768.0))

log("=== 官方实现 ===")
base = {}
for name, seg in cases:
    ts = []
    for _ in range(2):
        t0 = time.time()
        r = asr.transcribe(["x"], [(sr, seg)])[0]
        ts.append(time.time() - t0)
    base[name] = (min(ts), r)
    log(f"  {name:20s} {min(ts):6.3f}s RTF={min(ts)/(len(seg)/sr):.3f} [{r.get('text','')[:36]}]")

gbs = firede_graph.patch_graph_decode(asr, verbose=True)
log("\n=== CUDA Graph 实现 ===")
all_same = True
for name, seg in cases:
    ts = []
    for _ in range(5):
        t0 = time.time()
        r = asr.transcribe(["x"], [(sr, seg)])[0]
        ts.append(time.time() - t0)
    b_t, b_r = base[name]
    steady = min(ts[1:]) if len(ts) > 1 else ts[0]
    same_text = r.get("text") == b_r.get("text")
    same_ts = (r.get("timestamp") or []) == (b_r.get("timestamp") or [])
    dconf = abs((r.get("confidence") or 0) - (b_r.get("confidence") or 0))
    all_same = all_same and same_text
    log(f"  {name:20s} 首次 {ts[0]:6.3f}s 稳态 {steady:6.3f}s RTF={steady/(len(seg)/sr):.3f}  "
        f"加速 {b_t/steady:5.1f}x  文本{'一致' if same_text else '**不一致**'} "
        f"时间戳{'一致' if same_ts else '差异'} dconf={dconf:.4f}")
    if not same_text:
        log(f"      官方: {b_r.get('text','')}")
        log(f"      Graph: {r.get('text','')}")

log(f"\n文本全部一致 = {all_same}；已捕获图数量 = {len(gbs.graphs)}")
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT, flush=True)

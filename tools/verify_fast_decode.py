# -*- coding: utf-8 -*-
"""验证 firede_fast 增量解码：输出是否与官方完全一致 + 提速幅度。"""
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
import firede_fast                                                    # noqa: E402

OUT = os.path.join(BASE, "verify_fast_result.txt")
lines: list = []


def log(s):
    print(s, flush=True)
    lines.append(s)


asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))

cases = []
wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav *= 32768.0
for a, b in [(20, 30), (20, 40), (35, 65), (0, 12)]:
    cases.append((f"test_call {a}-{b}s", wav[a * sr:b * sr]))
for name in ["assets/hello_zh.wav", "assets/hello_en.wav"]:
    w2, sr2 = sf.read(os.path.join(REPO, name), dtype="float32")
    cases.append((os.path.basename(name), w2 * 32768.0))


def run(seg, n):
    ts, out = [], None
    for _ in range(n):
        t0 = time.time()
        out = asr.transcribe(["x"], [(sr, seg)])[0]
        ts.append(time.time() - t0)
    return min(ts), out


log("=== 官方实现 ===")
base_out = {}
base_t = {}
for name, seg in cases:
    t, r = run(seg, 2)
    base_out[name] = r
    base_t[name] = t
    log(f"  {name:20s} {t:6.3f}s RTF={t/(len(seg)/sr):.3f}  [{r.get('text','')[:40]}]")

assert firede_fast.patch_fast_decode(), "patch failed"
torch.cuda.empty_cache()
log("\n=== firede_fast 增量解码 ===")
same_all = True
for name, seg in cases:
    t, r = run(seg, 5)
    b = base_out[name]
    same_text = r.get("text") == b.get("text")
    same_ts = (r.get("timestamp") or []) == (b.get("timestamp") or [])
    dconf = abs((r.get("confidence") or 0) - (b.get("confidence") or 0))
    same_all = same_all and same_text
    log(f"  {name:20s} {t:6.3f}s RTF={t/(len(seg)/sr):.3f}  加速 {base_t[name]/t:5.1f}x  "
        f"文本{'一致' if same_text else '**不一致**'} 时间戳{'一致' if same_ts else '差异'} dconf={dconf:.4f}"
        f"  [{r.get('text','')[:40]}]")
    if not same_text:
        log(f"      官方: {b.get('text','')}")
        log(f"      快速: {r.get('text','')}")

log(f"\n结论：文本全部一致 = {same_all}")
avg_speedup = np.mean([base_t[n] / max(run(cases[i][1], 1)[0], 1e-6)
                       for i, (n, _) in enumerate(cases)])
log(f"平均加速（首测）≈ {avg_speedup:.1f}x")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT, flush=True)

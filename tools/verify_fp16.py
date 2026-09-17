# -*- coding: utf-8 -*-
"""fp16 精度验证：同一批音频，fp32 与 fp16 各跑一遍（存 JSON），再逐字比对。

用法: python verify_fp16.py fp32|fp16
"""
import json
import os
import re
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

PREC = sys.argv[1] if len(sys.argv) > 1 else "fp32"
if PREC == "fp16":
    torch.cuda.is_bf16_supported = lambda *a, **k: False

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402
import firede_graph                                                   # noqa: E402

torch.set_grad_enabled(False)
asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=(PREC == "fp16"), beam_size=3, nbest=1,
                      return_timestamp=True))
print(f"[{PREC}] dtype={next(asr.model.parameters()).dtype} "
      f"VRAM={torch.cuda.memory_allocated()/2**30:.2f}G", flush=True)

wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
wav *= 32768.0
cases = [(f"call {a}-{b}s", wav[a * sr:b * sr]) for a, b in
         [(20, 30), (20, 40), (35, 65), (0, 12), (40, 60)]]
for nm in ["assets/hello_zh.wav", "assets/hello_en.wav"]:
    w, _ = sf.read(os.path.join(REPO, nm), dtype="float32")
    cases.append((os.path.basename(nm), w * 32768.0))
voc, _ = sf.read(os.path.join(BASE, "test_4spk.wav"), dtype="float32")
cases.append(("4spk 70-95s", voc[70 * sr:95 * sr] * 32768.0))

out = {}
firede_graph.patch_graph_decode(asr, verbose=False)   # 走 CUDA Graph 路径（与线上一致）
for name, seg in cases:
    ts = []
    for _ in range(3):
        t0 = time.time()
        r = asr.transcribe(["x"], [(sr, seg)])[0]
        ts.append(time.time() - t0)
    out[name] = {"text": r.get("text", ""), "conf": r.get("confidence"),
                 "n_ts": len(r.get("timestamp") or []),
                 "ts_head": (r.get("timestamp") or [])[:3],
                 "time": round(min(ts), 4), "dur": round(len(seg) / sr, 2)}
    print(f"  {name:16s} {min(ts):5.3f}s RTF={min(ts)/(len(seg)/sr):.3f} "
          f"[{out[name]['text'][:34]}]", flush=True)

path = os.path.join(BASE, f"verify_fp16_{PREC}.json")
json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"written {path}", flush=True)

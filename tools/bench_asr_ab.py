# -*- coding: utf-8 -*-
"""ASR 层面对照：FireRedASR2-AED vs faster-whisper large-v3-turbo，同一段音频比准确率与速度。

音频: test_call.wav（中文客服通话）。片段: 10s / 20s / 30s（同一时间区间）。
"""
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
os.environ.setdefault("USERNAME", "00")
os.environ.setdefault("USER", "00")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

WAV = os.path.join(BASE, "test_call.wav")
OUT = os.path.join(BASE, "bench_ab_result.txt")
wav, sr = sf.read(WAV, dtype="float32")
base = wav[20 * sr:50 * sr]
segs = {"10s": base[:10 * sr], "20s": base[:20 * sr], "30s": base[:30 * sr]}
lines: list = []


def log(s):
    print(s, flush=True)
    lines.append(s)


# ---------- faster-whisper ----------
log("=== faster-whisper large-v3-turbo (int8, cuda, beam=5) ===")
try:
    from faster_whisper import WhisperModel
    t0 = time.time()
    wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    log(f"  load {time.time()-t0:.1f}s")
    for name, seg in segs.items():
        ts = []
        for _ in range(3):
            t0 = time.time()
            parts, _ = wm.transcribe(seg, beam_size=5, vad_filter=False, language="zh")
            txt = "".join(p.text for p in parts).strip()
            ts.append(time.time() - t0)
        log(f"  whisper {name:4s}: {min(ts):5.2f}s RTF={min(ts)/(len(seg)/sr):.3f} [{txt}]")
    del wm
    torch.cuda.empty_cache()
except Exception as e:
    log(f"  whisper FAILED: {e}")

# ---------- FireRedASR2-AED ----------
log("\n=== FireRedASR2-AED (fp32, cuda, beam=3, timestamps) ===")
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config  # noqa: E402

t0 = time.time()
asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))
log(f"  load {time.time()-t0:.1f}s, VRAM {torch.cuda.memory_allocated()/1e9:.1f}GB")
asr.transcribe(["w"], [(sr, segs["10s"])])
for name, seg in segs.items():
    ts = []
    for _ in range(3):
        t0 = time.time()
        r = asr.transcribe(["b"], [(sr, seg * 32768.0)])[0]
        ts.append(time.time() - t0)
    log(f"  firered {name:4s}: {min(ts):5.2f}s RTF={min(ts)/(len(seg)/sr):.3f} [{r.get('text','')}]")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nwritten {OUT}", flush=True)

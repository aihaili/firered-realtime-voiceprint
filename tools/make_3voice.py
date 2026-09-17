# -*- coding: utf-8 -*-
"""构造"一句话里三种声音"的复现用例（复现用户遇到的：句内换人没识别出来）。

素材：campplus 自带 speaker1 / speaker2 两个中文说话人样本 + JFK 英文演讲（第三个说话人）。
段间只留 0.15s 间隔 → 服务端的 0.7s 静音断句不会切开 → 整段会被当成"一句话"，
只有句内声学分段才能切出三个说话人。
"""
import glob
import os

import numpy as np
import soundfile as sf

BASE = os.path.dirname(os.path.abspath(__file__))
CAM = glob.glob(os.path.join(
    os.path.expanduser("~"), ".cache", "modelscope", "models",
    "iic--speech_campplus_sv_zh-cn_16k-common", "snapshots", "*", "examples", "*.wav"))
s1 = next(p for p in CAM if "speaker1_a" in p)
s2 = next(p for p in CAM if "speaker2_a" in p)


def load16k(path):
    w, sr = sf.read(path, dtype="float32")
    if w.ndim > 1:
        w = w.mean(axis=1)
    if sr != 16000:
        idx = np.arange(0, len(w), sr / 16000.0)
        w = np.interp(idx, np.arange(len(w)), w).astype(np.float32)
    return w


gap = np.zeros(int(0.15 * 16000), dtype=np.float32)      # 0.15s：不触发 0.7s 静音断句
parts = [
    ("说话人A(女)", load16k(s1)),                          # 3.7s
    ("说话人B(男)", load16k(s2)),                          # 5.3s
    ("说话人C(英语)", load16k(os.path.join(BASE, "test_audio.flac"))[:9 * 16000]),
    ("说话人A(女)回来", load16k(s1)[:3 * 16000]),
]
out = []
for name, w in parts:
    print(f"  + {name}: {len(w)/16000:.1f}s", flush=True)
    out.extend([w, gap])
audio = np.concatenate(out)
sf.write(os.path.join(BASE, "test_3voice_onesentence.wav"), audio, 16000)
print(f"写出 test_3voice_onesentence.wav: {len(audio)/16000:.1f}s", flush=True)

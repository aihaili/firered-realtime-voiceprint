# -*- coding: utf-8 -*-
"""kernel 级 profiling：确认解码每步是"启动开销瓶颈"还是"算力瓶颈"。"""
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
firede_fast.patch_fast_decode()
torch.set_grad_enabled(False)

asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))
m = asr.model
wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
seg = (wav[20 * sr:40 * sr] * 32768.0).astype(np.float32)
feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
feats, lengths = feats.cuda(), lengths.cuda()
eo, el, em = m.encoder(feats, lengths)
torch.cuda.synchronize()

# 预热
m.decoder.batch_beam_search(eo, em, 3, 1, 0, 1.25, 0.6, 1.0, None, 0.0)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity  # noqa: E402

with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    m.decoder.batch_beam_search(eo, em, 3, 1, 0, 1.25, 0.6, 1.0, None, 0.0)
    torch.cuda.synchronize()

steps = firede_fast.LAST_STEPS
print(f"steps={steps}", flush=True)
print("\n=== 按 CUDA kernel 总耗时 top15 ===", flush=True)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
evts = prof.key_averages()
tot_us = sum(e.cuda_time_total for e in evts)
n_calls = sum(e.count for e in evts)
print(f"\nCUDA kernel/op 调用总数 = {n_calls}  (每步 {n_calls/max(steps,1):.0f} 次)", flush=True)
print(f"CUDA 总耗时 = {tot_us/1000:.0f} ms  (每步 {tot_us/1000/max(steps,1):.1f} ms)", flush=True)

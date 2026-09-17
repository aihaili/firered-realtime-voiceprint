# -*- coding: utf-8 -*-
"""正确版耗时拆解（全程 no_grad 防激活图占显存）+ 解码步数统计。"""
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

OUT = os.path.join(BASE, "breakdown2_result.txt")
lines = []


def log(s):
    print(s, flush=True)
    lines.append(s)


asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))
m = asr.model
torch.set_grad_enabled(False)

wav, sr = sf.read(os.path.join(BASE, "test_call.wav"), dtype="float32")
seg = (wav[20 * sr:40 * sr] * 32768.0).astype(np.float32)
t0 = time.time()
feats, lengths, durs, _, _ = asr.feat_extractor([(sr, seg)], ["x"])
log(f"fbank extraction         {(time.time()-t0)*1000:7.1f} ms  (T={feats.shape[1]})")
feats, lengths = feats.cuda(), lengths.cuda()


def bench(fn, n=5):
    r = fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.time()
        r = fn()
        torch.cuda.synchronize()
        ts.append(time.time() - t0)
    return min(ts), r


t, (eo, el, em) = bench(lambda: m.encoder(feats, lengths))
log(f"encoder                  {t*1000:7.1f} ms")
t, hyps = bench(lambda: m.decoder.batch_beam_search(eo, em, 3, 1, 0, 1.25, 0.6, 1.0, None, 0.0))
log(f"decoder 官方             {t*1000:7.1f} ms   steps={firede_fast.LAST_STEPS}")

firede_fast.patch_fast_decode()
t, hyps = bench(lambda: m.decoder.batch_beam_search(eo, em, 3, 1, 0, 1.25, 0.6, 1.0, None, 0.0))
log(f"decoder fast             {t*1000:7.1f} ms   steps={firede_fast.LAST_STEPS}   "
    f"(每步 {t/max(firede_fast.LAST_STEPS,1)*1000:.2f} ms)")

t, _ = bench(lambda: m.get_token_timestamp_torchaudio(eo, el, hyps))
log(f"ctc timestamps           {t*1000:7.1f} ms")

asr.config.return_timestamp = False
t, _ = bench(lambda: asr.transcribe(["x"], [(sr, seg)]))
log(f"FULL (fast, 无时间戳)      {t*1000:7.1f} ms  RTF={t/(len(seg)/sr):.3f}")
asr.config.return_timestamp = True
t, _ = bench(lambda: asr.transcribe(["x"], [(sr, seg)]))
log(f"FULL (fast, 有时间戳)      {t*1000:7.1f} ms  RTF={t/(len(seg)/sr):.3f}")

firede_fast.unpatch()
asr.config.return_timestamp = True
t, _ = bench(lambda: asr.transcribe(["x"], [(sr, seg)]), n=3)
log(f"FULL (官方, 有时间戳)      {t*1000:7.1f} ms  RTF={t/(len(seg)/sr):.3f}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT, flush=True)

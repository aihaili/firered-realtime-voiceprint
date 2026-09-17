# -*- coding: utf-8 -*-
"""显存实测：whisper 栈 vs FireRed 栈，各模型单独加载后报告 allocated/reserved/free。

用 torch.cuda.mem_get_info() 取驱动侧真实数据（本机 nvidia-smi/NVML 不可用）。
"""
import gc
import os
import sys
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("USERNAME", "00")
os.environ.setdefault("USER", "00")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, REPO)
OUT = os.path.join(BASE, "vram_result.txt")
lines: list = []


def gb(x):
    return x / 1024 ** 3


def report(tag):
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    resv = torch.cuda.memory_reserved()
    line = (f"{tag:34s} torch_alloc={gb(alloc):5.2f}GB  reserved={gb(resv):5.2f}GB  "
            f"driver_free={gb(free):5.2f}GB / total={gb(total):.2f}GB  "
            f"used(incl.baseline)={gb(total - free):5.2f}GB")
    print(line, flush=True)
    lines.append(line)
    return gb(total - free)


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(1)


report("baseline (empty cuda ctx)")

# ---------------- 旧系统 whisper 栈 ----------------
try:
    from faster_whisper import WhisperModel
    t0 = time.time()
    wm = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8")
    print(f"(whisper load {time.time()-t0:.1f}s)", flush=True)
    report("whisper large-v3-turbo int8")
    wav = torch.zeros(16000 * 10).numpy()
    list(wm.transcribe(wav, beam_size=5, vad_filter=False)[0])
    report("whisper + 10s inference")
    del wm
except Exception as e:
    lines.append(f"whisper FAILED: {str(e)[:200]}")
cleanup()
report("after whisper freed")

# ---------------- 共享模型 campplus ----------------
try:
    from modelscope.pipelines import pipeline
    sv = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_zh-cn_16k-common")
    report("+ campplus (shared by both)")
    del sv
except Exception as e:
    lines.append(f"campplus FAILED: {str(e)[:200]}")
cleanup()

# ---------------- CT-Transformer 标点（旧系统） ----------------
try:
    from funasr import AutoModel
    pm = AutoModel(model="iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch", device="cuda")
    report("+ CT-Transformer punc (old sys)")
    del pm
except Exception as e:
    lines.append(f"CT-Transformer FAILED: {str(e)[:200]}")
cleanup()
report("after old-sys aux freed")

# ---------------- 新系统 FireRed 栈 ----------------
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config          # noqa: E402
from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig     # noqa: E402

asr = FireRedAsr2.from_pretrained(
    "aed", os.path.join(REPO, "pretrained_models", "FireRedASR2-AED"),
    FireRedAsr2Config(use_gpu=True, use_half=False, beam_size=3, nbest=1, return_timestamp=True))
v = report("FireRedASR2-AED fp32")
import numpy as np
seg = np.zeros(16000 * 20, dtype=np.float32)
asr.transcribe(["x"], [(16000, seg)], )
report("AED + 20s inference (peak)")

punc = FireRedPunc.from_pretrained(os.path.join(REPO, "pretrained_models", "FireRedPunc"),
                                   FireRedPuncConfig(use_gpu=True))
report("+ FireRedPunc")

from modelscope.pipelines import pipeline                                    # noqa: E402
sv = pipeline(task="speaker-verification", model="iic/speech_campplus_sv_zh-cn_16k-common")
report("+ campplus  =  NEW SYSTEM TOTAL")

# fp16 选项
asr.model.half()
cleanup()
report("AED converted to fp16 (same session)")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("written", OUT, flush=True)

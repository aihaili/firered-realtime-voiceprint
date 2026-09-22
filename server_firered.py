# -*- coding: utf-8 -*-
"""FireRedASR2S 实时转写 web 服务器：FireRedASR2-AED + FireRedPunc + campplus 声纹说话人分段。

架构（与 backup_whisper_20260917/server.py 的 whisper 版同构，ASR/标点后端换成 FireRedASR2S）：
- 浏览器发来 16kHz PCM16 单声道，服务端按 30ms 帧逐帧推进状态机（与到达速度无关）
- 说话开始 → 累积当前句（带 0.4s 前 padding）
- 句子结束（静音 0.7s / 超 20s / 超最大字数 / 挂起分段）→ 整句转写 + 声纹匹配，发 final
- 说话中每 1.2s 转写一次发 live（实时预览）
- ASR：FireRedASR2-AED（beam=3，词级时间戳，输入上限 60s，直接吃 numpy 无需落盘）
- 标点：FireRedPunc（BERT，替代原 CT-Transformer，F1 更高），失败降级规则补标点
- 声纹：campplus 192 维嵌入，实时比对声纹库（speakers_firered.json），词级分段
- 前端直接复用 index.html + audio-worklet.js（协议完全一致）

Usage:
    python server_firered.py
    端口 8766（FIRERED_PORT 可覆盖）
"""
import asyncio
import gc
import json
import os
import re
import shutil
import struct
import sys
import time
from collections import deque

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

BASE = os.path.dirname(os.path.abspath(__file__))
FIRERED_ROOT = os.path.join(BASE, "FireRedASR2S")
sys.path.insert(0, FIRERED_ROOT)

# 防御：DSH 等精简环境启动的进程没有 USER/USERNAME 环境变量，
# getpass.getuser() 会回退到 import pwd（Windows 无此模块）→ torch._inductor
# 缓存目录初始化失败 → transformers 导入链崩溃。这里兜底补上。
if not os.environ.get("USERNAME"):
    os.environ["USERNAME"] = "dsh"
if not os.environ.get("USER"):
    os.environ["USER"] = "dsh"

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config          # noqa: E402
from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig     # noqa: E402

HOST = "0.0.0.0"
PORT = int(os.environ.get("FIRERED_PORT", "8766"))
AED_DIR = os.path.join(FIRERED_ROOT, "pretrained_models", "FireRedASR2-AED")
PUNC_DIR = os.path.join(FIRERED_ROOT, "pretrained_models", "FireRedPunc")
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 精度：fp16 省一半显存、解码快 1.2~1.35x（实测 8/8 用例输出逐字一致）。
# ⚠️ 必须走 fp16 而不是 bf16：torchaudio.forced_align 不支持 bf16，而官方此处是裸 except，
# 会让词级时间戳静默退化成"均匀假时间戳"，拖累声纹切分。故先封掉 bf16 分支。
USE_HALF = os.environ.get("FIRERED_HALF", "1") == "1"
if USE_HALF:
    torch.cuda.is_bf16_supported = lambda *a, **k: False

# live 预览后端：aed=与 final 同质量；ctc=CTC 贪心（约 50x 快但会漏字）；
# hybrid=短句用 AED（预览准确、成本低），句子超过 LIVE_AED_MAX_SEC 后改用 CTC
# （AED live 的开销随句子长度二次方增长，20s 句子能吃掉一半 GPU，拖慢 final）
LIVE_ENGINE = os.environ.get("FIRERED_LIVE_ENGINE", "hybrid").lower()
LIVE_AED_MAX_SEC = float(os.environ.get("FIRERED_LIVE_AED_MAX_SEC", "6"))

# 端点检测参数（与原 whisper 系统一致，作为调优对照基线）
FRAME_SEC = 0.03
FRAME_SAMPLES = int(FRAME_SEC * SAMPLE_RATE)
SPEECH_RMS = 0.012
SILENCE_RMS = 0.006
SILENCE_END_SEC = 0.7
# 句子长度上限：AED 编码器开销随长度二次增长，13~20s 长句会 ① 显存暴涨（encoder
# 激活 + 缓存池不释放）② 束搜索解码器重复塌缩（同一短语反复输出）。10s 上限内实测
# 全部干净（12s 以下无塌缩），长独白会被强制切成 ≤10s 的连续气泡。
MAX_SENT_SEC = float(os.environ.get("FIRERED_MAX_SENT_SEC", "10.0"))
PRE_PAD_SEC = 0.4
LIVE_POLL_SEC = 1.2
MIN_LIVE_SEC = float(os.environ.get("FIRERED_MIN_LIVE_SEC", "0.8"))  # 置很大可关闭 live 预览
ASR_TIMING = os.environ.get("FIRERED_ASR_TIMING") == "1"   # 打印 fbank/模型耗时拆分
_feat_sec = [0.0]
MIN_FINAL_SEC = 0.5
PCM_WINDOW_SEC = 10
CAPTURE_MAX_SEC = 600

# 声纹 / 分段参数（沿用原系统已调好的阈值，均可用环境变量覆盖）
MAX_CHARS = int(os.environ.get("FIRERED_MAX_CHARS", "60"))
MATCH_THR = float(os.environ.get("FIRERED_MATCH_THR", "0.60"))    # >= 此值 → 归为同一人，否则新建
UPDATE_THR = float(os.environ.get("FIRERED_UPDATE_THR", "0.72"))  # 只有 >= 此值才更新质心（防污染）
MERGE_THR = float(os.environ.get("FIRERED_MERGE_THR", "0.78"))    # >= 此值 → 合并/吸收，不新建
MATCH_FLOOR = float(os.environ.get("FIRERED_MATCH_FLOOR", "0.52"))  # 灰区下限（弱质心容忍）
# 新建说话人的证据规则：长段证据充分可直接建；短段需第二次相似证据（防碎片化）
NEW_SPK_MIN_SEC = float(os.environ.get("FIRERED_NEW_SPK_MIN_SEC", "5.0"))
NEW_SPK_CONFIRM_THR = float(os.environ.get("FIRERED_NEW_SPK_CONFIRM", "0.65"))
PENDING_TTL = float(os.environ.get("FIRERED_PENDING_TTL", "120"))   # 待确认候选有效期（秒）
MERGE_PERIOD = float(os.environ.get("FIRERED_MERGE_PERIOD", "45"))  # 周期性自愈合并间隔（秒）

# ---- 定版（离线全局说话人聚类）----
# 在线判定受限于"只有一两秒的短段 + 只能看局部"，必然有错；停止录音后对整段会话做
# 全局聚类 + 重嵌入（把同一人的多段音频拼起来重算声纹）+ 库匹配，回发 revision 改正结果。
FINALIZE_ON_STOP = os.environ.get("FIRERED_FINALIZE_ON_STOP", "1") == "1"
CLUSTER_THR = float(os.environ.get("FIRERED_CLUSTER_THR", "0.55"))       # 全局聚类阈值（平均链接）
CLUSTER_MERGE_THR = float(os.environ.get("FIRERED_CLUSTER_MERGE_THR", "0.62"))  # 簇重嵌入后互并阈值
REEMBED_MIN_SEC = float(os.environ.get("FIRERED_REEMBED_MIN_SEC", "1.5"))  # 重嵌入最少音频
SESSION_MAX_SEC = 3600      # 会话音频保留上限（秒）
FINALIZE_PERIOD = float(os.environ.get("FIRERED_FINALIZE_PERIOD", "60"))  # 录制中自动定版间隔（0=只在停止时）
# 离线 diarization：滑窗声纹 + 全局聚类 + 把簇标签映射回每个词（再切分句子）
DIAR_WIN_SEC = float(os.environ.get("FIRERED_DIAR_WIN_SEC", "1.5"))
DIAR_HOP_SEC = float(os.environ.get("FIRERED_DIAR_HOP_SEC", "0.75"))
DIAR_MIN_RMS = 0.004
# 窗口"对谁都解释不了"的判定阈值：低于此余弦 → 可能是段级聚类漏掉的说话人（句内第二人）
UNEXPLAINED_THR = float(os.environ.get("FIRERED_UNEXPLAINED_THR", "0.35"))
UNEXPLAINED_MIN_WIN = int(os.environ.get("FIRERED_UNEXPLAINED_MIN_WIN", "4"))     # 最少窗口数
UNEXPLAINED_COHERE = float(os.environ.get("FIRERED_UNEXPLAINED_COHERE", "0.55"))  # 簇内自洽度
UNEXPLAINED_FAR = float(os.environ.get("FIRERED_UNEXPLAINED_FAR", "0.55"))        # 与已有说话人最小距离
MEDIAN_FILTER_SEC = float(os.environ.get("FIRERED_MEDIAN_FILTER_SEC", "3.0"))    # 段级中值滤波阈值
MIN_NEW_SPK_SEC = 2.0   # 音频短于此值不新建说话人，继承上一句
WORD_SV_MARGIN = 0.05   # 词级切分滞回：换人需领先此值
WORD_SV_MIN_SEG_SEC = 0.8  # 短于此值的段并入前段
# 句内声学变点分段（库无关）：词嵌入与当前段质心相似度低于此值 → 认为是换人
VOICE_CHANGE_THR = float(os.environ.get("FIRERED_VOICE_CHANGE_THR", "0.55"))
VOICE_MIN_SEG_SEC = float(os.environ.get("FIRERED_VOICE_MIN_SEG_SEC", "1.2"))
MIN_EMBED_SEC = 0.6     # 计算声纹的最短音频
PENDING_SPLIT_PAUSE = 0.2     # 挂起分段：出现 0.2s 静音即切
PENDING_SPLIT_MAX_SEC = 10.0  # 挂起分段兜底：10s 无停顿强制切
# 声纹嵌入模型：ERes2NetV2（3D-Speaker，192 维）。在本机 6 通电话实测中
# 关键段声纹对分离度 +0.084、EER 1.1%，优于 campplus（+0.061）。
# 可用环境变量回退：FIRERED_SV_MODEL=iic/speech_campplus_sv_zh-cn_16k-common
SV_MODEL_NAME = os.environ.get(
    "FIRERED_SV_MODEL", "iic/speech_eres2netv2_sv_zh-cn_16k-common")
SPEAKERS_FILE = os.environ.get("FIRERED_DB") or os.path.join(BASE, "speakers_firered.json")
LEGACY_SPEAKERS_FILE = os.path.join(BASE, "speakers.json")
SPK_COLORS = ["#07c160", "#1989fa", "#ff9800", "#e91e63", "#9c27b0", "#00bcd4", "#ff5722", "#607d8b"]

app = FastAPI()
asr_model = None        # FireRedASR2-AED
punc_model = None       # FireRedPunc
sv_model = None         # 声纹 pipeline（ERes2NetV2 / campplus，192 维）
clients: dict[int, dict] = {}

# ---- 声纹库（内存）----
speaker_db: dict[str, dict] = {}   # id -> {name, embeddings[list], centroid[np], samples[list], color}
_spk_counter = 0


# ===================== 转写 =====================
_CN_PUNCT = "，。！？；：、"
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


def normalize_punct(text: str, is_final: bool = False) -> str:
    """中文语境标点归一化（安全网）：
    - 中文相邻的半角逗号/问号/叹号/分号 → 全角
    - final 气泡结尾缺标点时补句号（live 预览不加）
    """
    if not text:
        return text
    out = []
    for i, ch in enumerate(text):
        if ch in ",!?;":
            prev_c = _CN_CHAR_RE.match(text[i - 1]) if i > 0 else None
            next_c = _CN_CHAR_RE.match(text[i + 1]) if i + 1 < len(text) else None
            if prev_c or next_c:
                out.append({",": "，", "?": "？", "!": "！", ";": "；"}[ch])
                continue
        out.append(ch)
    text = "".join(out).strip()
    if is_final and text and text[-1] not in _CN_PUNCT + ".,!?;":
        text += "。"
    return text


def _to_aed_input(audio: np.ndarray) -> np.ndarray:
    """AED 的 fbank 期望 **int16 量级**的采样值（kaldiio 读 wav 即此约定）。
    服务端音频是 [-1,1] float32 → 必须 ×32768，否则模型会当成静音（实测无输出）。
    注意：campplus 声纹走 [-1,1] 原始值，不要转换。"""
    return (audio * 32768.0).astype(np.float32)


def transcribe_live(audio: np.ndarray, sent_sec: float) -> str:
    """live 预览：按引擎选择后端。hybrid 模式下，句子超过 LIVE_AED_MAX_SEC 用 CTC。"""
    eng = LIVE_ENGINE
    if eng == "hybrid":
        eng = "ctc" if sent_sec > LIVE_AED_MAX_SEC else "aed"
    if eng == "ctc":
        try:
            t = transcribe_text_ctc(audio)
            if t:
                return t
        except Exception as e:
            print(f"[asr] ctc live error, 回退 AED: {e}", flush=True)
    return transcribe_text(audio)


def _release_vram() -> None:
    """释放 torch 缓存分配器持有的显存（每次推理后调用，避免空闲显存虚高）。

    torch 的 caching allocator 会保留推理峰值期间分配的所有块；不显式
    empty_cache() 的话，推理后这些块会一直占着显存。这里在每次推理后
    清空缓存 + 强制 GC，把空闲显存还给系统。
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[vram] release error: {e}", flush=True)


def transcribe_text(audio: np.ndarray) -> str:
    """AED 转写（live 预览用），返回原始文本。"""
    if asr_model is None:
        return ""
    try:
        results = asr_model.transcribe(["live"], [(SAMPLE_RATE, _to_aed_input(audio))])
    except Exception as e:
        print(f"[asr] live transcribe error: {e}", flush=True)
        return ""
    if not results:
        return ""
    _release_vram()
    return (results[0].get("text") or "").strip()


def transcribe_text_ctc(audio: np.ndarray) -> str:
    """CTC 分支贪心解码（一次前向出全文，约 1/50 于自回归解码），仅供 live 预览。
    实测 RTF 0.003~0.005，字符准确率约 90~95%（会漏个别字）。"""
    feats, lengths, _durs, _w, _u = asr_model.feat_extractor(
        [(SAMPLE_RATE, _to_aed_input(audio))], ["live"])
    if feats is None:
        return ""
    feats, lengths = feats.cuda(), lengths.cuda()
    if USE_HALF:
        feats = feats.half()
    eo, _el, _em = asr_model.model.encoder(feats, lengths)
    ids = asr_model.model.ctc(eo)[0].argmax(dim=-1).tolist()
    out, prev = [], None
    for i in ids:                       # CTC 折叠：去连续重复 + 去 blank(0)
        if i != prev and i != 0:
            out.append(i)
        prev = i
    text = asr_model.tokenizer.detokenize(out)
    _release_vram()
    return re.sub(r"(<blank>)|(<sil>)", "", text).lower().strip()


def transcribe_words(audio: np.ndarray) -> tuple[str, list]:
    """FireRedASR2-AED 转写 + 词级时间戳（final 用）。
    返回 (文本, words=[(start, end, word)])，words 与 whisper 版同构，供词级声纹切分。"""
    if asr_model is None:
        return "", []
    t0 = time.time()
    try:
        results = asr_model.transcribe(["final"], [(SAMPLE_RATE, _to_aed_input(audio))])
    except Exception as e:
        print(f"[asr] final transcribe error: {e}", flush=True)
        return "", []
    if ASR_TIMING:
        _free, _tot = torch.cuda.mem_get_info()
        print(f"[asr-time] dur={len(audio)/SAMPLE_RATE:.1f}s feat={_feat_sec[0]:.2f}s "
              f"total={time.time() - t0:.2f}s rtf={(time.time() - t0)/max(len(audio)/SAMPLE_RATE, 0.1):.3f} "
              f"vram_alloc={torch.cuda.memory_allocated()/2**30:.2f}G "
              f"reserved={torch.cuda.memory_reserved()/2**30:.2f}G "
              f"free={_free/2**30:.2f}G", flush=True)
    if not results:
        return "", []
    _release_vram()
    r = results[0]
    text = (r.get("text") or "").strip()
    words: list = []
    for tok, st, en in r.get("timestamp") or []:
        t = (tok or "").strip()
        if t:
            words.append((float(st), float(en), t))
    return text, words


def restore_punct(text: str) -> str:
    """final 文本标点恢复：优先 FireRedPunc（BERT），失败降级为规则补标点。"""
    if not text:
        return text
    if punc_model is not None:
        try:
            r = punc_model.process([text])
            out = r[0]["punc_text"] if isinstance(r, list) and r else ""
            if out and out.strip():
                return normalize_punct(out.strip(), is_final=True)
        except Exception as e:
            print(f"[punc] restore failed, fallback: {e}", flush=True)
    return normalize_punct(text, is_final=True)


# ===================== 声纹 =====================
def speaker_embed(audio: np.ndarray) -> np.ndarray | None:
    """192 维 L2 归一化声纹嵌入（ERes2NetV2 / campplus）。"""
    if sv_model is None:
        return None
    r = sv_model([audio], output_emb=True)
    emb = np.array(r["embs"][0], dtype=np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 0 else None


def _centroid(embs: list) -> np.ndarray | None:
    if not embs:
        return None
    m = np.mean(np.asarray(embs, dtype=np.float32), axis=0)
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else None


def _next_spk_id() -> str:
    global _spk_counter
    while True:
        sid = f"spk_{_spk_counter}"
        _spk_counter += 1
        if sid not in speaker_db:
            return sid


def _next_color() -> str:
    used = {sp["color"] for sp in speaker_db.values()}
    for col in SPK_COLORS:
        if col not in used:
            return col
    return SPK_COLORS[len(speaker_db) % len(SPK_COLORS)]


def match_speaker(emb: np.ndarray) -> tuple[str | None, float]:
    best, best_id = -1.0, None
    for sid, sp in speaker_db.items():
        c = sp["centroid"]
        if c is None:
            continue
        sim = float(np.dot(emb, c))
        if sim > best:
            best, best_id = sim, sid
    return best_id, best


def _match_decision(emb: np.ndarray) -> tuple[str | None, float, bool, bool]:
    """声纹归属判定。返回 (best_sid, sim, accept, gray)。
    - sim >= MATCH_THR           → 归类
    - 灰区 [MATCH_FLOOR, MATCH_THR) 且该人样本 < 3 条（质心不可靠，短句建库）→ 容忍归类并自愈质心
    - 其余（真陌生人，实测 0.07~0.35）→ 新建
    灰区容忍是为了修「首句只有 1.8s 就建库 → 质心差 → 后续同一人的 13s 句子 sim 0.585
    差 0.015 没达标 → 被拆成新说话人」这类碎片化。"""
    sid, sim = match_speaker(emb)
    if sid is None:
        return None, sim, False, False
    if sim >= MATCH_THR:
        return sid, sim, True, False
    n_samples = len(speaker_db[sid]["embeddings"])
    if sim >= MATCH_FLOOR and n_samples < 3:
        return sid, sim, True, True
    return sid, sim, False, False


def assign_speaker(emb: np.ndarray, text: str | None = None, force_update: bool = False) -> str:
    """把嵌入归入已有说话人或新建。更新声纹库并持久化。
    防污染：只有高置信度（>= UPDATE_THR）匹配才更新质心；灰区自愈情形才放宽。
    force_update=True（定版用）：簇嵌入是多秒音频拼接出来的高质量样本，匹配上就直接入库。
    新建时会先尝试与已有说话人合并（>= MERGE_THR），避免同一人被拆成多个 ID。"""
    sid, sim, accept, gray = _match_decision(emb)
    print(f"[sv] match best={sid} sim={sim:.3f} (thr={MATCH_THR})"
          f"{' 灰区自愈' if gray else ''}", flush=True)
    if accept:
        sp = speaker_db[sid]
        if sim >= UPDATE_THR or gray or force_update:
            sp["embeddings"].append(emb.tolist())
            if len(sp["embeddings"]) > 50:
                sp["embeddings"].pop(0)
            sp["centroid"] = _centroid(sp["embeddings"])
        if text:
            _add_sample(sid, text)
        save_speaker_db()
        return sid
    # 新建：先看是否与已有说话人足够像（碎片合并），否则真正新建
    if sid is not None and sim >= MERGE_THR:
        sp = speaker_db[sid]
        sp["embeddings"].append(emb.tolist())
        if len(sp["embeddings"]) > 60:
            sp["embeddings"] = sp["embeddings"][-60:]
        sp["centroid"] = _centroid(sp["embeddings"])
        if text:
            _add_sample(sid, text)
        save_speaker_db()
        print(f"[sv] absorb into {sid} (sim={sim:.3f} >= MERGE_THR={MERGE_THR})", flush=True)
        return sid
    sid = _next_spk_id()
    # 默认标签 speaker01/02/...：匿名但稳定，UI 直接显示，用户可随时改名
    try:
        _disp = f"speaker{int(sid.split('_')[1]):02d}"
    except Exception:
        _disp = sid
    speaker_db[sid] = {
        "name": _disp, "embeddings": [emb.tolist()], "centroid": emb.copy(),
        "samples": [], "color": _next_color(), "created": time.time(),
    }
    if text:
        _add_sample(sid, text)
    save_speaker_db()
    print(f"[sv] NEW speaker {sid}", flush=True)
    return sid


def _add_sample(sid: str, text: str):
    sp = speaker_db[sid]
    sp["samples"].append({"text": text[:60], "time": time.strftime("%H:%M:%S")})
    if len(sp["samples"]) > 5:
        sp["samples"].pop(0)


def _embed_range(audio: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    """取 [t0, t1] 音频段的 campplus 嵌入（词级声纹切分用）。"""
    if sv_model is None:
        return None
    a = audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)]
    if len(a) < MIN_EMBED_SEC * SAMPLE_RATE:
        return None
    r = sv_model([a], output_emb=True)
    emb = np.array(r["embs"][0], dtype=np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 0 else None


def _assign_words_to_speakers(sm: list, centroids: dict) -> list:
    """把每个词归到质心集合里最像的说话人（带滞回：换人需领先 WORD_SV_MARGIN）。"""
    ids = list(centroids)
    out = [None] * len(sm)
    cur = None
    for i, e in enumerate(sm):
        if e is None:
            continue
        if cur is None:
            cur = max(ids, key=lambda sid: float(np.dot(e, centroids[sid])))
        best_id, best_sim = cur, -2.0
        for sid in ids:
            sim = float(np.dot(e, centroids[sid]))
            if sid == cur:
                sim += WORD_SV_MARGIN
            if sim > best_sim:
                best_id, best_sim = sid, sim
        out[i] = best_id
    return out


def _merge_short_segments(n_w: int, marks: list, words: list) -> list:
    """按每词的说话人标记组装段；短于 WORD_SV_MIN_SEG_SEC 的段并入前段。"""
    valid = [i for i in range(n_w) if marks[i] is not None]
    if not valid:
        return []
    segs = []
    cur_start, cur_spk = valid[0], marks[valid[0]]
    for i in valid[1:]:
        if marks[i] != cur_spk:
            segs.append([cur_start, i, cur_spk])
            cur_start, cur_spk = i, marks[i]
    segs.append([cur_start, valid[-1] + 1, cur_spk])
    segs[0][0] = 0
    segs[-1][1] = n_w
    merged = []
    for s, e, sp in segs:
        if merged and (words[e - 1][1] - words[s][0]) < WORD_SV_MIN_SEG_SEC:
            merged[-1][1] = e
        else:
            merged.append([s, e, sp])
    return merged


def segment_by_voice(audio: np.ndarray, words: list) -> list:
    """把句子按**声学变化**切成说话人同质的段 —— 不依赖声纹库，因此能切出"句内出现陌生人"。

    返回 [(start, end, text, None)]；无法判断时返回整句一段。
    算法（顺序变点检测，支持任意人数）：
      1. 每个词取 ±0.3s 窗口算 campplus 嵌入，再 ±1 词平滑
      2. 顺序扫描：与"当前段质心"相似度 < VOICE_CHANGE_THR → 开新段
      3. 合并质心相似的相邻段（同一人隔开出现，如被噪声打断）
      4. 短于 VOICE_MIN_SEG_SEC 的段并入最像的邻居（避免把"嗯""对"切成独立段）

    为什么必须"库无关"：库内质心分类只能区分**已知的人**；一句里有几个陌生人时，
    每个词都会被塞给库里最像的那个人 → 整句被判成同一个已知说话人（实测踩到：
    女声+男声+英语三种声音的一句话被全标成"说话人3"）。
    """
    total = len(audio) / SAMPLE_RATE
    if len(words) < 3:
        txt = "".join(w[2] for w in words).strip()
        return [(0.0, total, txt, None)] if total else []
    n_w = len(words)
    embs = [_embed_range(audio, max(0.0, st - 0.3), min(total, en + 0.3))
            for st, en, _w in words]
    if sum(1 for e in embs if e is not None) < 3:
        txt = "".join(w[2] for w in words).strip()
        return [(0.0, total, txt, None)]
    sm = []
    for i in range(n_w):
        win = [embs[j] for j in range(max(0, i - 1), min(n_w, i + 2)) if embs[j] is not None]
        if not win:
            sm.append(None)
            continue
        m = np.mean(np.asarray(win, dtype=np.float32), axis=0)
        n = float(np.linalg.norm(m))
        sm.append(m / n if n > 0 else None)

    # ---- 2) 顺序变点检测 ----
    valid = [i for i in range(n_w) if sm[i] is not None]
    groups: list[list[int]] = [[valid[0]]]
    for i in valid[1:]:
        c = _centroid([sm[j] for j in groups[-1]])
        sim = float(np.dot(sm[i], c)) if c is not None else 1.0
        if sim < VOICE_CHANGE_THR:
            groups.append([i])          # 变点：开新段
        else:
            groups[-1].append(i)
    # ---- 3) 合并质心相似的相邻段（同一人隔开出现）----
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for k in range(len(groups) - 1):
            ca = _centroid([sm[j] for j in groups[k]])
            cb = _centroid([sm[j] for j in groups[k + 1]])
            if ca is None or cb is None:
                continue
            if float(np.dot(ca, cb)) >= VOICE_CHANGE_THR:
                groups[k] = groups[k] + groups[k + 1]
                del groups[k + 1]
                merged = True
                break
    # ---- 4) 短段并入最像的邻居 ----
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for k, g in enumerate(groups):
            dur = words[g[-1]][1] - words[g[0]][0]
            if dur >= VOICE_MIN_SEG_SEC:
                continue
            cg = _centroid([sm[j] for j in g])
            best, best_k = -2.0, None
            for k2 in (k - 1, k + 1):
                if 0 <= k2 < len(groups):
                    c2 = _centroid([sm[j] for j in groups[k2]])
                    if cg is not None and c2 is not None:
                        s = float(np.dot(cg, c2))
                        if s > best:
                            best, best_k = s, k2
            if best_k is not None:
                groups[best_k] = sorted(groups[best_k] + g)
                del groups[k]
                changed = True
                break
    # ---- 组装 ----
    out = []
    for k, g in enumerate(groups):
        s, e = g[0], g[-1]
        txt = "".join(words[j][2] for j in g).strip()
        if txt:
            out.append((words[s][0], words[e][1], txt, None))
    return out or [(0.0, total, "".join(w[2] for w in words).strip(), None)]


def merge_speakers(include_provisional: bool = True) -> int:
    """合并说话人，消除碎片化（启动时 + 周期性调用）。保留 id 小的一方。
    - 质心相似度 >= MERGE_THR 的两者 → 合并
    - 只有 1 条嵌入的「临时说话人」→ 只要与其他说话人 >= MATCH_THR 就并进去
      （这类条目多来自 2~5s 短段或噪声，是碎片化的主要来源）"""
    if len(speaker_db) < 2:
        return 0
    ids = sorted(speaker_db, key=lambda s: int(s.split("_")[1]) if s.startswith("spk_") else 0)
    merged = 0
    for i, a in enumerate(ids):
        if a not in speaker_db:
            continue
        for b in ids[i + 1:]:
            if b not in speaker_db:
                continue
            ca, cb = speaker_db[a]["centroid"], speaker_db[b]["centroid"]
            if ca is None or cb is None:
                continue
            sim = float(np.dot(ca, cb))
            na, nb = len(speaker_db[a]["embeddings"]), len(speaker_db[b]["embeddings"])
            thr = MERGE_THR
            if include_provisional and (na <= 1 or nb <= 1):
                thr = MATCH_THR          # 临时条目放宽到归类阈值
            if sim < thr:
                continue
            keep, drop = speaker_db[a], speaker_db[b]
            keep["embeddings"].extend(drop["embeddings"])
            if len(keep["embeddings"]) > 60:
                keep["embeddings"] = keep["embeddings"][-60:]
            keep["centroid"] = _centroid(keep["embeddings"])
            if not keep["name"] and drop["name"]:
                keep["name"] = drop["name"]
            for s in drop["samples"]:
                if s not in keep["samples"]:
                    keep["samples"].append(s)
            keep["samples"] = keep["samples"][-5:]
            del speaker_db[b]
            merged += 1
            if merged <= 6:
                print(f"[sv] merged {b} → {a} (sim={sim:.3f}, thr={thr:.2f}, "
                      f"n={na}/{nb})", flush=True)
    if merged:
        save_speaker_db()
        print(f"[sv] merged {merged} speakers -> {len(speaker_db)} total", flush=True)
    return merged


def load_speaker_db():
    global _spk_counter
    # 首次启动：从旧 whisper 系统的声纹库迁移（不改动原文件）
    if not os.path.exists(SPEAKERS_FILE) and os.path.exists(LEGACY_SPEAKERS_FILE):
        shutil.copy(LEGACY_SPEAKERS_FILE, SPEAKERS_FILE)
        print(f"[sv] seeded {SPEAKERS_FILE} from {LEGACY_SPEAKERS_FILE}")
    if not os.path.exists(SPEAKERS_FILE):
        return
    try:
        with open(SPEAKERS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        for sid, sp in data.get("speakers", {}).items():
            embs = sp.get("embeddings", [])
            speaker_db[sid] = {
                "name": sp.get("name", ""),
                "embeddings": embs,
                "centroid": _centroid(embs),
                "samples": sp.get("samples", []),
                "color": sp.get("color", _next_color()),
                "created": sp.get("created", time.time()),
            }
            num = int(sid.split("_")[1]) if sid.startswith("spk_") else 0
            _spk_counter = max(_spk_counter, num + 1)
        print(f"[sv] loaded {len(speaker_db)} speakers from {SPEAKERS_FILE}")
        merge_speakers()
    except Exception as e:
        print(f"[sv] load error: {e}")


def save_speaker_db():
    out = {"speakers": {}}
    for sid, sp in speaker_db.items():
        out["speakers"][sid] = {
            "name": sp["name"], "embeddings": sp["embeddings"],
            "samples": sp["samples"], "color": sp["color"], "created": sp["created"],
        }
    try:
        with open(SPEAKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)
    except Exception as e:
        print(f"[sv] save error: {e}")


def speaker_list() -> list:
    out = []
    for sid, sp in speaker_db.items():
        name = sp["name"]
        if not name:
            # 旧条目无默认名 → 补 speakerNN 风格显示名（不改库，仅展示）
            try:
                name = f"speaker{int(sid.split('_')[1]):02d}"
            except Exception:
                name = sid
        out.append({
            "id": sid, "name": name, "color": sp["color"],
            "count": len(sp["embeddings"]), "samples": sp["samples"][-3:],
        })
    out.sort(key=lambda x: x["id"])
    return out


# ===================== 定版：离线全局说话人聚类 =====================
def _pcm_slice(session_ba: bytearray, s0: int, s1: int) -> np.ndarray:
    """从会话录音里取 [s0, s1) 采样区间（会话时间轴，单位=采样点）。"""
    a, b = max(0, int(s0)) * 2, max(0, int(s1)) * 2
    return np.frombuffer(bytes(session_ba[a:b]), dtype=np.int16).astype(np.float32) / 32768.0


def _cluster_agglomerative(embs: list, thr: float) -> list:
    """平均链接层次聚类（余弦相似度 > thr 合并）。返回 [[idx,...], ...]。
    embs 里允许有 None（无效嵌入），会被单独放一边。"""
    idx_valid = [i for i, e in enumerate(embs) if e is not None]
    if not idx_valid:
        return []
    M = np.stack([embs[i] for i in idx_valid])
    n = len(idx_valid)
    sim = M @ M.T
    clusters = [[k] for k in range(n)]

    def avg_sim(a: list, b: list) -> float:
        return float(sim[np.ix_(a, b)].mean())

    while len(clusters) > 1:
        best, bi, bj = -2.0, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                s = avg_sim(clusters[i], clusters[j])
                if s > best:
                    best, bi, bj = s, i, j
        if best < thr:
            break
        clusters[bi] = clusters[bi] + clusters[bj]
        del clusters[bj]
    return [[idx_valid[k] for k in c] for c in clusters]


def _concat_cluster_audio(segs: list, cluster: list, session_ba: bytearray) -> np.ndarray:
    parts = []
    for i in sorted(cluster):
        s = _pcm_slice(session_ba, segs[i]["s0"], segs[i]["s1"])
        if len(s):
            parts.append(s)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    gap = np.zeros(int(0.25 * SAMPLE_RATE), dtype=np.float32)
    out = []
    for p in parts:
        out.extend([p, gap])
    return np.concatenate(out)


def diarize_windows(audio: np.ndarray, cache: dict | None = None) -> list:
    """对整段会话做滑窗声纹嵌入，返回 [(中心采样点, emb)]。带增量缓存（只算新音频）。
    这是"真离线 diarization"的第一步：细粒度（默认 1.5s/0.75s 跳）比"按句"更能捕捉换人。"""
    w = int(DIAR_WIN_SEC * SAMPLE_RATE)
    h = int(DIAR_HOP_SEC * SAMPLE_RATE)
    out = list(cache.get("embs", [])) if cache is not None else []
    upto = cache.get("upto", 0) if cache is not None else 0
    start = max(0, upto - w)                    # 从最后一个未覆盖的位置继续
    n = len(audio)
    s = start
    while s + w <= n:
        seg = audio[s:s + w]
        if float(np.sqrt(np.mean(seg ** 2))) >= DIAR_MIN_RMS:
            try:
                e = speaker_embed(seg)
            except Exception:
                e = None
            if e is not None:
                out.append((s + w // 2, e))
        s += h
    if cache is not None:
        cache["embs"] = out
        cache["upto"] = n
    return out


def finalize_session(segs: list, audio: np.ndarray, win_cache: dict | None = None,
                     session_start_ts: float = 0.0, verbose: bool = True) -> list:
    """离线定版（两段式，实测比纯窗口聚类稳得多）：

    A. **说话人集合**：对 final 段做全局聚类（段音频通常几秒到几十秒，声纹稳定）
       → 每簇拼接音频重嵌入 → 簇间互并 → 与声纹库比对/建库。
       （纯窗口聚类试过：1.5s 短窗抖动大，85s 会碎成 19 簇 —— 所以说话人必须按段定）
    B. **句内切分**：对整段会话滑窗算嵌入，把每个窗口分配到最近的那个说话人质心，
       再把标签映射到**每个词**（词中点找最近窗口）→ 按说话人重新切句。
       （在线按"阈值判定"切不出的事，这里靠全局质心 + 词级映射做到）

    返回气泡列表 [{"text","spk"}]，可能比原 final 段更多（一句里换人会拆开）。
    """
    if not segs:
        return []

    # ---------- A. 段级聚类 → 说话人集合 ----------
    clusters = _cluster_agglomerative([s.get("emb") for s in segs], CLUSTER_THR)
    if not clusters:
        return [{"text": s.get("text", ""), "spk": s.get("spk")} for s in segs]

    def _seg_audio(idx_list: list) -> np.ndarray:
        parts = [audio[max(0, segs[i]["s0"]):max(0, segs[i]["s1"])] for i in sorted(idx_list)]
        parts = [x for x in parts if len(x)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _embed(idx_list: list) -> np.ndarray | None:
        cat = _seg_audio(idx_list)
        if len(cat) >= REEMBED_MIN_SEC * SAMPLE_RATE:
            try:
                e = speaker_embed(cat)
                if e is not None:
                    return e
            except Exception as ex:
                print(f"[finalize] re-embed failed: {ex}", flush=True)
        return _centroid([segs[i].get("emb") for i in idx_list])

    cents = [_embed(c) for c in clusters]
    merged = True
    while merged and len(clusters) > 1:
        merged = False
        for i in range(len(clusters)):
            if cents[i] is None:
                continue
            for j in range(i + 1, len(clusters)):
                if cents[j] is None:
                    continue
                s_ij = float(np.dot(cents[i], cents[j]))
                if s_ij >= CLUSTER_MERGE_THR:
                    if verbose:
                        print(f"[finalize] 簇合并 sim={s_ij:.3f}", flush=True)
                    clusters[i] = sorted(clusters[i] + clusters[j])
                    cents[i] = _embed(clusters[i])
                    del clusters[j], cents[j]
                    merged = True
                    break
            if merged:
                break

    cl_spk, used = [], set()
    for ci, cl in enumerate(clusters):
        if cents[ci] is None:
            cl_spk.append(None)
            continue
        sample = segs[cl[0]].get("text", "")[:40]
        dur = sum(segs[k]["s1"] - segs[k]["s0"] for k in cl) / SAMPLE_RATE
        if dur < MIN_NEW_SPK_SEC and speaker_db:
            # 短簇（<2s）：嵌入噪声大、易低于阈值 → 新建即碎片（实测 1.3s「我操」
            # sim=0.557 被误建成 spk_6）。不新建，直接归最佳匹配（词级窗口仍按
            # 质心分配，真·第二人由 B 阶段"未解释窗口"路径兜底挖出）。
            sid, sim = match_speaker(cents[ci])
            if verbose:
                print(f"[finalize] 说话人簇#{ci}: {len(cl)} 段 {dur:.1f}s 短簇 → "
                      f"归 {sid} (sim={sim:.3f})  「{sample}」", flush=True)
            cl_spk.append(sid)
            if sid:
                used.add(sid)
            continue
        sid = assign_speaker(cents[ci], sample, force_update=True)
        cl_spk.append(sid)
        used.add(sid)
        if verbose:
            print(f"[finalize] 说话人簇#{ci}: {len(cl)} 段 {dur:.1f}s → {sid}  「{sample}」",
                  flush=True)

    # ---------- B. 窗口级分配 → 词级标签 → 重新切句 ----------
    live_cents = [c for c in cents if c is not None]
    live_ids = [cl_spk[i] for i, c in enumerate(cents) if c is not None]
    wins = diarize_windows(audio, win_cache) if live_cents else []
    win_t = np.array([w[0] for w in wins], dtype=np.float64) if wins else np.zeros(0)
    win_lab = []
    if wins:
        W = np.stack([w[1] for w in wins])
        # 1) 先按段级簇分配
        if live_cents:
            C = np.stack(live_cents)
            S = W @ C.T
            win_lab = S.argmax(axis=1).tolist()
            best = S.max(axis=1)
        else:
            C = np.zeros((0, W.shape[1]), dtype=np.float32)
            win_lab = [-1] * len(wins)
            best = np.zeros(len(wins))
        # 2) "对谁都解释不了"的窗口 → 自己聚类成新说话人
        #    （关键：如果一句话里混了第二个人，他在段级聚类里从未独立成簇，
        #      只有这一步才能把他挖出来。阈值取"跨人相似度上界"附近）
        odd = np.where(best < UNEXPLAINED_THR)[0]
        if len(odd) >= 2:
            sub_clusters = _cluster_agglomerative([wins[i][1] for i in odd], CLUSTER_THR)
            if verbose:
                print(f"[finalize] 未解释窗口 {len(odd)}/{len(wins)} → "
                      f"{len(sub_clusters)} 个新候选", flush=True)
            for sc in sub_clusters:
                if len(sc) < UNEXPLAINED_MIN_WIN:     # 太小 → 大概率是过渡/噪声
                    continue
                idx = [int(odd[k]) for k in sc]
                # 保守化门槛①：簇内必须自洽（平均互相似度够高）
                if len(idx) >= 2:
                    E = np.stack([wins[i][1] for i in idx])
                    Sim = E @ E.T
                    iu = np.triu_indices(len(idx), 1)
                    if float(Sim[iu].mean()) < UNEXPLAINED_COHERE:
                        continue
                parts = [audio[max(0, wins[i][0] - int(DIAR_WIN_SEC * SAMPLE_RATE) // 2):
                               wins[i][0] + int(DIAR_WIN_SEC * SAMPLE_RATE) // 2] for i in idx]
                parts = [x for x in parts if len(x)]
                cat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                e = None
                if len(cat) >= REEMBED_MIN_SEC * SAMPLE_RATE:
                    try:
                        e = speaker_embed(cat)
                    except Exception:
                        e = None
                if e is None:
                    e = _centroid([wins[i][1] for i in idx])
                if e is None:
                    continue
                # 保守化门槛②：必须与所有已有说话人质心都明显不像
                if live_cents and max(float(np.dot(e, c)) for c in live_cents) >= UNEXPLAINED_FAR:
                    continue
                sid = assign_speaker(e, "", force_update=True)
                used.add(sid)
                t0 = wins[min(idx)][0] / SAMPLE_RATE
                sample = ""
                for sg in segs:
                    if sg["s0"] / SAMPLE_RATE <= t0 <= sg["s1"] / SAMPLE_RATE:
                        sample = sg.get("text", "")[:24]
                        break
                if verbose:
                    print(f"[finalize] 新说话人 {sid}（{len(idx)} 窗 ≈"
                          f"{len(idx)*DIAR_HOP_SEC:.1f}s）来源：「{sample}」", flush=True)
                live_ids.append(sid)
                live_cents.append(e)
        # 3) k-means 式迭代精炼：用"分配到的窗口音频"重嵌入质心，再重新分配。
        #    必要性：段级质心可能是"混合质心"（整段里混了两个人），只分配不精炼会切不准。
        def _reembed_win_idx(idx) -> np.ndarray | None:
            parts = [audio[max(0, wins[i][0] - int(DIAR_WIN_SEC * SAMPLE_RATE) // 2):
                           wins[i][0] + int(DIAR_WIN_SEC * SAMPLE_RATE) // 2] for i in idx]
            parts = [x for x in parts if len(x)]
            if not parts:
                return None
            cat = np.concatenate(parts)
            if len(cat) >= REEMBED_MIN_SEC * SAMPLE_RATE:
                try:
                    e = speaker_embed(cat)
                    if e is not None:
                        return e
                except Exception:
                    pass
            return _centroid([wins[i][1] for i in idx])

        for _it in range(2):
            C = np.stack(live_cents)
            lab = (W @ C.T).argmax(axis=1)
            new_cents, new_ids = [], []
            for ci in range(len(live_cents)):
                idx = np.where(lab == ci)[0]
                e = _reembed_win_idx(idx) if len(idx) >= 1 else None
                if e is None:
                    continue
                new_cents.append(e)
                new_ids.append(live_ids[ci])
            if not new_cents:
                break
            live_cents, live_ids = new_cents, new_ids
        # 4) 最终分配 + 窗口级众数滤波（抑制相邻窗口在两人间抖动）
        C = np.stack(live_cents)
        win_lab = (W @ C.T).argmax(axis=1).tolist()
        if len(win_lab) >= 3:
            sm = []
            for i in range(len(win_lab)):
                vals = win_lab[max(0, i - 2):i + 3]
                sm.append(max(set(vals), key=vals.count))
            win_lab = sm

    bubbles = []
    for sg in segs:
        words = sg.get("words") or []
        if not words or not wins:
            bubbles.append({"text": sg.get("text", ""), "spk": sg.get("spk")})
            continue
        s0 = sg["s0"]
        labs = []
        for (st, en, _tok) in words:
            mid = s0 + int((st + en) / 2 * SAMPLE_RATE)
            k = int(np.argmin(np.abs(win_t - mid)))
            labs.append(live_ids[win_lab[k]])
        groups = []
        for i, lab in enumerate(labs):
            if groups and groups[-1][0] == lab:
                groups[-1][2] = i + 1
            else:
                groups.append([lab, i, i + 1])
        # 短组并入邻居（避免"嗯""对"单独成句）
        changed = True
        while changed and len(groups) > 1:
            changed = False
            for gi, g in enumerate(groups):
                if words[g[2] - 1][1] - words[g[1]][0] >= VOICE_MIN_SEG_SEC:
                    continue
                for gj in (gi - 1, gi + 1):
                    if 0 <= gj < len(groups):
                        groups[gj][1] = min(groups[gj][1], g[1])
                        groups[gj][2] = max(groups[gj][2], g[2])
                        del groups[gi]
                        changed = True
                        break
                if changed:
                    break
        # 段级中值滤波：A-B-A 且中间段较短 → 视为抖动，三并为一（用外层标签）
        changed2 = True
        while changed2 and len(groups) > 2:
            changed2 = False
            for gi in range(1, len(groups) - 1):
                if groups[gi - 1][0] != groups[gi + 1][0]:
                    continue
                dur = words[groups[gi][2] - 1][1] - words[groups[gi][1]][0]
                if dur >= MEDIAN_FILTER_SEC:
                    continue
                groups[gi - 1][2] = groups[gi + 1][2]
                del groups[gi + 1]
                del groups[gi]
                changed2 = True
                break
        for (lab, i0, i1) in groups:
            txt = "".join(words[j][2] for j in range(i0, i1)).strip()
            if not txt:
                continue
            # 词表本身没有标点 → 逐子段补标点（与在线路径一致，也顺带修中英混排空格）
            try:
                txt = restore_punct(txt)
            except Exception:
                pass
            bubbles.append({"text": txt, "spk": lab or sg.get("spk")})

    # ---------- 清理本次会话产生的在线碎片 ----------
    if session_start_ts > 0:
        junk = [sid for sid, sp in list(speaker_db.items())
                if sp.get("created", 0) >= session_start_ts and sid not in used]
        for sid in junk:
            del speaker_db[sid]
        if junk:
            save_speaker_db()
            print(f"[finalize] 清理在线碎片 {len(junk)} 个：{junk}", flush=True)
    if verbose:
        print(f"[finalize] 定版：{len(segs)} 段 → {len(bubbles)} 气泡，"
              f"{len({b['spk'] for b in bubbles if b['spk']})} 个说话人，库共 {len(speaker_db)} 人",
              flush=True)
    return bubbles


def _finalize_by_segment(segs: list, audio: np.ndarray, session_start_ts: float,
                         verbose: bool) -> list:
    """降级路径：窗口数不足时，退化为"按 final 段"的全局聚类（不拆句内换人）。"""
    embs = [s.get("emb") for s in segs]
    clusters = _cluster_agglomerative(embs, CLUSTER_THR)
    cents, keep = [], []
    for cl in clusters:
        parts = [audio[max(0, segs[i]["s0"]):max(0, segs[i]["s1"])] for i in sorted(cl)]
        parts = [x for x in parts if len(x)]
        cat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        e = None
        if len(cat) >= REEMBED_MIN_SEC * SAMPLE_RATE:
            try:
                e = speaker_embed(cat)
            except Exception:
                e = None
        if e is None:
            e = _centroid([embs[i] for i in cl])
        if e is not None:
            cents.append(e)
            keep.append(cl)
    merged = True
    while merged and len(keep) > 1:
        merged = False
        for i in range(len(keep)):
            for j in range(i + 1, len(keep)):
                if float(np.dot(cents[i], cents[j])) >= CLUSTER_MERGE_THR:
                    keep[i] = sorted(keep[i] + keep[j])
                    cents[i] = _centroid([embs[k] for k in keep[i]]) or cents[i]
                    del keep[j], cents[j]
                    merged = True
                    break
            if merged:
                break
    out, used = {}, set()
    for ci, cl in enumerate(keep):
        if cents[ci] is None:
            continue
        sid = assign_speaker(cents[ci], segs[cl[0]].get("text", "")[:40], force_update=True)
        used.add(sid)
        for k in cl:
            out[k] = sid
    if session_start_ts > 0:
        junk = [sid for sid, sp in list(speaker_db.items())
                if sp.get("created", 0) >= session_start_ts and sid not in used]
        for sid in junk:
            del speaker_db[sid]
        if junk:
            save_speaker_db()
    bubbles = [{"text": s.get("text", ""), "spk": out.get(i) or s.get("spk")}
               for i, s in enumerate(segs)]
    if verbose:
        print(f"[finalize] (按段降级) {len(segs)} 段 -> {len(bubbles)} 气泡，"
              f"库共 {len(speaker_db)} 人", flush=True)
    return bubbles


# ===================== 音频工具 =====================
def save_wav(path: str, pcm: bytes):
    n = len(pcm)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + n))
        f.write(b"WAVEfmt ")
        f.write(struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", n))
        f.write(pcm)


def _window_from(ba: bytearray) -> np.ndarray:
    return np.frombuffer(bytes(ba), dtype=np.int16).astype(np.float32) / 32768.0


def _window(c: dict, sec: float) -> np.ndarray:
    raw = bytes(c["buf"])
    n = min(int(sec * SAMPLE_RATE), len(raw) // 2)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw[-(n * 2):], dtype=np.int16).astype(np.float32) / 32768.0


def _stats(c: dict) -> dict:
    a = _window(c, 0.1)
    rms = float(np.sqrt(np.mean(a * a))) if len(a) else 0.0
    return {"rms": round(rms, 4), "recording": c["recording"], "in_speech": c["in_speech"]}


def new_client_state() -> dict:
    return {
        "ws": None,
        "buf": bytearray(),
        "history": bytearray(),
        "capture": deque(),
        "sentence": bytearray(),
        "in_speech": False,
        "silence_samples": 0,
        "cur_spk": None,
        "last_spk": None,
        "pending_split": False,
        "pending_spk_change": False,
        "pending_since": 0.0,
        "pending_new": None,      # 短段"待确认新说话人"候选（防碎片化）
        # ---- 定版用：会话录音 + 段清单（时间轴=采样点，绝对计数）----
        "session_audio": bytearray(),
        "session_segs": [],
        "session_len": 0,
        "session_base": 0,
        "recording": False,
        "busy": False,
        "lock": asyncio.Lock(),
    }


# ===================== 句子结束处理 =====================
async def emit_final_for(c: dict, ws, sentence_ba: bytearray):
    """对给定音频段：FireRedASR2-AED 转写 + 词级声纹切分 + FireRedPunc 标点 + 逐段发 final。"""
    audio = _window_from(sentence_ba)
    if len(audio) < MIN_FINAL_SEC * SAMPLE_RATE:
        return
    audio_sec = len(audio) / SAMPLE_RATE
    t0 = time.time()
    raw_text, words = await asyncio.to_thread(transcribe_words, audio)
    if not raw_text:
        return

    # 句内声学分段（库无关）：先按"声音变没变"切段，能切出句内出现的陌生人；
    # 每段的说话人仍由后面的段级复核按阈值判定（认识就归类，不认识就新建）。
    if sv_model is not None and audio_sec >= 2.0 and len(words) >= 3:
        segs = await asyncio.to_thread(segment_by_voice, audio, words)
    else:
        segs = [(0.0, audio_sec, raw_text, None)]
    n_seg = len(segs)
    if n_seg > 1:
        print(f"[sv] 句内声学分段：{n_seg} 段（检测到换人）", flush=True)

    # 整句嵌入只用于日志/参考（不再作为"是否切分"的门槛：
    # 三种声音混在一句时整句嵌入是大杂烩，用它当门槛会把整句判给一个人）
    sent_emb, sent_sim = None, -1.0
    if sv_model is not None and audio_sec >= MIN_EMBED_SEC:
        sent_emb = await asyncio.to_thread(_embed_range, audio, 0.0, audio_sec)
        if sent_emb is not None and speaker_db:
            _, sent_sim = await asyncio.to_thread(match_speaker, sent_emb)
            print(f"[sv] 整句 sim={sent_sim:.3f}（仅参考，不作为切分门槛）", flush=True)

    prev_spk = c.get("cur_spk") or c.get("last_spk")
    for t0s, t1s, seg_txt, seg_spk in segs:
        text = await asyncio.to_thread(restore_punct, seg_txt)
        if not text:
            continue
        seg_sec = t1s - t0s
        spk = None
        # 段级复核：不采信词级切分给出的库内 ID，一律用本段嵌入按阈值重新判定，
        # 相似度不够（< MATCH_THR）且段够长 → 走 assign_speaker 新建说话人。
        emb = None
        if sv_model is not None and seg_sec >= MIN_EMBED_SEC:
            is_whole = t0s <= 0.01 and abs(t1s - audio_sec) < 0.01
            emb = sent_emb if (is_whole and sent_emb is not None) else None
            if emb is None:
                emb = await asyncio.to_thread(_embed_range, audio, t0s, t1s)
        if emb is not None:
            sid, sim, accept, _gray = await asyncio.to_thread(_match_decision, emb)
            if accept:
                spk = await asyncio.to_thread(assign_speaker, emb, text)
            elif prev_spk is None:
                # 首句无上一句可继承 → 即使短也建档，否则第一个气泡没有标识
                spk = await asyncio.to_thread(assign_speaker, emb, text)
            elif seg_sec >= (VOICE_MIN_SEG_SEC if n_seg > 1 else NEW_SPK_MIN_SEC):
                # 句内声学变点是强证据 → 段够 VOICE_MIN_SEG_SEC 就直接新建；
                # 整句一段时仍用 5s 门槛 + 二次确认，防短段噪声造碎片
                spk = await asyncio.to_thread(assign_speaker, emb, text)
            else:
                # 短段未匹配：先记候选，需第二次相似证据才新建，否则继承上一句。
                # 防的是「同一人被 2~5s 短段的噪声嵌入拆成十几个 ID」这种碎片化。
                cands = [x for x in (c.get("pending_new") or [])
                         if time.time() - x["t"] < PENDING_TTL]
                hit = next((x for x in cands
                            if float(np.dot(emb, x["emb"])) >= NEW_SPK_CONFIRM_THR), None)
                if hit is not None:
                    print(f"[sv] 短段二次确认（与候选 sim="
                          f"{float(np.dot(emb, hit['emb'])):.3f}）→ 新建说话人", flush=True)
                    spk = await asyncio.to_thread(assign_speaker, emb, text)
                    # 回溯纠正：之前暂时继承错的气泡，按新质心复核后改标
                    ctr = speaker_db[spk]["centroid"]
                    fixed = 0
                    for x in cands:
                        if not x.get("relabel"):
                            continue
                        if float(np.dot(x["emb"], ctr)) >= NEW_SPK_CONFIRM_THR:
                            try:
                                await ws.send_text(json.dumps(
                                    {"type": "relabel", "text": x["text"], "spk": spk}))
                                fixed += 1
                            except Exception:
                                pass
                    if fixed:
                        print(f"[sv] 回溯纠正 {fixed} 个气泡 → {spk}", flush=True)
                    c["pending_new"] = []
                else:
                    cands.append({"emb": emb, "t": time.time(), "text": text,
                                  "relabel": True})
                    c["pending_new"] = cands[-6:]
                    print(f"[sv] seg {t0s:.1f}-{t1s:.1f}s sim={sim:.3f} 短段未匹配 → "
                          f"先继承，等二次确认", flush=True)
        if spk is None and seg_spk and not seg_spk.startswith("_"):
            spk = seg_spk          # 兜底：无法计算嵌入时用词级切分结果
        if spk is None:
            spk = prev_spk
        if spk:
            prev_spk = spk
            c["last_spk"] = spk
        # 定版用：记录本段（会话绝对时间轴 + 嵌入 + 在线判定结果）
        try:
            s1_abs = int(c.get("session_len", 0))
            s0_abs = s1_abs - len(sentence_ba) // 2
            c.setdefault("session_segs", []).append({
                "s0": s0_abs + int(t0s * SAMPLE_RATE),
                "s1": s0_abs + int(t1s * SAMPLE_RATE),
                "text": text, "emb": emb, "spk": spk,
                # 词级时间戳（段内相对秒）：定版时把簇标签映射到每个词，才能拆开句内换人
                "words": [(w[0] - t0s, w[1] - t0s, w[2]) for w in words
                          if t0s - 0.01 <= w[0] and w[1] <= t1s + 0.01],
            })
        except Exception as e:
            print(f"[finalize] record seg failed: {e}", flush=True)
        print(f"[final] spk={spk} t={t0s:.1f}-{t1s:.1f}s asr={time.time() - t0:.2f}s text={text}",
              flush=True)
        await ws.send_text(json.dumps({"type": "final", "text": text, "spk": spk}))


async def finalize_sentence(c: dict, ws):
    """当前句结束：转写+词级声纹切分+逐段发 final，并重置句子状态。"""
    sentence_ba = c["sentence"]
    c["sentence"] = bytearray()
    c["in_speech"] = False
    c["silence_samples"] = 0
    c["cur_spk"] = None
    c["pending_split"] = False
    c["pending_spk_change"] = False
    c["pending_since"] = 0.0
    await emit_final_for(c, ws, sentence_ba)


async def process_frames(c: dict):
    """把 buf 中所有完整 30ms 帧逐帧推进状态机。"""
    buf = c["buf"]
    frame_bytes = FRAME_SAMPLES * 2
    pre_bytes = int(PRE_PAD_SEC * SAMPLE_RATE) * 2
    while len(buf) >= frame_bytes:
        raw = bytes(buf[:frame_bytes])
        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        del buf[:frame_bytes]
        # 会话录音（定版用）：只记录"被状态机消费的帧"，保证与句子时间轴严格对齐
        sa = c["session_audio"]
        sa.extend(raw)
        c["session_len"] += FRAME_SAMPLES
        cap_bytes = int(SESSION_MAX_SEC * SAMPLE_RATE * 2)
        if len(sa) > cap_bytes:
            drop = len(sa) - cap_bytes
            del sa[:drop]
            c["session_base"] += drop // 2
        rms = float(np.sqrt(np.mean(frame * frame)))
        if not c["in_speech"]:
            if rms >= SPEECH_RMS:
                c["sentence"] = bytearray(c["history"])
                c["sentence"].extend(raw)
                c["in_speech"] = True
                c["silence_samples"] = 0
                c["cur_spk"] = None
        else:
            c["sentence"].extend(raw)
            if rms < SILENCE_RMS:
                c["silence_samples"] += FRAME_SAMPLES
            else:
                c["silence_samples"] = 0
            done = (c["silence_samples"] >= SILENCE_END_SEC * SAMPLE_RATE
                    or len(c["sentence"]) // 2 >= MAX_SENT_SEC * SAMPLE_RATE)
            if not done and c["pending_split"]:
                paused = c["silence_samples"] >= PENDING_SPLIT_PAUSE * SAMPLE_RATE
                timed_out = (c["pending_since"] > 0
                             and time.time() - c["pending_since"] >= PENDING_SPLIT_MAX_SEC)
                if paused or timed_out:
                    done = True
            if done:
                await finalize_sentence(c, c["ws"])
        c["history"].extend(raw)
        if len(c["history"]) > pre_bytes:
            del c["history"][: len(c["history"]) - pre_bytes]


async def run_finalize(c: dict, ws, reason: str = "stop") -> int:
    """对当前会话做一次离线全局定版（窗口级 diarization），回发整份改正后的气泡。"""
    segs = c.get("session_segs") or []
    if not segs or not FINALIZE_ON_STOP:
        return 0
    try:
        base = c.get("session_base", 0)
        shifted = [dict(s, s0=s["s0"] - base, s1=s["s1"] - base) for s in segs]
        audio = (np.frombuffer(bytes(c["session_audio"]), dtype=np.int16)
                 .astype(np.float32) / 32768.0)
        print(f"[finalize] ({reason}) 开始定版：{len(shifted)} 段 / "
              f"{len(audio)/SAMPLE_RATE:.1f}s 音频", flush=True)
        t_f = time.time()
        bubbles = await asyncio.to_thread(
            finalize_session, shifted, audio, c.setdefault("win_cache", {}),
            c.get("session_start_ts", 0.0), True)
        try:
            await ws.send_text(json.dumps({"type": "revision", "full": True,
                                           "bubbles": bubbles}))
        except Exception as e:
            print(f"[finalize] 回发失败（客户端可能已断开）：{e}", flush=True)
        print(f"[finalize] ({reason}) 完成 {time.time()-t_f:.1f}s，"
              f"定版 {len(bubbles)} 气泡", flush=True)
        return len(bubbles)
    except Exception as e:
        import traceback
        print(f"[finalize] failed: {e}", flush=True)
        traceback.print_exc()
        return -1


async def work_loop():
    last_merge = time.time()
    while True:
        await asyncio.sleep(LIVE_POLL_SEC)
        # 周期性声纹库自愈：把碎片（1 条嵌入的临时说话人）并回真实说话人
        if sv_model is not None and time.time() - last_merge >= MERGE_PERIOD:
            last_merge = time.time()
            try:
                await asyncio.to_thread(merge_speakers, True)
            except Exception as e:
                print(f"[sv] periodic merge error: {e}", flush=True)
        if asr_model is None:
            continue
        for cid in list(clients):
            c = clients[cid]
            try:
                await c["ws"].send_text(json.dumps({"type": "stats", **_stats(c)}))
            except Exception:
                pass
            if not c["recording"] or c["busy"]:
                continue
            # 录制中的周期性定版（只跑聚类+重嵌入，很轻）；周期随会话时长自适应，
            # 使定版开销始终约占会话时长的 1/10，长会话也不会被定版拖住
            if FINALIZE_PERIOD > 0 and c.get("session_segs"):
                spent = time.time() - c.get("last_finalize",
                                            c.get("session_start_ts", time.time()))
                period = max(FINALIZE_PERIOD, len(c["session_audio"]) / 2 / SAMPLE_RATE / 10)
                if spent >= period:
                    c["last_finalize"] = time.time()
                    try:
                        await run_finalize(c, c["ws"], "periodic")
                    except Exception as e:
                        print(f"[finalize] periodic error: {e}", flush=True)
            c["busy"] = True
            try:
                async with c["lock"]:
                    await process_frames(c)
                    if not c["in_speech"]:
                        continue
                    sent_sec = len(c["sentence"]) // 2 / SAMPLE_RATE
                    if sent_sec < MIN_LIVE_SEC:
                        continue
                    raw = await asyncio.to_thread(
                        transcribe_live, _window_from(c["sentence"]), sent_sec)
                    text = normalize_punct(raw, is_final=False) if raw else ""
                    if text:
                        await c["ws"].send_text(json.dumps({"type": "live", "text": text}))
                    if len(text) >= MAX_CHARS and not c["pending_split"]:
                        c["pending_split"] = True
                        c["pending_since"] = time.time()
                        c["pending_spk_change"] = False
            except Exception as e:
                import traceback
                print(f"work_loop error: {e}", flush=True)
                traceback.print_exc()
            finally:
                c["busy"] = False


# ===================== HTTP / WS =====================
@app.get("/")
async def index():
    return FileResponse(os.path.join(BASE, "index.html"))


@app.get("/audio-worklet.js")
async def worklet():
    return FileResponse(os.path.join(BASE, "audio-worklet.js"), media_type="application/javascript")


@app.get("/speakers")
async def get_speakers():
    return {"speakers": speaker_list()}


@app.post("/speakers/{spk_id}/name")
async def rename_speaker(spk_id: str, request: Request):
    body = await request.json()
    name = (body.get("name") or "").strip()[:20]
    if spk_id in speaker_db:
        speaker_db[spk_id]["name"] = name
        save_speaker_db()
        return {"ok": True, "id": spk_id, "name": name}
    return {"ok": False, "error": "not found"}


@app.post("/speakers/reset")
async def reset_speakers():
    speaker_db.clear()
    global _spk_counter
    _spk_counter = 0
    save_speaker_db()
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cid = id(ws)
    c = new_client_state()
    c["ws"] = ws
    clients[cid] = c
    await ws.send_text(json.dumps({"type": "status", "model_ready": asr_model is not None,
                                   "model": "FireRedASR2-AED", "sv_ready": sv_model is not None}))
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"]:
                data = msg["bytes"]
                c["buf"].extend(data)
                max_buf = int(PCM_WINDOW_SEC * SAMPLE_RATE * 2)
                if len(c["buf"]) > max_buf:
                    del c["buf"][: len(c["buf"]) - max_buf]
                if c["recording"]:
                    c["capture"].append(data)
                    cap_max = int(CAPTURE_MAX_SEC * SAMPLE_RATE * 2)
                    while c["capture"] and sum(len(x) for x in c["capture"]) > cap_max:
                        c["capture"].popleft()
            elif "text" in msg and msg["text"]:
                data = json.loads(msg["text"])
                if data.get("cmd") == "start":
                    async with c["lock"]:
                        c["buf"].clear()
                        c["history"].clear()
                        c["capture"].clear()
                        c["session_audio"] = bytearray()
                        c["session_segs"] = []
                        c["session_len"] = 0
                        c["session_base"] = 0
                        c["session_start_ts"] = time.time()
                        c["sentence"] = bytearray()
                        c["in_speech"] = False
                        c["silence_samples"] = 0
                        c["cur_spk"] = None
                        c["pending_split"] = False
                        c["pending_spk_change"] = False
                        c["pending_since"] = 0.0
                        c["recording"] = True
                        await ws.send_text(json.dumps({"type": "started"}))
                elif data.get("cmd") == "stop":
                    async with c["lock"]:
                        c["recording"] = False
                        try:
                            await process_frames(c)
                            if c["in_speech"]:
                                await finalize_sentence(c, ws)
                        except Exception as e:
                            print("stop flush error:", e)
                        c["sentence"] = bytearray()
                        c["in_speech"] = False
                        c["silence_samples"] = 0
                        c["cur_spk"] = None
                        c["pending_split"] = False
                        c["pending_spk_change"] = False
                        c["pending_since"] = 0.0
                        if c["capture"]:
                            cap = b"".join(c["capture"])
                            if len(cap) >= SAMPLE_RATE * 2:
                                save_wav(os.path.join(BASE, "mic_debug.wav"), cap)
                                print(f"[debug] saved mic audio ({len(cap)} bytes, "
                                      f"{len(cap)/2/SAMPLE_RATE:.1f}s)", flush=True)
                        # ---- 定版：整段会话离线全局聚类，回发改正后的说话人 ----
                        if FINALIZE_ON_STOP and c.get("session_segs"):
                            await run_finalize(c, ws, "stop")
                            c["session_segs"] = []
                        await ws.send_text(json.dumps({"type": "stopped"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import traceback
        print(f"ws_endpoint error: {e}", flush=True)
        traceback.print_exc()
    finally:
        clients.pop(cid, None)


async def main():
    global asr_model, punc_model, sv_model
    print(f"loading FireRedASR2-AED on {DEVICE} (half={USE_HALF}) ...", flush=True)
    asr_model = FireRedAsr2.from_pretrained(
        "aed", AED_DIR,
        FireRedAsr2Config(use_gpu=(DEVICE == "cuda"), use_half=USE_HALF,
                          beam_size=3, nbest=1, return_timestamp=True))
    print(f"ASR (FireRedASR2-AED) ready  VRAM {torch.cuda.memory_allocated()/2**30:.2f}G", flush=True)
    # ---- 解码加速 ----
    # graph: CUDA Graph 解码（最快，~4x；极少数句子 beam 打平时与官方差 1~2 字）
    # fast : 等价增量解码（~1.5x，与官方逐字一致）
    # official: 官方原版（基准）
    mode = os.environ.get("FIRERED_DECODE", "fast").lower()
    if mode in ("graph", "fast"):
        try:
            import firede_fast
            if firede_fast.patch_fast_decode():
                print("[decode] 增量解码已启用（cross-attn K/V 预算 + self-attn KV 缓存）", flush=True)
        except Exception as e:
            print(f"[decode] 增量解码不可用: {e}", flush=True)
    if mode == "graph":
        try:
            import firede_graph
            gbs = firede_graph.patch_graph_decode(
                asr_model, max_graphs=int(os.environ.get("FIRERED_MAX_GRAPHS", "24")),
                verbose=True)
            print(f"[decode] CUDA Graph 解码已启用（按 Ti 缓存图，上限 {gbs.max_graphs}，"
                  f"每图约 180MB）", flush=True)
        except Exception as e:
            print(f"[decode] CUDA Graph 不可用，回退增量解码: {e}", flush=True)
    print(f"[decode] mode={mode}  half={USE_HALF}  live={LIVE_ENGINE}  "
          f"VRAM {torch.cuda.memory_allocated()/2**30:.2f}G", flush=True)
    if ASR_TIMING:
        _orig_feat_call = asr_model.feat_extractor.__call__

        def _timed_feat_call(*a, **k):
            t = time.time()
            r = _orig_feat_call(*a, **k)
            _feat_sec[0] = time.time() - t
            return r

        asr_model.feat_extractor.__call__ = _timed_feat_call
        print("[asr-time] fbank 计时已开启", flush=True)
    try:
        punc_model = FireRedPunc.from_pretrained(PUNC_DIR, FireRedPuncConfig(use_gpu=(DEVICE == "cuda")))
        print("punctuation (FireRedPunc) ready", flush=True)
    except Exception as e:
        print(f"[punc] punctuation model load failed: {e} (继续，用规则补标点)", flush=True)
    try:
        from modelscope.pipelines import pipeline
        sv_model = pipeline(task="speaker-verification", model=SV_MODEL_NAME)
        print("speaker model ready", flush=True)
    except Exception as e:
        print(f"[sv] speaker model load failed: {e} (继续，无声纹分段)", flush=True)
    load_speaker_db()
    asyncio.get_event_loop().create_task(work_loop())
    await uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, log_level="warning",
                                        ws_ping_interval=30, ws_ping_timeout=300)).serve()


if __name__ == "__main__":
    asyncio.run(main())

# -*- coding: utf-8 -*-
"""FireRedASR2-AED beam search 的 CUDA Graph 实现（数学等价 + 消除 Python 派发开销）。

背景（本机实测）：
  - 解码一步 ≈ 240 个小算子，eager 耗时 ~18ms，其中 GPU 实际执行仅 0.8~5ms
  - 本机每算子派发开销约 42µs（Windows/WDDM + torch dispatcher），CPU 侧占 93%
  - 微基准：同样 240 个算子，CUDA Graph 回放 0.80ms vs eager 11.04ms → 13.8x
  - 官方实现另有致命浪费：每步每层都重投影整段编码器输出的 cross-attn K/V

本实现把**整步解码**（16 层 + beam 剪枝 + KV 重排）全部塞进一张 CUDA Graph：
  - 所有状态（KV 缓存、token、step、attn mask、scores）都是预先分配的静态缓冲，
    图内原地更新 → 回放之间状态自然延续，host 每步只做一次 graph.replay()
  - cross-attn K/V 每句只投影一次；self-attn 用增量 KV（尾部填充 + 加性 mask）
  - 图按编码器帧数 Ti 缓存（每种长度一张），上限可配

用法：
    from firede_graph import GraphBeamSearch
    gbs = GraphBeamSearch(aed_model)                 # aed_model = FireRedAsr2 实例
    gbs.patch()                                      # 之后 model.transcribe 自动走 graph
"""
import torch

CAP = 320          # 最大解码步数（模型 output_length_max=300，留余量）
TI_BUCKET = 64     # 编码器帧数按此粒度分桶（同一桶复用同一张图，避免每句重捕获）
NEG = -1e10        # 加性 mask 的屏蔽值（fp32；与官方 self.INF = 1e10 同量级）


def _neg_val(dtype):
    """加性 mask 的屏蔽值必须落在目标 dtype 范围内：
    fp16 最大 65504，直接用 -1e10 会 overflow 报错 → 用 -3e4（exp(-3e4)=0，效果相同）。"""
    return NEG if dtype == torch.float32 else -3.0e4


class _StepGraph:
    """单个 Ti（编码器帧数）对应的一张解码图 + 其静态状态缓冲。"""

    def __init__(self, dec, Ti, B, device, dtype=torch.float32):
        self.dec = dec
        self.Ti = Ti
        self.B = B
        self.NB = B                      # N=1 utterance
        self.device = device
        self.dtype = dtype
        H = dec.tgt_word_emb.embedding_dim
        self.H = H
        layers = dec.layer_stack
        self.n_layers = len(layers)
        heads = layers[0].self_attn.n_head
        d_k = layers[0].self_attn.d_k
        self.heads, self.d_k = heads, d_k

        z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)     # noqa: E731
        # ---- 每句只算一次的 cross-attn K/V（N=1，未复制到 beam）----
        self.ck = [z(1, heads, Ti, d_k) for _ in range(self.n_layers)]
        self.cv = [z(1, heads, Ti, d_k) for _ in range(self.n_layers)]
        # cross-attention 的加性 mask：Ti 桶对齐后，尾部填充帧要屏蔽掉
        self.cross_mask = z(1, 1, 1, Ti)

        # ---- 解码状态（静态缓冲，图内原地更新）----
        self.sk = [z(self.NB, heads, CAP, d_k) for _ in range(self.n_layers)]
        self.sv = [z(self.NB, heads, CAP, d_k) for _ in range(self.n_layers)]
        neg = _neg_val(dtype)
        self.neg = neg
        self.attn_mask = torch.full((1, 1, 1, CAP), neg, device=device, dtype=dtype)
        self.step_in = torch.zeros(1, dtype=torch.long, device=device)
        self.tok_in = torch.zeros(self.NB, 1, dtype=torch.long, device=device)
        self.ys_buf = torch.zeros(self.NB, CAP, dtype=torch.long, device=device)
        # ⚠️ scores/conf 必须 fp32：官方 batch_beam_search 的 scores 是 .float()（fp32），
        # 每步累加 fp16 的 t_topB_scores 时自动提升为 fp32。若用 fp16 累加，
        # 分数在 |s|≈20~30 处 ulp≈0.016，50 步随机游走误差可达 ~0.05，
        # 足以翻转 beam topk 的接近比分 → 轨迹漂移 → 在循环吸引子音频上塌缩。
        self.scores = torch.zeros(self.NB, 1, device=device, dtype=torch.float32)
        self.conf = torch.zeros(self.NB, CAP, device=device, dtype=torch.float32)
        self.is_fin = z(self.NB, 1)
        # 调试缓冲：每步的 t_scores 分布（供逐步 diff 定位数值分歧），52KB 可忽略
        self.t_scores_buf = z(self.NB, dec.tgt_word_emb.num_embeddings)
        self.fin_mask = torch.tensor([0.0] + [neg] * (B - 1), device=device,
                                     dtype=dtype).view(1, B).repeat(self.NB, 1)
        self.stride = (B * torch.arange(1, device=device)).view(1, 1).repeat(1, B).reshape(self.NB)

        # 位置编码表 (CAP, H)：与官方 PositionalEncoding 同一批数值
        pe_full = dec.positional_encoding(torch.zeros(1, CAP, dtype=torch.long, device=device))
        self.pe_all = pe_full[0].to(dtype).contiguous()               # (CAP, H)

        self.graph = None
        self.param = {}          # 捕获时固化的超参
        self.reset()

    # ---------------- 状态复位 ----------------
    def reset(self):
        for t in (self.sk, self.sv):
            for x in t:
                x.zero_()
        self.attn_mask.fill_(self.neg)
        self.step_in.zero_()
        self.tok_in.fill_(self.dec.sos_id)
        self.ys_buf.zero_()
        self.ys_buf[:, 0] = self.dec.sos_id
        # 官方初值: scores = [0, -INF, -INF]（N=1 时仅 beam0 存活，其余 -INF）。
        # 注意取 fin_mask 的**第一行**（[0, neg, neg]）：若误取第一列 fin_mask[:, :1]
        # 会得到 [0, 0, 0] → 三个 beam 从第 0 步起历史/分数全同 → topk 永远选中同一
        # token → beam 搜索永久退化为贪心解码，在低质量音频上直接循环塌缩。
        self.scores.copy_(self.fin_mask[0].view(self.NB, 1))
        self.conf.zero_()
        self.is_fin.zero_()

    # ---------------- 图内一步 ----------------
    def _step(self, softmax_smoothing, eos_penalty, length_penalty):
        dec = self.dec
        B, NB, device = self.B, self.NB, self.device
        t = self.step_in                                       # (1,)

        self.attn_mask.scatter_(3, t.view(1, 1, 1, 1), 0.0)     # 揭示位置 t

        pe = self.pe_all.index_select(0, t)                     # (1, H)
        x = dec.tgt_word_emb(self.tok_in) * dec.scale + pe.unsqueeze(0)   # (NB,1,H)

        for i, layer in enumerate(dec.layer_stack):
            # ---- self-attention（增量 KV，尾部填充 + 加性 mask）----
            residual = x
            h = layer.self_attn_norm(x)
            att = layer.self_attn
            q = att.w_qs(h).view(NB, 1, att.n_head, att.d_k).transpose(1, 2)
            k_new = att.w_ks(h).view(NB, 1, att.n_head, att.d_k).transpose(1, 2)
            v_new = att.w_vs(h).view(NB, 1, att.n_head, att.d_k).transpose(1, 2)
            self.sk[i].index_copy_(2, t, k_new)
            self.sv[i].index_copy_(2, t, v_new)
            o = torch.matmul(q, self.sk[i].transpose(-2, -1)) / att.attention.temperature
            o = torch.softmax(o + self.attn_mask, dim=-1)
            o = torch.matmul(o, self.sv[i])
            o = o.transpose(1, 2).reshape(NB, 1, att.d_model)
            x = residual + att.fc(o)

            # ---- cross-attention（K/V 已预算，单 utterance 广播到 beam）----
            residual = x
            h = layer.cross_attn_norm(x)
            catt = layer.cross_attn
            qc = catt.w_qs(h).view(NB, 1, catt.n_head, catt.d_k).transpose(1, 2)
            kc = self.ck[i].expand(NB, -1, -1, -1)
            vc = self.cv[i].expand(NB, -1, -1, -1)
            oc = torch.matmul(qc, kc.transpose(-2, -1)) / catt.attention.temperature
            oc = oc + self.cross_mask          # (1,1,1,Ti) 广播：屏蔽 Ti 桶对齐的填充帧
            oc = torch.softmax(oc, dim=-1)
            oc = torch.matmul(oc, vc)
            oc = oc.transpose(1, 2).reshape(NB, 1, catt.d_model)
            x = residual + catt.fc(oc)

            # ---- MLP ----
            x = x + layer.mlp(layer.mlp_norm(x))

        x = dec.layer_norm_out(x)
        logits = dec.tgt_word_prj(x[:, 0])                       # (NB, V)
        t_scores = torch.log_softmax(logits / softmax_smoothing, dim=-1)
        if eos_penalty != 1.0:
            t_scores[:, dec.eos_id] = t_scores[:, dec.eos_id] * eos_penalty
        self.t_scores_buf.copy_(t_scores)

        t_top_scores, t_top_ys = torch.topk(t_scores, k=B, dim=1)          # (NB,B)
        # finished 的 beam：分数归零 / 强制 EOS（与官方 set_finished_* 等价）
        t_top_scores = t_top_scores * (1 - self.is_fin) + self.fin_mask * self.is_fin
        t_top_ys = t_top_ys * (1 - self.is_fin.long()) + dec.eos_id * self.is_fin.long()

        # ⚠️ 全部原地写入：CUDA Graph 捕获的是缓冲地址，Python 重绑定会让状态跨回放失效
        acc = (self.scores + t_top_scores).reshape(1, B * B)               # (1, B*B)
        s2, ids = torch.topk(acc, k=B, dim=1)                             # (1,B)
        self.scores.copy_(s2.reshape(self.NB, 1))
        row = (ids // B).view(self.NB) + self.stride                       # (NB,)

        # ---- 按 beam 重排全部状态（图内原地）----
        self.ys_buf.copy_(self.ys_buf.index_select(0, row))
        self.conf.copy_(self.conf.index_select(0, row))
        for i in range(self.n_layers):
            self.sk[i].copy_(self.sk[i].index_select(0, row))
            self.sv[i].copy_(self.sv[i].index_select(0, row))

        t_ys = torch.gather(t_top_ys.reshape(1, B * B), 1, ids).reshape(self.NB, 1)
        self.ys_buf.index_copy_(1, (t + 1).view(1), t_ys)
        t_cf = torch.gather(t_top_scores.reshape(1, B * B), 1, ids).reshape(self.NB, 1)
        # 官方: t_confidences = exp(fp16) 后 cat 进 fp32 confidences（值仍是 fp16 量化）
        self.conf.index_copy_(1, (t + 1).view(1), torch.exp(t_cf).float())

        self.is_fin.copy_(t_ys.eq(dec.eos_id).float())
        self.tok_in.copy_(t_ys)
        self.step_in.add_(1)

    # ---------------- 捕获 ----------------
    def capture(self, softmax_smoothing, eos_penalty, length_penalty):
        self.param = dict(ss=softmax_smoothing, ep=eos_penalty, lp=length_penalty)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                 # 预热（避免捕获期惰性初始化）
            for _ in range(3):
                self._step(softmax_smoothing, eos_penalty, length_penalty)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.reset()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step(softmax_smoothing, eos_penalty, length_penalty)
        self.graph = g
        torch.cuda.synchronize()
        self.reset()

    def set_cross(self, eo, ck, cv, valid):
        """填入本句的 cross-attn K/V。Ti 桶比实际帧数长时，尾部补零并用 mask 屏蔽。"""
        if valid < self.Ti:
            self.cross_mask[..., valid:].fill_(self.neg)
        else:
            self.cross_mask.fill_(0.0)
        d = min(valid, self.Ti)
        for i in range(self.n_layers):
            self.ck[i][:, :, :d].copy_(ck[i][:, :, :d])
            self.cv[i][:, :, :d].copy_(cv[i][:, :, :d])
            if d < self.Ti:
                self.ck[i][:, :, d:].zero_()
                self.cv[i][:, :, d:].zero_()

    def run(self, max_steps, check_every=4):
        """回放解码（cross K/V 由 set_cross 预先填好）。"""
        self.reset()
        torch.cuda.synchronize()
        steps = min(max_steps, CAP - 1)
        done = 0
        for t in range(steps):
            self.graph.replay()
            done = t + 1
            if (t + 1) % check_every == 0:
                if bool(self.is_fin.all().item()):
                    break
        torch.cuda.synchronize()
        return self.ys_buf[:, :done + 1].clone(), self.scores.clone(), self.conf[:, :done + 1].clone(), done


class GraphBeamSearch:
    """把 FireRedASR2AED 的 beam search 换成 CUDA Graph 版（按 Ti 缓存图）。"""

    graph_runs = 0
    fallbacks = 0

    def __init__(self, aed, max_graphs=24, check_every=4, verbose=True):
        self.aed = aed
        self.dec = aed.model.decoder
        self.enc = aed.model.encoder
        self.max_graphs = max_graphs
        self.check_every = check_every
        self.verbose = verbose
        self.graphs: dict[int, _StepGraph] = {}
        self.device = next(aed.model.parameters()).device

    # 每句一次：编码器 + cross K/V 投影
    def _encode(self, feats, lengths):
        eo, el, em = self.enc(feats, lengths)
        ck, cv = [], []
        for layer in self.dec.layer_stack:
            catt = layer.cross_attn
            k = catt.w_ks(eo).view(1, -1, catt.n_head, catt.d_k).transpose(1, 2)
            v = catt.w_vs(eo).view(1, -1, catt.n_head, catt.d_k).transpose(1, 2)
            ck.append(k.contiguous())
            cv.append(v.contiguous())
        return eo, el, em, ck, cv

    def _get_graph(self, Ti):
        g = self.graphs.get(Ti)
        if g is None:
            if len(self.graphs) >= self.max_graphs:
                oldest = next(iter(self.graphs))
                del self.graphs[oldest]
            B = max(1, int(self.aed.config.beam_size))     # 与官方 beam 数一致（勿写死）
            dtype = next(self.dec.parameters()).dtype     # 跟随模型精度（fp32/fp16）
            g = _StepGraph(self.dec, Ti, B, self.device, dtype)
            g.capture(self.aed.config.softmax_smoothing,
                      self.aed.config.eos_penalty,
                      self.aed.config.aed_length_penalty)
            self.graphs[Ti] = g
            if self.verbose:
                print(f"[graph] captured Ti={Ti} beam={B} ({len(self.graphs)} graphs)", flush=True)
        return g

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path):
        """签名与 FireRedAsr2.transcribe 一致（AED 路径）。"""
        import re
        batt_uttid = list(batch_uttid)
        feats, lengths, durs, wavs, uttids = self.aed.feat_extractor(batch_wav_path, batch_uttid)
        if feats is None:
            return [{"uttid": u, "text": ""} for u in batt_uttid]
        feats, lengths = feats.cuda(), lengths.cuda()
        assert feats.size(0) == 1, "graph 版仅支持 batch=1"
        # 半精度模型要同步转换输入（官方 asr.py 在 use_half 时做同样的事）
        if self.aed.config.use_half:
            feats = feats.half()

        eo, el, em, ck, cv = self._encode(feats, lengths)
        # 有 padding 帧时加性 mask 不成立 → 退回快速 eager 解码
        if bool(em.eq(0).any()):
            torch.cuda.synchronize()
            type(self).fallbacks += 1
            if self.verbose and type(self).fallbacks <= 3:
                print(f"[graph] 检测到 padding 帧 → 回退 eager（第 {type(self).fallbacks} 次）",
                      flush=True)
            return self.aed._graph_fallback(batch_uttid, batch_wav_path)
        type(self).graph_runs += 1

        Ti = eo.size(1)
        Ti_b = ((Ti + TI_BUCKET - 1) // TI_BUCKET) * TI_BUCKET   # 桶对齐 → 大幅减少图捕获次数
        g = self._get_graph(Ti_b)
        g.set_cross(eo, ck, cv, Ti)
        cfg = self.aed.config
        # 与官方 batch_beam_search 对齐: maxlen = decode_max_len 或 Ti（实际编码器帧数），
        # 而不是固定 CAP-1（319 步比官方的 Ti≈25×秒数 宽松，会放大循环空间）
        max_steps = cfg.decode_max_len if cfg.decode_max_len > 0 else Ti
        max_steps = min(max_steps, CAP - 1)
        ys, scores, conf, steps = g.run(max_steps, self.check_every)

        # ---- 收尾：与官方 batch_beam_search 尾段 + asr.py 完全对齐 ----
        dec = self.dec
        ys_lengths = (ys != dec.eos_id).sum(dim=-1).int().float()      # (NB,)
        sc = scores.view(-1)                                          # (B,)
        if cfg.aed_length_penalty > 0.0:
            pen = torch.pow((5 + ys_lengths) / 6.0, cfg.aed_length_penalty)
            sc = sc / pen
        best = int(torch.argmax(sc).item())                            # nbest=1
        yseq = ys[best][1:int(ys_lengths[best].item())].contiguous()
        cf = conf[best][1:int(ys_lengths[best].item())]
        confidence = float(cf.mean().item()) if cf.numel() else 0.0
        dur = sum(durs)

        hyp = {"yseq": yseq, "confidence": torch.tensor(confidence)}
        if cfg.return_timestamp:
            # 与官方一致：用 CTC 强制对齐拿词级时间戳
            self.aed.model.get_token_timestamp_torchaudio(eo, el, [[hyp]])
        hyp_ids = [int(v) for v in yseq.cpu()]
        text = dec_tokenizer(self.aed, hyp_ids)

        out = {"uttid": batt_uttid[0], "text": text,
               "confidence": round(confidence, 3),
               "dur_s": round(dur, 3), "rtf": "0.0000"}
        if isinstance(wavs[0], str):
            out["wav"] = wavs[0]
        if cfg.return_timestamp:
            out["timestamp"] = self.aed._get_and_fix_timestamp(hyp, hyp_ids, dur)
        return [out]


def dec_tokenizer(aed, ids):
    """用官方 tokenizer 反解文本（与 asr.py 的 detokenize + lower 一致）。"""
    import re
    text = aed.tokenizer.detokenize(ids)
    return re.sub(r"(<blank>)|(<sil>)", "", text).lower()


def patch_graph_decode(aed, **kw):
    """给已加载的 FireRedAsr2 实例挂上 CUDA Graph 解码。"""
    gbs = GraphBeamSearch(aed, **kw)
    aed._graph_fallback = aed.transcribe          # 保留原路径做降级
    aed.graph_beam_search = gbs
    aed.transcribe = gbs.transcribe
    return gbs

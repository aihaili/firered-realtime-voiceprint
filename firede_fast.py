# -*- coding: utf-8 -*-
"""FireRedASR2-AED 解码器增量优化（数学等价，不改变输出）。

官方 `TransformerDecoder.batch_beam_search` 每一步都对**整段编码器输出**重算
cross-attention 的 K/V 投影（16 层 × T=1998 帧 × d=1280），每步约 210 GFLOP，
100 步就是 20 TFLOP —— 实测占了整条转写链路 2.1s 中的 ~2.0s（编码器只要 0.059s）。

本模块用标准增量解码重写：
1. cross-attn K/V 每句只投影一次，且**不给每个 beam 复制**（用 (N,B) 广播算 attention）
2. self-attn 用增量 KV 缓存（官方每步重算整个前缀）
3. 前缀输出缓存（DecodeLayer 的 cat(cache, x)）官方只在最后取 [-1] 用，直接省掉

用法：
    from firede_fast import patch_fast_decode
    patch_fast_decode()          # 之后再调 model.transcribe 即走快速路径
"""
import torch
import torch.nn.functional as F

# 诊断计数器：本次解码跑了多少步（maxlen 是上限，实际步数受"全部 beam 出 EOS"控制）
LAST_STEPS = 0


def _fast_batch_beam_search(self, encoder_outputs, src_masks, beam_size=1, nbest=1,
                            decode_max_len=0, softmax_smoothing=1.0, length_penalty=0.0,
                            eos_penalty=1.0, elm=None, elm_weight=0.0):
    if elm is not None and elm_weight > 0.0:
        raise NotImplementedError("fast decode 不支持外部 LM，请用原始实现")

    B = beam_size
    N, Ti, H = encoder_outputs.size()
    device = encoder_outputs.device
    maxlen = decode_max_len if decode_max_len > 0 else Ti
    assert eos_penalty > 0.0
    NB = N * B

    # ---------- 1) cross-attention K/V：每个 utterance 只投影一次 ----------
    # 官方把 encoder_outputs 沿 beam 复制成 (N*B, Ti, H) 并**每步重投影**；
    # 同一 utterance 的 B 个 beam 内容完全相同 → 只保留 (N, heads, Ti, d_k)。
    enc0 = encoder_outputs.reshape(N, Ti, H)                        # (N, Ti, H)，未复制到 beam
    cross_k, cross_v = [], []
    for layer in self.layer_stack:
        att = layer.cross_attn
        k = att.w_ks(enc0).view(N, -1, att.n_head, att.d_k).transpose(1, 2)
        v = att.w_vs(enc0).view(N, -1, att.n_head, att.d_k).transpose(1, 2)
        cross_k.append(k)                                          # (N, heads, Ti, d_k)
        cross_v.append(v)
    # 官方 mask: src_masks (N,1,Ti) → unsqueeze(1) → (N*B,1,1,Ti) → eq(0)
    cross_mask = src_masks.reshape(N, 1, Ti).eq(0).view(N, 1, 1, 1, Ti)   # True = 屏蔽
    # ⚠️ 只在循环外做一次同步判断：循环内 bool(tensor) 会每一步每层触发 GPU→CPU 同步
    #（实测 16 层 × 86 步 = 1376 次同步 → 每步 18.7ms，比真实算力慢 200 倍）
    use_cross_mask = bool(cross_mask.any())

    def cross_attn_out(i, q):
        """q: (NB, heads, 1, d_k) → out: (NB, heads, 1, d_k)，利用 (N,B) 广播避免复制 K/V"""
        att = self.layer_stack[i].cross_attn
        qn = q.view(N, B, att.n_head, 1, att.d_k)
        k = cross_k[i].unsqueeze(1)                                 # (N,1,heads,Ti,d_k)
        scores = torch.matmul(qn, k.transpose(-2, -1)) / att.attention.temperature
        if use_cross_mask:
            scores = scores.masked_fill(cross_mask, -att.attention.INF)
            scores = torch.softmax(scores, dim=-1).masked_fill(cross_mask, 0.0)
        else:
            scores = torch.softmax(scores, dim=-1)
        v = cross_v[i].unsqueeze(1)
        out = torch.matmul(scores, v)                               # (N,B,heads,1,d_k)
        return out.reshape(NB, att.n_head, 1, att.d_k)

    # ---------- 2) 初始化 beam ----------
    ys = torch.ones(NB, 1).fill_(self.sos_id).long().to(device)
    t_ys = ys.clone()
    confidences = torch.zeros(NB, 1).float().to(device)
    self_kv = [None] * self.n_layers            # 增量 self-attn KV
    scores = torch.tensor([0.0] + [-self.INF] * (B - 1)).float().to(device)
    scores = scores.repeat(N).view(NB, 1)
    is_finished = torch.zeros_like(scores)

    stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(NB).to(device)

    # ---------- 3) 增量自回归解码 ----------
    global LAST_STEPS
    LAST_STEPS = 0
    for t in range(maxlen):
        LAST_STEPS = t + 1
        # 只嵌入最新 token（官方每步重新嵌入整段前缀）
        pe = self.positional_encoding(ys)[:, -1:, :]
        x = self.dropout(self.tgt_word_emb(ys[:, -1:]) * self.scale + pe)   # (NB,1,H)

        for i, layer in enumerate(self.layer_stack):
            # --- self-attention（增量 KV）---
            residual = x
            h = layer.self_attn_norm(x)
            att = layer.self_attn
            q = att.w_qs(h).view(NB, -1, att.n_head, att.d_k).transpose(1, 2)
            k_new = att.w_ks(h).view(NB, -1, att.n_head, att.d_k).transpose(1, 2)
            v_new = att.w_vs(h).view(NB, -1, att.n_head, att.d_k).transpose(1, 2)
            if self_kv[i] is None:
                self_kv[i] = (k_new, v_new)
            else:
                self_kv[i] = (torch.cat([self_kv[i][0], k_new], dim=2),
                              torch.cat([self_kv[i][1], v_new], dim=2))
            k_all, v_all = self_kv[i]
            # query 是最后一个位置，因果掩码天然满足 → 无需 mask
            o = torch.matmul(q, k_all.transpose(-2, -1)) / att.attention.temperature
            o = torch.matmul(torch.softmax(o, dim=-1), v_all)
            o = o.transpose(1, 2).contiguous().view(NB, -1, att.d_model)
            x = residual + att.dropout(att.fc(o))

            # --- cross-attention（K/V 已预算）---
            residual = x
            h = layer.cross_attn_norm(x)
            catt = layer.cross_attn
            qc = catt.w_qs(h).view(NB, -1, catt.n_head, catt.d_k).transpose(1, 2)
            oc = cross_attn_out(i, qc)
            oc = oc.transpose(1, 2).contiguous().view(NB, -1, catt.d_model)
            x = residual + catt.dropout(catt.fc(oc))

            # --- MLP ---
            residual = x
            x = residual + layer.mlp(layer.mlp_norm(x))

        x = self.layer_norm_out(x)
        t_logit = self.tgt_word_prj(x[:, -1])
        t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
        if eos_penalty != 1.0:
            t_scores[:, self.eos_id] = t_scores[:, self.eos_id] * eos_penalty

        t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
        t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
        t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

        scores = scores + t_topB_scores
        scores = scores.view(N, B * B)
        scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
        scores = scores.view(-1, 1)

        row_in_B = torch.div(topB_score_ids, B).view(NB)
        topB_row_number_in_ys = row_in_B.long() + stride.long()

        ys = ys[topB_row_number_in_ys]
        t_ys = torch.gather(t_topB_ys.view(N, B * B), dim=1, index=topB_score_ids).view(NB, 1)
        ys = torch.cat((ys, t_ys), dim=1)

        confidences = confidences[topB_row_number_in_ys]
        t_conf = torch.gather(t_topB_scores.view(N, B * B), dim=1, index=topB_score_ids).view(NB, 1)
        t_conf = torch.exp(t_conf)
        confidences = torch.cat((confidences, t_conf), dim=1)

        # 增量 self-attn 缓存按 beam 重排
        self_kv = [(k[topB_row_number_in_ys], v[topB_row_number_in_ys]) for k, v in self_kv]

        is_finished = t_ys.eq(self.eos_id)
        if is_finished.sum().item() == NB:
            break

    # ---------- 4) 收尾（与官方一致）----------
    scores = scores.view(N, B)
    ys = ys.view(N, B, -1)
    ys_lengths = self.get_ys_lengths(ys)
    if length_penalty > 0.0:
        penalty = torch.pow((5 + ys_lengths.float()) / (5.0 + 1), length_penalty)
        scores = scores / penalty
    nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
    nbest_scores = -1.0 * nbest_scores
    index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
    nbest_ys = ys.view(N * B, -1)[index.view(-1)]
    nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
    nbest_ys_lengths = ys_lengths.view(N * B)[index.view(-1)].view(N, -1)
    nbest_confidences = confidences.view(N * B, -1)[index.view(-1)].view(
        N, nbest_ids.size(1), -1)

    nbest_hyps = []
    for n in range(N):
        n_nbest_hyps = []
        for i, score in enumerate(nbest_scores[n]):
            confidence = nbest_confidences[n, i, 1:nbest_ys_lengths[n, i]]
            confidence = confidence.mean()
            n_nbest_hyps.append({
                "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]],
                "confidence": confidence,
            })
        nbest_hyps.append(n_nbest_hyps)
    return nbest_hyps


_patched = False


def patch_fast_decode():
    """把优化后的 beam search 挂到 TransformerDecoder 上（幂等）。"""
    global _patched
    if _patched:
        return False
    try:
        from fireredasr2s.fireredasr2.models.module.transformer_decoder import TransformerDecoder
    except Exception as e:
        print(f"[fast] patch skipped: {e}", flush=True)
        return False
    TransformerDecoder._orig_batch_beam_search = TransformerDecoder.batch_beam_search
    TransformerDecoder.batch_beam_search = _fast_batch_beam_search
    _patched = True
    return True


def unpatch():
    global _patched
    if not _patched:
        return False
    from fireredasr2s.fireredasr2.models.module.transformer_decoder import TransformerDecoder
    if hasattr(TransformerDecoder, "_orig_batch_beam_search"):
        TransformerDecoder.batch_beam_search = TransformerDecoder._orig_batch_beam_search
    _patched = False
    return True

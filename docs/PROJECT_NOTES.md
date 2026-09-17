# FireRedASR2S 实时转写 — 交接文档

> 更新：2026-09-18。新会话先读这份文档再动手。
> 本系统与旧 whisper 系统**并存**，旧系统完整保留（见下方"备份与文件清单"），未覆盖任何原文件。
> 📌 **可迁移的优化结论汇总在 `OPTIMIZATION_SUMMARY.md`**（不依赖本项目上下文，其它程序可直接照搬：
> 三步诊断法、CUDA Graph 实现要点、坑清单速查、适用边界与复现脚本）。

## ⭐ 声纹判人的最终形态：在线粗略 + 停止/周期定版（2026-09-18）

> 用户要求：**声纹识别可以不用实时，但最终定版输出必须准确**。据此改成两遍式（商用产品的通行做法）。

### 在线（实时，够用即可）
1. **句内声学分段**（`segment_by_voice`）：库无关的顺序变点检测 —— 逐词嵌入与"当前段质心"相似度低于
   `VOICE_CHANGE_THR`(0.55) 就开新段，再合并相似段、吸收 <1.2s 短段。
   **关键**：早期版本用"库内质心给每个词归类"，只能区分**已知的人**；一句里出现陌生人时，
   每个词都会被塞给库里最像的人 → 整句判成同一个已知说话人（实测踩到：女声+男声+英语的一句话全标成"说话人3"）。
2. **段级复核**：每段用本段嵌入按阈值判定（认识就归类、不认识就新建），不采信切分阶段的结论。
3. 防碎片规则：短段二次证据、灰区自愈、新建前先吸收、周期合并（见下节）。

### 定版（离线，追求准确）
`run_finalize()` → `finalize_session()`，在**停止录音**时和**录制中每 ~60s**（周期随会话时长自适应）执行：

| 步骤 | 做法 | 为什么 |
|---|---|---|
| A. 说话人集合 | 对 final 段全局聚类（平均链接，0.55）→ 每簇把成员音频**拼接后重嵌入** → 簇间互并 → 与库比对/整簇建库（force_update） | 段音频几秒到几十秒，声纹稳定；纯窗口聚类试过会碎（85s → 19 簇） |
| B. 找漏掉的人 | 窗口嵌入分配到段级质心；**相似度 < 0.35 的"未解释窗口"**自己聚类 → 保守化门槛（≥4 窗、簇内自洽 ≥0.55、与已有说话人相似 <0.55）→ 通过则建新说话人 | 只出现在混合段里的人（句内第二人）在段级聚类里从未独立成簇，只有这一步能挖出来。实测挖出 16 窗 ≈12s 的隐藏说话人 |
| C. 迭代精炼 | k-means 式：用"分配到的窗口音频"重嵌入质心，再重新分配（2 轮） | 段级质心可能是**混合质心**（整段混了两人），只分配不精炼会切不准 |
| D. 时序平滑 | 窗口标签众数滤波(±1.5s) + **段级中值滤波**（A-B-A 且中间段 <3s → 三并为一） | 抑制抖动式过切分（不加时会出现"哦。/ 那。"这种碎片气泡） |
| E. 词级映射切句 | 每个词取中点 → 最近窗口 → 说话人 → 连续同人合并 → **逐子段补标点**(FireRedPunc) | 把一句话按说话人真正拆开 |
| F. 清理碎片 | 删掉本次会话新建、定版后未被任何簇用到的说话人 | 在线判定产生的碎片自动清掉 |

回发 `{"type":"revision","full":true,"bubbles":[...]}` → 前端**整份重建**气泡（定版可能把一句拆成多句）。

### 实测效果（合成用例，干净库）

**4 人录音（通话 A/B + 小说朗读 + 英文演讲）**：定版后
`spk_1(A)` → `spk_0(B)` → `spk_2(朗读)` → `spk_3(英文)` → `spk_1(A 回来正确重认)` ✓ 4 人全对

**"三种声音在一句话里"（女声+男声+英语，句间仅 0.15s 静音）**：定版拆成 `spk_5 / spk_6 / spk_3` ✓
（且英语说话人跨用例正确匹配到同一人，说明入库质心可用）

### 相关开关
| 变量 | 默认 | 含义 |
|---|---|---|
| `FIRERED_FINALIZE_ON_STOP` | 1 | 停止时定版 |
| `FIRERED_FINALIZE_PERIOD` | 60 | 录制中自动定版间隔秒（0=只在停止时）；实际周期 = max(此值, 会话时长/10) |
| `FIRERED_DIAR_WIN_SEC` / `_HOP_SEC` | 1.5 / 0.75 | 离线窗口大小/跳距 |
| `FIRERED_CLUSTER_THR` / `_MERGE_THR` | 0.55 / 0.62 | 段级聚类 / 簇互并阈值 |
| `FIRERED_UNEXPLAINED_THR` / `_MIN_WIN` / `_COHERE` / `_FAR` | 0.35 / 4 / 0.55 / 0.55 | "未解释窗口→新说话人"的保守门槛 |
| `FIRERED_MEDIAN_FILTER_SEC` | 3.0 | 段级中值滤波阈值 |
| `FIRERED_VOICE_CHANGE_THR` / `_MIN_SEG_SEC` | 0.55 / 1.2 | 在线句内声学分段阈值 |

**耗时**：85s 音频定版约 5~7s（主要是滑窗声纹，~110 窗；窗口嵌入带增量缓存，周期定版只算新增部分）。

---

## 两套系统并存

| 系统 | 入口 | 端口 | ASR | 标点 | 声纹库 | 状态 |
|---|---|---|---|---|---|---|
| 旧（whisper） | `server.py` | 8765 | faster-whisper large-v3-turbo (cuda int8) | CT-Transformer (funasr) | `speakers.json` | 保留可用 |
| **新（FireRed）** | `server_firered.py` | 8766 | **FireRedASR2-AED**（beam=3 + 词级时间戳） | **FireRedPunc**（BERT） | `speakers_firered.json` | 本系统 |

- 前端**共用同一份**：`index.html` + `audio-worklet.js`（WebSocket 协议完全一致，两套服务各自 serve 同一份文件，无需改动）
- 声纹库**已迁移**：首次启动 `speakers_firered.json` 从 `speakers.json` 复制一份（原文件不动），此后各自独立演化

## 架构（一条音频的完整链路）

```
浏览器 30ms PCM 帧 → 帧序状态机(30ms/帧，10s 缓冲上限)
  → 句子切分：0.7s 静音 / 20s 上限 / 挂起式分段(超60字等自然停顿)
  → final: FireRedASR2-AED 转写(beam=3, 词级时间戳)
          → 词级声纹切分(campplus 质心分类 + 滞回 + 碎片吸收)
          → FireRedPunc 逐段标点(BERT，失败降级规则)
          → 实时比对声纹库(余弦 >= 0.60 归类，>= 0.72 才更新质心)
  → live: 每 1.2s AED beam=3 转写 + 规则标点归一化
```

三个模型独立加载、独立降级。FireRedASR2S 源码与权重都在 `FireRedASR2S/` 下（`sys.path` 直接挂载，未 pip 安装）。

## 启动方式

```powershell
cd .
python server_firered.py            # 或双击 start_firered.bat
# 浏览器打开 http://localhost:8766
```

就绪标志（`server_firered.log`）：`ASR (FireRedASR2-AED) ready` / `punctuation (FireRedPunc) ready` / `speaker model ready`
启动耗时约 15s（AED 8s + Punc 1.5s + campplus 5s），显存约 5.3GB（AED 4.8 + Punc 0.4）。

---

## ⚠️ 三个必知的坑

### 1. 采样值标度：AED 要 **int16 量级**，不是 [-1,1]（最隐蔽，实测踩到）

| 输入方式 | 数值范围 | AED 结果 |
|---|---|---|
| `kaldiio.load_mat(wav)` | int16 量级（如 -30368~24207） | ✅ 正确 |
| `soundfile.read()` | [-1,1] float32 | ❌ **输出空文本**（当成静音） |
| `soundfile.read() × 32768` | int16 量级 | ✅ 与 kaldiio 结果逐字一致（conf 同 0.927） |

- 官方 `asr_feat.ASRFeatExtractor` 走 `kaldiio.load_mat`，所以官方脚本没事；我们直接把内存里的 numpy 喂进去，就踩了这个坑
- 症状：中文通话录音整段返回空 text + 只有一个 `<sil>` token（conf 0.24~0.88 忽高忽低），而干净响亮的示例音频（`assets/hello_zh.wav`）仍能出字 → **极具迷惑性**
- 修复：`server_firered.py` 的 `_to_aed_input()` 统一 ×32768
- ⚠️ campplus 声纹仍然吃 [-1,1] 原始值，**不要**一起转换（两个模型约定相反）

### 2. `getpass.getuser()` → `import pwd` 崩溃（Windows + 精简环境）

- 精简环境（DSH 启动的 pwsh 等）没有 `USER`/`USERNAME` 环境变量 → `getpass.getuser()` 回退到 `import pwd`（Windows 无此模块）→ `torch._inductor` 缓存目录初始化失败 → **transformers 导入链崩溃**
- 报错长相：`ModuleNotFoundError: Could not import module 'AutoModelForCausalLM' / 'GenerationMixin' / 'PreTrainedModel'`
- 修复：脚本顶部 `os.environ.setdefault` 补 `USERNAME`/`USER`（server_firered.py 已内置）
- 副作用提示：**旧 whisper 系统没有这个补丁**，在精简环境下启动会 `[sv] speaker model load failed` + `[punc] punctuation model load failed`（降级成无声纹、无标点）。启动前先 `$env:USERNAME="00"` 即可，无需改代码

### 3. AED 输入长度上限 60s

官方 FAQ：**>60s 可能幻觉，>200s 位置编码报错**。我们 `MAX_SENT_SEC=20`，安全。

---

## 参数速查（与旧系统一致，作为调优对照基线）

| 参数 | 值 | 含义 |
|---|---|---|
| SILENCE_END_SEC | 0.7 | 静音断句 |
| MAX_SENT_SEC | 20 | 单句硬上限（AED 上限 60s，留足余量） |
| MAX_CHARS | 60 | 单气泡字数（挂起切句触发） |
| PENDING_SPLIT_PAUSE | 0.2 | 挂起分段：出现此静音即切 |
| PENDING_SPLIT_MAX_SEC | 10 | 挂起分段兜底 |
| MATCH_THR | 0.60 | 声纹归类阈值（低于此值新建 ID） |
| UPDATE_THR | 0.72 | 质心更新阈值（防污染） |
| MERGE_THR | 0.78 | 启动时合并重复说话人 |
| MIN_NEW_SPK_SEC | 2.0 | 短句不跑声纹，继承上一句 |
| WORD_SV_MARGIN | 0.05 | 词级换人滞回 |
| WORD_SV_MIN_SEG_SEC | 0.8 | 词级短段并入前段 |
| PRE_PAD_SEC / LIVE_POLL_SEC | 0.4 / 1.2 | 前 padding / live 刷新周期 |

差异：`MIN_NEW_SPK_SEC` 旧文档写 1.5、代码实际 2.0（新系统沿用代码值 2.0）。

## 实测验证（2026-09-17）

### 冒烟
| 音频 | 结果 |
|---|---|
| `FireRedASR2S/assets/hello_zh.wav` | `你好世界` ✅ |
| `test_audio.flac`（英文演讲 11s） | `and so my governor ask not what your country can do for you...` ✅ |

### e2e（同一段 87.7s 客服通话 `test_call.wav`，2 倍速灌流，两套系统同音频对照）

| 维度 | FireRedASR2S (8766) | faster-whisper large-v3-turbo (8765) |
|---|---|---|
| final 段数 | 8 | 8 |
| 说话人识别 | spk_1 / spk_2 ✅ | spk_1 / spk_2 ✅ |
| 关键句 | 「要是**七折优惠**就不需要了是吗？」✅ | 「**记者优惠**就不需要了」❌ |
| 关键句 | 「改成**折扣**是其他都可以取消掉」✅ | 「改成**这个地方**，其他都可以取消」❌ |
| 幻觉 | 无 | 「**630**这个折扣的话…」「这下事情调得不了」「这个**汽车用会**」❌ |
| live 刷新次数 | 10 | 19 |

**结论**：中文场景 FireRedASR2-AED 准确率明显优于 whisper large-v3-turbo（无数字/近音词幻觉，"七折优惠"这类业务词正确），符合官方中文 SOTA 声明；代价是转写速度慢 2~5 倍（live 刷新次数 10 vs 19，final 晚约 1~2s），句子完整性二者相当。

### 复现命令
```powershell
# 两套系统同时跑（显存合计约 8.3GB，24GB 卡充裕）
python server_firered.py                                   # 8766
$env:USERNAME="00"; python server.py                       # 8765（旧系统需要这个环境变量）
python e2e_firered.py test_call.wav 2 8766     # 灌进新系统
python e2e_firered.py test_call.wav 2 8765     # 灌进旧系统
```

### ASR 层对照（`test_call.wav` 20~50s 区间，单跑无 GPU 争用，min of 3）

| 片段 | faster-whisper large-v3-turbo (int8, beam5) | FireRedASR2-AED (fp32, beam3, 时间戳) |
|---|---|---|
| 10s | 0.38s  RTF **0.038** | 1.00s  RTF **0.100** |
| 20s | 0.45s  RTF **0.022** | 2.14s  RTF **0.107** |
| 30s | 1.56s  RTF **0.052** | 3.25s  RTF **0.108** |

速度：whisper 快 2~5 倍（两者都远超实时，FireRed 约 10 倍实时，一段 20s 句子的 final 延迟约 2s）。
精度（30s 片段逐字对比，**FireRed 5:0 全胜**）：

| 实际内容 | whisper 输出 | FireRed 输出 |
|---|---|---|
| 办不了 | 断不了 ❌ | 办不了 ✅ |
| 七折优惠 | 汽车运会 ❌ | 七折优惠 ✅ |
| 最低档 | 最底层 ❌ | 最低档 ✅ |
| 改成折扣 | 改成这个套餐 ❌ | 改成折扣 ✅ |
| （结尾） | 幻觉"谢谢大家" ❌ | 无 ✅ |

### 已实测排除的提速方向
- `use_half`（fp16/bf16）与 `beam_size`(1 vs 3) **都不影响速度**（RTF 恒 0.11~0.13）→ 瓶颈是 Conformer 编码器，不是解码
- fp16 显存减半（4.8→2.4GB）但速度无提升；**bf16 会让 `torchaudio.forced_align` 报错**，而该错误在官方代码里是裸 `except` → 词级时间戳静默退化成"均匀分布假时间戳"，会拖累声纹切分 → 保持默认 fp32

### 服务器内实测（插桩 `FIRERED_ASR_TIMING=1`，见 server_firered.py）
```
[asr-time] dur=12.0s feat=0.00s total=1.44s rtf=0.119
[asr-time] dur=20.0s feat=0.00s total=2.06s rtf=0.103
```
fbank 提取可忽略（<5ms），耗时全在模型前向；服务器 RTF 与 ASR 层一致。
⚠️ **坑**：两套系统同时跑时 FireRed 的 RTF 会涨到 0.3~0.7（GPU 争用），**测速务必单跑**。
可用旋钮：`FIRERED_MIN_LIVE_SEC`（默认 0.8，置很大可关闭 live 预览换更低 final 延迟）、`FIRERED_ASR_TIMING=1`（打印耗时拆分）。

## 声纹：4 人只认出 2 人 → 已修（2026-09-17 晚）

**症状**：录音前半是两人通话（A/B，正常分开），后半插入另一个人的朗读，朗读却被标成 spk_1/spk_2 —— 整段录音 4 人只认出 2 人。

**根因（两处，都在 `emit_final_for`）**：
1. `split_speaker_segments()` 把每个词归给"最像的库内质心"，**从不检查相似度是否达标**；而 `emit_final_for` 直接采信它给出的库内 ID，**绕过了 `assign_speaker` 的新人判定**。于是朗读声（与库里两人只有 0.07~0.35 相似度）仍被硬塞给 spk_1/spk_2。
2. 陌生人整句做词级切分 → 他的词被拆给库里不同的已知人 → 一句被拆成别人的几段。

**修复**：
- **整句门槛**：先算整句嵌入判定"是否认识"；只有已知说话人的句子才做词级切分（用于抓句内插话），陌生人整句不切分，直接进段级判定。
- **段级复核**：不再采信词级切分给出的库内 ID，每段都用本段嵌入按阈值重新判定（`_match_decision`）。
- **新建前先吸收**：新建说话人前，若与已有说话人 sim ≥ MERGE_THR 则吸收，不产生碎片。
- **灰区容忍 + 弱质心自愈**：sim 落在 [MATCH_FLOOR=0.52, MATCH_THR=0.60) 且该人样本 < 3 条（质心不可靠）→ 归入该人并补强质心。
  修的是实测到的碎片化：首句仅 1.8s 就建库 → 质心差 → 同一人后续 13s 句子 sim **0.585**（差 0.015 没达标）被拆成新说话人。
- 首句无上一句可继承时，即使偏短也建档，避免第一个气泡没有说话人标识。

**实测验证**（合成 4 人音频 `test_4spk.wav`：通话A/B + 小说朗读C + 英文演讲D + 回到B）：

| 音频段 | 结果 | 判定 |
|---|---|---|
| 通话 A | spk_0 | 首句建档 |
| 通话 B | spk_1 | sim 0.357 → 新建 |
| A 后续 13s | spk_0 | sim 0.585 → **灰区自愈**（不碎片化） |
| 小说朗读 C | spk_2 | sim 0.271 → 新建 |
| 英文演讲 D | spk_3 | sim 0.117 → 新建 |
| 回到通话 B | spk_1 | sim 0.788 → **正确重认** |

**4 人全部分开，陌生人不再被并进旧说话人，说话人回来时能正确重认。**

声纹参数全部可用环境变量覆盖（双侧规则，见上节取舍表）：

| 变量 | 默认 | 含义 |
|---|---|---|
| `FIRERED_MATCH_THR` | 0.60 | 归类阈值；低于此值倾向新建（调高 → 人更多、更少误并） |
| `FIRERED_MATCH_FLOOR` | 0.52 | 灰区下限（弱质心容忍，防碎片） |
| `FIRERED_UPDATE_THR` | 0.72 | 质心更新阈值（防污染） |
| `FIRERED_MERGE_THR` | 0.78 | 合并阈值（临时条目放宽到 MATCH_THR） |
| `FIRERED_NEW_SPK_MIN_SEC` | 5.0 | 段长 ≥ 此值且未匹配 → 直接新建；更短则需二次确认 |
| `FIRERED_NEW_SPK_CONFIRM` | 0.65 | 短段二次确认所需与候选的相似度 |
| `FIRERED_PENDING_TTL` | 120 | 待确认候选有效期（秒） |
| `FIRERED_MERGE_PERIOD` | 45 | 周期性自愈合并间隔（秒） |
| `FIRERED_DB` | speakers_firered.json | 声纹库文件路径（换文件=从零开始） |

复现测试：`python make_4spk.py`（合成 4 人音频）→ `python e2e_firered.py test_4spk.wav 1.0 8766`。
⚠️ 服务端每次 stop 会把本次录音写进 `mic_debug.wav`（覆盖），测试前先复制留底。

---

## 声纹：碎片化（实测 17 个说话人）→ 双侧规则已补齐（2026-09-17 深夜）

修完"4 人被并成 2 人"后，用真实综艺/多人音频实测又出现**另一侧**问题：**同一人被拆成 17 个 ID**
（其中 11 个只有 1 条嵌入，多来自 2~5s 短段 —— `spk_5 sim=0.539`、`spk_4 sim=0.568` 这种差一点没达标的
短段，每个都新建了一个说话人）。**声纹判人本质是双侧约束，必须同时守住**：

| 侧 | 风险 | 规则 |
|---|---|---|
| 别把陌生人并进旧人 | 4 人认成 2 人 | 整句门槛 + 段级复核 + 低于阈值才新建（见上一节） |
| 别把同一人拆成多人 | 1 人认成 17 人 | **短段需二次证据才新建** + **弱质心灰区自愈** + **周期性合并** |

### 新增的三条规则（都可用环境变量调）

1. **短段二次证据**（`FIRERED_NEW_SPK_MIN_SEC=5.0` / `FIRERED_NEW_SPK_CONFIRM=0.65`）：
   段长 ≥5s 且未匹配 → 证据充分，直接新建；段长 <5s 且未匹配 → 先记为**候选**并暂时继承上一句，
   等下一段与候选相似度 ≥0.65（`FIRERED_PENDING_TTL=120`s 内）才真正新建。
2. **回溯纠正（relabel）**：候选被确认、新建了说话人后，服务端把**之前暂时继承错的气泡**按新质心复核，
   给前端补发 `{"type":"relabel","text":...,"spk":...}`，前端把那个气泡的说话人标签改回来。
   （`index.html` 里新增 `relabelBubble()`；旧 whisper 服务端从不发此消息，两套系统共用不受影响）
3. **周期性自愈合并**（`FIRERED_MERGE_PERIOD=45`s）：`merge_speakers()` 从"只在启动时跑"改为循环里周期跑，
   并且对**只有 1 条嵌入的临时说话人**放宽到 `MATCH_THR`(0.60) 即并入他人 —— 碎片会自动被清掉。

### 实测（合成 4 人音频，干净库）

```
spk_0 ← 通话A            （首句建档）
spk_0 ← 通话B 第 1 句    （短段未匹配 → 先继承，等二次确认）
spk_1 ← 通话B 第 2 句    （与候选 sim 0.773 → 二次确认新建）
        + 回溯纠正 1 个气泡 → spk_1   ← B 的第 1 句标签被改回来
spk_0 ← 通话A 后续        （sim 0.585 灰区自愈）
spk_2 ← 小说朗读C        （20s 长段，sim 0.242 → 直接新建）
spk_3 ← 英文演讲D        （12s 长段，sim 0.120 → 直接新建）
spk_1 ← 回到通话B        （sim 0.672 → 正确重认）
```

**4 人全部分开 + 首句标签也会被纠正 + 短段碎片不再产生。**

### 两侧规则的取舍（调参时看这张表）

| 现象 | 调什么 |
|---|---|
| 陌生人被并进旧人（人少了） | 调高 `FIRERED_MATCH_THR`（0.60→0.65）、调低 `FIRERED_MATCH_FLOOR`、调高 `FIRERED_NEW_SPK_MIN_SEC` |
| 同一人被拆成多人（人多了） | 调低 `FIRERED_MATCH_THR`、调高 `FIRERED_MATCH_FLOOR`、调低 `FIRERED_MERGE_THR`、调高 `FIRERED_NEW_SPK_CONFIRM` 之外的项 |
| 想立刻重新开始 | 前端"👥 声纹 → 清空声纹库"，或换 `FIRERED_DB` 指向新文件 |

**已知无法根治的区间**：跨录音的同一人 sim 只有 0.4~0.5，与部分陌生人 0.5~0.6 重叠 ——
单阈值必然在某一侧出错。要再进一步需上"两段证据 + margin 双条件"或引入更强的声纹模型（如 3D-Speaker ERes2NetV2）。

---

## 解码加速第二轮：fp16 + 长度分桶 + live 引擎（2026-09-18）

第一轮（增量解码 + CUDA Graph）把 RTF 从 0.125 压到 0.031。**瓶颈由此从"算子派发"变成"权重带宽"**，
于是第一轮被判定"无效"的 fp16 重新有了价值：

| 优化 | 实测 | 说明 |
|---|---|---|
| **fp16**（`.half()`） | 20s 句子 0.62s → **0.47s（1.26x）**；显存 4.47G → **2.18G** | **8/8 用例文本+时间戳逐字一致**，置信度零差异 |
| **编码器 fp16** | 62ms → **32~44ms** | 编码器是大矩阵，吃 tensor core |
| **Ti 长度分桶**（`TI_BUCKET=64` 帧） | 图捕获 **24+ → 8 次** | 修的是实测卡顿：每遇新句长就捕获一张图（0.5~1s），24 张上限一到就淘汰重捕 |
| **live 引擎可选**（`FIRERED_LIVE_ENGINE`） | final 0.89s → **0.44s**（CTC live） | live 预览的 GPU 开销随句长二次方增长，会拖慢 final |

### 三个必踩的 fp16 坑

1. **必须走 fp16，不能走 bf16**：`use_half=True` 在 Ampere 上会走 `bfloat16()`，而
   `torchaudio.forced_align` 只接受 fp16/fp32/fp64 → 官方此处是**裸 except** → 词级时间戳
   **静默退化成均匀假时间戳**（文本正常，但声纹切分用的是假时间戳）。
   修法：加载前 `torch.cuda.is_bf16_supported = lambda *a, **k: False` 逼出 `.half()`。
2. **mask 值要按 dtype 取**：`-1e10` 超出 fp16 范围（max 65504）→ `value cannot be converted to
   type at::Half without overflow`。fp32 用 -1e10，fp16 用 -3e4（exp(-3e4)=0，效果相同）。
3. **输入特征要同步转**：只 `model.half()` 会报 `Input type (float) and bias type (Half)`；
   graph 路径里也要补 `feats.half()`（官方 `asr.py` 在 use_half 时会转）。

### live 引擎三选一（`FIRERED_LIVE_ENGINE`）

| 取值 | live 文本质量 | 20s 句子的 final 延迟 | 适用 |
|---|---|---|---|
| `aed`（旧默认） | 与 final 同质量 | 0.89~0.98s | 只看重预览准确 |
| `ctc` | 约 90~95%（会漏字，如"保号"→"饱"） | **0.44s** | 只看重 final 速度 |
| **`hybrid`（当前默认）** | ≤6s 短句走 AED（准确），更长走 CTC | 0.45~0.86s | 兼顾 |

原理：live 预览每 1.2s 重转写整句，累计处理量 ≈ **D²/2.4**（D=20s → 相当于 167s 音频的算力），
会持续抢 GPU。CTC 分支贪心解码（一次前向出全文）**只要 51~99ms，比自回归解码快 50~100 倍**，
代价是漏字——所以用它只做预览，final 仍走 AED beam search 保准确率。

---

## 解码加速：把转写从 RTF 0.13 压到 0.03（4 倍）

官方实现慢得离谱，而且**慢的地方和直觉相反**（逐层排查记录，别重复踩）：

| 环节 | 耗时（20s 音频） | 说明 |
|---|---|---|
| fbank 特征 | 27 ms | 可忽略 |
| **编码器（16 层 Conformer）** | **62 ms** | 不是瓶颈！torch.compile/fp16 都没用 |
| **解码器 beam search** | **2400 ms** | ← 真正瓶颈，占 95% |
| CTC 时间戳对齐 | 5 ms | 可忽略 |

解码器每步 ~240 个小算子，实测**每个算子派发开销约 42µs**（Windows/WDDM + torch dispatcher）：
- GPU 实际执行 0.8~5 ms/步，CPU 派发 **10 ms/步** → CPU 才是大头
- 微基准：同样 240 个算子，eager 11.04ms vs **CUDA Graph 回放 0.80ms（13.8x）**
- 另有官方致命浪费：`DecoderLayer` 每步每层都对**整段编码器输出**重算 cross-attention 的 K/V 投影
  （16 层 × 1998 帧 × 1280 维 ≈ 105 GFLOP/步），而同一句的 3 个 beam 内容完全一样

于是做了两版加速（都在本仓库，**不改官方源码**）：

| 解码后端 | 开关 | 20s 句子 | RTF | 与官方输出 |
|---|---|---|---|---|
| official（基准） | `FIRERED_DECODE=official` | 2.50 s | 0.125 | — |
| **fast**（等价增量解码） | `FIRERED_DECODE=fast` | 1.67 s | 0.084 | **逐字一致**（含时间戳/置信度） |
| **graph**（CUDA Graph，默认） | `FIRERED_DECODE=graph` | **0.53~0.78 s** | **0.027~0.039** | 多数逐字一致，少数差 1~2 字 |

- `firede_fast.py`：CRAN 式增量解码——cross-attn K/V 每句只投影一次、self-attn 用增量 KV 缓存、
  去掉官方只用于取 `[-1]` 的前缀输出拼接。数学等价，已 6/6 用例逐字一致。
- `firede_graph.py`：把**整步解码**（16 层 + beam 剪枝 + KV 重排）全部塞进一张 CUDA Graph，
  所有状态用预分配静态缓冲**原地更新**，host 每步只做一次 `graph.replay()`。按编码器帧数 Ti
  缓存图（LRU 上限 24，每图约 180MB）。**5.8~6.1x**（对官方）。

### CUDA Graph 的两个致命坑（我踩过）
1. **状态必须原地更新**：`self.scores = ...` 这种 Python 重绑定，图里捕获的是**旧缓冲地址**，
   回放时新对象根本没被写入 → 分数不累积、finished 标志不更新 → EOS 后继续瞎生成
   （症状：文本里出现 `<eos><eos>他`）。必须用 `self.scores.copy_(...)` / `.add_()` / `.index_copy_()`。
2. **捕获期禁止同步**：任何 `bool(tensor)` / `.item()`（除循环外的固定检查）都会让捕获失败或
   每步多一次 GPU→CPU 同步。我第一版在 cross-attn 里写了 `if bool(cm.any())`，
   16 层 × 86 步 = 1376 次同步 → 每步 18.7ms（比真实算力慢 200 倍）。判断要提到循环外做一次。
   另：beam 数必须取 `config.beam_size`（我一开始写死 1，跑成了贪心解码，文本自然对不上）。

### 其他被实测排除的提速方向
- `use_half`（fp16/bf16）、`beam_size=1` **都不提速**（瓶颈在派发与解码步数，不在算力）
- fp16 省一半显存（4.8→2.4GB）但速度一样；**bf16 会让 `torchaudio.forced_align` 报错**，
  而官方此处是裸 `except` → 词级时间戳静默退化成"均匀假时间戳"，会拖累声纹切分 → 保持 fp32
- 编码器加 torch.compile：编码器只占 62ms，收益可忽略

### 环境开关（server_firered.py）
| 变量 | 默认 | 含义 |
|---|---|---|
| `FIRERED_DECODE` | `graph` | 解码后端：graph / fast / official |
| `FIRERED_HALF` | 1 | fp16（显存减半 + 解码 1.26x，输出逐字一致）；置 0 回 fp32 |
| `FIRERED_LIVE_ENGINE` | hybrid | live 预览后端：aed / ctc / hybrid |
| `FIRERED_LIVE_AED_MAX_SEC` | 6 | hybrid 模式下超过此句长改用 CTC |
| `FIRERED_MAX_GRAPHS` | 24 | CUDA Graph 缓存上限（Ti 分桶后 8~16 张足够；每图约 180MB） |
| `FIRERED_MIN_LIVE_SEC` | 0.8 | 置很大可关闭 live 预览（省算力换更低 final 延迟） |
| `FIRERED_ASR_TIMING` | 0 | 打印每句转写耗时 + 显存（`[asr-time]` 行） |
| `FIRERED_PORT` / `FIRERED_MAX_CHARS` | 8766 / 60 | 端口 / 单气泡字数 |

显存实况（24GB 卡）：AED 4.5GB + Punc 0.4GB + 声纹 0.05GB + 图缓存约 4.3GB ≈ **10GB**，
连续多轮 e2e 后稳定不涨（LRU 淘汰有效，无泄漏）。

### 与 whisper 的最终对决（20s 句子，单跑）
| | whisper large-v3-turbo | FireRed+graph |
|---|---|---|
| 耗时 | 0.45 s（RTF 0.022） | **0.53 s（RTF 0.027）** |
| 中文准确率 | 30s 片段错 5 处 | **全对** |

**结论：加了 CUDA Graph 解码后，FireRed 的速度已追平 whisper，准确率保持领先。**

---

## 后续可做（未实施，按需）

1. **FireRedVAD 流式 VAD 替换 RMS 状态机**：模型已下载在 `FireRedASR2S/pretrained_models/FireRedVAD/Stream-VAD`，官方 VAD 比能量阈值抗噪（外放混音场景会更稳）。当前沿用 RMS 是为了与原系统参数可比。
2. **FireRedLID 自动语种/方言识别**：模型未下载（约 300MB）。当前不分语种，AED 本身中英混说都能转。
3. **TensorRT-LLM 加速**：官方 `runtime/triton_tensorrt` 是 Linux/Docker 路线（Triton 推理服务），Windows 不实用；当前 CUDA Graph 版已到 RTF 0.027，收益空间不大。
4. **live 预览提速**：live 与 final 共用 beam=3。解码快了 4 倍后 live 开销占比已大幅下降；若要更顺滑可再加 beam=1 的 AED 实例专供 live（多占 4.8GB 显存）。
5. **声纹库增强**：
   - 目前每人 1 条嵌入起步、靠 >=UPDATE_THR 匹配自动累积质心；可加"注册模式"（本人念 3~5 句一次性灌入多条）。
   - 跨录音的同一人 sim 只有 0.4~0.5，与陌生人 0.5~0.6 存在重叠区。当前用「单阈值 + 灰区自愈」；若要更稳可上「需两段证据才新建」或「margin + 绝对阈值」双条件。
   - 库里人多了以后建议周期性跑 `merge_speakers()`（现在只在启动时跑）。
   - 未做的方向：把 FireRedVAD 的 mVAD（说话人无关的语音活动/事件检测）与声纹结合，先切段再判人。

## 文件清单

```

├── server_firered.py            ← 新系统服务端（本系统唯一新增逻辑）
├── firede_fast.py               ← 解码加速①：等价增量解码（cross K/V 预算 + self-attn KV 缓存）
├── firede_graph.py              ← 解码加速②：CUDA Graph 整步回放（默认，最快）
├── start_firered.bat            ← 新系统启动脚本（8766）
├── FIRERED_NOTES.md             ← 本文档
├── speakers_firered.json        ← 新系统声纹库（首启从 speakers.json 迁移）
├── speakers_firered.json.bak_before4spk ← 4 人修复前的库备份
├── speakers_empty.json          ← 空库（测试用：从零建库验证 4 人分离）
├── FireRedASR2S/                ← 官方源码（git clone）+ pretrained_models/
│   └── pretrained_models/       ← FireRedASR2-AED(4.5GB) / FireRedPunc(1.3GB) / FireRedVAD
├── server.py                    ← 旧 whisper 系统（原样未动）
├── index.html / audio-worklet.js← 前端（两套系统共用，原样未动）
├── speakers.json                ← 旧声纹库（原样未动）
├── test_call.wav / test_4spk.wav / make_4spk.py  ← 测试音频（通话 / 合成 4 人）
├── verify_fast_decode.py / verify_graph.py       ← 解码优化等价性验证
├── diag_speakers.py / ground_truth.py            ← 声纹相似度诊断 / 录音真值
├── check_mask.py / bench_op_overhead.py / profile_step.py / measure_vram.py ← 各项定位脚本
├── ab_4spk_synth3.txt           ← 4 人分离的最终验证输出
├── backup_whisper_20260917/     ← 旧系统完整备份（server.py/index.html/worklet/notes/speakers.json/start.bat）
├── test_firered.py / diag_*.py  ← 冒烟与诊断脚本
└── e2e_firered.py               ← e2e 灌流脚本（可指定 wav / 倍速 / 端口）
```

## 测试方法

```powershell
# 冒烟：模型 + 转写 + 标点
python test_firered.py

# e2e：把 wav 按 30ms 帧灌进服务器（模拟浏览器）
#   参数：wav路径  倍速  端口
python e2e_firered.py test_call.wav 2 8766     # 新系统
python e2e_firered.py test_call.wav 2 8765     # 旧系统对照
```
- 长录音直接看服务端日志的 `[final]` 行更可靠（`server_firered.log`），WS 客户端收集易超时
- 服务端每次 stop 会把本次录音存成 `mic_debug.wav`（覆盖），要留底先复制成别的名字

# 使用 Agent 助力 SGLang 开发的初步探索

日期：2026-06-19

在 SGLang 开发中，越来越多问题已经超出单点代码修改：同一个仓库里既有 LLM serving、distributed runtime、GPU kernel，也有 diffusion pipeline、model-specific execution path 和生产排障。过去很多工作依赖开发者个人记忆：某个模型该怎么启动、profile trace 该怎么看、CUDA crash 应该先打哪个日志、一个性能 PR 到底应该补哪些 benchmark。Agent 工具成熟以后，这些经验可以被整理成可执行的 `SKILL.md`、脚本、benchmark contract 和 review loop。

围绕 SGLang agent 开发，已经有一批面向 LLM 和 diffusion 的 skills：

- [SGLang `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills) 维护在 SGLang 仓库内部，沉淀 repo-level 的开发流程，例如 CUDA crash debug、kernel 集成、测试、CI 和 profiling。
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills) 面向 diffusion 相关流程，包括新增 diffusion 模型、benchmark/profile denoise 路径、调优性能选项，以及验证量化 pipeline。
- [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) 覆盖 serving benchmark、profile analysis、production incident triage、SOTA loop 等流程。
- [kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) 是 KDA 项目，也是 MLSys 2026 FlashInfer Kernel Contest 的 winning solution。
- [BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot) 将 KDA 风格的 agent kernel workflow 用到 SGLang 上。B200 diffusion 侧目前覆盖 7 个 SGLang kernel task，在 extracted production rows 上的 wall-geomean speedup 从 `1.1341x` 到 `2.7499x`。

把这些工作放在一起看，可以看到一个共同方向：Agent 的价值来自流程化的工程经验，包括可执行步骤、可复现实验和可审查证据。

## 1. TL;DR

- Agent 在 SGLang 里最有用的地方，是按一条清楚的流程持续推进。benchmark、profile、kernel API logging、diffusion pipeline 添加、生产 incident replay、SOTA loop，都可以写进 skills。
- SGLang skill 是一份可执行的开发规程。以 `debug-cuda-crash`、`sglang-diffusion-benchmark-profile`、`llm-torch-profiler-analysis` 为例，里面沉淀的是 preflight、失败门禁、artifact contract、复现命令和结果格式。
- 性能优化的证据主要来自 profile。SGLang profiler skill 固定输出 kernel table、overlap-opportunity table、fuse-pattern table；KDA-Pilot 则把 profile 继续推到 baseline/candidate 同 ABI、真实 workload、correctness gate、NCU 证据和 per-shape 结果。
- 长期优化已经开始进入 Loop Engineering 阶段。SGLang SOTA Performance Loop 把“追 SOTA”拆成公平 benchmark、gap decision、profile、patch、revalidate；Humanize/RLCR 提供更强的外部审查，Codex Goal 也可以完整平替这个 loop，以更低成本持续迭代。
- Review 会变得更重要。Agent 能跑更多实验，也会生成更多看起来合理但需要审查的改动。开发者的工作会更多转向定义问题、选择证据、设计流程和判断结果能否进入生产路径。

## 2. 为什么 SGLang 适合使用 Agent 辅助开发

SGLang 是一个面向 LLM 和多模态模型的高性能 serving framework。模型类型和硬件路径变多后，开发里会遇到几类常见问题：

- LLM 路径复杂：同一个性能问题可能跨 Python runtime、scheduler、CUDA graph、Triton/CUDA kernel、FlashInfer/FlashAttention、distributed collective 和 model-specific wrapper。
- Diffusion 路径也复杂：一个 denoise 变慢的问题，可能和 pipeline/stage 划分、DiT block、attention backend、torch.compile graph break、CFG/SP 并行、VAE 或自定义 fused kernel 有关。
- 验证成本高：很多修改必须在 H100/H200/B200 或 RTX 5090 上跑真实模型、真实 workload，单靠本地单元测试不够。
- profile 难以手工复用：一次 trace 里可能有几百个 kernel launch，仅靠人工看 Perfetto 很容易遗漏 kernel 到 Python source 的映射，也容易混淆 prefill 和 decode。人类在读 profiler 的过程中会积累很多 know-how，比如哪些 kernel 名字对应哪段模型逻辑、哪些 launch pattern 暗示 graph break、哪些 NCCL/attention/MLP 排布是正常的；但这些经验如果只留在个人脑子里，很难被下一个任务复用。
- 性能结论非常依赖上下文：GPU 型号、shape、batch、parallelism、precision、backend、compile 状态都会改变结论。一个孤立 microbench 很可能无法说明真实模型端收益，所以还需要建立端到端的长测试过程，用固定 workload 反复验证吞吐、延迟、显存、精度和稳定性。这件事本身就很费力，也很耗时。

这些问题恰好适合交给 Agent 处理。启动 server、固定 workload、采集 trace、初筛 profile row、补测试、记录实验结果，本质上都有明确输入输出，也很适合脚本化和重复执行。开发者需要提供的是边界：同一个 benchmark 设置、同一套 profile 解释规则、同一组 accuracy gate，以及什么情况下应该停止继续改代码。

因此，这里讨论的 Agent 是被工程流程约束的执行器。SGLang 开发里反复出现的流程适合沉淀成 skills，让 Agent 承担重复执行、证据收集和状态记录；开发者则负责定义目标、判断证据、审查改动能否进入真实 serving 路径。

## 3. 从 Prompt 工程到 SKILL：协议和案例

在 SGLang 框架里，一个有用的 skill 至少要交代清楚几件事：

| 问题 | Skill 中需要沉淀的内容 |
| --- | --- |
| 什么时候使用 | 触发场景、适用模型、适用硬件、哪些情况必须停止 |
| 怎么开始 | preflight、环境变量、repo 状态、依赖检查、模型配置 |
| 怎么验证 | benchmark 命令、profile 命令、测试入口、accuracy gate |
| 怎么判断 | 输出表格、失败模式、优先级、risk 分类、fallback 条件 |
| 怎么交付 | artifact 目录、结果 schema、PR 描述、复现命令、审查要求 |

这些 SGLang agent 相关 skills 覆盖的范围不同：有的贴近源码修改，比如 debug、test、diffusion model 添加和 benchmark/profile；有的面向跨框架 benchmark、profile 分析、生产排障、PR 优化知识库和 Humanize/RLCR 这类上层 workflow。

### 3.1 当前 Skill 栈

目前比较常用的 SGLang agent 相关 skills 包括以下几类。

| 层级 | 代表 skill / 项目 | 解决的问题 |
| --- | --- | --- |
| CUDA crash | [`debug-cuda-crash`](https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash) | 在 custom op/kernel API 边界记录输入、异常和 dump，把瞬时 crash 转成可离线分析样本 |
| LLM benchmark | [`llm-serving-auto-benchmark`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-auto-benchmark) | 对 SGLang、vLLM、TensorRT-LLM 做公平、有限预算、可恢复的 serving benchmark search |
| Trace triage | [`llm-torch-profiler-analysis`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-torch-profiler-analysis) | 固定输出 kernel table、overlap-opportunity table、fuse-pattern table，并把 kernel 映射回 Python source |
| Pipeline/layer analysis | [`llm-pipeline-analysis`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-pipeline-analysis) | 把 torch profiler trace 切到 forward pass、layer 和 kernel flow，定位稳态 pass、瓶颈层类型和 Perfetto 时间范围 |
| Diffusion benchmark/profile | [`sglang-diffusion-benchmark-profile`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile) | 采集 denoise latency、perf dump、torch profiler，并先检查是否真正走 native SGLang diffusion backend |
| Diffusion 添加模型 | [`sglang-diffusion-add-model`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model) | 从 Diffusers/reference pipeline 出发，按 SGLang pipeline/stage/model/config 结构添加新 diffusion 模型 |
| Diffusion 性能调参 | [`sglang-diffusion-performance`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance) | 选择 torch.compile、warmup、SP/CFG parallel、offload、attention backend、quantization 等性能参数 |
| 生产排障 | [`sglang-prod-incident-triage`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage) | 从 live server 收集 bundle、保存 failing request、replay，再接 crash/hang/profile 等专项工具 |
| SGLang SOTA Performance Loop（Loop Engineering） | [`sglang-sota-humanize-loop`](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-sota-humanize-loop) | 先公平比较 SGLang/vLLM/TensorRT-LLM，再把 gap decision、profile、patch、复测放进 Humanize/RLCR loop |

这些条目把容易遗漏的步骤写成协议，让流程能执行，能恢复，也能被人审查。

### 3.2 几个近期优化和流程案例

下面几个案例来自最近已经合并的 SGLang PR。表格关注完整工程链路：benchmark、profile、定位、修改、测试和复测。

| 案例 | 结果 | 关键点 |
| --- | --- | --- |
| Router 长上下文 tokenization 去重，[SGLang PR #28744](https://github.com/sgl-project/sglang/pull/28744) | 在 DeepSeek-V4-Flash 部署上，60k/125k token prompt 的 idle TTFT 分别下降约 `29%` / `41%`；60k token under load 场景 TTFT 下降 `34%–49%` | Agent 把 cache-aware routing、chat encoder parity、engine `input_ids` fallback 和 proxy body 构造一起处理，避免 router 和 engine 重复 tokenize |
| Qwen3-Next FlashInfer allreduce fusion，[SGLang PR #22664](https://github.com/sgl-project/sglang/pull/22664) | H100 TP=4 上 request throughput `5.49 req/s -> 9.41 req/s`，约 `+71.4%`；mean TTFT `456.24 ms -> 167.54 ms` | 这是一个 profile 驱动的 LLM collective 优化：prefill 里未融合的 cross-device reduce 是主要热点，打开 fused allreduce 后还补了 MMLU/GSM8K accuracy 验证 |
| Cohere2Moe NVFP4 fused-MoE 路径，[SGLang PR #27401](https://github.com/sgl-project/sglang/pull/27401) | `CohereLabs/command-a-plus-05-2026-w4a4` 在 1x B300 上，相比 SGLang 原默认路径 request throughput 提升 chat `+26%`、summarization `+21%`，并在该实验中超过 vLLM `+4.1%` / `+6.8%` | 这次改动补齐了 routing semantics，使已有 `flashinfer_trtllm` NVFP4 fused-MoE kernel 能在真实模型路径里被正确启用，并补全 GSM8K/MMLU 验证 |
| Kimi Delta Attention (KDA) CuteDSL prefill kernel on SM100，[SGLang PR #27488](https://github.com/sgl-project/sglang/pull/27488) | `moonshotai/Kimi-Linear-48B-A3B-Instruct` 的 KDA prefill 在 B200 上相比 Triton 快 `1.08x–1.52x`，GSM8K `0.915 -> 0.920`，并新增真实 gate magnitude 的回归测试 | 这个 kernel 任务需要同时处理模型 gate 分布、数值溢出、host overhead、真实模型 accuracy 和 unit test，最后才形成可合并的优化 |
| Spectral Progressive Diffusion，[SGLang PR #27524](https://github.com/sgl-project/sglang/pull/27524) | FLUX.1、FLUX.2、Z-Image、Wan、Qwen-Image 上的 denoising speedup 分别达到 `1.63x`、`1.77x`、`2.07x`、`2.32x`、`1.6x`（RTX A6000，文中设置） | 这是 diffusion 侧的系统级优化：利用低频先行的 denoising 特性，把早期步骤放到低分辨率 latent 上，再用 GPU DCT upsample 回全分辨率 |
| LTX-2 VAE decode channels-last-3d，[SGLang PR #27431](https://github.com/sgl-project/sglang/pull/27431) | LTX-2 decode stage `5.41 s -> 3.84 s`，约 `1.41x`；peak reserved memory `71.81 GiB -> 62.12 GiB`，减少约 `9.7 GiB` | 这个例子体现了 profile 到实现的闭环：瓶颈在 Conv3d 和 layout conversion，于是修 causal padding 的 memory format，并把 loader policy 接到单 GPU LTX-2 |

这些例子里，Agent 的主要作用体现在流程执行上：跑 benchmark、读 profile、定位 Python source、改代码、补测试、复测、整理 PR 描述。没有 skills 的时候，很多步骤依赖人工提醒；写进 skills 后，流程更容易重复。

## 4. Profile、Humanize 与 Loop Engineering

SGLang 性能优化里最常见的误区，是只看一个总耗时或者只打开 Perfetto 浏览几分钟，然后凭感觉判断“应该 fuse 一下”。这对 Agent 更危险，因为它很容易把一个看起来热的 kernel 误判为真正 bottleneck。

实际分析时，通常先用两个 profiler skills 配合。`llm-torch-profiler-analysis` 负责第一层 trace triage，把全局 profile 固定成三张表：

- `Kernel Table`：按 stage 统计 GPU time share、launch count、kernel category，并尽量映射回 Python source 和 CPU op。
- `Overlap Opportunity Table`：根据 exclusive/hidden 时间占比、依赖风险、kernel 类别判断哪里还有 overlap/headroom。
- `Fuse Pattern Table`：用 source-backed 的 pattern catalog 对照 SGLang、vLLM、FlashInfer、TensorRT-LLM 中已有或正在推进的 fusion/overlap path。

这三张表先回答最基本的问题：哪个 stage 的哪个 kernel 占了多少 GPU time，映射到哪行 Python，是否已经有可借鉴的 fuse/overlap path。如果 SGLang 落后于 vLLM 或 TensorRT-LLM，应先让 profiler table 解释 gap，再进入代码修改。

再往下，就是 `llm-pipeline-analysis`。全局热点知道以后，还要知道热点落在哪个 forward pass、哪类 layer、哪条 kernel flow 上。这个 skill 会读取 Chrome trace JSON 和模型 `config.json`，用 layer-boundary anchor kernel 把 trace 切成 forward passes 和 layers，然后输出更适合深挖的几类表：

- `Forward pass summary`：区分 cold-start 和 steady-state，避免把预热阶段误当成优化目标。
- `Per-layer timeline`：按 layer 给出 wall time、sum duration，以及 MLA、MoE、GEMM、NCCL、MHC、Hadamard 等类别占比。
- `Layer cluster statistics`：对有交替层结构的模型尤其有用，比如带 `compress_ratios` 的 NSA/混合注意力模型，可以看出 C4_LIGHT、C128_HEAVY、HASH 等层类型谁在拖慢。
- `Compute flow table`：选中代表性 layer 后，展开到具体 kernel flow，给出 hotness、相对时间戳和输入维度，方便直接跳回 Perfetto。

这样 profile 分析会变成两步：先用 `llm-torch-profiler-analysis` 找到全 trace 的主要矛盾，再用 `llm-pipeline-analysis` 把问题落到稳态 forward pass、代表性 layer 和具体 kernel flow。前者避免凭感觉选方向，后者避免只盯着一个全局 hot kernel，却忽略模型结构里的层类型差异。

### 4.1 Humanize/RLCR: 给 Loop 加外部审查

Humanize 处理的是长期任务里的状态和审查问题。一个高风险的 SGLang 性能任务通常不会“一轮实现”就结束，它会经历多轮 benchmark、profile、patch、revert、换方向、再验证。Humanize 把这些动作拆成两个阶段：

1. 先 gen-plan。用 `humanize-gen-plan` 把草稿需求整理成结构化 `plan.md`，里面包含 goal description、acceptance criteria、positive/negative tests、path boundaries、milestones 和 implementation notes。
2. 再跑 RLCR loop。用 `humanize-rlcr` 从 `plan.md` 启动循环。每轮 Claude Code 读取 `.humanize/rlcr/<timestamp>/round-<N>-prompt.md`，实现、提交、写 summary；随后由 Codex Review 检查状态文件、summary、git clean、review 结果、open question、max iteration 等条件，不能靠一句“任务完成”跳过。

这套机制给后面的 SGLang SOTA Performance Loop 提供了执行和审查底座。Claude Code 负责跑 benchmark、读 profile、改 SGLang 代码和复测；Codex Review 负责在每轮结束时检查证据、状态和风险。它适合会进入 PR、会影响 serving 正确性、或者需要多天多轮实验的任务。

实际使用时，命令顺序最好写得很明确，避免 Agent 直接跳进实现：

```text
1. Write a task draft under artifact_root/draft.md.
2. Run humanize-gen-plan to generate artifact_root/plan.md.
3. Start humanize-rlcr from artifact_root/plan.md.
4. Keep all decisions, summaries, and review state in the local Humanize workspace.
```

### 4.2 SGLang SOTA Performance Loop（Loop Engineering）

单个 skill 可以把一次任务做稳。但十几轮实验之后，另一个问题会冒出来：哪个候选版本最好，哪些思路已经失败，上一轮 NCU 说明了什么，benchmark 是否还对齐 baseline，什么时候应该停下。这些状态不能只靠聊天上下文记着。

SGLang SOTA Performance Loop 就是建立在 Humanize/RLCR 上的一套 Loop Engineering。这里的 SOTA 指固定实验条件下的可复现最好结果：同一个模型、硬件、GPU 数量、precision、workload、SLA、framework commit 和 serving 参数，SGLang 能否达到当前最好的可复现实验结果。

![SGLang SOTA Performance Loop](https://raw.githubusercontent.com/BBuf/AI-Infra-Auto-Driven-SKILLS/main/docs/assets/sglang-sota-performance-loop.svg)

图 1：SGLang SOTA Performance Loop。固定公平 benchmark 先给出可复现 baseline，后续 gap decision、profile、pipeline analysis、patch 和 revalidate 由 Humanize/RLCR loop 持续推进。

一个完整的 SGLang SOTA Performance Loop 包含以下阶段：

1. 定义目标边界。例如 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`、single-node 2x B200、FP8、SGLang TP=2、同样 2 卡预算下比较 vLLM 和 TensorRT-LLM。
2. 先做公平 search。在 patch SGLang 之前，先用相同 workload 和资源预算搜索 SGLang/vLLM/TensorRT-LLM 的可复现最好命令。
3. 判断 gap。如果 SGLang 已经持平或领先，就记录完成证据；如果稳定落后超过阈值，进入 profile。
4. 用 profile 解释 gap。先别急着改代码，先产出 kernel table、pipeline table、overlap/fuse table，必要时补 NCU。
5. 只 patch 有证据支持的路径。例如 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph、MoE/EP、quant kernel 或 model wrapper。
6. 回到同一 workload 复测。每一轮都记录 benchmark、profile、accuracy、失败尝试、环境信息和清理动作。

对于 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` 这类 2x B200 目标，loop 的价值在于把 benchmark 结果、profile trace、失败 patch 和中间结论都绑定到同一个模型、硬件、workload 和 framework commit 上。类似任务如果拆成多个独立 prompt，很容易忘记哪个命令产出了哪个结果，或者后续 profile 是否还对应最初 baseline。放进带证据和审查的 loop 里，更容易保持前后条件一致。

### 4.3 Codex Goal: 成本更低的完整平替

上面的 SGLang SOTA Performance Loop 采用双角色协作：Claude Code 执行 benchmark、profile、patch、revalidate，Codex Review 在每轮结束后做审查。这种方式适合严肃 PR，但每轮都要同时消耗执行模型和 review 模型，成本会更高，等待链路也更长。

Codex Goal 给了另一种实现方式。把“公平 benchmark -> gap decision -> profile -> patch -> revalidate -> artifact ledger”写进一个持续 Goal 后，可以让一个 Codex Goal 在同一个目标里完成执行、自检和复测，不再依赖双角色执行/审查设置。这样仍然保留 SGLang SOTA Performance Loop 的核心约束：固定 workload、证据驱动 patch、同一实验条件复测、每轮更新 artifact manifest。

两种方式的差异如下：

| 维度 | Humanize/RLCR SOTA Loop | Codex Goal |
| --- | --- | --- |
| 执行方式 | Claude Code 做实现和实验，Codex Review 做每轮审查 | 一个 Codex Goal 持续执行、自检和复测 |
| 状态位置 | `.humanize/rlcr/...` 里的 plan、prompt、summary、review result | 当前 Goal 线程 + `artifact_root` 里的 manifest/evidence |
| 审查方式 | Stop hook + Codex Review + git/state/schema 检查 | Goal 内自检 + artifact contract + 人工抽查 |
| 成本 | 两个模型角色参与，单轮成本更高 | 单个 Goal 承载执行和检查，成本更低 |
| 主要风险 | loop 配置复杂、每轮等待更长 | 容易目标漂移或过早完成，需要写硬停止条件 |

下面是 [AI-Infra-Auto-Driven-SKILLS/prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts) 中的一个 2x B200 模型优化 prompt 示例。

Humanize/RLCR 版本：

```text
Use the sglang-sota-humanize-loop workflow.

Task:
Optimize SGLang serving performance for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
on a single node with 2 NVIDIA B200 GPUs, FP8 precision, and initial SGLang
TP=2. SGLang should match or exceed the best reproducible vLLM/TensorRT-LLM
result under the same 2-GPU budget, workload, SLA, model, precision, and
environment constraints.

Required workflow:
1. Create a draft task document under artifact_root.
2. Run humanize-gen-plan to turn the draft into a structured plan.md.
3. Start humanize-rlcr from that plan.md in the Claude Code session.
4. Keep benchmark, profile, patch, and revalidation decisions inside the same
   Humanize workspace.

Evidence and safety requirements:
- Before patching, run a fair bounded search for SGLang, vLLM, and TensorRT-LLM.
- Check relevant open PRs in sgl-project/sglang and BBuf/sglang before choosing
  the SGLang baseline.
- If SGLang is behind by more than 1%, profile before patching.
- Prioritize evidence around hybrid attention, Mamba/GDN, radix cache, target
  verify, and CUDA graph.
- Record benchmark commands, profile artifacts, failed attempts, and cleanup
  evidence for every round.
- Patch only evidence-supported SGLang code paths.
- If a PR is needed, push/open it only against BBuf/sglang and include benchmark,
  GSM8K, and full MMLU accuracy tables.

artifact_root:
/Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_instruct_fp8_sota_humanize
```

Codex Goal 版本：

```text
/goal Keep optimizing SGLang serving for
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on a single node with 2 NVIDIA B200
GPUs until SGLang matches or exceeds the best reproducible vLLM/TensorRT-LLM
result under the same 2-GPU budget, FP8 precision, workload, SLA, model, and
environment constraints. The current Codex Goal is the loop: fixed fair
benchmarking, gap decision, profiling, pipeline analysis, evidence-backed
patching, revalidation, final report, and optional PR preparation all happen
inside this Goal. Completion requires benchmark evidence, profile evidence when
SGLang was behind, correctness/accuracy evidence, a final artifact manifest,
and no regression in environment safety constraints.

model_id: Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
root_dir: /Users/bbuf/工作目录/Common
target_hardware: single-node 2x NVIDIA B200
minimum_gpu_count: 2
precision_quantization: FP8
initial_deployment: SGLang TP=2
artifact_root:
/Users/bbuf/工作目录/Common/opt_model/b200_qwen3_next_80b_a3b_instruct_fp8_sota_goal

Requirements:
- Use the current Codex Goal as the only persistent loop.
- Before patching, run a fair bounded search for SGLang, vLLM, and TensorRT-LLM
  under the same 2-GPU budget.
- If SGLang is behind by more than 1%, profile in the same Goal, then use
  llm-torch-profiler-analysis, llm-pipeline-analysis, and ncu-report-skill when
  needed before patching.
- Focus on hybrid attention, Mamba/GDN, radix cache, target verify, and CUDA graph.
- Update the artifact manifest, benchmark evidence, profile evidence, failed
  attempts, and next-step decision after every round.
- Stop and report a blocker if resources are unavailable, evidence is
  untrustworthy, the budget is exhausted, or no defensible next patch exists.
```

Goal 版本保留同一套 benchmark、profile、accuracy 和 artifact 要求，区别在于执行和审查都收进一个持续目标里。只要硬停止条件写清楚，它就可以完整承担 SGLang SOTA Performance Loop。

## 5. 基于 KDA 的 SGLang 系统 CUDA kernel 优化

LLM 和 diffusion 的模型级优化之外，kernel 优化还有更强的规模化压力。不存在一个脱离硬件和 workload 的通用最优 kernel：同一个 operator 在 H100、H200、B200、B300 上可能需要不同实现；不同模型结构会带来不同 tensor shape 和 layout 约束；serving workload 又会改变 batch size、sequence length、precision format、wrapper overhead、同步行为和 fallback path。实际面对的是硬件 x 模型 x workload 定义的组合空间。

这个组合空间会迅速变成很重的人力成本。每个 candidate kernel 都要抽取真实生产 row，搭 same-ABI harness，做 A/B measurement，跨 shape bucket 检查 correctness，读 NCU 指标，判断某个 bucket 是否值得 specialize，最后还要回到真实 SGLang 路径里重新验证。如果每个硬件、模型和负载组合都靠人手工推进，成本会非常高。这类重复、证据密集、对隐藏变量敏感的流程，正适合交给 agent kernel workflow 去执行；人类则负责定义不变量、判断 tradeoff，并审查最终路径。

但直接让 Agent 写 CUDA，很容易出现 benchmark reward hacking：改 benchmark、走更轻的 wrapper、打开 baseline 没有的 fast math、只优化一个 shape、破坏数值语义，或者在真实 SGLang 路径里没有任何收益。

KDA-Pilot 的思路是把 kernel 优化拆成独立任务，避免 Agent 在 SGLang 大仓里随意改：

- workload 来自真实 SGLang diffusion 模型，先跑 20 个 diffusion 模型并汇总实际 kernel meta。
- baseline 从 upstream SGLang main 复制，记录 source lineage。
- baseline 和 candidate 必须走同一个本地 ABI、同一个 build/export path。
- benchmark 使用固定 production rows、A/B interleaving、CUDA event 或 wall timing。
- correctness 覆盖 production rows、canonical regression grid、NaN/Inf、poison output 和 fallback contract。
- 每轮迭代刷新 task prompt、benchmark evidence、KernelWiki 和 ncu-report-skill。
- 允许 shape-specialized dispatch，但必须写清楚每个 bucket 的条件、路径、latency 和 fallback。

一个具体快照是：KDA-Pilot 已经在 B200 上优化了 7 个 SGLang diffusion kernel task，在 extracted production rows 上的 wall-geomean speedup 从 `1.1341x` 到 `2.7499x`。

这些结果也开始进入 upstream 路径。[SGLang PR #27392](https://github.com/sgl-project/sglang/pull/27392) 给 Qwen-Image-2512 增加 B200 native diffusion norm-scale-shift fast path，在单张 B200 上报告 full request `1.081x`、denoise wall `1.093x` 加速。[SGLang PR #28051](https://github.com/sgl-project/sglang/pull/28051) 把 B200 `fused_inplace_qknorm_rope` 路径拆成单独改动，profiler 里 target qknorm+RoPE CUDA work 从 `24.087 ms / 1440 calls` 变为 `18.081 ms / 720 calls + 1.896 ms / 720 calls`，约 `1.21x` kernel-level speedup。

![KDA-Pilot B200 diffusion kernel results](https://raw.githubusercontent.com/BBuf/how-to-optim-algorithm-in-cuda/master/large-language-model/sglang/assets/kda-pilot-b200-speedups.svg)

图 2：KDA-Pilot 在 B200 上对 7 个 SGLang diffusion kernel task 的 wall-geomean speedup。这里的 wall time 包含 Python dispatch、wrapper、kernel launch 和 `cuda.synchronize()` 能观察到的同步开销，比单纯 kernel device time 更接近真实调用路径。

| Kernel task | B200 wall geomean | 主要优化方向 |
| --- | ---: | --- |
| `qknorm_rope` | `1.1341x` | shared RoPE staging、Q/K 复用、large rows fast path |
| `norm_infer` | `1.3523x` | warp-row RMS、tiled persistent RMS、8B/16B vector path |
| `rotary_embedding` | `1.4912x` | 128-bit vector I/O、cos/sin hoisting、LTX2 block matching |
| `cutedsl_norm_tanh_mul_add` | `1.4953x` | row-invariant math hoisting、launch-bounds tuning、exact tanh |
| `cutedsl_norm_scale_shift` | `1.3201x` | operand-class dispatch、16B/32B vector、two-pass variance |
| `fuse_scale_shift` | `2.7499x` | rowgrid/flatvec/exact-C paths、cache hints、one-pass reduction |
| `group_norm_silu` | `2.3118x` | split-group stats、channels-last direct path、fallback for giant rows |

这些数字要看清实验设置：它们是 extracted production rows 上的 kernel task speedup，不能等同于完整模型端到端提升。即便如此，它们仍然有价值。只要 baseline、workload、correctness、profile 和 review 都固定住，Agent 就能在真实框架 kernel 上做出可审查的增量。

KDA-Pilot 的实验里，有两条规则最值得保留：

- 不要给 Agent 留 benchmark reward hacking 的空间。baseline/candidate ABI 不一致、fast math 不对齐、wrapper 路径不同，结果都会失真。还有一种常见问题是看完结果以后再换 benchmark shape 集合，比如把 candidate 变慢的 shape 从表里删掉，这种结果也不能用。
- 接近 Roofline 的 bucket 应该允许 no-go 或 fallback。好的 kernel 优化任务不应该逼 Agent 每个 shape 都赢。对于 giant contiguous bucket 或接近带宽上限的路径，记录 fallback 可能比继续堆复杂度更正确。

## 6. 几条实践规则

1. 先定义任务边界，再启动 Agent。
“优化 SGLang”太大，“给 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` 在 2x B200、固定 `1000->1000` 和 `8000->1000` workload 下追平 vLLM”才是一个可执行目标。

2. 先固定 benchmark，再看 profile。
如果 workload 可以在看到结果后改变，Agent 很容易无意中优化到一个更简单的问题。SOTA loop 和 KDA-Pilot 都把固定 workload 放在 patch 之前。

3. 看 NCU 结果要先判断 kernel 的计算性质。
memory-bound kernel 重点看 DRAM/L2 throughput、load/store efficiency、memory pipe utilization；compute-bound GEMM/attention kernel 重点看 Tensor Core utilization、SM busy、eligible warps 和主要 stall 原因；小而碎的 latency-bound kernel 则要看 launch count、单 kernel duration、同步点，以及有没有融合空间。只贴一张 trace 截图不够，至少要说清楚哪个指标支持下一步改动。

4. 先确认 backend 和 fallback gate，再相信 profile。
LLM 路径如果静默切换 attention backend、关闭 CUDA graph，或者走了和 benchmark 不一致的 wrapper，trace 描述的就不再是目标 serving path。Diffusion 也是一样：如果日志里已经 fallback 到 diffusers backend，这段 trace 不能作为 native SGLang diffusion 的证据。这样的硬停止条件应该写进 skill。

5. Kernel 优化要同 ABI、同 wrapper、同 compile flags。
尤其不要让 candidate 偷偷走更轻路径，也不要单边开启 `--use_fast_math`。

6. Review 能力比以前更重要。
Agent 能制造更多 PR，也能制造更多看似合理的错误。SGLang 这类高性能系统的 review 需要关注 shape、dtype、distributed、CUDA graph、fallback、accuracy、serving API、metrics 和 benchmark 设置。

Agent 时代的 SGLang 开发，不会把开发者从系统里拿掉。更现实的变化是：把开发者的经验写成流程，把重复执行交给 Agent，把判断、设计和审查留给人。省下来的时间可以继续投入到更难的性能问题、模型路径和生产稳定性，也可以反过来优化 Agent 流程本身。对开源推理框架来说，这类基础设施值得长期投入。

## 7. 致谢

感谢参与 SGLang agent skills 建设的 SGLang Team 成员和贡献者：Xiaoyu Zhang (BBuf), Lianmin Zheng, Liangsheng Yin, Ke Bao, fzyzcjy, Kangyan Zhou, DarkSharpness, Mick, Alison Shao, Baizhou Zhang, Bingxu Chen, Cheng Wan, Ratish P, shuwenn, ykcai-daniel, Yuhao Yang，以及 Artem Savkin.

感谢 KDA 团队：Dongyun Zou, Ligeng Zhu, Sihao Liu, Junxian Guo, Yixin Dong, Zijian Zhang, Hao Kang，以及 Song Bian.

感谢 Humanize 团队和贡献者：Sihao Liu, Ligeng Zhu, Zijian Zhang, Zenus Zhang, shinan6, DYZhang, Chao Liu, Zhou Yaoyang, gyy0592, AcrossForest, Emin, Qiming Chu, jiaxiaoyu, tastynoob，以及 zhenwei.

## 8. 参考资料

- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills)
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills)
- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
- [AI-Infra-Auto-Driven-SKILLS prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts)
- [Kernel Design Agents (KDA)](https://github.com/mit-han-lab/kernel-design-agents)
- [KernelWiki skill](https://github.com/mit-han-lab/KernelWiki)
- [ncu-report-skill](https://github.com/DongyunZou/ncu-report-skill)
- [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)
- [SGLang Diffusion Advanced Optimizations, LMSYS Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)
- [OpenAI Codex Prompting: Goal mode](https://developers.openai.com/codex/prompting#goal-mode)
- [Humanize](https://github.com/PolyArch/humanize)

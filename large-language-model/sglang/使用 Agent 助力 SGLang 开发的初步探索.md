# 使用 Agent 助力 SGLang 开发的初步探索

日期：2026-06-19

在 SGLang 开发中，越来越多问题已经不是单点代码修改：同一个仓库里既有 LLM serving、multi-node runtime、attention/MoE/quantization kernel，也有 diffusion pipeline、torch.compile、ModelOpt 量化、视频模型并行和生产排障。过去很多工作依赖开发者个人记忆：某个模型该怎么启动、profile trace 该怎么看、CUDA crash 应该先打哪个日志、一个性能 PR 到底应该补哪些 benchmark。Agent 工具成熟以后，这些经验可以被整理成可执行的 `SKILL.md`、脚本、benchmark contract 和 review loop。

SGLang 主仓已经维护了一批面向 LLM 和 diffusion 开发的 skills。[BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) 进一步覆盖 serving benchmark、profile analysis、production incident triage、SOTA loop 等流程；[BBuf/KDA-Pilot](https://github.com/BBuf/KDA-Pilot) 则围绕 SGLang diffusion kernel 做自动优化实验。把这些工作放在一起看，可以看到一个共同方向：Agent 的价值不只在生成代码，更在于把 SGLang 开发中的工程经验整理成能执行、能复现、能审查的流程。

## 1. TL;DR

- Agent 在 SGLang 里最有用的地方，是按一条清楚的流程持续推进。benchmark、profile、kernel API logging、diffusion pipeline 添加、生产 incident replay、SOTA loop，都可以写进 skills。
- SGLang skill 是一份可执行的开发规程。以 `debug-cuda-crash`、`sglang-diffusion-benchmark-profile`、`llm-torch-profiler-analysis` 为例，里面沉淀的是 preflight、失败门禁、artifact contract、复现命令和结果格式。
- 性能优化的证据主要来自 profile。SGLang profiler skill 固定输出 kernel table、overlap-opportunity table、fuse-pattern table；KDA-Pilot 则把 profile 继续推到 baseline/candidate 同 ABI、真实 workload、correctness gate、NCU 证据和 per-shape 结果。
- 长期优化已经开始进入 Loop Engineering 阶段。`sglang-sota-performance` 把“追 SOTA”拆成公平 benchmark、gap decision、profile、patch、revalidate；Humanize/RLCR 提供更强的外部审查，Codex Goal 可以承担一部分低成本持续迭代。
- Review 会变得更重要。Agent 能跑更多实验，也会生成更多看起来合理但需要审查的改动。开发者的工作会更多转向定义问题、选择证据、设计流程和判断结果能否进入生产路径。

## 2. 为什么 SGLang 适合被 Agent 化

SGLang 是一个面向 LLM 和多模态模型的高性能 serving framework。模型类型和硬件路径变多后，开发里会遇到几类常见问题：

- LLM 路径复杂：同一个性能问题可能跨 Python runtime、scheduler、CUDA graph、Triton/CUDA kernel、FlashInfer/FlashAttention、distributed collective 和 model-specific wrapper。
- Diffusion 路径也复杂：一个 denoise 变慢的问题，可能和 pipeline/stage 划分、DiT block、attention backend、torch.compile graph break、CFG/SP 并行、VAE 或自定义 fused kernel 有关。
- 验证成本高：很多修改必须在 H100/H200/B200 或 RTX 5090 上跑真实模型、真实 workload，单靠本地单元测试不够。
- profile 难以手工复用：一次 trace 里可能有几百个 kernel launch，仅靠人工看 Perfetto 很容易遗漏 kernel 到 Python source 的映射，也容易混淆 prefill 和 decode。人类在读 profiler 的过程中会积累很多 know-how，比如哪些 kernel 名字对应哪段模型逻辑、哪些 launch pattern 暗示 graph break、哪些 NCCL/attention/MLP 排布是正常的；但这些经验如果只留在个人脑子里，很难被下一个任务复用。
- 性能结论非常依赖上下文：GPU 型号、shape、batch、parallelism、precision、backend、compile 状态都会改变结论。一个孤立 microbench 很可能无法说明真实模型端收益，所以还需要建立端到端的长测试过程，用固定 workload 反复验证吞吐、延迟、显存、精度和稳定性。这件事本身就很费力，也很耗时。

这些问题恰好适合交给 Agent 处理。启动 server、固定 workload、采集 trace、初筛 profile row、补测试、记录实验结果，本质上都有明确输入输出，也很适合脚本化和重复执行。开发者需要提供的是边界：同一个 benchmark 设置、同一套 profile 解释规则、同一组 accuracy gate，以及什么情况下应该停止继续改代码。

因此，这里讨论的 Agent 是被工程流程约束的执行器。SGLang 开发里反复出现的流程适合沉淀成 skills，让 Agent 承担重复执行、证据收集和状态记录；开发者则负责定义目标、判断证据、审查改动能否进入真实 serving 路径。

## 3. 从 Prompt 到 Skill

在 SGLang 框架里，一个有用的 skill 至少要交代清楚几件事：


| 问题     | Skill 中需要沉淀的内容                             |
| ------ | ------------------------------------------ |
| 什么时候使用 | 触发场景、适用模型、适用硬件、哪些情况必须停止                    |
| 怎么开始   | preflight、环境变量、repo 状态、依赖检查、模型配置           |
| 怎么验证   | benchmark 命令、profile 命令、测试入口、accuracy gate |
| 怎么判断   | 输出表格、失败模式、优先级、risk 分类、fallback 条件          |
| 怎么交付   | artifact 目录、结果 schema、PR 描述、复现命令、审查要求      |


因此，SGLang 主仓和外部 skill 仓库会自然分工。主仓更适合放和源码一起演进的流程，比如 debug、test、diffusion model 添加和 benchmark/profile；外部仓库更适合放跨框架 benchmark、profile 分析、生产排障、PR 优化知识库和 Humanize/RLCR 这类上层 workflow。

## 4. 当前 Skill 栈

目前比较常用的几类如下。有的已经在 [sgl-project/sglang](https://github.com/sgl-project/sglang) 主仓维护，有的放在 [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) 和 [KDA-Pilot](https://github.com/BBuf/KDA-Pilot) 里。


| 层级                                             | 代表 skill / 项目                                                                                                                                                          | 解决的问题                                                                                      |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| CUDA crash                                     | [debug-cuda-crash](https://github.com/sgl-project/sglang/tree/main/.claude/skills/debug-cuda-crash)                                                                  | 在 custom op/kernel API 边界记录输入、异常和 dump，把瞬时 crash 转成可离线分析样本                                 |
| LLM benchmark                                  | [llm-serving-auto-benchmark](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-serving-auto-benchmark)                                        | 对 SGLang 和其它开源推理框架做公平、有限预算、可恢复的 serving benchmark search                          |
| Trace triage                                   | [llm-torch-profiler-analysis](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-torch-profiler-analysis)                                      | 固定输出 kernel table、overlap-opportunity table、fuse-pattern table，并把 kernel 映射回 Python source |
| Pipeline/layer analysis                        | [llm-pipeline-analysis](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/llm-pipeline-analysis)                                                  | 把 torch profiler trace 切到 forward pass、layer 和 kernel flow，定位稳态 pass、瓶颈层类型和 Perfetto 时间范围  |
| Diffusion benchmark/profile                    | [sglang-diffusion-benchmark-profile](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-benchmark-profile) | 采集 denoise latency、perf dump、torch profiler，并先检查是否真正走 native SGLang diffusion backend      |
| Diffusion 添加模型                                 | [sglang-diffusion-add-model](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-add-model)                 | 从 Diffusers/reference pipeline 出发，按 SGLang pipeline/stage/model/config 结构添加新 diffusion 模型  |
| Diffusion 性能调参                                 | [sglang-diffusion-performance](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills/sglang-diffusion-performance)             | 选择 torch.compile、warmup、SP/CFG parallel、offload、attention backend、quantization 等性能参数       |
| 生产排障                                           | [sglang-prod-incident-triage](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-prod-incident-triage)                                      | 从 live server 收集 bundle、保存 failing request、replay，再接 crash/hang/profile 等专项工具              |
| SGLang SOTA Performance Loop（Loop Engineering） | [sglang-sota-humanize-loop](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/skills/sglang-sota-humanize-loop)                                          | 先公平比较 SGLang 和指定的其它开源推理框架，再把 gap decision、profile、patch、复测放进 Humanize/RLCR loop       |


这张表不是在堆工具。它想强调的是：把容易遗漏的步骤写成协议，让流程能执行，能恢复，也能被人审查。

## 5. 几个优化案例

下面几个 SGLang PR 展示了 Agent + skills 在真实开发任务中的作用。表中关注完整流程：benchmark、profile、定位、修改和复测。


| 案例                                                                                                          | 结果                                                                                              | 关键点                                                                                                    |
| ----------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| FlashInfer RoPE graph break 定位，[SGLang PR #20699](https://github.com/sgl-project/sglang/pull/20699)         | profile 中单 block 约 `5.763 ms -> 4.592 ms`，Qwen-Image-2512 H200 1024x1024 实验中观测到明显模型端提升          | Agent 用 `torch._dynamo.explain` 发现 kernel 不在 torch.compile region 中，然后把调用改成 compile-friendly custom op |
| Z-Image compile fallback 修复，[SGLang PR #20962](https://github.com/sgl-project/sglang/pull/20962)            | `Z-Image-Turbo` 9-step denoise `1362.583 ms -> 744.656 ms`，50-step `6297.997 ms -> 4652.565 ms` | 关键问题在 fp32 RMSNorm + wrapper 导致 compile fallback，而不是某个 kernel 本身慢                                      |
| Diffusion Q/K RMSNorm + RoPE 融合，[SGLang PR #21440](https://github.com/sgl-project/sglang/pull/21440)        | `Qwen-Image` denoise `14.43 s -> 12.36 s`，约 `14.35%`                                            | Agent 在现有 profile 和 fused pattern 基础上实现 JIT CUDA kernel，并接入多个 diffusion DiT 路径                         |
| Qwen3-Next FlashInfer allreduce fusion，[SGLang PR #22664](https://github.com/sgl-project/sglang/pull/22664) | H100 TP=4 上 throughput `5.49 req/s -> 9.41 req/s`，约 `71.4%`                                     | 这类改动依赖 model path、collective fusion、benchmark 和真实 serving 验证，需要 profile 和复测闭环                          |
| LTX-2.3 one-stage denoise 优化，[SGLang PR #22861](https://github.com/sgl-project/sglang/pull/22861)           | B200 E2E `36.71 s -> 22.69 s`，denoise `34.00 s -> 19.78 s`                                      | 把 guider forward 合并、缓存静态张量、接入 fused kernel，体现模型级优化不只是单 kernel 优化                                       |


这些例子里，Agent 的主要作用体现在流程执行上：跑 benchmark、读 profile、定位 Python source、改代码、补测试、复测、整理 PR 描述。没有 skills 的时候，很多步骤依赖人工提醒；写进 skills 后，流程更容易重复。

## 6. Profiler Skills: 从 Trace 到 Layer

SGLang 性能优化里最常见的误区，是只看一个总耗时或者只打开 Perfetto 浏览几分钟，然后凭感觉判断“应该 fuse 一下”。这对 Agent 更危险，因为它很容易把一个看起来热的 kernel 误判为真正 bottleneck。

实际分析时，通常先用两个 profiler skills 配合。`llm-torch-profiler-analysis` 负责第一层 trace triage，把全局 profile 固定成三张表：

- `Kernel Table`：按 stage 统计 GPU time share、launch count、kernel category，并尽量映射回 Python source 和 CPU op。
- `Overlap Opportunity Table`：根据 exclusive/hidden 时间占比、依赖风险、kernel 类别判断哪里还有 overlap/headroom。
- `Fuse Pattern Table`：用 source-backed 的 pattern catalog 对照 SGLang、其它开源推理框架和 kernel library 中已有或正在推进的 fusion/overlap path。

这三张表先回答最基本的问题：哪个 stage 的哪个 kernel 占了多少 GPU time，映射到哪行 Python，是否已经有可借鉴的 fuse/overlap path。如果 SGLang 落后于其它推理框架，第一步不是立刻改代码，而是先让 profiler table 解释 gap。

再往下，就是 `llm-pipeline-analysis`。全局热点知道以后，还要知道热点落在哪个 forward pass、哪类 layer、哪条 kernel flow 上。这个 skill 会读取 Chrome trace JSON 和模型 `config.json`，用 layer-boundary anchor kernel 把 trace 切成 forward passes 和 layers，然后输出更适合深挖的几类表：

- `Forward pass summary`：区分 cold-start 和 steady-state，避免把预热阶段误当成优化目标。
- `Per-layer timeline`：按 layer 给出 wall time、sum duration，以及 MLA、MoE、GEMM、NCCL、MHC、Hadamard 等类别占比。
- `Layer cluster statistics`：对有交替层结构的模型尤其有用，比如带 `compress_ratios` 的 NSA/混合注意力模型，可以看出 C4_LIGHT、C128_HEAVY、HASH 等层类型谁在拖慢。
- `Compute flow table`：选中代表性 layer 后，展开到具体 kernel flow，给出 hotness、相对时间戳和输入维度，方便直接跳回 Perfetto。

这样 profile 分析会变成两步：先用 `llm-torch-profiler-analysis` 找到全 trace 的主要矛盾，再用 `llm-pipeline-analysis` 把问题落到稳态 forward pass、代表性 layer 和具体 kernel flow。前者避免凭感觉选方向，后者避免只盯着一个全局 hot kernel，却忽略模型结构里的层类型差异。

## 7. SGLang SOTA Performance Loop（Loop Engineering）

单个 skill 可以把一次任务做稳。但十几轮实验之后，另一个问题会冒出来：哪个候选版本最好，哪些思路已经失败，上一轮 NCU 说明了什么，benchmark 是否还对齐 baseline，什么时候应该停下。这些状态不能只靠聊天上下文记着。

这就是 Loop Engineering 要解决的问题。`sglang-sota-performance` 是一个面向 SGLang serving 的例子。这里的 SOTA 指固定实验条件下的可复现最好结果：同一个模型、硬件、GPU 数量、precision、workload、SLA、framework commit 和 serving 参数，SGLang 能否达到当前最好的可复现实验结果。

SGLang SOTA Performance Loop

图 1：SGLang SOTA Performance Loop。固定公平 benchmark 先给出可复现 baseline，后续 gap decision、profile、pipeline analysis、patch 和 revalidate 由 Humanize/RLCR loop 持续推进。

一个完整的 SGLang SOTA Performance Loop 包含以下阶段：

1. 定义目标边界。例如 `Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`、single-node 2x B200、FP8、SGLang TP=2，并在同样 2 卡预算下和指定的其它开源推理框架比较。
2. 先做公平 search。在 patch SGLang 之前，先用相同 workload 和资源预算搜索 SGLang 以及每个指定开源推理框架的可复现最好命令。
3. 判断 gap。如果 SGLang 已经持平或领先，就记录完成证据；如果稳定落后超过阈值，进入 profile。
4. 用 profile 解释 gap。先别急着改代码，先产出 kernel table、pipeline table、overlap/fuse table，必要时补 NCU。
5. 只 patch 有证据支持的路径。例如 hybrid attention、Mamba/GDN、radix cache、target verify、CUDA graph、MoE/EP、quant kernel 或 model wrapper。
6. 回到同一 workload 复测。每一轮都记录 benchmark、profile、accuracy、失败尝试、环境信息和清理动作。

在 Qwen3.6-35B-A3B-FP8 的 B200/H200 实验中，同一模型在不同硬件上的瓶颈并不相同。B200 上，固定 workload 下 SGLang baseline 已经超过一个同类开源推理框架；继续 profile 后仍然发现 GDN prefill split 存在优化空间，patch 后 chat 和 long 场景 output tok/s 均继续提升约 `2.6%`。H200 上，需要结合 FP8 MoE Triton configs、CUTLASS scaled-mm replacement path、GDN backend 默认值等改动，才能追平并超过同类开源基线。类似任务如果拆成多个独立 prompt，benchmark、profile、失败尝试和中间结论很容易丢失；放进带证据和审查的 loop 里，更容易保持前后条件一致。

## 8. Humanize/RLCR：给 Loop 加外部审查

Humanize 处理的是长期任务里的状态和审查问题。一个高风险的 SGLang 性能任务通常不会“一轮实现”就结束，它会经历多轮 benchmark、profile、patch、revert、换方向、再验证。Humanize 把这些动作拆成两个阶段：

1. 先 gen-plan。用 `humanize-gen-plan` 把草稿需求整理成结构化 `plan.md`，里面包含 goal description、acceptance criteria、positive/negative tests、path boundaries、milestones 和 implementation notes。
2. 再跑 RLCR loop。用 `humanize-rlcr` 从 `plan.md` 启动循环。每轮 Agent 读取 `.humanize/rlcr/<timestamp>/round-<N>-prompt.md`，实现、提交、写 summary，然后由 native Stop hook 触发 Codex review。hook 会检查状态文件、summary、git clean、review 结果、open question、max iteration 等条件，不能靠一句“任务完成”跳过。

`sglang-sota-humanize-loop` 先由 SOTA performance workflow 固定 baseline 和 gap，再把后续 gap decision、profile、pipeline analysis、patch、revalidate 交给 Humanize/RLCR 托管。这样做的成本更高，但适合会进入 PR、会影响 serving 正确性、或者需要多天多轮实验的任务。

实际使用时，命令顺序最好写得很明确，避免 Agent 直接跳进实现：

```text
1. Write a task draft under artifact_root/draft.md.
2. Run humanize-gen-plan to generate artifact_root/plan.md.
3. Start humanize-rlcr from artifact_root/plan.md.
4. Keep all decisions, summaries, and review state in the local Humanize workspace.
```

## 9. Codex Goal: 成本更低的轻量平替

Humanize 的强审查很适合严肃 PR，但启动成本也不低：要生成 plan、维护 `.humanize` 状态、每轮提交、等待 Stop-hook review。对于一些探索性 SOTA loop，尤其是想在 B200/B300 上快速把一组模型跑起来时，可以用 Codex Goal 做一个轻量平替。

Codex manual 对 Goal mode 的定义很适合这类任务：Goal 给 Codex 一个跨多步任务持续推进的目标，目标文本同时承担起始 prompt 和完成条件；好的 goal 应该包含具体 outcome、可度量目标或测试标准。换到 SGLang 框架里，就是把“公平 benchmark -> gap decision -> profile -> patch -> revalidate -> artifact ledger”全部写进一个 `/goal`，由当前 Goal 承载循环状态。

Goal 版适合快速自动跑，成本更低，状态也更贴近当前 Codex 线程；Humanize 版适合高风险改动、长期任务和需要强 review gate 的 PR。实际使用时，可以把同一类 SOTA prompt 改成新的硬件目标后交给 Codex Goal 继续跑。它不能替代审查，但在目标、证据、停止条件和 artifact contract 写清楚的情况下，可以承载一部分 SOTA Loop 的自动迭代。

两种方式的差异如下：


| 维度   | Humanize/RLCR                                             | Codex Goal                                        |
| ---- | --------------------------------------------------------- | ------------------------------------------------- |
| 启动方式 | 先 `humanize-gen-plan` 生成 `plan.md`，再从 `plan.md` 启动 RLCR   | 一个 `/goal` 承载目标、约束和完成条件                           |
| 状态位置 | `.humanize/rlcr/...` 里的 plan、prompt、summary、review result | 当前 Goal 线程 + `artifact_root` 里的 manifest/evidence |
| 审查强度 | Stop hook + Codex review + git/state/schema 检查            | 依赖 goal 文本、artifact contract 和人工抽查                |
| 成本   | 更高，适合长周期和高风险 PR                                           | 更低，适合快速实验和模型优化探索                                  |
| 主要风险 | loop 配置复杂、每轮成本高                                           | 容易目标漂移或过早完成，需要写硬停止条件                              |


下面是 [AI-Infra-Auto-Driven-SKILLS/prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts) 中的一个 2x B200 模型优化例子，并改写成英文 prompt。

Humanize 版采用“先写草稿，再让 Humanize 生成计划并托管循环”的方式：

```text
Use the sglang-sota-humanize-loop workflow.

Task:
Optimize SGLang serving performance for Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
on a single node with 2 NVIDIA B200 GPUs, FP8 precision, and initial SGLang
TP=2. SGLang should match or exceed the best reproducible result from the
requested open-source inference frameworks under the same 2-GPU budget,
workload, SLA, model, precision, and environment constraints.

Required workflow:
1. Create a draft task document under artifact_root.
2. Run humanize-gen-plan to turn the draft into a structured plan.md.
3. Start humanize-rlcr from that plan.md in the local Codex session.
4. Keep benchmark, profile, patch, and revalidation decisions inside the same
   Humanize workspace.

Evidence and safety requirements:
- Before patching, run a fair bounded search for SGLang and the requested
  open-source inference framework set.
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

Goal 版则把循环直接压进 `/goal`，重点是把完成条件和停止条件写硬：

```text
/goal Keep optimizing SGLang serving for
`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8` on a single node with 2 NVIDIA B200
GPUs until SGLang matches or exceeds the best reproducible result from the
requested open-source inference frameworks under the same 2-GPU budget, FP8
precision, workload, SLA, model, and environment constraints. The current Codex Goal is the loop: fixed fair
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
- Before patching, run a fair bounded search for SGLang and the requested
  open-source inference frameworks under the same 2-GPU budget.
- If SGLang is behind by more than 1%, profile in the same Goal, then use
  llm-torch-profiler-analysis, llm-pipeline-analysis, and ncu-report-skill when
  needed before patching.
- Focus on hybrid attention, Mamba/GDN, radix cache, target verify, and CUDA graph.
- Update the artifact manifest, benchmark evidence, profile evidence, failed
  attempts, and next-step decision after every round.
- Stop and report a blocker if resources are unavailable, evidence is
  untrustworthy, the budget is exhausted, or no defensible next patch exists.
```

Humanize 版把“计划”和“审查”显式外置，适合比较严肃的交付；Goal 版把“循环”和“完成条件”写进一个持久目标，适合更轻量的自动探索。二者可以共存，不需要互相替代。

## 10. 基于 KDA 的 SGLang 系统 CUDA kernel 优化

LLM 和 diffusion 的模型级优化之外，更难的是 kernel 优化。直接让 Agent 写 CUDA，很容易出现 benchmark reward hacking：改 benchmark、走更轻的 wrapper、打开 baseline 没有的 fast math、只优化一个 shape、破坏数值语义，或者在真实 SGLang 路径里没有任何收益。

KDA-Pilot 的思路是把 kernel 优化拆成独立任务，而不是让 Agent 在 SGLang 大仓里随意改：

- workload 来自真实 SGLang diffusion 模型，先跑 20 个 diffusion 模型并汇总实际 kernel meta。
- baseline 从 upstream SGLang main 复制，记录 source lineage。
- baseline 和 candidate 必须走同一个本地 ABI、同一个 build/export path。
- benchmark 使用固定 production rows、A/B interleaving、CUDA event 或 wall timing。
- correctness 覆盖 production rows、canonical regression grid、NaN/Inf、poison output 和 fallback contract。
- 每轮迭代刷新 task prompt、benchmark evidence、KernelWiki 和 ncu-report-skill。
- 允许 shape-specialized dispatch，但必须写清楚每个 bucket 的条件、路径、latency 和 fallback。

截至 2026-06-27，KDA-Pilot 派生的优化已经有 3 个 upstream SGLang PR 落地。第一个是 [SGLang PR #27392](https://github.com/sgl-project/sglang/pull/27392)，为 Qwen-Image-2512 合入 B200 native diffusion norm-scale-shift CUDA fast path；这周又合入了 [SGLang PR #29281](https://github.com/sgl-project/sglang/pull/29281) 和 [SGLang PR #29361](https://github.com/sgl-project/sglang/pull/29361)，分别覆盖 Cosmos3 VAE causal Conv3D cat/pad copy path 和 LTX-2.3 residual-gate update path。

| Upstream PR | 目标路径 | Kernel 侧证据 | 模型路径证据 |
| --- | --- | --- | --- |
| [#27392](https://github.com/sgl-project/sglang/pull/27392) | Qwen-Image norm-scale-shift | profiler attribution 中 target kernel group 提升 `1.279x` | 单张 B200 上双方各 5 次 interleaved 运行，full request `1.125x`，denoise wall `1.130x` |
| [#29281](https://github.com/sgl-project/sglang/pull/29281) | Cosmos3 causal Conv3D cat/pad | B200 traced VAE decode 中 weighted kernel group `10.621 ms -> 5.240 ms`，约 `2.03x` | Cosmos3-Nano T2V 开启 `torch.compile` 后，median E2E `181.521 ms -> 177.687 ms`，约 `1.021x` |
| [#29361](https://github.com/sgl-project/sglang/pull/29361) | LTX-2.3 residual-gate update | B200 LTX-2.3 large rows 相比已有 Triton path 提升 `1.108x` 到 `1.130x`，相邻 diffusion rows 最高到 `2.587x` | LTX-2.3 HQ T2V E2E `46644.08 ms -> 45198.37 ms`，约 `1.032x` |

KDA-Pilot B200 diffusion kernel results

图 2：KDA-Pilot 在 B200 上跟踪了 10 个 SGLang diffusion kernel task。大部分行给出 KDA-Pilot benchmark ledger 里的 wall-geomean speedup；这里的 wall time 包含 Python dispatch、wrapper、kernel launch 和 `cuda.synchronize()` 能观察到的同步开销，比单纯 kernel device time 更接近真实调用路径。`residual_gate_add` 使用已经合并的 SGLang PR #29361 中 B200 LTX-2.3 Triton-row 的加速数据，图和表里按 `1.11x` 记录。


| Kernel task                 | B200 evidence | 主要优化方向                                                              |
| --------------------------- | ----------------- | ------------------------------------------------------------------- |
| `qknorm_rope`               | `1.1341x`         | shared RoPE staging、Q/K 复用、large rows fast path                     |
| `norm_infer`                | `1.3523x`         | warp-row RMS、tiled persistent RMS、8B/16B vector path                |
| `rotary_embedding`          | `1.4912x`         | 128-bit vector I/O、cos/sin hoisting、LTX2 block matching             |
| `cutedsl_norm_tanh_mul_add` | `1.4953x`         | row-invariant math hoisting、launch-bounds tuning、exact tanh         |
| `cutedsl_norm_scale_shift`  | `1.3201x`         | operand-class dispatch、16B/32B vector、two-pass variance             |
| `fuse_scale_shift`          | `2.7499x`         | rowgrid/flatvec/exact-C paths、cache hints、one-pass reduction        |
| `group_norm_silu`           | `2.3118x`         | split-group stats、channels-last direct path、fallback for giant rows |
| `attention_concat_copy`     | `1.30x`           | single-launch region copy、pitched 16B block gather、严格 layout/device 检查 |
| `causal_conv3d_cat_pad`     | `2.06x`           | flat chunking、16B vectorized stores、stride-aware fallback、bitwise-exact gate |
| `residual_gate_add`         | `1.11x`           | one-pass CUDA fusion、pinned-GPU correctness、SGLang PR #29361 的 B200 Triton-row re-benchmark |


这些数字要看清实验设置：它们是 extracted production rows 上的 kernel task speedup，不是完整模型端到端提升。即便如此，它们仍然有价值。只要 baseline、workload、correctness、profile 和 review 都固定住，Agent 就能在真实框架 kernel 上做出可审查的增量。

KDA-Pilot 的实验里，有两条规则最值得保留：

- 不要给 Agent 留 benchmark reward hacking 的空间。baseline/candidate ABI 不一致、fast math 不对齐、wrapper 路径不同，结果都会失真。还有一种常见问题是看完结果以后再换 benchmark shape 集合，比如把 candidate 变慢的 shape 从表里删掉，这种结果也不能用。
- 接近 Roofline 的 bucket 应该允许 no-go 或 fallback。好的 kernel 优化任务不应该逼 Agent 每个 shape 都赢。对于 giant contiguous bucket 或接近带宽上限的路径，记录 fallback 可能比继续堆复杂度更正确。

## 11. 几条实践规则

1. 先定义任务边界，再启动 Agent。

“优化 SGLang”太大，“给 `Qwen/Qwen3.6-35B-A3B-FP8` 在 1x H200、固定 `1000->1000` 和 `8000->1000` workload 下追平某个同类开源推理框架”才是一个可执行目标。

1. 先固定 benchmark，再看 profile。

如果 workload 可以在看到结果后改变，Agent 很容易无意中优化到一个更简单的问题。SOTA loop 和 KDA-Pilot 都把固定 workload 放在 patch 之前。

1. 看 NCU 结果要先判断 kernel 的计算性质。

memory-bound kernel 重点看 DRAM/L2 throughput、load/store efficiency、memory pipe utilization；compute-bound GEMM/attention kernel 重点看 Tensor Core utilization、SM busy、eligible warps 和主要 stall 原因；小而碎的 latency-bound kernel 则要看 launch count、单 kernel duration、同步点，以及有没有融合空间。只贴一张 trace 截图不够，至少要说清楚哪个指标支持下一步改动。

1. Diffusion 先确认没有 fallback。

如果日志里已经 fallback 到 diffusers backend，就不能把 trace 当成 native SGLang diffusion 的证据继续分析。这类硬停止条件应该写进 skill，而不是靠人每次提醒。

1. Kernel 优化要同 ABI、同 wrapper、同 compile flags。

尤其不要让 candidate 偷偷走更轻路径，也不要单边开启 `--use_fast_math`。

1. Review 能力比以前更重要。

Agent 能制造更多 PR，也能制造更多看似合理的错误。SGLang 这类高性能系统的 review 需要关注 shape、dtype、distributed、CUDA graph、fallback、accuracy、serving API、metrics 和 benchmark 设置。

Agent 时代的 SGLang 开发，不会把开发者从系统里拿掉。更现实的变化是：把开发者的经验写成流程，把重复执行交给 Agent，把判断、设计和审查留给人。对开源推理框架来说，这类基础设施值得长期投入。

## 12. 参考资料

- [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
- [SGLang main `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/.claude/skills)
- [SGLang diffusion `.claude/skills`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/.claude/skills)
- [AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)
- [AI-Infra-Auto-Driven-SKILLS prompts](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS/tree/main/prompts)
- [KDA-Pilot](https://github.com/BBuf/KDA-Pilot)
- [SGLang Diffusion Advanced Optimizations, LMSYS Blog](https://lmsys.org/blog/2026-02-16-sglang-diffusion-advanced-optimizations/)
- [OpenAI Codex Prompting: Goal mode](https://developers.openai.com/codex/prompting#goal-mode)
- [Humanize](https://github.com/PolyArch/humanize)

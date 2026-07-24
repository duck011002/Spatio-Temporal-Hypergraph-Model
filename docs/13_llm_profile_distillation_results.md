# LLM 语义画像蒸馏实验结果

## 1. 实验目标

本阶段在冻结 A4 主干与既有 MoE 的前提下，引入离线 LLM 教师。LLM 不参与在线推荐，也不接收用户轨迹、POI 标识或训练/验证/测试标签；其唯一职责是为 Foursquare 类别名称生成通用语义画像。部署时仅保留约 2.32 万个可训练参数的本地语义专家。

NYC 共包含 207 个去重类别名称。已有缓存覆盖 197 类，本阶段新增调用 qwen-flash 10 次，预计费用为 0.0052584 元，实际新增费用为 0.004104 元，低于 0.02 元硬预算。

## 2. 方法冻结过程

第一版采用联合交叉熵训练的双线性语义打分器。该方案在 step 500–2500 的所有 validation 点均低于 step 0 的 A4 基线；语义 scale 变为负值，gate 均值升至约 0.82，说明模型通过任意旋转或反向利用语义空间来拟合训练集，而不是保留 LLM 画像的原始含义。因此该版本在 step 2500 后终止，未进入 test。

第二版改为两阶段画像蒸馏：

1. 前 2000 batches 仅使用 Smooth L1 损失，让轨迹状态预测真实下一 POI 的冻结 LLM 画像；此阶段语义 logit 残差严格为零。
2. warmup 后启用非负语义残差。若语义方向无效，scale 可以保持为零并退回 A4，但不允许学习负 scale 反向使用画像。
3. 使用 validation 的 selection/calibration/audit 固定划分。selection 用于选择快照，calibration 用于选择非负残差倍率，audit 不参与选择。
4. 最终选择续训 step 2000 快照，总计约 7002 个语义训练 batches；calibration 选择残差倍率 1.5。

A4 公共参数在加载前后逐张量完全一致。warmup 快照的平均语义残差为 0，且冻结 A4/MoE 始终处于 eval 模式。

## 3. Paired validation 结果

Paired 评估在同一次骨干前向中同时计算 `base logits` 和 `base + semantic delta`，从而消除 PyG/CUDA 稀疏算子在不同前向之间约 1–3 个样本的排序波动。

| 分区 | 方法 | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---:|---:|---:|---:|---:|
| Selection | A4 | 0.252381 | 0.497619 | 0.590476 | 0.635714 | 0.364400 |
| Selection | A4 + LLM | 0.255952 | 0.498810 | 0.590476 | 0.636905 | 0.366354 |
| Calibration | A4 | 0.278571 | 0.550000 | 0.617857 | 0.671429 | 0.400465 |
| Calibration | A4 + LLM | 0.282143 | 0.553571 | 0.617857 | 0.675000 | 0.401754 |
| Audit | A4 | 0.271429 | 0.482143 | 0.585714 | 0.639286 | 0.369010 |
| Audit | A4 + LLM | 0.275000 | 0.482143 | 0.585714 | 0.639286 | 0.370988 |
| 全部 validation | A4 | 0.261429 | 0.505000 | 0.595000 | 0.643571 | 0.372535 |
| 全部 validation | A4 + LLM | 0.265000 | 0.506429 | 0.595000 | 0.645000 | 0.374361 |

全部 validation 上，LLM 语义专家使 R@1 提升 0.003571、R@5 提升 0.001429、R@20 提升 0.001429、MRR 提升 0.001825。selection、calibration 和 audit 三个分区的 R@1 与 MRR 均未下降。由于没有重复种子或显著性检验，这些差异只能视为小幅正向证据，不能表述为统计显著。

## 4. 画像消融

消融均固定使用同一快照和 1.5 倍残差，不重新调参。

| 画像设置 | validation R@1 变化 | validation R@10 变化 | validation MRR 变化 | 结论 |
|---|---:|---:|---:|---|
| 真实 LLM 画像 | +0.003571 | 0.000000 | +0.001825 | 小幅正向 |
| 全零画像 | 0.000000 | 0.000000 | 0.000000 | 只产生候选间相同偏置，不改变排序 |
| POI 间打乱画像 | -0.001429 | -0.001429 | -0.000738 | 破坏类别—POI 对齐后性能下降 |

全零画像不改变任何分区的排序，而打乱画像在 validation 全集及 audit 上降低 MRR。这说明正向 validation 结果依赖正确对齐的 LLM 类别语义，而不是新增参数量或任意 logit 扰动。

## 5. 与规则教师的关系

冻结规则教师仍是当前最可靠的精度提升来源。既有 A4 规则教师 test 结果为 R@1=0.275427、MRR=0.392920，分别较对应 A4 候选基线提升 0.014105 和 0.013389。

直接把冻结规则权重串接到已校准的 LLM 分数后，validation 全集 R@1 变化为 -0.000714、MRR 变化为 -0.000323，虽然 R@5/R@10/R@20 有小幅提升，但不满足主排序指标约束。因此当前不选择“规则 + LLM 串联”作为部署方案；论文中应将规则教师与 LLM 语义教师分别报告。

## 6. 唯一一次冻结 test

在快照、残差倍率和组合方式全部冻结后，仅执行一次 paired test，且没有使用 test 调整任何参数。

| 方法 | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| Paired A4 | 0.259837 | 0.526355 | 0.611730 | 0.679287 | 0.378355 |
| Paired A4 + LLM | 0.259094 | 0.527840 | 0.611730 | 0.679287 | 0.377954 |
| 变化 | -0.000742 | +0.001485 | 0.000000 | 0.000000 | -0.000400 |

LLM 分支在 test 上将 R@5 提升 2/1347，但 R@1 减少 1/1347，MRR 小幅下降。与此同时，同一 A4 checkpoint 在不同 CUDA 稀疏前向之间曾出现约 2 个 R@1 样本的波动。因此，test 证据支持“LLM 语义有助于候选覆盖”的有限结论，但不支持“稳定提升 Top-1 或 MRR”的结论。不得根据该 test 结果继续选择倍率或重复 test。

## 7. 论文建议表述

可以据实表述为：

> 本文提出一种无需在线 LLM 推理的类别语义画像蒸馏方法。该方法通过画像回归 warmup 与非负残差约束，将 qwen-flash 的通用类别知识蒸馏到约 2.32 万参数的本地专家中。Paired validation 与画像打乱消融表明，正确对齐的 LLM 语义能够改善候选排序；在唯一一次冻结 test 中，方法提升了 Recall@5，但 Top-1 与 MRR 的增益未稳定复现。

不应表述为：

- LLM 在 test 上全面优于 A4；
- LLM 已显著提升 R@1；
- 规则教师与 LLM 串联后进一步提升所有指标。

## 8. 可复现资产

- 冻结语义快照：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/llm_semantic_selected_step_2000.pt`
- 快照 SHA-256：`bb78d31943e5ddf10c8d8051c254b447b79ae8d5cdeac83b7bab897a596b0f19`
- 快照大小：581294 bytes
- 校准 validation：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/calibrated_validation.json`
- 零画像消融：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/ablation_zero.json`
- 打乱画像消融：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/ablation_shuffle.json`
- 规则组合验证：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/paired_rule_validation.json`
- 唯一 test：`server_artifacts/nyc_llm_semantic_profile_distill_20260724/frozen_test_once.json`

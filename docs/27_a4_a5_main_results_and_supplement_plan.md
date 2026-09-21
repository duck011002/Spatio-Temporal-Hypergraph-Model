# A4/A5 主实验整理与补实验计划

更新时间：2026-09-21。本文只登记已经落到本地、可复核的结果；没有把 validation 数字写成 test，也没有把旧版 NYC Jev 协议冒充当前正式 A5。

## 1. 当前主线结论

三套数据的最终 A4 全量训练都已经完成：NYC 使用已有 A4 结果，CA/TKY 使用本轮 AutoDL 全量训练结果。CA 的正式 A5-Jev 已完成唯一 paired test；TKY 的正式 A5-Jev 只完成 validation，因预注册准入条件未通过而没有发起 test 请求。

因此当前最接近论文主表的状态是：

| 数据集 | A4 全量 test | 当前 A5-Jev test | 结论状态 |
|---|---|---|---|
| NYC | 已有 | 有历史类别融合 test，但不是当前 `a5_jev` 正式协议 | 可作历史参考，不能与 CA 直接并列为同协议 A5 |
| CA | 已完成 | 已完成，validation 准入、paired test 均通过 | 当前最完整的 A4→A5 主实验 |
| TKY | 已完成 | validation 有正向总体现象，但 calibration Top-1 净变化为 -1，test 被正确阻止 | 主实验缺正式 test |

## 2. A4 全量 test 基线

下表的 CA/TKY 数字来自本轮 A4 训练日志；NYC 数字来自已有 A4 候选缓存对应的 paired 基线。A4 重新独立导出候选时，NeighborSampler 的随机状态可能造成少量末位差异，所以 A5 差值必须以同一份候选缓存中的 backbone 为准。

| 数据集 | N | R@1 | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NYC A4 | 1347 | 0.261321 | 0.526355 | 0.611730 | 0.676318 | 0.402178 | 0.429987 | 0.446764 | 0.379531 |
| CA A4 | 2780 | 0.148921 | 0.338129 | 0.411151 | 0.474101 | 0.249252 | 0.272989 | 0.288804 | 0.238283 |
| TKY A4 | 6868 | 0.289855 | 0.508099 | 0.591645 | 0.654163 | 0.406296 | 0.433486 | 0.449312 | 0.391489 |

NDCG@1 与 R@1 相同；完整 MAP 也已经包含在原始日志和候选报告中。NDCG 不需要重训，可以从保存的 Top-20 候选、标签和原始 rank 直接离线计算。

## 3. 当前正式 A5-Jev 结果

### 3.1 CA：唯一正式 paired test

CA 的 A5 使用 `weight=0.35, temperature=2`，validation 准入为 true。下面的 backbone 是与 A5 完全同一份候选导出的配对基线，不是 A4 训练日志中的另一份随机采样。

| 指标 | paired A4 | A5-Jev | 差值 |
|---|---:|---:|---:|
| R@1 | 0.153597 | 0.161871 | +0.827 pp |
| R@5 | 0.337770 | 0.344964 | +0.719 pp |
| R@10 | 0.411870 | 0.414388 | +0.252 pp |
| R@20 | 0.471583 | 0.471583 | 0.000 pp |
| NDCG@5 | 0.250697 | 0.258482 | +0.778 pp |
| NDCG@10 | 0.274736 | 0.281050 | +0.631 pp |
| NDCG@20 | 0.289891 | 0.295558 | +0.567 pp |
| MRR | 0.240460 | 0.247678 | +0.722 pp |

配对变化为 rescued=79、lost=56、net Top-1=+23。因为 A5 只重排已有 Top-20，R@20 不变是结构性质，不应把它当成失败指标；NDCG@20 仍可因名次变化而提升。

### 3.2 TKY：validation 已有正向信号，但 test 被阻止

当前冻结参数为 `weight=0.2, temperature=1`。全 validation 相对配对 backbone 的变化为：R@1 +0.612 pp、R@5 +0.539 pp、R@10 +0.801 pp、R@20 0、NDCG@5 +0.578 pp、NDCG@10 +0.647 pp、NDCG@20 +0.450 pp、MRR +0.556 pp。

但 calibration 分区为 net Top-1=-1、ΔMRR=+0.001225，所以按预先写入代码的准入条件，TKY test 被禁用。这是有效的负面结果记录，不应直接绕过门槛发 test。

同一份 validation 响应的离线网格诊断显示，若事先把“calibration 不降后再冻结参数、audit 只作最终准入”写成新的 A5.1 规则，则 `(0.2, 2)` 是一个可候选参数：calibration net Top-1=+1、audit net Top-1=+4、全体 net Top-1=+28。它目前只是候选，不是正式 test 结果。

### 3.3 NYC：历史结果只能作参考

NYC 历史类别融合 test 为 R@1 +2.004 pp、R@5 +1.262 pp、R@10 +0.965 pp、R@20 0，NDCG@5 +1.435 pp、NDCG@10 +1.368 pp、NDCG@20 +1.119 pp、MRR +1.434 pp。这个结果使用的是旧的 arithmetic/category 融合实验协议，虽然方向很好，但 prompt、分区和 log-fusion 实现与当前 `a5_jev` 不完全一致。因此它可以放在历史探索结果或补充材料，不能直接和 CA 的正式 A5 并列宣称“三数据集当前 A5 均已验证”。

## 4. NDCG 的本地复核

本次已经把服务器的以下内容同步到本地：

- `server_artifacts/a5_review_20260921/artifacts/ca_a4_candidates/`
- `server_artifacts/a5_review_20260921/artifacts/tky_a4_candidates/`
- `server_artifacts/a5_review_20260921/artifacts/ca_a5_jev/`
- `server_artifacts/a5_review_20260921/artifacts/tky_a5_jev/`
- `server_artifacts/a5_review_20260921/remote_runs/`

本地用相同的 `evaluate_ranking` 重新读取 CA/TKY 候选，得到的 backbone 指标与 A5 报告逐项一致；因此后续可以在本地补算 NDCG、MAP、MRR、paired delta 和用户级 bootstrap 区间，不需要重新训练，也不需要重新调用 Jev。服务器原始 A4 日志、候选 manifest 和数据哈希也一并保留。

本次实际重算输出位于 `docs/results/a4_a5_recomputed_20260921/summary.json`，执行入口为 `recompute_a4_a5_metrics.py`。重算使用 2,000 次、seed=`20260920` 的用户聚类 bootstrap：

| 结果 | ΔR@1 95% CI | ΔMRR 95% CI | rescued/lost | 备注 |
|---|---|---|---:|---|
| CA A5 formal test | [-0.000722, 0.017182] | [0.001460, 0.012934] | 79 / 56 | 点估计为正；R@1 区间跨 0，MRR 区间为正 |
| CA A5 validation | [-0.001695, 0.012764] | [0.000174, 0.009606] | 91 / 72 | 仅 validation |
| TKY 当前参数 validation | [0.001594, 0.010411] | [0.002774, 0.008174] | 143 / 101 | 不能替代 test |
| NYC 历史类别融合 test | [0.006675, 0.033779] | [0.005213, 0.023783] | 59 / 32 | 旧协议、历史 test |

这些区间是配对用户聚类 bootstrap 的描述性区间，不是多种子统计显著性结论。CA 的 NDCG、MAP、MRR、paired delta 也已逐样本重算；NYC 历史报告则用本地保存的历史候选和响应缓存重放得到完全一致的指标。

## 5. 补实验优先级

### P0：先完成主表整理（不耗 GPU、不耗 Jev）

1. 固定三数据集的 A4 主表，统一报告 R@1/5/10/20、NDCG@1/5/10/20、MAP@1/5/10/20、MRR。
2. 对 CA A5、TKY validation 和 NYC 历史 A5 计算用户级 paired bootstrap；这一步已完成，TKY 仍不能写成 test。
3. 在每个结果旁边标注协议版本：`A5-formal`、`NYC-historical-category`、`validation-only`，防止论文表格混口径。

### P1：优先补真正缺失的主实验

1. **TKY A5.1 paired test**：先把新的安全选择规则写入独立版本并在已有 validation 缓存上冻结，例如选择满足 calibration 不降的候选中 selection objective 最高者，audit 只作准入；然后只发起一次 TKY test。当前离线诊断预期首选 `(0.2, 2)`，但必须在规则冻结后再确认。
2. **NYC current-formal A5**：复用已有 NYC A4 候选/数据，按当前 `a5_jev` 的 prompt、log fusion、selection/calibration/audit 协议重新走一遍。它不需要重训 A4，但需要新的 validation/test Jev 响应，完成后才能形成真正同协议的 NYC/CA/TKY A5 主表。

这两项完成后，主实验才具备“同一 A5 协议、三个数据集、A4 paired baseline”的完整结构。CA 不需要重跑。

### P2：主实验稳健性与方法对照

1. 在最终 A4/A5 结构冻结后，优先给三数据集各补 2 个随机种子；若资源有限，先补 TKY 和 CA，再补 NYC。报告均值和标准差，不能把单种子提升写成统计显著。
2. 用最终 A4 候选补一套 CA/TKY 的 rule-teacher paired 对照；现有 CA/TKY rule-teacher 数字基于 R0 候选，不能当作最终 A4 联合结果。
3. 若论文要声称 Jev 优于规则教师，再补 `A4`、`A4+rule teacher`、`A4+Jev` 三路同候选比较；`A4+rule+Jev` 可作为次要组合消融，不应先于主表。

### P3：可放补充材料的实验

- Jev 类别准确率、outside-category 比例、uniform-category 对照。
- rescued/lost 案例和按用户活跃度、候选类别数的分桶。
- weight/temperature 敏感性与旧 arithmetic/log-pool/adaptive 融合对照。
- 规则教师组件消融、专家流量和路由熵等机制分析。

## 6. 当前最小执行清单

如果目标是尽快得到可写入论文的主结果，下一轮只需要：

1. 使用已生成的本地 A4/A5/NDCG/MRR 总表和置信区间；
2. 冻结并实现 A5.1，补 TKY 一次 test；
3. 按当前正式协议补 NYC 一次 A5 validation + test；
4. 再决定是否投入多种子和 rule-teacher 联合消融。

不需要重新训练 CA/TKY A4，也不需要重新计算 NDCG，更不应重跑已经完成的 CA Jev 请求。

## 7. GPU 开机后的明确工作

GPU 开机后不需要再重训已经完成的 CA/TKY A4。待执行的服务器工作按以下顺序固定：

1. 将 A5.1 的安全选参规则写入独立版本，使用本地已经重算的 TKY validation 缓存做冻结检查；不触碰原 `tky_a5_jev` 目录，不覆盖原失败准入记录。
2. 在 AutoDL 上只对 TKY 发起一次新的 test 请求并生成 `tky_a5_jev_v1` 的 paired test 报告；如果新规则未通过 audit，立即停止，不发 test。
3. 为 NYC 准备当前正式 `a5_jev` 所需的 A4 candidate manifest；完成 validation、select 后，只有准入通过才发起唯一 test 请求，目录与历史 `jev_category_nyc_20260920` 分开。
4. 拉回 TKY/NYC 的冻结文件、test predictions、test report、ledger 和日志，重新运行本地离线汇总。

GPU 侧明确不做：CA A5 重跑、CA/TKY A4 重训、已完成 CA Jev 请求重发、观察 test 后调参。多种子 A4 和 rule-teacher 联合消融放在上述主实验闭环之后。

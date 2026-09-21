# Method 写作依据：与当前代码逐项对应

## 1. 任务、继承与贡献边界

预测用户下一次实际签到的 POI。沿用 STHGCN 的数据划分、时空信息编码、轨迹超图与超边交互编码器；该部分必须引用 Yan et al., SIGIR 2023，不能写成本文发明。README 保留上游说明，不是本文实验台账。

本文实现层面的主线：在超图表示后加超图条件的分组 MoE；以外部 Jev 的 next-category 概率修正 POI 排序。可将“类别决策任务构造”和“保持类别内次序的融合”作为两个技术环节，但能否作为独立创新点还需文献对比和消融支持，不能仅以 API 新颖性宣称方法原创。

## 2. A4：超图条件分组 MoE

`model/sthgcn.py::forward` 提供局部表示 h_L 与超边协作后表示 h_H。路由输入是 `[h_L; h_H; h_H-h_L]`，经 LayerNorm 和 router 得到 softmax 权重 g。12 个 rank-32 routed experts、1 个 shared expert，每条样本选两个 routed experts。

`model/sparse_moe.py::_update_adaptive_grouping` 在前 3000 个训练 batch 累计路由概率的 Gram 矩阵。以专家路由向量的余弦相似度构建配对分数；12 个专家、6 组的当前设置调用 `_optimal_pair_groups`，用动态规划求总相似度最大的二元配对。之后组别冻结。**不是每个输入动态重新聚类，也不是使用 Jev 决定组别。**

分组后 `_select_topk` 先取每组最高选择分数，再挑两个组，各组取一名专家。专家选择分数为 g+b，但混合权重由未加 b 的 g 在所选专家上重新归一化。训练时 b 根据专家实际负载与平均负载之差符号更新，更新率 .001；推理不更新 b。当前 A4 的两项均衡 loss 权重为 0，不能写成增加了一个可微均衡损失。

令 e_s 为共享专家输出，e_r 为两个专家的加权和，a、r 为样本相关的 sigmoid 门：

\[
\tilde h=h_H+r(x)\{a(x)e_s(h_H)+(1-a(x))e_r(h_H)\}.
\]

分类器对全部 POI 输出 logits，使用交叉熵训练。r 初始 .5、共享门基础初始 .15。sigmoid 的范围为 (0,1)，旧文档的“无界 sigmoid”不准确；本模型只是没有额外的 [.35,.65] tanh 限幅。r 在已测运行中接近饱和，不能仅凭存在门控就宣称其保持丰富的样本适应性。

## 3. Jev next-category 请求

`a5_jev/data.py::Dataset.request` 构造状态：查询小时、星期、历史长度、最近 12 次已观察签到的类别及相对小时数；选项为 Top-20 候选的不同类别，确定性打乱选项顺序，并加入 outside-category。POI→类别表仅来自训练事件。

历史条件为时间不晚于 last_checkin 且严格早于当前 target。属于滚动历史评估：可观察的早期 validation/test 签到可能进入后续查询历史；不是把所有 test 标签一律视为永远不可见。是否已知下一次签到的查询时间属于任务假设，应与所有基线保持一致，并通过额外“去查询时间”API 对照评估依赖程度。

请求没有 POI ID、真实目标标签或模型排名分数。Jev 是运行时指定 `jev-1.13.0` 的外部黑箱服务；我们没有训练 Jev，也没有其内部网络/训练损失的复现证据。API 返回概率不能直接称为在本任务上已校准，需以可靠性图与 Brier/ECE 衡量。

## 4. 类别质量修正的对数融合

对固定候选集合 C（|C|=20），令 p_i=softmax(z)_i，P_c=∑_{j∈C:c_j=c}p_j。将 Jev 在候选类别上的概率条件化并平滑：

\[
\bar q_c=(1-\epsilon)\frac{q_c}{\sum_{d\in\mathcal C}q_d}+\frac{\epsilon}{|\mathcal C|},\quad\epsilon=.02.
\]

`a5_jev/core.py::fuse` 实际排序分数为

\[
s_i=\log p_i+\lambda\left(\frac{\log\bar q_{c_i}}{T}-\log P_{c_i}\right).
\]

相同类别的候选共享同一修正量，因此类别内次序保持；候选集合不变，所以 R@20 不变，但 NDCG@20 与 MRR 可变化。lambda=0 或候选类别概率总和为 0 时精确回退。

outside 概率在当前实现中只通过条件化参与：非零候选质量会重新归一化，**没有利用 outside/confidence 动态减小 lambda**。也没有 session 长度门控、规则教师串联、Ridge 门控或语义画像蒸馏。每个数据集的 lambda/T 是 validation 冻结的常数。uniform 类别质量仍会改变 P_c，因此 uniform 对照十分必要。

## 5. 训练/选参与三数据集政策差异

A4 checkpoint 依据 validation 的 4R@1+R@20 选择；早停按完整 epoch，patience=5。融合网格 lambda={0,.05,.1,.2,.35,.5}、T={1,2,4}，lambda=0 只用 T=1。选择目标 R@1+.25MRR。

- CA：原 selection 选参，calibration 与 audit 检查后 test；lambda=.35,T=2。
- TKY：原政策失败后增加 calibration-safe 选参，lambda=.2,T=2，再执行 test。旧 validation/audit 已观察，不能称 audit 是全新未见留出集。
- NYC：原政策 lambda=.5,T=1，audit Top-1 净变化 -1；用户要求保持参数补跑 test，作为明确记录的 gate override。test 执行本身可以报告；不能抹去政策例外或依据 test 重新制定门槛。

三者共享融合公式，但不是完全相同的模型选择程序。后续确认性实验需先统一程序，在新训练重复或新的时间留出上验证。多随机种子不会自动恢复已反复观察的 test 的独立性。

## 6. 代码清单与排除项

| 模块 | 当前实现入口 | 论文定位 |
|---|---|---|
| STHGCN backbone | `model/sthgcn.py`, `layer/conv.py`, `layer/sampler.py` | 继承并引用 |
| 分组 MoE | `model/sparse_moe.py` | 当前主体 |
| A4 三数据集配置 | `conf/best_conf/nyc_r1_6_aux_free_dual_gate.yml`, `ca_a4.yml`, `tky_a4.yml` | 复现配置 |
| Jev 状态/融合/调用 | `a5_jev/data.py`, `core.py`, `client.py` | 外部语义辅助排序 |
| 选参与评估 | `run_a5_jev.py`, `freeze_a5_jev_safe.py` | 政策必须单独说明 |
| 规则教师 | `rule_teacher/core.py` | 对照，不属于最终推理链 |
| `llm_semantic_expert.py` / profile / distillation | 历史代码保留 | 排除于当前 Method |
| HSID-lite / B-Tanh / R1 adapter | 历史试验分支 | 仅附录或研究历史 |

根目录 `部分思路.md` 为用户草稿，未删除；其中有关创新性、Jev 内部能力及 outside 不确定性门控的表述须以上述代码审计为准。

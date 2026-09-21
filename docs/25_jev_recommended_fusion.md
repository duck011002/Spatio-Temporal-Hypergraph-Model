# A5-Jev（当前 A5）：平滑后的对数类别融合

版本命名：按用户确认，自本次登记起，当前讨论中的 **A5** 指 **A5-Jev**，即 **A4 + Jev 类别概率平滑、温度调整与对数类别融合**。不包含规则门控，也不包含本轮比较的 Ridge 样本级自适应融合。

历史编号说明：docs/04_moe_ablation_and_teacher_baseline.md 已将有界 tanh 残差门消融命名为 A5；该历史实验保留原编号。跨文档、结果表及后续运行标识使用完整名称 `A5-Jev`，避免与历史 A5 混淆。

当前状态：方案已命名，隔离离线原型和 validation 用户交叉验证已完成；正式独立推理模块、CA/TKY A4 配置和离线测试已准备，执行交接见 docs/26_a5_jev_server_runbook.md。尚未通过服务器 GPU smoke，尚无此对数融合版本的最终 test 结果。之前算术类别融合的 historical test 不能标为 A5-Jev 的测试结果。CA/TKY 历史预验证使用 R0 骨干，不是完整 A4 + Jev 的 A5 结果。

根据 docs/24_jev_fusion_redesign_results.md，当前优先保留平滑后的对数类别融合作为下一步候选。学习每条样本的融合收益也已实测，但其跨划分波动更大，尚未显示稳定的额外收益。

设 p_i 是 backbone 在 Top-20 内的概率，P_c 是类别 c 的 backbone 概率总和，q_c 是 Jev 在候选类别内归一化后的概率，K 是候选类别数。

```text
q_smooth_c = 0.98 * q_c + 0.02 / K
q_temperature_c = normalize(q_smooth_c ** (1 / T))
C_c = normalize(P_c ** (1 - lambda) * q_temperature_c ** lambda)
p_final_i = C_category(i) * p_i / P_category(i)
```

历史原型在 Jev 给全部候选类别的概率和为0时，以 P 代替 q；正式模块对此边界直接精确回退 backbone，避免平滑/温度带来无证据干预。lambda=0精确还原backbone。所有类别内的POI相对排序保持不变。outside概率表示候选列表外的类别，其数值不直接作为准确率或手写门控阈值。

旧算术融合对候选 i 的相对权重为 `(1-alpha) + alpha*q_c/P_c`；新公式的相对权重与 `(q_temperature_c/P_c)**lambda` 成比例。在本轮lambda不超过0.5的范围内，极端概率比的放大更缓和。温度T和融合系数lambda分别控制Jev分布形状与干预强度，二者不能合并解读为旧alpha。

本轮15个外层训练折选择的参数：

| 数据集/骨干 | lambda、T 的选择 | 含义 |
|---|---|---|
| NYC/A4 | 15/15选择 (0.5, 1) | 本网格下参数选择一致；0.5在网格边界，不代表全局最优 |
| CA/R0 | (0.2, 2) 9次；(0.1, 1) 3次；(0.35, 2) 3次 | 参数仍存在波动，不能宣布稳定最优 |
| TKY/R0 | 15/15选择 (0.35, 2) | 本网格下参数选择一致 |

这里的参数频率用于提出下一阶段候选，不是另一次测试结果。三次交叉验证反复使用同一批validation；NYC/TKY重复结果相同是因为各训练折恰好选择了相同参数，不能将其解释为独立多种子训练复现。

独立模块现已落地到 a5_jev；正式流程在 validation 内按用户划分 selection/calibration/audit，冻结后才能评估 test，详见 docs/26。已观察 validation 的再划分不能视为独立新证据。本次准备未调用 API 或进行新的 test 效果评估。

本轮同时改变了评估划分及统一选参目标（R@1 + 0.25 MRR），因此新表中的旧算术融合结果不同于之前报告；方法优劣只能在本轮相同评估框架内比较。类别输入、提示词和骨干在城市间并不完全一致，不能仅凭提升大小断言城市语义差异。

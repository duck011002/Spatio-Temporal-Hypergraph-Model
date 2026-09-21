# CA/TKY：Jev 类别预测辅助 A4/R0 排序

固定于 2026-09-20，作为 NYC 类别融合的跨数据集快速验证。CA/TKY 使用已归档的 R0 Top-20 候选缓存；不重新跑 backbone，不改变模型、规则教师或训练配置。

对每条 validation/test query，Jev 只预测候选 Top-20 中的类别分布；A4/R0 保留同类别内部原排序。输入为查询时间、最近 12 条严格历史类别以及候选类别名，不输入 POI ID、目标、A4 分数、规则分数、距离或未来事件。CA 的序列化多级类别统一取数据中的首个 category name。

融合为：

```text
P_c = sum(p_backbone_i for candidate i in category c)
r_i = q_Jev_c * p_backbone_i / P_c
p_final = (1-alpha) * p_backbone + alpha * r
```

每个数据集只在自己的 selection 用户分区上从 `[0,.01,.025,.05,.1,.2,.35,.5,.75,1]` 选择 alpha；同时报告均匀类别对照。validation 验收条件为 calibration 的 R@1/MRR 不下降，audit 的 R@1 上升且 MRR 不下降。只有通过才执行一次冻结参数的历史 test。

这是快速探索，不是独立确认性基准：CA/TKY 的历史 test 已出现在规则教师结果中。每个数据集的请求、缓存和账本隔离于 `server_artifacts/jev_category_cross_20260920/`，并发上限 2，实验费用上限 `$2`。主模型和配置不修改；无论结果如何都不自动启用。

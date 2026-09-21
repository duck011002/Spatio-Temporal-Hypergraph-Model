# CA/TKY：Jev 类别预测辅助 A4/R0 排序结果

本轮将 NYC 已验证的类别融合结构迁移到 CA/TKY。CA/TKY 使用归档 R0 Top-20 候选；结果是探索性，因为历史 test 已在规则教师实验中使用。

## 总结

| 数据集 | validation alpha | validation净Top-1 | validation MRR变化 | test状态 | test净Top-1 | test MRR变化 |
|---|---:|---:|---:|---|---:|---:|
| CA R0 | 0.1 | +11 | +0.002690 | withdrawn | +5 | +0.002464 |
| TKY R0 | 0.1 | +3 | +0.002757 | withdrawn | — | — |

## 结果细节

### CA R0

- Validation：A4/R0 Top-1 712/3529 → 723/3529，净变化 +11；R@1 0.201757 → 0.204874；MRR 0.285075 → 0.287765。
- Validation 分区：selection +8，calibration +1，audit +2。
- 纯类别分布替换相对 backbone 净变化 -262；本方案必须保留同类内部 R0 排序。
- 冻结 test：R0 Top-1 469/2780 → 474/2780，净变化 +5；R@1 0.168705 → 0.170504；MRR 0.254011 → 0.256475。
- Test 挽救16、损失11；R@1用户聚类bootstrap区间 [-0.0018034123267182953, 0.005634902388068647]，MRR差值 +0.002464。
- 类别诊断：Jev类别准确率 24.06%，backbone聚合类别准确率 27.15%；真实 outside 类别比例 33.78%。

### TKY R0

- Validation：A4/R0 Top-1 1556/6868 → 1559/6868，净变化 +3；R@1 0.226558 → 0.226995；MRR 0.327935 → 0.330692。
- Validation 分区：selection +4，calibration +1，audit -2。
- 纯类别分布替换相对 backbone 净变化 -253；本方案必须保留同类内部 R0 排序。
- 未执行 test：validation 未通过预先固定的 audit 条件，方案撤回。
- 类别诊断：Jev类别准确率 43.78%，backbone聚合类别准确率 46.51%；真实 outside 类别比例 19.50%。

## 判断

CA 在 validation 上有小幅正向信号，但冻结 test 只净增5条，R@1区间跨0，未满足保留条件，因此撤回。TKY validation 仅净增3条且 audit 未通过，未跑 test，直接撤回。
这说明 NYC 的类别融合效果不能直接外推到 CA/TKY；当前更像是数据集依赖的弱互补信号。CA/TKY 没有启用任何 Jev 分支，主模型、规则教师和配置均未修改。

## 成本与复现

- 成功请求 13177；不确定尝试 8；输入 token 15588217；估算费用 $0.654705。
- p50 0.338s，p95 0.528s；并发上限2。
- 协议：docs/21_jev_category_cross_dataset_protocol.md。
- 汇总：docs/results/jev_category_cross_20260920/summary.json。
- 完整请求缓存、账本、冻结参数和逐样本结果：server_artifacts/jev_category_cross_20260920/。
- API key仅通过进程环境提供，未写入源码或缓存。

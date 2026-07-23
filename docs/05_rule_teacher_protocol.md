# 数据集自适应规则教师：实现与实验协议

## 1. 当前阶段

本阶段只实现显式规则教师，不接入 LLM，也不重新训练 STHGCN 或 MoE 主干。

规则教师对主模型生成的 Top-20 候选进行离线重排。候选集合保持不变，因此规则教师的直接目标是在保持 Recall@20 的同时改善 Recall@1、Recall@5、MRR 与 NDCG。

## 2. 跨数据集设计

NYC、TKY 与 CA 共享五类规则骨架：

1. 空间可达性。
2. 时间匹配。
3. 类别转移。
4. 用户偏好。
5. POI 热度。

统计量和融合权重不跨数据集共享。每个数据集分别从自己的训练事件计算：

- POI 的类别与地理位置；
- 数据集与用户在同一轨迹内的典型移动距离；
- 工作日和周末的小时级类别分布；
- 类别转移概率；
- 用户类别偏好与重复访问偏好；
- POI 训练集热度。

空间规则使用用户历史移动距离的中位数作为衰减尺度；用户统计不足时回退到该数据集的中位移动距离。这避免使用对 NYC、TKY 与 CA 都相同的固定公里阈值。

## 3. 防止数据泄漏

- 所有规则先验只使用 `sample.csv` 中 `SplitTag=train` 的事件拟合。
- validation 用于选择数据集专属规则权重与总规则强度。
- test 不参与权重搜索，只在 validation 参数冻结后评估一次。
- 验证和测试样本的上一签到只作为预测时已经发生的查询上下文使用。
- 权重搜索结果必须按数据集分别保存。

## 4. 规则融合

每条查询内部，模型分数与五类规则分数分别执行 Min-Max 归一化。最终分数为：

```text
final_score = (1 - lambda) * model_score
              + lambda * sum(rule_weight[k] * rule_score[k])
```

其中规则权重非负且和为 1，`lambda` 位于 `[0, 1]`。第一版使用数据集级全局权重，不使用用户级或样本级可学习门。

validation 选择目标固定为：

```text
4 * Recall@1 + Recall@5 + MRR + NDCG@5
```

若所有规则组合都无效，搜索允许选择 `lambda=0`，即退回原模型排序。

## 5. 运行方式

以 NYC A4 为例：

```bash
python run_rule_teacher.py \
  -f best_conf/nyc_r1_6_aux_free_dual_gate.yml \
  --checkpoint /root/autodl-tmp/STHGCN-git/tensorboard/20260723_155322/nyc/checkpoint.pt \
  --output-dir rule_teacher_runs/nyc_a4
```

第一次运行会导出 validation 与 test 的 Top-20。后续只修改规则实现或搜索设置时，可跳过 GPU 推理：

```bash
python run_rule_teacher.py \
  -f best_conf/nyc_r1_6_aux_free_dual_gate.yml \
  --output-dir rule_teacher_runs/nyc_a4 \
  --reuse-candidates
```

## 6. 产物

每次运行生成：

- `validation_candidates.npz`：validation 的 Top-20、模型分数和真实标签原始排名。
- `test_candidates.npz`：test 的对应导出结果。
- `test_reranked.npz`：使用 validation 冻结参数后的 test 重排结果。
- `rule_teacher_summary.json`：数据集统计、规则权重、原模型指标、规则教师指标及差值。

## 7. 第一轮判据

第一轮只在 NYC 固定种子上筛选。进入 CA、TKY 前至少满足：

- Recall@20 与 A4 保持一致；
- Recall@1 或 MRR 高于 A4；
- NDCG@5 不出现明显下降；
- `rule_teacher_summary.json` 能完整复现实验参数与结果。

NYC 有效后，CA 与 TKY 使用相同代码和相同搜索流程，但重新拟合各自统计量与权重。三数据集规则教师冻结后，再进入 LLM 教师阶段。

# STHGCN 实验成果与结果台账

## 1. 文档用途

本文记录 STHGCN 与后续 HC-Shared-MoE、LLM 教师模块的正式实验结果。所有结论必须来自实际日志、配置、checkpoint 或逐样本预测文件。缺失证据统一标记为 `[NEEDS DATA]`。

当前实验命名：

- R0：原始 STHGCN，可复现服务器基线。
- R1：STHGCN + Hypergraph-Conditioned Shared Sparse MoE，尚未实现。
- R2：R1 + 规则教师，尚未实现。
- R3：R1 + LLM 离线教师，尚未实现。

开发阶段默认遵循单城市、单种子策略：优先 NYC，固定 `seed=80786525`。多种子仅在最终结构冻结后补做。

## 2. R0 数据与配置

| 数据集 | 用户数 | POI 数 | 类别数 | 签到数 | 轨迹数 | 训练样本 | 验证样本 | 测试样本 | 固定种子 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NYC | 1,048 | 4,981 | 318 | 103,941 | 14,130 | 72,206 | 1,400 | 1,347 | 80786525 |
| CA | 3,957 | 9,690 | 296 | 238,369 | 45,123 | 154,253 | 3,529 | 2,780 | 27486607 |
| TKY | 2,282 | 7,833 | 290 | 405,000 | 65,499 | 274,597 | 6,868 | 7,038 | 54607333 |

共同训练设置：

- 模型：STHGCN。
- batch size：64。
- learning rate：0.0001。
- epoch：20，TKY 的快速配置另有 2 epoch 与 5 epoch 版本。
- checkpoint 选择指标：`4 × Recall@1 + Recall@20`。
- 单目标 next-POI 评估，指标越高越好。
- 当前基线提交：`0e65a3d chore: establish reproducible server baseline`。

## 3. R0 结果状态

| 数据集 | 当前状态 | 证据等级 | 说明 |
|---|---|---|---|
| NYC | 完整 test 已归档 | 已验证 | 本地含完整日志、配置和 checkpoint，日志包含最终 `Test evaluation result`。 |
| CA | 训练与 validation 已归档 | 部分验证 | 本地日志到 epoch 19、step 47000，但下载版本缺少最终 test 段。不能把 validation 数字写成 test。 |
| TKY | 用户报告已运行 | `[NEEDS DATA]` | 本地尚无完整训练日志和 checkpoint；当前只保留此前 epoch 3 的阶段性 validation 摘要。 |

### 3.1 NYC R0 正式 test 结果

运行标识：`20260722_160336`。

| 指标 | 数值 |
|---|---:|
| Recall@1 | 0.2672605813 |
| Recall@5 | 0.5285820365 |
| Recall@10 | 0.6050482392 |
| Recall@20 | 0.6688938141 |
| NDCG@1 | 0.2672605813 |
| NDCG@5 | 0.4087910950 |
| NDCG@10 | 0.4337376952 |
| NDCG@20 | 0.4500625134 |
| MAP@1 | 0.2672605813 |
| MAP@5 | 0.3686835169 |
| MAP@10 | 0.3791040778 |
| MAP@20 | 0.3836795389 |
| MRR | 0.3860895932 |
| 参数量 | 27,820,021 |

证据位置：

- 日志：`server_artifacts/nyc_20260722_160336_log/nyc/train.log`。
- 配置快照：`server_artifacts/nyc_20260722_160336_log/nyc/config.json`。
- checkpoint：`server_artifacts/nyc_20260722_160336_tensorboard/nyc/checkpoint.pt`。

日志时间范围为 `2026-07-22 16:03:37` 至 `16:28:42`，可观察区间约 25 分 05 秒。该区间从预处理日志开始，不等同于包含远程排队和传输的完整作业时间。

观察：NYC R0 已完整跑通，并与保存的 test 汇总一致，可作为后续单模块快速筛选的固定对照。当前只有一个训练种子，不作统计显著性声明。

### 3.2 CA R0 当前可验证的 validation 结果

运行标识：`20260722_173109`。

本地日志中较优的一次 validation 位于 epoch 19、step 46500：

| 指标 | 数值 |
|---|---:|
| Recall@1 | 0.2009067684 |
| Recall@5 | 0.3828279972 |
| Recall@10 | 0.4474355280 |
| Recall@20 | 0.5123264194 |
| NDCG@20 | 0.3344252110 |
| MRR | 0.2856908739 |

证据位置：

- 日志：`server_artifacts/ca_20260722_173109_log/ca/train.log`。
- 配置快照：`server_artifacts/ca_20260722_173109_log/ca/config.json`。
- checkpoint：`server_artifacts/ca_20260722_173109_tensorboard/ca/checkpoint.pt`。

限制：当前下载日志未出现 `Start evaluating on test set` 与 `Test evaluation result`。CA test 结果标记为 `[NEEDS DATA]`，待从服务器补齐日志尾部或重新加载 checkpoint 执行 test。

### 3.3 TKY R0 阶段性记录

当前本地未归档最终 artifact。此前阶段性摘要记录到 epoch 3、step 16000：

| 指标 | 数值 |
|---|---:|
| Valid Loss | 4.9744 |
| Recall@1 | 0.2292 |
| Recall@5 | 0.4371 |
| Recall@10 | 0.5098 |
| Recall@20 | 0.5796 |
| NDCG@20 | 0.3814 |
| MRR | 0.3264 |

上述数字是阶段性 validation，不是最终 test。TKY 完整日志、配置快照、checkpoint 与最终 test 指标均标记为 `[NEEDS DATA]`。

### 3.4 NYC R1 HC-Shared-MoE

状态：代码实现中，服务器结果待回填。

| 字段 | 当前记录 |
|---|---|
| 分支 | `feature/hc-shared-moe` |
| 基线起点 | `0e65a3d` |
| 完整配置 | `conf/best_conf/nyc_r1_moe.yml` |
| smoke 配置 | `conf/smoke_conf/nyc_r1_moe_smoke.yml` |
| 数据集 | NYC |
| 运行策略 | 沿用配置中的单个 seed，不使用 `--multi_run_mode` |
| 结构 | hypergraph-conditioned Router、4 routed experts、Top-2、1 shared expert、低秩 rank 32 |
| balance loss | 记录诊断但训练权重为 0 |
| smoke | `[NEEDS DATA]` |
| 完整 test | `[NEEDS DATA]` |

本次先得到一个可运行的 R1 候选，不在同一轮加入 LLM，也不启动多种子矩阵。服务器运行完成后必须补充 commit、参数量、运行时间、最佳 validation、完整 test 和 Router 诊断。

## 4. 后续结果登记模板

每个新实验追加以下字段：

| 字段 | 内容 |
|---|---|
| 实验编号 | R1、R2、R3 或消融编号 |
| Git commit | 精确提交哈希 |
| 数据集与种子 | 开发期默认 NYC、80786525 |
| 唯一结构变量 | 本次只改变的模块 |
| 配置文件 | 仓库相对路径 |
| 日志与 checkpoint | 仓库或服务器相对路径 |
| 最佳 validation step | step、epoch 与选择指标 |
| test 指标 | Recall、NDCG、MAP、MRR |
| 效率 | 参数量、训练时间、推理时间、显存 |
| Router 诊断 | 专家流量、熵、Top-2、shared alpha |
| 观察 | 数字直接支持的现象 |
| 解释 | 机制假设，不冒充事实 |
| 限制 | 单种子、缺失分桶或未完成检验 |

## 5. 当前结论

R0 的 NYC 基线已经完成并可作为 R1 的固定效果门槛。CA 与 TKY 的最终 test artifact 仍需补齐。R1 已进入实现和运行阶段，但在服务器完整 test 结束前不声明收益；LLM 教师仍未开始，相关收益为 `[NEEDS DATA]`。

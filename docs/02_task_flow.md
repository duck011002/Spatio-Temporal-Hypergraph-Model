# STHGCN-MoE-LLM 工程任务流

## 1. 项目目标

以已复现的 STHGCN 为稳定骨干，逐步实现 Hypergraph-Conditioned Shared Sparse MoE，并在结构稳定后接入 LLM 离线语义教师，用于 next-POI 推荐。

核心原则：

- 原始 STHGCN 路径必须可关闭新模块后严格恢复。
- 每次只增加一个结构变量，先跑通、再判断效果。
- 开发阶段只用 NYC 与固定单种子，不启动多种子矩阵。
- CA 用于中等规模泛化检查，TKY 只在最终候选冻结后运行。
- smoke 只判断链路正确，完整训练才判断推荐效果。
- test 不用于反复调参；结构选择优先看 validation，冻结后再报告 test。
- 正式训练连续 5 个 epoch 未刷新最佳 validation 时早停，并回载最佳 checkpoint 做 test。
- R1 开发期小成功条件为 Recall@1、Recall@5、Recall@10、Recall@20 中至少两项超过 R0；同时完整记录 NDCG@1/5/10/20、MAP 与 MRR。

## 2. 当前工程基线

当前活动仓库：`D:\project\STHGCN`。

当前基线：

- 基线分支：`main`；当前融合分支：`feature/hc-shared-moe`。
- 基线提交：`0e65a3d`；R1 第一版提交：`e267a29`；调参配置提交：`b6ca105`。
- 远程：`autodl/main` 与本地 `main` 当前对齐。
- R0：原始 STHGCN。
- NYC R0 test 已完整归档。
- CA R0 本地归档缺最终 test 段。
- TKY 最终 artifact 尚未同步到本地。

已完成工程改动：

1. `7a53f5f`：将局部 Jaccard 从完整 Gram 计算改为只计算采样器实际消费的矩形区域，并加入等价性 smoke。
2. `0e65a3d`：建立可复现服务器基线，修复数据读取、embedding 边界、采样器、预处理、配置和运行报告。
3. 新增 `utils/run_reporter.py`，按运行保存阶段耗时、validation、test 与配置快照。
4. 新增 TKY 2 epoch、5 epoch 快速配置。
5. 本地保存 NYC、CA 的服务器日志和 checkpoint artifact。
6. `e267a29`：加入超图条件 Router、Top-2 稀疏专家、shared expert、低秩 adapter 与路由诊断。
7. R1 服务器 smoke 通过，NYC 单种子完整训练与 test 已归档。
8. 正式配置加入 `early_stop_patience: 5`，并固定完整指标记录与 Recall@1/@5/@10/@20 二过四判据。
9. R1.1：4 专家、Top-2、rank 32、残差缩放 0.5，NYC 单种子完整训练结束。
10. R1.2：8 专家、Top-2、rank 32、残差缩放 0.5，NYC 单种子完整训练结束。
11. 单 Adapter 3000-step 最省计算控制组已完成，只记录 validation，不执行 test。
12. A1：8 专家、`moe_loss_weight=0.01`、无分组，NYC 单种子完整训练结束。
13. A2：8 专家、`moe_loss_weight=0.01`、3000-batch warm-up 后按路由共激活余弦自适应分组，NYC 单种子完整训练结束。
14. A3 / R1.5：12 专家、3000-batch warm-up 后全局最优配对为 6 个二元组，分组后将负载权重从 0.01 降至 0.003；服务器测试与 NYC 单种子完整训练均已完成。

## 3. 方法结构约定

计划中的核心表示：

- `h_L`：经过时间过滤的 check-in-to-trajectory 局部表示。
- `h_H`：完成跨轨迹超图协作后的最终表示。
- `c_H`：超图协作增量，初版定义为 `h_H - h_L`，必要时加入投影保证维度一致。
- Router 输入：`[h_L || h_H || c_H]`。
- Expert 输入：默认使用归一化后的 `h_H`。
- 输出：在原始 `h_H` 上增加低秩 MoE 残差，再进入原始预测头。

初版公式：

```text
router_input = concat(h_L, h_H, h_H - h_L)
top2 = Top2Softmax(MLP(router_input))
moe_delta = alpha * shared_adapter(h_H)
          + (1 - alpha) * sum(top2_prob[e] * routed_adapter[e](h_H))
output = h_H + beta * moe_delta
```

第一版暂定：

- 4 个 routed experts。
- global Top-2。
- 1 个 always-active shared expert。
- 低秩 adapter，不使用完整大 FFN。
- `moe_loss_weight=0.0`，只记录路由诊断。
- LLM 关闭。

## 4. 分阶段任务

### D0：项目记忆文档

状态：已完成。

- 建立 R0 成果台账。
- 建立工程任务流。
- 建立本地与服务器运行手册。

### D1：融合分支与干净起点

状态：已完成。

- 已从可复现基线创建 `feature/hc-shared-moe`。
- 不把本地密码、服务器 artifact 或临时脚本提交进 Git。
- 记录分支起点与基线 commit。

### M1：单低秩 Adapter

状态：基础能力已纳入 R1 代码，并完成 3000-step、无 test 的最省计算短跑；暂不追加完整训练。

目标：先验证预测头前的低秩残差 adapter 能正常训练，作为增加参数的控制组。

执行范围：

- 只改模型与配置开关。
- 本地 NYC smoke。
- 服务器 NYC、固定 seed 80786525、一次完整训练。
- 不跑 CA、TKY，不跑多种子。

### M2：局部 Router Sparse MoE

状态：路由基础设施已实现，可通过 `moe_router_context: local` 做后续消融。

目标：迁移 Top-2 稀疏 Router 与 routed adapters，Router 暂只读取 `h_L` 或 `h_H`，不加入超图协作增量。

必须记录：

- Top-1 与 Top-2 专家分布。
- Router entropy。
- 实际 expert evaluation 数量。
- 参数量、训练时间、显存。

### M3：超图条件化 Router

状态：R1 第一版已实现并完成 NYC smoke 与完整训练。

目标：将 Router 输入切换为 `[h_L || h_H || c_H]`，验证超图协作上下文是否改善路由。

快速效果门槛：先在 NYC 单种子与 M2 严格比较。没有正向证据时不扩大实验。

### M4：Shared Expert

状态：R1 第一版已实现并完成 NYC smoke 与完整训练。

目标：增加 always-active shared adapter，并通过 scalar gate 与 routed 输出融合。

必须保持：

- M3 的已有参数初始化不受新增 shared expert 扰动。
- shared alpha、shared output norm、routed output norm 可记录。
- balance loss 继续关闭。

### M5：无 LLM 结构冻结

状态：已有两个不同目标下的候选，暂缓最终冻结。R1.2 的 Recall@20 最高；A3 / R1.5 的 Recall@1 最高、MRR 接近 R0，并将 12 个专家全部有效利用。

目标：选出 R1，并冻结 MoE 结构、Router 输入、adapter rank 和训练配置。

扩展顺序：

1. NYC 固定单种子筛选。
2. R1 明确优于 R0 后，运行 CA 单种子。
3. CA 没有明显退化后，运行 TKY 单种子。
4. 三城结构冻结后，再决定毕业论文最终多种子预算。

### L1：规则教师

状态：待执行。

固定语义维度：

- spatial：local movement 与 long-range displacement。
- temporal：routine 与 irregular。
- semantic：semantic continuity 与 intent shift。
- behavioral：repeat 与 exploration。

规则标签只用于验证教师接口与蒸馏链路。R2 必须与 R1 使用相同模型结构。

### L2：LLM 离线教师

状态：待执行。

约束：

- LLM 不进入在线推理。
- 每条标签使用稳定 `sample_id` 对齐。
- 学生只读取预测时可见的历史和超图表示。
- 若使用真实下一跳生成训练教师标签，必须明确写为训练期 privileged information。
- validation 与 test 不加载教师标签。
- LLM 结果必须与无教师和规则教师分别比较。

### F0：最终实验矩阵

状态：后置。

只在方法与 LLM 方案冻结后执行：

- R0：STHGCN。
- R1：HC-Shared-MoE。
- R2：R1 + 规则教师。
- R3：R1 + LLM 教师。
- NYC、CA、TKY。
- 多种子数量由时间和毕业要求最终确认。

## 5. 每次开发的最小闭环

1. 更新本文档当前阶段状态。
2. 新增一个配置开关，默认关闭。
3. 执行静态检查和 NYC smoke。
4. 运行 NYC 固定单种子完整训练。
5. 将数字写入 `01_r0_baseline_results.md` 的后续结果区。
6. 根据 validation 和 test 现象决定保留、修复或回退。
7. 提交代码时记录 commit、配置、日志与 artifact 路径。

## 6. 当前下一步

单 Adapter 短跑、A1 负载均衡、A2 自适应分组和 A3 / R1.5 的 12 专家二元分组均已完成。A3 使用固定的分组前 0.01、分组后 0.003 负载权重，在不增加每样本专家计算量的前提下达到 12/12 专家活跃，最大 Top-1 专家占比约 13.96%。其 Recall@1 为当前最高，MRR 接近 R0，但 Recall@5/10 低于 R1.2，Recall@20 与 A2 持平。

下一步不继续搜索专家数或负载权重。先根据毕业论文主目标确定主模型：若强调长列表候选覆盖，保留 R1.2；若强调头部准确率、排序质量、专家专业化和方法完整性，优先 A3 / R1.5。未来 LLM 仍作为离线教师，可以监督 6 个组的语义分布或组属性；若后续改变专家数，只需重新执行 warm-up 分组，不需要重写 LLM 教师接口。主模型确定前不运行 CA、TKY，也不增加种子。

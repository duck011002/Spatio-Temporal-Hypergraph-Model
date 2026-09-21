# 论文写作入口

更新：2026-09-21。当前论文主线为 **STHGCN 表示 + A4 超图条件分组 MoE + Jev 类别概率融合**。所有新的论文写作以 `paper/` 为入口；`archive/` 只用于核查研究历史。

1. [实验章节草稿](paper/EXPERIMENTS.md)：已完成的结果、表格、证据等级与解释。
2. [方法与代码映射](paper/METHOD.md)：最终实现、数学公式、继承部分与排除模块。
3. [复现配置与运行时间](paper/REPRODUCIBILITY.md)：数据、配置、版本、硬件及计时口径。
4. [补实验执行清单](paper/EXPERIMENT_PLAN.md)：优先主实验，之后结构消融与机制验证。
5. [KAIS 期刊对标](paper/KAIS_REVIEW.md)：官方要求、已核实论文及本项目差距。
6. [全文写作大纲](paper/MANUSCRIPT_OUTLINE.md)：贡献、章节、图表、声明。
7. [离线补充分析方案](paper/SUPPLEMENT_PROTOCOL.md)与[自动生成的配对结果](paper/generated/main_results.md)。

术语固定：A4 指分组 MoE 主干；A5-Jev 指类别概率融合；A5.1 指 TKY 修订后的选参政策；历史 bounded-tanh “A5”统一称 **B-Tanh**，绝不等同 A5-Jev。Jev 是外部语义概率源，未替换 MoE router，也未作为端到端蒸馏教师。

本轮新增分析是 CPU 离线分析。单 checkpoint 的配对收益、旧模型演进对照、事后消融、多种子确认性实验分别标注。不能把“所有表都有数字”写成“所有实验均已充分验证”。

旧文档和失败路线见[归档索引](archive/README.md)。原始预测、配置和日志本地保存在 `server_artifacts/paper_snapshot_20260921/`；Git 只保存分析代码和论文所需汇总。原始数据与 API 密钥不上传。

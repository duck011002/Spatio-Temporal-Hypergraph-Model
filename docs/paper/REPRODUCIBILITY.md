# 运行配置、时间与证据来源

## 快照与完整性

CPU 模式 AutoDL 已同步到本地 `server_artifacts/paper_snapshot_20260921/`。包含配置、训练 summary、日志、候选、预测、缓存、账本和相关源码；不重新训练、不新增 Jev 调用、不下载大 checkpoint 或发布原始轨迹。

归档 `evidence.tgz` SHA256：`49978254d18f5dab126632ffba2430fb668832a7f45a0605c5f2b15af95465f9`。全部文件逐项核验见 [provenance.json](generated/provenance.json)；完整原始清单在快照 `snapshot_manifest.json`。共享仓库只提交汇总与重算代码，原始快照被忽略。

服务器记录 Git HEAD 为 `0cbb8c39…`，工作树包含部署修改，不能仅凭这一提交号宣称精确复现。应以快照源码哈希和保存的 config 为准；新文档整理提交不代表训练时代码提交。`provenance.json` 同时检查当前本地源码是否与服务器快照一致。

## 训练设置

实际每数据集配置全量保存于 [provenance.json](generated/provenance.json)，概要见 [training_runs.md](generated/training_runs.md)。共同设置：batch/eval batch=64、学习率 1e-4、最多20 epochs/100000 steps、early stop patience=5、workers=4、embedding=128；具体停止点以日志为准。

A4：12 routed experts、rank32、top2、6 groups、group warmup3000 steps、shared expert、hypergraph router context、可学习 shared/residual sigmoid gate、aux-loss-free bias 更新率 .001。学习率 warmup 与分组 warmup 是不同参数。

| 数据集 | seed | 邻居采样 | LR warmup | validation interval | cooldown |
| --- | --- | --- | --- | --- | --- |
| NYC | 80786525 | 300-500 | 8000 | 500 | 1.5 |
| CA | 27486607 | 300-600 | 14000 | 500 | 1.4 |
| TKY | 54607333 | 400-240 | 48000 | 4000 | 1.5 |

历史训练硬件识别为单卡 NVIDIA vGPU-32GB、32760 MiB；这不足以判定物理显卡型号。当前 CPU 模式采集的 CPU/内存/无 GPU 状态不能当作训练期间利用率或峰值显存证据。尚无严格同步的推理延迟、峰值显存、FLOPs benchmark，不编造。

快照环境：Python3.12.3、torch2.5.1+cu124、torch-geometric2.8.0.post1、torch-sparse0.6.18+pt25cu124、torch-scatter2.1.2+pt25cu124、numpy2.1.3、pandas3.0.3、scipy1.18.0、scikit-learn1.9.0。完整原始环境见快照 `snapshot_environment.json`；这些是采集时版本，不保证数月前 NYC 训练环境未改变。

## 实测耗时

| 阶段 | 耗时 | 口径 |
| --- | --- | --- |
| NYC A4 | 29分38秒 | summary start–finish，含运行内验证测试 |
| CA A4 | 2时15分34秒 | 同上 |
| TKY A4 | 5时40分26秒 | 同上 |
| CA/TKY 候选导出 | 23秒 / 72秒 | pipeline 阶段包络 |
| TKY A5.1 test | 24分08秒 | 20:06:43–20:30:51，含报告 |
| NYC override test | 6分58秒 | 21:04:08–21:11:06，含报告 |

时间戳按原日志口径呈现；不要将服务器日志直接标成 UTC。API 请求服务延迟、并发墙钟时间与训练时间分开报告。[API汇总](generated/api_efficiency.md)的费用是历史 usage 估算，不是支付账单；目录覆盖 validation/test 范围不同，不直接横比总成本。重复运行有缓存命中，因此上述不是无缓存完整流程时长。

## 指标复算与随机性

训练日志测试与后续候选导出可能因邻居采样产生差异。Jev 收益必须对比同一导出的 A4 排序；禁止用训练日志较低值当分母制造收益。Top20 外标签的完整主干 rank 保存在候选数据中，因此可以复算全排名 MRR；若未来只有截断 Top20 而无原始 rank，只能报告截断 MRR，不能冒充完整 MRR。

CPU 重算：`python scripts/build_paper_evidence.py`，随后 `python scripts/summarize_paper_runs.py`。需要本地三数据集 sample/validate/test CSV 和已解压快照，依赖 numpy/pandas 及项目导入依赖；不需要 GPU/API key。脚本验证请求哈希、候选与数据一致性、冻结配置和原报告指标，不会重调 Jev 主方案。

复现公开版仍需：锁依赖文件、训练时源码归档、统一种子清单、环境恢复命令、数据许可核查与匿名化。API 模型可能随服务变化；缓存/请求哈希可保证本地分析复核，但不保证未来重新请求完全一致。

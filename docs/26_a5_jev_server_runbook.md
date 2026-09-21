# A5-Jev：CA / TKY 正式实验执行交接

## 状态与边界（2026-09-20）

本轮只准备本地代码、配置和离线测试；未开启服务器、未训练全量模型、未调用 Jev API。后续由用户确认开启 GPU 服务器并切换 Luna 后执行本文。不要创建新任务或自动开机。不要把通过单元测试理解成已经通过服务器 smoke。

当前 A5 指 A4 + Jev 对数类别融合，不是历史 bounded-tanh A5，也不是 docs/14 中的语义画像蒸馏。先完成 CA/TKY 各自 A4，再在同一份候选上比较 A4 与 A5。不加载旧 R0 checkpoint，不混入规则教师、语义蒸馏或 Ridge 门控。

## 实现与预注册选择

- 训练配置：`conf/best_conf/{ca,tky}_a4.yml`。保留各数据集原始采样参数、随机种子、学习率与 warmup；MoE 与 NYC A4 一致。early-stop patience 为 5。
- 安全 smoke：`conf/smoke_conf/{ca,tky}_a4_smoke.yml`。完整模型维度、完整邻居采样尺寸；batch/eval batch=2，max_steps=4，每次评估最多两批。分组 warmup=2，用于覆盖分组切换。现有循环从 step=0 计数，上限可能执行五次更新，仍为有限小测试。
- checkpoint 新增 `experiment_metadata`；候选导出检查数据集和 MoE 设置。旧 checkpoint 缺少元信息时拒绝正式导出，不要手动伪造标记。
- `a5_jev/core.py`：固定 smoothing=.02，lambda 网格 `{0,.05,.1,.2,.35,.5}`，非零 lambda 配合 T `{1,2,4}`。按 `R@1 + .25*MRR` 选参，同分优先较小 lambda 和 T。lambda=0 精确还原；候选类别概率全零也精确还原 backbone。类别内顺序及 Top20 成员不变。
- validation 以用户哈希固定分为约 60% selection、20% calibration、20% audit（不是样本数严格 6:2:2）。仅 selection 搜索参数；calibration 要求净 Top1 和 MRR 不降，audit 要求净 Top1 正增且 MRR 不降，否则禁用 test API。这是实验准入检查，不是逐样本规则门控。实际任一分区为空会停止。
- 选参后冻结参数、数据/代码/候选指纹及 validation 响应指纹；test 不选参、最终报告不覆盖。`active_integration=False`，生成报告不会自动替换 A4。若最终 test 不升，保留原始结果并继续用 A4，不删除不利结果或反复调整 test。
- 新请求统一采用 `a5-category-v1`，仅历史类别、查询时间及打乱的候选类别；不发送标签、POI ID、backbone 分数。历史必须早于目标且不晚于 last_checkin；没有历史的用户使用空历史。POI 类别表只由训练事件构造。

新统一 prompt、分区规则和全零回退与历史研究原型并非完全相同，因此旧提升数字不能当成本实现实测结果。历史 validation 已反复用于方法开发，重新划分它不会使其成为完全独立留出集。正式论文需如实声明这一点，并使用冻结后的 test 与后续多种子结果判断可靠性。

## 1. 同步及环境检查

优先在 GitHub 对齐代码；先保存并比较 AutoDL 未提交修改，不强制覆盖。当前准备的本地文件尚未因本文自动 commit/push。实际部署必须包含新文件，而非仅拉取旧 HEAD；记录本地与服务器 commit、dirty diff、配置和数据哈希。不要将密钥、原始数据、大 checkpoint 推到 GitHub。

在已确认的服务器仓库根目录执行，先核实当前环境，不盲目重装 CUDA：

```bash
git status --short
nvidia-smi
python -c "import torch, torch_geometric, torch_sparse; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -v
```

确认 `data/ca/preprocessed` 和 `data/tky/preprocessed` 对应现有固定划分，含 sample.csv、validate_sample.csv、test_sample.csv 和图文件。任何划分或预处理变化必须另开实验记录。

## 2. 顺序做 smoke，不并行抢显存

```bash
python run.py -f smoke_conf/ca_a4_smoke.yml
python run.py -f smoke_conf/tky_a4_smoke.yml
```

每个命令完成后检查 loss 有限、反向传播与优化成功、分组完成、验证和测试均只处理两批、checkpoint 能回读。记录 `runs/` 对应运行，以及日志中真实 checkpoint 目录，禁止用模糊的“最新文件”猜测。

另做 batch=64 / eval batch=64 的短容量测试：复制该 smoke 配置为独立 capacity 配置，仅调整 batch 大小，保留 smoke_only、步数和评估上限；记录修改后的配置。batch=2 通过不代表 batch=64 安全。出现 OOM 时停止，先报告显存与采样情况；不要默默缩小正式采样、模型或 batch 后声称同一实验。

使用实际 CA smoke checkpoint 做独立候选导出回读测试（替换变量值；TKY 同理）：

```bash
CA_SMOKE_CKPT='实际的CA-smoke-checkpoint目录'
python export_a5_candidates.py -f smoke_conf/ca_a4_smoke.yml --checkpoint "$CA_SMOKE_CKPT" --output artifacts/ca_a4_smoke_candidates --max-batches 2
```

导出应产生 validation/test NPZ 与 smoke_only=true 的 manifest。此候选禁止进入正式 A5 prepare。不要续训 smoke checkpoint 来充当全量初始状态。

## 3. 全量 A4（只在上述检查通过且获准后）

```bash
python run.py -f best_conf/ca_a4.yml
python run.py -f best_conf/tky_a4.yml
```

顺序运行，保留日志、checkpoint、配置与环境版本。正式配置无局部评估上限。每次完成后检查最佳验证 checkpoint 和完整 test 报告；异常退出不能当完成。不要擅自用 init_checkpoint 恢复训练，其断点续训路径不在本轮验收范围内。

导出示例：

```bash
CA_A4_CKPT='实际的CA正式A4-checkpoint目录'
TKY_A4_CKPT='实际的TKY正式A4-checkpoint目录'
python export_a5_candidates.py -f best_conf/ca_a4.yml --checkpoint "$CA_A4_CKPT" --output artifacts/ca_a4_candidates
python export_a5_candidates.py -f best_conf/tky_a4.yml --checkpoint "$TKY_A4_CKPT" --output artifacts/tky_a4_candidates
```

输出目录必须全新。随机邻居采样可能使再次独立推理略有波动，因此 A5 的配对基线必须使用这份同一候选导出，不直接减去另一次训练日志里的 test 指标。

## 4. Jev：先估算，再验证，最后测试

以下以 CA 为例；TKY 替换数据集及对应目录。首次 prepare 固定整次预算（含 validation + test + 重试），后续重新传 --cap-usd 不会修改它。

```bash
python run_a5_jev.py prepare --dataset ca --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev --cap-usd 10
python run_a5_jev.py query-validation --dataset ca --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
python run_a5_jev.py select --dataset ca --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
```

prepare 不调用 API；先检查 example_request、数据计数和估价。密钥通过 `TYPESAFE_API_KEY` 环境变量安全提供，不能写入代码、命令参数、日志或 Git。开启 API 前核实真实账户余额和当前官方价格；代码沿用历史实验输入 token 估价 $0.042/M，不是计费保障。若价格变化，先更新代码、重做 prepare，不能在同一冻结实验中改源文件。代码不查询账户余额；每个 run 的 cap 独立，跨目录不共享预算。建议至少购买 $10，并给 CA、TKY 各自使用 $10 的独立 cap（合计最多 $20），避免重试、服务端计费变化或输入估算偏差提前停止；实际消耗会按 usage 记录，不会主动花满 cap。

固定并发 2；每次尝试先按请求体估算输入 token（最低预留 1,536 token），成功后按 usage 修正；超时及未知失败保留预留，不假设免费。每请求最多三次尝试；429/5xx 有限退避；鉴权失败、schema 错误、余额保护触发即停。这个本地估算保护不是服务端硬消费限额；仍需官方账单核对。

中断后同一目录可续跑，完整响应按请求哈希复用。`--offline` 禁止网络并要求已有缓存。强杀留下 `.active.lock` 时先确认原进程已退出，再手动清理该锁；禁止并发操作同一 run。输入或源码改动则拒绝沿用冻结结果。

只有 frozen.json 中 eligible_for_test=true 才执行：

```bash
python run_a5_jev.py query-test --dataset ca --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
python run_a5_jev.py report --dataset ca --data data/ca/preprocessed --candidates artifacts/ca_a4_candidates --output artifacts/ca_a5_jev
```

归档 prepared.json、frozen.json、候选 manifest、validation_report.json、test_report.json、cache 与 ledger。报告含 R@1/5/10/20、MRR、救回/损失/净 Top1；单次点估计不等于显著提升，论文结论还需用户级配对置信区间及多种子验证。本准备不承诺 CA/TKY 一定提升。

## 本地验收

本地 Python 3.11 / torch 2.9.1 环境完成全套 unittest：49 项通过（含新增 A5 数学公式、极端 logits、schema、离线/预算保护、标签及未来事件不进入请求、模拟缓存完整流程）。模拟响应只验证管线，绝不是 Jev 效果证据。另核对两集配置与原始非 MoE 参数、NYC A4 MoE 参数一致。尚未执行服务器 GPU smoke、正式全量训练或本实现的新 Jev 效果实验。

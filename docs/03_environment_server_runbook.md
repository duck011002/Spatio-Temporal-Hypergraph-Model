# STHGCN 本地与服务器运行手册

## 1. 安全与凭据

本文档可以提交 Git，因此不保存明文密码。

当前服务器密码只保存在本地未跟踪文件和既有本地辅助脚本中：

- 仓库根目录：`ssh`。
- 若干未跟踪的 Paramiko 辅助脚本目前也包含明文密码。

安全约定：

- `ssh`、`.env`、`.env.local` 和任何 `*.local.md` 不得提交。
- 未清理明文密码前，不对服务器辅助脚本执行批量 `git add .`。
- 新对话需要连接服务器时，可从本地私密文件读取凭据，但不得在聊天、日志或提交中回显。
- 公开文档只记录主机、端口、用户、路径和操作流程。

## 2. 路径与连接信息

| 项目 | 值 |
|---|---|
| 本地项目目录 | `D:\project\STHGCN` |
| 服务器 SSH 主机 | `connect.westb.seetacloud.com` |
| SSH 端口 | `45559` |
| SSH 用户 | `root` |
| 服务器旧项目目录 | `/root/autodl-tmp/STHGCN`，只保留旧训练和数据，不直接修改 |
| 服务器当前 Git 工作目录 | `/root/autodl-tmp/STHGCN-git` |
| Conda 初始化脚本 | `/root/miniconda3/etc/profile.d/conda.sh` |
| 当前远程环境 | `base` |
| 本地 Git 服务器远程名 | `autodl` |
| 服务器裸仓库 | `/root/autodl-tmp/STHGCN.git` |
| 当前数据链接 | `/root/autodl-tmp/STHGCN-git/data -> ../STHGCN/data` |
| PuTTY host key | `ssh-ed25519 255 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0` |

当前 Git 远程：

```text
origin  https://github.com/duck011002/Spatio-Temporal-Hypergraph-Model.git
autodl  ssh://root@198.18.0.53:45559/root/autodl-tmp/STHGCN.git
```

`connect.westb.seetacloud.com` 是外部 SSH 入口；`198.18.0.53` 是当前 Git remote 中的连接地址。服务器实例重启后，入口端口或内部地址可能变化，连接失败时先核对平台面板。

## 3. 本地环境准备

在 PowerShell 中进入项目：

```powershell
Set-Location 'D:\project\STHGCN'
```

基础依赖：

```powershell
python -m pip install -r requirements.txt
```

PyTorch、CUDA、PyTorch Geometric、`torch_sparse` 与 `torch_scatter` 需要按照当前 CUDA 和 PyTorch 版本匹配安装。仓库中保留 `install_pyg_libs.py` 用于辅助检查和安装。安装后先执行：

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import torch_geometric; import torch_sparse; print(torch_geometric.__version__)"
```

## 4. 本地 smoke 流程

开发阶段只使用 NYC，不主动运行 CA、TKY，也不启用 `--multi_run_mode`。

### 4.1 语法检查

```powershell
python -m compileall model layer dataset preprocess utils run.py
```

### 4.2 NYC 预处理 smoke

```powershell
python -c "from utils import Cfg; from preprocess import preprocess; preprocess(Cfg('best_conf/nyc.yml'))"
```

预期：生成或复用 `data/nyc/preprocessed`，无异常退出。该步骤可能改写本地预处理缓存，运行前确认正在使用正确 NYC 原始数据。

### 4.3 NYC 局部 Jaccard 等价性 smoke

```powershell
python smoke_rectangular_jaccard.py --sample-file data/nyc/preprocessed/sample.csv --batch-size 64 --fanout 300 --seed 80786525 --threshold 0.005 --repeats 1
```

通过条件：

- `exact_jaccard_values=true`。
- `exact_diagonal=true`。
- `identical_threshold_decisions=true`。

### 4.4 NYC 单种子完整运行

只有本地 GPU 与 PyG 环境完整时执行：

```powershell
python run.py -f best_conf/nyc.yml
```

开发期固定配置文件内的单种子，不添加 `--multi_run_mode`。

## 5. 服务器登录与环境检查

交互式登录：

```powershell
ssh -p 45559 root@connect.westb.seetacloud.com
```

登录后：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/STHGCN-git
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import torch_geometric; import torch_sparse; print(torch_geometric.__version__)"
nvidia-smi
git status --short --branch
git rev-parse HEAD
```

本地也可运行 `python check_gpu_env.py`，但该脚本当前含本地明文凭据，不得提交或公开输出。

## 6. 服务器单模块运行流程

### 6.1 同步代码

优先使用 Git 同步已跟踪代码。每次运行前确认服务器 commit 与成果文档记录一致。

本地：

```powershell
git status --short --branch
git push autodl feature/hc-shared-moe
```

若服务器工作仓库采用自动更新配置，推送后检查 HEAD；否则登录服务器后切换到对应分支。数据集、checkpoint 与大体积 artifact 不通过 Git 同步。

### 6.2 前台 NYC 运行

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/STHGCN-git
python run.py -f best_conf/nyc.yml
```

### 6.3 后台 NYC 运行

```bash
cd /root/autodl-tmp/STHGCN-git
mkdir -p logs
nohup python run.py -f best_conf/nyc.yml > logs/nyc_single_seed.launcher.log 2>&1 &
```

记录返回的 PID，并将本次 Git commit、配置文件和启动日志名称写入任务流。

### 6.4 查看进度

```bash
tail -n 80 -F /root/autodl-tmp/STHGCN-git/logs/nyc_single_seed.launcher.log
```

正式训练指标日志位于：

```text
/root/autodl-tmp/STHGCN-git/log/<timestamp>/nyc/train.log
```

辅助检查：

```bash
nvidia-smi
ps -ef | grep '[p]ython run.py'
find log -name train.log -type f | sort -r | head
```

### 6.5 完成判定

一次完整运行必须同时满足：

- launcher 无 Traceback、OOM、KeyboardInterrupt。
- 正式日志包含 `Start evaluating on test set`。
- 正式日志包含 `Test evaluation result`。
- `tensorboard/<timestamp>/nyc/checkpoint.pt` 存在。
- `runs` 下对应运行包含 finish 与 test metrics。

仅有 validation 指标不算完整 test 结果。

## 7. 结果与 artifact 位置

服务器：

```text
log/<timestamp>/<dataset>/train.log
tensorboard/<timestamp>/<dataset>/checkpoint.pt
runs/<dataset>/<model>/<run_id>/
```

本地归档：

```text
server_artifacts/<dataset>_<timestamp>_log/
server_artifacts/<dataset>_<timestamp>_tensorboard/
```

`server_artifacts` 已被 Git 忽略。成果文档必须记录 artifact 相对路径、运行时间、seed、commit 和配置快照。

## 8. 模块开发运行政策

逐步加模块时采用以下顺序：

1. 本地语法检查。
2. 本地 NYC smoke。
3. 服务器 NYC 固定单种子完整训练。
4. 与上一阶段严格比较。
5. 有正向证据后才进入下一个模块。
6. 不在逐模块阶段跑多种子。
7. 不在逐模块阶段运行 TKY。
8. 最终模型冻结后再安排 CA、TKY 和多种子。

当前 R1 第一版已经完成。下一轮只做一个轻量结构或强度调整，仍先在 NYC 单种子上筛选；未通过 Recall@1/@5/@10 二选三门槛前，不运行 CA、TKY，也不加入 LLM。

## 9. 常见问题

### 配置路径错误

配置文件实际位于仓库的 `conf` 目录，但 `Cfg` 会自动补上这一层，因此命令参数不能再次包含 `conf/`。正确命令是：

```text
python run.py -f best_conf/nyc.yml
```

若写成 `conf/best_conf/nyc.yml`，程序会错误查找 `conf/conf/best_conf/nyc.yml`。

### R1 MoE 的 NYC 命令

服务器 smoke：

```bash
python run.py -f smoke_conf/nyc_r1_moe_smoke.yml
```

smoke 通过后的单次完整 NYC：

```bash
python run.py -f best_conf/nyc_r1_moe.yml
```

两条命令均不添加 `--multi_run_mode`，沿用配置文件中的单个 seed。

R1 正式配置使用 `early_stop_patience: 5`。早停以 epoch 为单位：连续 5 个完整 epoch 没有刷新原有 checkpoint 选择分数 `4 × Recall@1 + Recall@20` 时停止训练，随后仍执行最佳 checkpoint 的完整 test。结果验收同时对比 Recall@1/5/10/20、NDCG@1/5/10/20、MAP@1/5/10/20 与 MRR；工程快速成功要求 Recall@1、@5、@10 至少两项超过 R0。

### CA 日志只有 validation

若日志缺少 test 段，先确认远程训练进程是否完成，再检查 checkpoint。必要时使用相同配置加载 checkpoint 单独执行 test，不得把 validation 值写成 test。

### TKY 图太大

TKY 只用于最终候选。开发期避免重复预处理和多种子，优先使用 NYC；需要中间泛化检查时先使用 CA。

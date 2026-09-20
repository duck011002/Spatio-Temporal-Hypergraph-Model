# STHGCN：AutoDL 剩余两个数据集实验交接

> 交接基线：feature/hc-shared-moe，当前代码提交 9f22947（2026-09-20 复核）。
>
> 本文按当前仓库证据编写，不把论文草稿、validation 数字或未归档的服务器目录当作完成证明。

## 1. 先说结论：现在做到哪里了

当前分支已经完成的是 **NYC 的最终 LLM 语义画像蒸馏链路**。后面要补的是 **CA 和 TKY 两个数据集**，但它们不是把 NYC 的数字复制过去，而是要各自重新完成：

~~~
最终 12 专家 A4 骨干
  -> 只导出一次 validation/test Top-20 候选
  -> 生成本数据集全部类别的离线语义画像
  -> 构造 POI 对齐的 24 维 profile matrix
  -> 冻结 A4，只训练本地语义专家
  -> validation 的 selection / calibration / audit
  -> 参数冻结后唯一一次 paired test
~~~

### 1.1 已确认完成

| 项目 | 数据集 | 当前证据 | 判定 |
|---|---|---|---|
| 原始 R0 完整训练与 test | NYC | server_artifacts/nyc_20260722_160336_*，含日志、配置、checkpoint、test | 已完成 |
| 12 专家 A4 / HC-Shared-MoE 及后续 NYC 调参 | NYC | docs/01_r0_baseline_results.md，多个 server_artifacts/nyc_* | 已完成单种子开发实验 |
| 规则教师 | NYC A4、CA R0、TKY R0 | docs/06_rule_teacher_cross_dataset_results.md 及候选缓存 | 已完成“规则教师预验证”，不是最终三数据集联合模型 |
| LLM 类别语义画像蒸馏 | NYC | docs/13_llm_profile_distillation_results.md，paired validation、画像消融、唯一 frozen test | 已完成 |

### 1.2 不能误报为已完成

| 项目 | 当前状态 |
|---|---|
| CA 最终 A4 + 语义专家 | 未完成。已有 CA R0 checkpoint 和 validation 日志，但本地日志没有最终 test 段，也没有 CA 的 A4 语义蒸馏结果。 |
| TKY 最终 A4 + 语义专家 | 未完成。当前本地没有可复核的完整 TKY 训练日志、最终 checkpoint 或语义蒸馏结果。 |
| CA/TKY 的 LLM profile matrix | 未生成。当前 server_artifacts/llm_category_profiles/ 只有 NYC matrix。 |
| CA/TKY 多种子与显著性检验 | 未做。NYC 也只有单种子，所有结果只能作为工程/研究证据，不能声明统计显著。 |

已有的 CA/TKY 规则教师数字只说明：在 **R0 候选** 上规则重排有正向信号；它们不能替代 CA/TKY 的最终 A4 骨干和语义蒸馏实验。

## 2. NYC 已完成实验的可复核锚点

这些路径用于确认新机器上的代码和评估口径，不要把它们当作 CA/TKY 的输入 checkpoint：

- 代码配置：conf/best_conf/nyc_r1_6_aux_free_dual_gate_llm_semantic.yml。
- NYC profile matrix：server_artifacts/llm_category_profiles/nyc_semantic_profile_matrix_v2.npz。
- 语义专家快照：server_artifacts/nyc_llm_semantic_profile_distill_20260724/llm_semantic_selected_step_2000.pt。
- 快照 SHA-256：bb78d31943e5ddf10c8d8051c254b447b79ae8d5cdeac83b7bab897a596b0f19。
- NYC paired validation / 消融 / frozen test：同一目录下的 calibrated_validation.json、ablation_zero.json、ablation_shuffle.json、paired_rule_validation.json、frozen_test_once.json。
- NYC A4 backbone：文档记录的服务器路径为 /root/autodl-tmp/STHGCN-git/tensorboard/20260723_155322/nyc/checkpoint.pt；新机器不要假定该路径已经存在，先在 AutoDL 上核对。

NYC 语义实验的关键口径：profile 宽度为 24；画像只来自类别名称，不发送用户轨迹、POI ID、validation/test 标签；A4 主干冻结；先做 profile regression warmup，再启用非负语义残差；test 在所有超参数冻结后只运行一次。

## 3. 在另一台电脑拉取源代码

### 3.1 推荐从 GitHub 拉当前分支

本次交接提交完成并推送后，在新电脑执行：

~~~bash
git clone --branch feature/hc-shared-moe --single-branch \
  https://github.com/duck011002/Spatio-Temporal-Hypergraph-Model.git STHGCN
cd STHGCN
git rev-parse --short HEAD
git status --short --branch
~~~

要求：HEAD 必须等于交接提交中记录的 commit，工作树应干净。不要把 GitHub token、AutoDL 密码或 key.txt 写进 clone URL、脚本或 Markdown。

如果 GitHub 访问受限，可以从已有 Git 仓库导出同一分支；不要使用本地的 STHGCN_project.tar.gz 作为唯一源码来源，因为它可能包含旧代码、生成文件和过期配置。

### 3.2 AutoDL 工作目录

服务器上建议使用独立目录，不覆盖旧训练目录：

~~~bash
cd /root/autodl-tmp
git clone --branch feature/hc-shared-moe --single-branch \
  https://github.com/duck011002/Spatio-Temporal-Hypergraph-Model.git STHGCN-git-new
cd /root/autodl-tmp/STHGCN-git-new
git rev-parse HEAD
git status --short --branch
~~~

如果服务器已有 /root/autodl-tmp/STHGCN-git，先只读核对：

~~~bash
cd /root/autodl-tmp/STHGCN-git
git status --short --branch
git rev-parse HEAD
~~~

只有确认旧目录没有未保存的用户改动、且 commit 正确后，才可以在该目录执行 git fetch / git switch。原始数据、预处理缓存、checkpoint 和日志不通过 Git 同步。

### 3.3 本次应该进 Git 的文档与不应该进 Git 的文件

本次同步范围是：

- docs/01—docs/14 实验台账、环境手册、LLM 协议与本交接文档；
- docs/results/ 下已经形成的可审计 JSON/CSV 汇总；
- 当前分支中已经跟踪的模型、评估脚本和配置。

不要提交：

- key.txt、ssh、.env* 和任何包含服务器凭据的辅助脚本；
- server_artifacts/、log/、tensorboard/、runs/、raw/preprocessed 数据；
- .tar.gz、临时目录、IDE 目录、编译生成的 PDF/aux/log，除非另有明确归档要求。

当前本地有若干未跟踪 Paramiko 辅助脚本包含明文凭据；它们不属于交接源码。建议在把凭据交给另一台电脑前轮换 AutoDL 密码，并使用 SSH key 或密码管理器注入凭据。

## 4. AutoDL 环境检查

不要在聊天或日志中打印密码。登录方式使用平台当前分配的 SSH 入口；当前历史记录中的入口为 connect.westb.seetacloud.com:45559，实例重启后必须以平台面板为准。

~~~bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/STHGCN-git-new

python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
import torch_geometric
print("pyg", torch_geometric.__version__)
PY
nvidia-smi
python -m compileall model layer dataset preprocess utils run.py
~~~

依赖安装要以现有 CUDA/PyTorch 为准，先检查再安装；不要无条件升级 PyTorch。若缺少 PyG 扩展，按 requirements.txt 和 install_pyg_libs.py 检查当前版本匹配关系。

## 5. 运行前需要准备的非 Git 资产

### 5.1 数据

确认 data/ca/raw、data/tky/raw 存在；如果采用共享旧数据目录，可建立经过核对的链接，但要保证当前代码读取到的是 CA/TKY 对应原始数据：

~~~bash
ls -lah data/ca data/tky
ls -lah data/ca/raw data/tky/raw
~~~

预处理会生成被 .gitignore 忽略的目录：

~~~bash
python - <<'PY'
from utils import Cfg
from preprocess import preprocess
for name in ("ca", "tky"):
    preprocess(Cfg(f"best_conf/{name}.yml"))
PY
~~~

预处理完成后检查 data/ca/preprocessed/sample.csv 和 data/tky/preprocessed/sample.csv，并记录行数、POI 数、类别数、train/validation/test 样本数。若数据统计与 docs/01_r0_baseline_results.md 不一致，先查数据版本，不要继续跑模型。

### 5.2 最终 A4 骨干 checkpoint

当前仓库没有现成的 CA/TKY A4 完整配置；只有 NYC 的 nyc_r1_6_aux_free_dual_gate.yml。因此下一台电脑上的 Codex 必须先生成两个数据集专用配置：

- 以 conf/best_conf/ca.yml / tky.yml 的数据集、图采样、seed、learning rate、warm-up、valid_steps 为基础；
- 注入 NYC A4 配置中的 12 experts、Top-2、shared expert、adaptive grouping、aux-free dual gate 全部 model_args；
- 保持 CA seed=27486607、TKY seed=54607333，不加 --multi_run_mode；
- 先做 smoke，再做完整训练并保留 do_test: true；
- 每次运行记录 commit、配置快照、完整日志、tensorboard/<timestamp>/<dataset>/checkpoint.pt 和 runs/ 摘要。

示意命令（配置文件名由 Codex 创建后再执行）：

~~~bash
python run.py -f smoke_conf/ca_r1_6_aux_free_dual_gate_smoke.yml
python run.py -f best_conf/ca_r1_6_aux_free_dual_gate.yml

python run.py -f smoke_conf/tky_r1_6_aux_free_dual_gate_smoke.yml
python run.py -f best_conf/tky_r1_6_aux_free_dual_gate.yml
~~~

完整运行验收必须同时满足：日志无 Traceback / CUDA out of memory / KeyboardInterrupt；存在 checkpoint.pt；日志含 Start evaluating on test set 和 Test evaluation result；runs/<dataset>/... 有完成状态和 test metrics。只有 validation 不算完成。

## 6. 导出候选与生成 profile 的顺序

对 CA、TKY 各自的最终 A4 checkpoint 执行一次候选导出。这里的候选缓存还会用于规则教师和后续 paired 评估，生成后尽量使用 --reuse-candidates，避免 NeighborSampler 的随机采样导致不配对。

~~~bash
python run_rule_teacher.py \
  -f best_conf/ca_r1_6_aux_free_dual_gate.yml \
  --checkpoint /ABS/PATH/CA_A4/checkpoint.pt \
  --output-dir server_artifacts/rule_teacher_final/ca_a4 \
  --top-k 20

python run_rule_teacher.py \
  -f best_conf/tky_r1_6_aux_free_dual_gate.yml \
  --checkpoint /ABS/PATH/TKY_A4/checkpoint.pt \
  --output-dir server_artifacts/rule_teacher_final/tky_a4 \
  --top-k 20
~~~

检查两个目录各有 validation_candidates.npz、test_candidates.npz 和 rule_teacher_summary.json。权重只从 validation 搜索；test 只能在协议冻结后报告，不能用来回调参数。

## 7. vLLM 拉起与离线画像生成

当前仓库的 generate_llm_category_profiles.py 不是直接调用 vLLM Python API，而是向 OpenAI-compatible endpoint 发送 POST /v1/chat/completions。vLLM 官方当前推荐用 vllm serve 启动兼容服务器；参数以服务器安装版本的 vllm serve --help 为准。参考：https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html。

### 7.1 GPU 分配

- 如果有两张 GPU：让 vLLM 使用一张，训练使用另一张；分别设置 CUDA_VISIBLE_DEVICES。
- 如果只有一张 GPU：先启动 vLLM 完成 profile 生成并关闭它，再运行 PyTorch 训练。不要让 vLLM 和 A4 训练抢同一张卡。
- vLLM 只在本机监听 127.0.0.1，不把端口暴露到公网。--api-key 不是完整的网络安全边界。

先确认版本和模型路径：

~~~bash
vllm --version
vllm serve --help | head -80
MODEL_PATH=/ABS/PATH/TO/A/QWEN-INSTRUCT-MODEL
~~~

### 7.2 启动服务

下面是模板，不要把 MODEL_PATH 当成已经存在的路径；模型必须是 AutoDL 上实际下载且有 chat template 的 Qwen instruct 模型。若只有一张卡，把 CUDA_VISIBLE_DEVICES=1 改成可用卡号，并在生成画像后停止服务。

~~~bash
export VLLM_LOCAL_KEY=local-vllm-only
CUDA_VISIBLE_DEVICES=1 nohup vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name qwen-local \
  --api-key "$VLLM_LOCAL_KEY" \
  --dtype auto \
  --gpu-memory-utilization 0.80 \
  --max-model-len 4096 \
  > logs/vllm_category_profiles.log 2>&1 &
echo $! > logs/vllm_category_profiles.pid
~~~

等待服务就绪并确认实际模型名：

~~~bash
curl -fsS http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${VLLM_LOCAL_KEY}"
tail -n 80 logs/vllm_category_profiles.log
~~~

如果 vllm serve 不识别某个参数，以当前版本 vllm serve --help 为准；不要为了迁就旧命令直接混用不兼容版本。

### 7.3 运行 profile 脚本

脚本需要一个本地 key 文件，但 key 只是本机 vLLM 的占位认证，不要提交：

~~~bash
printf '%s\n' "$VLLM_LOCAL_KEY" > key.txt

python generate_llm_category_profiles.py \
  --datasets ca tky \
  --include-all-poi-categories \
  --limit-per-dataset 0 \
  --model qwen-local \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-file key.txt \
  --cache-dir server_artifacts/llm_category_profiles \
  --output-dir docs/results/llm_category_profiles \
  --max-cost-cny 100 \
  --dry-run
~~~

先审查 dry-run 的 new_requests、两个数据集的类别数量和预计 token；确认不是 0，也确认没有把 test 样本放进 prompt，再去掉 --dry-run 正式执行：

~~~bash
python generate_llm_category_profiles.py \
  --datasets ca tky \
  --include-all-poi-categories \
  --limit-per-dataset 0 \
  --model qwen-local \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key-file key.txt \
  --cache-dir server_artifacts/llm_category_profiles \
  --output-dir docs/results/llm_category_profiles \
  --max-cost-cny 100
~~~

这里的 --max-cost-cny 是仓库脚本的预算闸门；使用本地 vLLM 时没有 DashScope 计费，但仍保留闸门并记录请求数。脚本会缓存每类画像，失败后可在同一 cache 上 resume。若返回不是严格 JSON，先检查模型 chat template / structured output 支持，不要手工把错误响应改成实验结果。

单卡 profile 生成结束后：

~~~bash
kill "$(cat logs/vllm_category_profiles.pid)" || true
unset VLLM_LOCAL_KEY
~~~

### 7.4 生成 CA/TKY profile matrix

~~~bash
python build_llm_semantic_matrix.py \
  --dataset ca \
  --profiles docs/results/llm_category_profiles/category_semantic_profile_v2_deduplicated_profiles.json \
  --output server_artifacts/llm_category_profiles/ca_semantic_profile_matrix_v2.npz

python build_llm_semantic_matrix.py \
  --dataset tky \
  --profiles docs/results/llm_category_profiles/category_semantic_profile_v2_deduplicated_profiles.json \
  --output server_artifacts/llm_category_profiles/tky_semantic_profile_matrix_v2.npz
~~~

验收：输出的 matrix 行数必须等于相应数据集运行时的 num_poi + 1；profile width 必须为 24；padding/unknown 和 legacy extra output 两行必须 unavailable；missing LLM profiles 必须为零。

## 8. CA/TKY 语义蒸馏训练

为 CA/TKY 各生成一份 semantic config，结构参照 conf/best_conf/nyc_r1_6_aux_free_dual_gate_llm_semantic.yml，但必须替换：

- dataset_args.dataset_name；
- 数据集专用的 sizes、inter_jaccard_threshold、seed、learning rate、warm-up 和 validation 间隔；
- llm_semantic_profile_path 为对应 matrix；
- do_test: false，避免训练阶段提前看 test；
- use_llm_semantic_expert: true、llm_semantic_freeze_backbone: true、llm_semantic_mode: profile_distillation；
- llm_semantic_distill_weight: 1.0、llm_semantic_distill_warmup_steps: 2000、llm_semantic_snapshot_steps: 500；
- 保持 max_scale: 0.5、rank=32，除非先在 validation 协议中明确记录新的选择。

训练命令必须显式给最终 A4 checkpoint：

~~~bash
python run.py \
  -f best_conf/ca_r1_6_aux_free_dual_gate_llm_semantic.yml \
  --backbone-checkpoint /ABS/PATH/CA_A4/checkpoint.pt

python run.py \
  -f best_conf/tky_r1_6_aux_free_dual_gate_llm_semantic.yml \
  --backbone-checkpoint /ABS/PATH/TKY_A4/checkpoint.pt
~~~

若没有 --backbone-checkpoint，当前 run.py 会拒绝冻结骨干的语义训练；不要用随机初始化模型代替 A4。保留 tensorboard/<timestamp>/<dataset>/llm_semantic_step_*.pt、最佳 checkpoint.pt、配置快照和 train.log。

## 9. Validation、画像消融、规则组合与唯一 test

每个数据集单独完成以下顺序：

1. 用 validation 的 selection 选择 semantic snapshot；
2. 只用 calibration 选择非负 semantic_delta_multiplier；
3. audit 只做验收，不参与选择；
4. 可在 validation 上执行 zero 和 shuffle profile ablation；
5. 参数、snapshot、倍率和组合方式全部冻结后，才执行一次 data-split test，且不加 --calibrate-multiplier。

validation 校准示例：

~~~bash
python evaluate_llm_semantic_validation.py \
  -f best_conf/ca_r1_6_aux_free_dual_gate_llm_semantic.yml \
  --backbone-checkpoint /ABS/PATH/CA_A4/checkpoint.pt \
  --semantic-snapshot /ABS/PATH/CA_SEMANTIC/llm_semantic_step_2000.pt \
  --output-json server_artifacts/ca_llm_semantic/calibrated_validation.json \
  --rule-summary server_artifacts/rule_teacher_final/ca_a4/rule_teacher_summary.json \
  --calibrate-multiplier
~~~

画像消融使用相同 validation 命令，分别加上：

~~~text
--profile-ablation zero
--profile-ablation shuffle
~~~

冻结后的 test 示例：

~~~bash
python evaluate_llm_semantic_validation.py \
  -f best_conf/ca_r1_6_aux_free_dual_gate_llm_semantic.yml \
  --backbone-checkpoint /ABS/PATH/CA_A4/checkpoint.pt \
  --semantic-snapshot /ABS/PATH/CA_SEMANTIC/llm_semantic_step_2000.pt \
  --output-json server_artifacts/ca_llm_semantic/frozen_test_once.json \
  --rule-summary server_artifacts/rule_teacher_final/ca_a4/rule_teacher_summary.json \
  --data-split test \
  --semantic-delta-multiplier <VALIDATION_SELECTED_VALUE>
~~~

TKY 完全同样执行。test 命令不能带 --calibrate-multiplier；不能观察 test 后重选 snapshot、倍率、规则权重或 profile。

## 10. 结果归档模板

每个数据集完成后，至少归档以下信息到 docs/14 的后续更新或新的结果文档：

~~~text
dataset / backbone config / semantic config
Git commit
seed
preprocess 数据摘要
vLLM 模型名、vLLM 版本、profile 请求数、cache 命中数
A4 checkpoint 路径与 SHA-256
profile JSON、matrix 路径与 matrix shape / coverage
semantic snapshot 路径与 SHA-256
selection / calibration / audit 的 R@1、R@5、R@10、R@20、MRR
zero / shuffle 消融
唯一 frozen test 的全部指标与样本数
日志、runs、tensorboard 路径
~~~

结果判定必须拆开写：

- “已完成”：日志、checkpoint、配置、结果 JSON 全部可复核；
- “仅 validation”：没有执行或没有保存 frozen test；
- “部分完成”：有 checkpoint 但缺日志尾部、配置或结果；
- “无法核实”：只有口头数字、截图或旧机器路径。

不要把 CA/TKY 的 R0 规则教师结果写成最终 A4+LLM 结果，也不要把 NYC 的 profile cache / snapshot 复制成其他数据集的证据。

## 11. 出问题时先查什么

- No profile found / Missing LLM profiles：确认已完成预处理、candidate cache、--include-all-poi-categories，且使用同一 profile JSON。
- profile rows != num_poi + 1：matrix 是用错数据集或错版本的预处理数据生成的，删除错误 matrix 并重建，不要 reshape。
- vLLM 连接失败：检查 127.0.0.1:8000/v1/models、PID、日志、--served-model-name 与 --model 参数；确认脚本 --base-url 末尾为 /v1。
- vLLM OOM：降低 --gpu-memory-utilization、--max-model-len 或改用另一张 GPU；profile 生成不应与训练并行抢卡。
- semantic training 报 frozen backbone 缺失：检查 --backbone-checkpoint 是否为最终 A4 checkpoint.pt，以及配置是否为 llm_semantic_freeze_backbone: true。
- test 数字与候选重排不一致：检查是否重复运行了随机 NeighborSampler；优先使用已经保存的候选和 paired evaluator。
- 任何 Traceback、OOM、进程被杀、磁盘不足：该数据集标记为“待收尾”，先保存现场路径和 commit，再决定是否续跑。

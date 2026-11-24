# 使用 conda + pip 配置 KL-in-LLM-RL 训练环境

本教程使用 **conda 创建 Python 环境、pip 安装计算库和本地 verl**，全程不需要 uv。命令面向 Linux Bash、NVIDIA GPU、FSDP + vLLM；依据本仓库 `8ff8a9f` 编写，资料核对日期为 2026-11-19。当前仅做文档和静态检查，尚未在 Linux GPU 服务器上验证整套安装与训练。

## 1. 选择版本与准备条件

### 一键安装并开始四卡训练

若 Linux x86_64 服务器已经配置 NVIDIA 驱动、CUDA 12.8 toolkit（nvcc）、gcc/g++，可直接在仓库根目录执行：

```bash
bash verl/rl-kl/install_conda_and_train_a100.sh
```

脚本不需要预先激活环境：优先使用已有 conda，没有时下载官方 Miniforge 安装器到临时文件并安装到 `$HOME/miniforge3`。创建或复用 `$CONDA_BASE/envs/kl-rl`（Python 3.12），通过 pip 安装约束文件 `verl/rl-kl/constraints-a100-cu128.txt` 对应的计算栈和本地源码，然后检查四卡 FlashAttention 前后向与数学奖励、准备模型和数据、处理 SwanLab 登录，最后调用四卡训练入口。系统驱动与 toolkit 由服务器预先提供，脚本不会更改系统 CUDA。

默认首次下载 `Qwen/Qwen3-1.7B-Base`、`FlippyDora/dapo-filter` 的 train split，以及根 README 列出的全部六个 `CorrectKLinRL` 评测集的 test split（olympiadbench、minerva_math、amc23、aime25、aime24、math500）；训练期间仍默认使用 math500 验证。在线资源是否可访问取决于运行服务器。数据准备器仅接受已识别的 verl schema 或 problem/question + answer schema，遇到其他结构会停止，避免猜测答案字段。已有本地 Parquet 会检查后复用，不自动覆盖。权重目录有 config 但下载不完整时会报错，需要先补全下载。设 `DOWNLOAD_BENCHMARKS=0` 可在安装流程中只准备训练集和 math500。

```bash
# 建议首次只跑 10 步，确认日志、保存、验证都成功
TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=10 \
  bash verl/rl-kl/install_conda_and_train_a100.sh

# 指定已有模型和数据；三个路径都应是本机绝对路径
MODEL_PATH=/absolute/models/Qwen3-1.7B-Base \
TRAIN_FILE=/absolute/data/dapo_filter.parquet \
VAL_FILE=/absolute/data/math500/test.parquet \
  bash verl/rl-kl/install_conda_and_train_a100.sh

# 安装并检查环境，不下载模型/数据，也不开始训练
INSTALL_ONLY=1 bash verl/rl-kl/install_conda_and_train_a100.sh

# 后续跳过 pip 安装；仍执行环境和数据检查，再开始训练
SKIP_INSTALL=1 TRAIN_MODE=prm bash verl/rl-kl/install_conda_and_train_a100.sh
```

可设置 `CONDA_ENV_NAME` 或 `CONDA_ENV_PREFIX` 选择独立环境。已有环境必须使用 Python 3.12，否则脚本停止，不会删除或强制重建。`INSTALL_ONLY=1` 也会做四卡计算检查，因此仍需分配四张 GPU。`SKIP_INSTALL=1` 不会跳过检查或自动修复缺包。

云端模式下，设置过 `SWANLAB_API_KEY` 时无需交互；否则运行 `swanlab login` 复用本地凭证或提示输入 Key。可以用 `SWANLAB_MODE=offline` 不联网记录，或 `TRAINER_LOGGER='[console]'` 只记录控制台。脚本不把密钥写入文件或回显。不要把真实 API Key 作为 Hydra 参数。

Base 模型缺少 chat template 时，准备器默认在本地模型副本创建简单纯文本模板，并保存 `tokenizer_config.before-bootstrap.json`。这是可运行基线，不保证与论文模板一致。用 `CHAT_TEMPLATE_FILE=/absolute/template.jinja` 显式指定模板，或设 `ALLOW_PLAIN_CHAT_TEMPLATE=0` 在缺模板时停止。已有模板且未指定新文件时不修改。模型版本可通过 `MODEL_REVISION` 固定；数据集与 split 可通过 `TRAIN_DATASET`、`TRAIN_SPLIT`、`VAL_DATASET`、`VAL_SPLIT` 和对应 `*_DATASET_REVISION` 固定。

环境记录保存在根目录 `environment-records/`，资产路径与显式版本参数记录在 `DATA_ROOT/bootstrap-assets.json`。默认 Miniforge 使用最新安装器；如需固定其版本，设置官方历史安装器的 `MINIFORGE_INSTALLER_URL`。当前脚本经过语法与模拟流程验证，尚未执行真实 GPU 安装和训练。

### 手动安装时的版本选择

本仓库内置了修改过的 verl，训练入口直接调用 Python；必须安装本仓库的 `verl/` 源码，以保留 PRM/PRL 和 KL 实现。

本文采用的候选组合如下，依据是本地 `setup.py` 与 `scripts/install_vllm_sglang_mcore.sh`，并非已验证的环境锁文件：

| 组件 | 本教程选择 |
| --- | --- |
| 操作系统 | Linux x86_64；建议 Ubuntu 22.04 或兼容系统 |
| Python | 3.12 |
| vLLM | 0.11.0 |
| PyTorch / torchvision / torchaudio | 2.8.0 / 0.23.0 / 2.8.0 |
| CUDA wheel | cu128 |
| FlashAttention | 2.8.1 |
| FlashInfer | 0.3.1 |
| numpy | <2.0 |
| pyarrow | >=19.0 |
| tensordict | >=0.8.0, <=0.10.0，排除 0.9.0 |
| transformers | >=4.51.0, <5；最终小版本以实测结果记录 |

vLLM 0.11.0 的 [CUDA 依赖文件](https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt)明确要求上述 torch 系列版本。原提交中 vLLM 0.8.4 镜像注释低于本地 setup.py 的支持范围，该过时注释现已移除。

需要可用的 NVIDIA 驱动，以及支持所选 BF16、FlashAttention2 kernel 的 GPU。GPU 型号未知，本教程不承诺显存需求或所有架构均兼容。原生 Windows PowerShell 不适用以下命令；建议在 Linux GPU 服务器执行，WSL2 需另行核实 GPU 和多卡支持。

在服务器检查：

```bash
nvidia-smi
uname -m
df -h
free -h
nvcc --version
gcc --version
```

`nvidia-smi` 中的 CUDA Version 是驱动支持能力，不是本机 toolkit 版本。PyTorch/vLLM wheel 会安装部分运行时库，但源码编译 FlashAttention 仍需要 CUDA toolkit 开发组件与编译器。本文按 **CUDA 12.8 toolkit** 编译环境准备；缺少 nvcc 时先按服务器的模块系统或管理员提供的安装方式配置，可参考 [NVIDIA CUDA 12.8 安装指南](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-linux/index.html)。不要把 `conda install cudatoolkit` 当成完整编译环境。

在使用标准 `/usr/local/cuda-12.8` 安装路径的机器上，可设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
nvcc --version
```

如果集群使用 `module load cuda/...`，按集群实际路径设置，不必覆盖成上述路径。

## 2. 创建干净的 conda 环境

先确认服务器已有 conda；若没有，按 [conda 官方 Linux 安装说明](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html)安装并初始化 Bash，再打开新终端。

```bash
conda --version
conda create -n kl-rl python=3.12 pip -y
conda activate kl-rl

# 防止误读用户目录中其他环境的 Python 包
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip setuptools wheel

which python
python --version
python -m pip --version
```

Python 和 pip 应指向同一个 `kl-rl` 环境。后文统一用 `python -m pip`，避免 shell 中的 pip 指向其他解释器。

这里 conda 只负责隔离环境和 Python。**torch、vLLM、FlashAttention 等统一用 pip 安装。**vLLM 官方提示，conda 安装的 PyTorch 静态链接 NCCL，可能影响 vLLM 的 NCCL 使用，见[该版本安装说明](https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html#create-a-new-python-environment)。不要先 `conda install pytorch` 再混装 pip 的计算库。

## 3. 定位本仓库，写入版本约束

若已经上传或克隆代码，直接进入已有目录；否则执行：

```bash
git clone https://github.com/ScaleML/KL-in-LLM-RL.git
cd KL-in-LLM-RL
```

以下在仓库根目录执行：

```bash
export REPO_ROOT="$PWD"
mkdir -p "$REPO_ROOT/environment-records"
git rev-parse HEAD

cat > "$REPO_ROOT/environment-records/constraints-conda-pip.txt" <<'EOF'
torch==2.8.0
torchvision==0.23.0
torchaudio==2.8.0
vllm==0.11.0
flash-attn==2.8.1
flashinfer-python==0.3.1
numpy<2.0.0
transformers>=4.51.0,<5
pyarrow>=19.0.0
tensordict>=0.8.0,<=0.10.0,!=0.9.0
EOF

export KL_CONSTRAINTS="$REPO_ROOT/environment-records/constraints-conda-pip.txt"
```

约束文件只限制被安装的包，不会单独安装其中全部依赖。`transformers<5` 是为该旧栈采用的兼容策略，不是本地 setup.py 的硬性要求；整个环境仍需预检与试跑。首次安装后保存实际 freeze，不要把范围约束视为完整复现锁定。

## 4. 安装 vLLM 与匹配的 PyTorch

先安装推理引擎及其匹配计算库，再安装本地训练框架，避免先装一份任意版本的 torch 又被替换。

```bash
python -m pip install -c "$KL_CONSTRAINTS" \
  'vllm==0.11.0' 'torch==2.8.0' \
  'torchvision==0.23.0' 'torchaudio==2.8.0' \
  --extra-index-url https://download.pytorch.org/whl/cu128

python -c "import torch, vllm; print('torch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('vLLM:', vllm.__version__); print('CUDA available:', torch.cuda.is_available())"
```

预期 torch 显示 2.8.0（可能带 `+cu128`），CUDA runtime 为 12.8，vLLM 为 0.11.0，CUDA available 为 True。这里不使用自动升级到最新 vLLM 的命令。

如果显示 CUDA 不可用，先检查 GPU 驱动、设备权限、容器/作业分配和 `CUDA_VISIBLE_DEVICES`，不要继续安装更多包来碰运气。

## 5. 安装本地 verl 和数学训练依赖

```bash
cd "$REPO_ROOT/verl"
python -m pip install -c "$KL_CONSTRAINTS" -e '.[vllm,math]' \
  'flashinfer-python==0.3.1' \
  latex2sympy2_extended liger-kernel ninja packaging psutil

python -c "import verl; print(verl.__file__)"
```

`verl.__file__` 必须指向当前项目下的 `verl/verl/__init__.py`。`-e` 是 editable 安装，后续修改源码无需重新拷贝包。`math` extra 提供 Math-Verify；本地 setup.py 会安装 Ray、Hydra、datasets、pyarrow、wandb 等基础依赖。

这条路线只安装 vLLM + FSDP 所需组件。不要同时装 `.[sglang]`：本地 SGLang extra 要求 torch 2.9.1，会与这里的 2.8.0 冲突。也不需要直接运行默认同时安装多个后端的 `install_vllm_sglang_mcore.sh`。

## 6. 安装 FlashAttention

默认训练脚本启用了 remove padding，需要可用的 FlashAttention。先核对 torch 的编译信息：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
nvcc --version
```

如果本环境尚未安装 FlashAttention，使用匹配开发环境进行安装：

```bash
MAX_JOBS=4 python -m pip install -c "$KL_CONSTRAINTS" \
  'flash-attn==2.8.1' --no-build-isolation

python -c "import flash_attn; print(flash_attn.__version__)"
```

`--no-build-isolation` 让构建读取当前环境的 torch；`MAX_JOBS=4` 限制编译并行度，可按主存调整。安装器可能获取预编译 wheel，也可能退回源码编译，后者需要较长时间。

若使用手动下载的 wheel，必须匹配 Python 3.12、Linux 架构、torch 2.8、CUDA 和 CXX11 ABI。本仓库安装脚本写死的 ABI FALSE wheel 不能无条件套用到当前环境。出现 `undefined symbol` 应先排查二进制匹配，而不是随意升级 torch。

## 7. 验证依赖、CUDA kernel 与数学奖励

先检查包关系：

```bash
python -m pip check
python -m verl.trainer.main_ppo --help
```

`pip check` 应无冲突；`--help` 应能加载本地配置入口。这两项都不等于训练通过。再创建一个实际调用 GPU kernel 的检查脚本：

```bash
cat > "$REPO_ROOT/environment-records/check_training_env.py" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import os

def main():
    import torch
    import verl
    from flash_attn import flash_attn_func
    from verl.utils.reward_score import default_compute_score

    for name in ('torch', 'vllm', 'transformers', 'ray', 'tensordict',
                 'flash-attn', 'flashinfer-python', 'pyarrow', 'math-verify'):
        print(name, metadata.version(name), flush=True)
    source = Path(verl.__file__).resolve()
    expected = Path(os.environ['REPO_ROOT']) / 'verl' / 'verl' / '__init__.py'
    print('verl source:', source, flush=True)
    assert source == expected.resolve(), '导入的不是本仓库 verl'
    assert torch.cuda.is_available(), 'CUDA 不可用'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('CUDA:', torch.version.cuda, 'GPU count:', torch.cuda.device_count(), flush=True)

    # 小规模实际前向 + 反向，检查 BF16 与 FlashAttention kernel
    q = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    out = flash_attn_func(q, k, v, causal=True)
    out.float().square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all().item()
    assert q.grad is not None and torch.isfinite(q.grad).all().item()
    print('FlashAttention forward/backward: PASS', flush=True)

    score = default_compute_score('dapo_filter', r'\boxed{2}', '2')
    print('math reward:', score, flush=True)
    assert score == 1.0, '检查 Math-Verify 的依赖与子进程异常'
    print('Environment checks: PASS', flush=True)

if __name__ == '__main__':
    main()
PY

python "$REPO_ROOT/environment-records/check_training_env.py"
```

这里使用真实文件及 `if __name__ == '__main__'`，以兼容数学奖励模块的 spawn 子进程。不要把这段检查改为 `python -` 从标准输入运行。通过检查说明基础计算与奖励链路可用，vLLM 的实际模型生成和训练分布式路径仍由下一步验证。

## 8. 准备模型和数据，进行两步训练

先按[训练环境配置指南第 5、6 节](训练环境配置指南.md)下载模型并检查 tokenizer chat template、准备 Parquet；这两节的命令不依赖 uv，可直接在当前 conda 环境执行。数据集字段不匹配时需要显式映射，不能只下载文件就假定格式正确。

假定文件位于：

```text
KL-in-LLM-RL/verl/
  models/Qwen3-1.7B-Base/
  data/dapo_filter.parquet
  data/math500/test.parquet
```

在当前环境运行以下小规模检查。它使用短上下文、小 batch、两步训练，验证整条链路，不能用于报告正式实验指标。

```bash
cd "$REPO_ROOT/verl"
export DATA_ROOT="$PWD"
export MODEL_PATH="$DATA_ROOT/models/Qwen3-1.7B-Base"
export TRAIN_FILE="$DATA_ROOT/data/dapo_filter.parquet"
export CKPTS_DIR="$DATA_ROOT/ckpts/conda-pip-smoke"

test -d "$MODEL_PATH"
test -f "$TRAIN_FILE"
test -f "$DATA_ROOT/data/math500/test.parquet"
# 以上检查都成功后再继续

export MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=512
export ROLLOUT_TP=1 ROLLOUT_N=2 VAL_N=1
export PPO_MINI_BATCH_SIZE=2
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.4 ROLLOUT_MAX_NUM_SEQS=4
export OFFLOAD=True

set -o pipefail
bash rl-kl/train.sh \
  "data.val_files=['$DATA_ROOT/data/math500/test.parquet']" \
  data.train_batch_size=2 data.train_max_samples=16 data.val_max_samples=2 \
  data.filter_overlong_prompts_workers=2 \
  trainer.nnodes=1 trainer.n_gpus_per_node=1 \
  trainer.total_training_steps=2 trainer.total_epochs=1 \
  trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=2 \
  trainer.experiment_name=conda-pip-smoke 'trainer.logger=[console]' \
  actor_rollout_ref.rollout.enforce_eager=True \
  2>&1 | tee "$REPO_ROOT/environment-records/conda-pip-smoke.log"
```

验收要看到真实两步更新、有限的 loss/grad norm、没有奖励解析异常，并在保存目录看到 `global_step_2/actor`。如果 1024 长度过滤后样本不足，增加 train_max_samples 或 prompt 上限。是否能在单卡装下仍取决于显存；OOM 时优先降低序列长度、生成并发与 token 预算。

默认脚本是 4 节点 × 8 GPU，所以这里必须覆盖节点和 GPU 数。验证集路径也必须显式覆盖，因为脚本中的 TEST_FILE 独立写死，不随 DATA_ROOT 改变。

测试 PRM 可改成 `train_prl.sh`；current-policy PRL 使用 `train_prl_current_policy.sh`，保留 `trainer.use_legacy_worker_impl=disable`（总配置已默认如此）。每种实验用不同保存目录，避免自动恢复到别的任务。正式参数、loss、导出与评测见[代码实现与训练说明](代码实现与训练说明.md)。

### 8.1 使用 SwanLab 记录四卡训练实验

本仓库已经在 `verl/verl/utils/tracking.py` 原生支持 SwanLab，trainer 会按训练 step 提交指标和配置，不需要改 loss 或增加手写 `swanlab.log()`。但 setup.py 未把 SwanLab 列为必装包，需要单独安装。官方也提供了 [verl 集成说明](https://docs.swanlab.cn/guide_cloud/integration/integration-verl.html)。

先在服务器当前训练环境中安装并登录：

```bash
conda activate kl-rl
python -m pip install -c "$KL_CONSTRAINTS" swanlab
python -m pip check
swanlab login
```

按交互提示输入 SwanLab 账号的 API Key。默认登录凭证保存在当前系统用户目录；在同一用户下启动的 Ray 任务可使用它，见 [SwanLab 登录文档](https://docs.swanlab.cn/api/cli-swanlab-login.html)。不需要把密钥写进脚本、文档或 Git。无交互作业也可使用平台注入的 `SWANLAB_API_KEY`，本地 tracking 实现会读取它；如果使用已经启动的远程 Ray 集群，要另外确认 trainer 所在节点收到认证配置。

以下是单机 **4 × A100 40GB** 的短程检查配置，模型与 Parquet 按上一节准备：

```bash
cd "$REPO_ROOT/verl"
export DATA_ROOT="$PWD"
export MODEL_PATH="$DATA_ROOT/models/Qwen3-1.7B-Base"
export TRAIN_FILE="$DATA_ROOT/data/dapo_filter.parquet"
export CKPTS_DIR="$DATA_ROOT/ckpts/a100x4-grpo-swanlab"

export SWANLAB_MODE=cloud
export SWANLAB_LOG_DIR="$REPO_ROOT/swanlog"
mkdir -p "$SWANLAB_LOG_DIR"

export MAX_PROMPT_LENGTH=2048 MAX_RESPONSE_LENGTH=2048
export PPO_MINI_BATCH_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1
export ROLLOUT_TP=1 ROLLOUT_N=5 VAL_N=1
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 ROLLOUT_MAX_NUM_SEQS=16
export OFFLOAD=True

set -o pipefail
bash rl-kl/train.sh \
  "data.val_files=['$DATA_ROOT/data/math500/test.parquet']" \
  data.train_batch_size=32 data.val_max_samples=16 \
  data.filter_overlong_prompts_workers=4 \
  trainer.nnodes=1 trainer.n_gpus_per_node=4 \
  trainer.total_training_steps=10 \
  trainer.val_before_train=False trainer.test_freq=10 trainer.save_freq=10 \
  'trainer.logger=[console,swanlab]' \
  trainer.project_name=kl-in-llm-rl \
  trainer.experiment_name=qwen3-1.7b-grpo-a100x4-check \
  trainer.log_val_generations=2 \
  actor_rollout_ref.rollout.enforce_eager=True \
  2>&1 | tee "$REPO_ROOT/environment-records/a100x4-grpo-swanlab.log"
```

这组参数是针对四卡的起步建议，未做实机显存或速度验证。原入口现在已默认 `console + swanlab`，也可通过 `TRAINER_LOGGER` 或末尾 Hydra 参数覆盖；以最终打印的配置为准。

| 配置 | 效果 |
| --- | --- |
| `trainer.logger=[console,swanlab]` | 控制台输出，并把已实现的训练/验证指标提交 SwanLab |
| `trainer.project_name` | SwanLab 项目名称 |
| `trainer.experiment_name` | 该次实验名称 |
| `SWANLAB_MODE=cloud` | 使用云端模式；本仓库显式读取并传给 swanlab.init |
| `SWANLAB_LOG_DIR` | 本地 SwanLab 日志目录，使用绝对路径 |
| `trainer.log_val_generations=2` | 验证时记录少量问题、回答和得分，表格键为 `val/generations` |

初始化成功后，终端通常会给出实验链接。检查项目中是否出现逐步更新的 reward、actor loss、耗时等曲线；第 10 步验证后才会有本例的验证指标和样例。不同算法产生不同指标：`train_prl.sh` 在 trainer 中修正优势，不会自动拥有 current-policy loss 专属的 `actor/prl_*` 指标；`train_prl_current_policy.sh` 则在 loss 中记录这些指标。没有启用的计算项不会凭空出现在面板中。

SwanLab 接收的指标和配置不等于完整原始终端输出；完整 stdout/stderr 同时由示例中的 `tee` 留存。训练 checkpoint 仍保存在 CKPTS_DIR，不能把实验面板当作模型文件备份。

切换算法时把脚本分别改为 `train_prl.sh`、`train_prl_current_policy.sh`，同时更换 experiment_name 和 CKPTS_DIR；current-policy 保持 `trainer.use_legacy_worker_impl=disable`。项目名相同、实验名不同，便于在同一项目比较。

如果服务器无法联网，可设 `SWANLAB_MODE=offline` 先保留本地记录，暂不实时上传；不要把 offline 模式误认为云端正在同步。有关模式与认证的说明见 [SwanLab 环境变量文档](https://docs.swanlab.cn/api/environment-variable.html)。

### 8.2 使用已经封装的四卡启动脚本

仓库现在提供 `verl/rl-kl/train_a100_40g_4gpu.sh`，可以代替手写第 8.1 节的长命令。先激活环境、安装并登录 SwanLab，在仓库根目录运行：

```bash
# 首次连通性检查：10 步，最后执行验证和保存
TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=10 \
  bash verl/rl-kl/train_a100_40g_4gpu.sh

# 正常启动，默认最多 3000 步
bash verl/rl-kl/train_a100_40g_4gpu.sh

# 切换算法
TRAIN_MODE=prm bash verl/rl-kl/train_a100_40g_4gpu.sh
TRAIN_MODE=prl_current_kl bash verl/rl-kl/train_a100_40g_4gpu.sh
```

启动器不自动安装包或激活 conda。缺少 swanlab 会打印安装提示；`SWANLAB_LOGIN=1` 可在启动时交互登录，否则复用已有凭证或环境中的 API Key。它会检查两份 Parquet 和至少四张可见 CUDA GPU，并将完整终端输出保存到 `verl/logs/`。

可用环境变量覆盖 `MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE`、`TRAIN_BATCH_SIZE`、`MAX_RESPONSE_LENGTH`、`PROJECT_NAME`、`EXPERIMENT_NAME`、`CKPTS_DIR`、`TOTAL_TRAINING_STEPS`、`TEST_FREQ`、`SAVE_FREQ`。默认验证只取最多 32 个样本、每题采样 1 次，属于轻量检查；完整验证可设 `VAL_MAX_SAMPLES=-1 VAL_N=16`，计算量会增加。默认长度为 2048/2048，不等同原脚本的 2048/8192 实验。

```bash
MODEL_PATH=/absolute/models/Qwen3-1.7B-Base \
TRAIN_FILE=/absolute/data/dapo_filter.parquet \
VAL_FILE=/absolute/data/math500/test.parquet \
  bash verl/rl-kl/train_a100_40g_4gpu.sh trainer.total_training_steps=100
```

脚本末尾的 Hydra 参数优先级最高。每次默认创建带算法名和时间戳的新实验，恢复训练需显式设置上次的 CKPTS_DIR；也建议保持 EXPERIMENT_NAME 相同。SwanLab 的训练指标恢复为新 run 还是原 run 由 SDK/接入实现决定，本仓库未显式设置 SwanLab resume ID。

### 8.3 六个基准的独立四卡评测

一键安装已默认准备全部六个评测集。若只想补下载评测数据，不检查 GPU 或下载模型，可以在当前训练环境中执行：

```bash
python verl/rl-kl/prepare_a100_training.py --benchmarks-only
```

默认输出到 `verl/data/<名称>/test.parquet`，数据集 ID、split、显式 revision 和路径记录到 `verl/benchmark-assets.json`。默认 test split 不存在时会停止，不会自动改用 train split。可用 `<大写名称>_SPLIT` 和 `<大写名称>_DATASET_REVISION` 指定五个非 math500 基准的 split 与版本；math500 继续使用 VAL_SPLIT/VAL_DATASET_REVISION/VAL_FILE。

在仓库根目录，把训练 checkpoint 的 actor 分片合并后再评测：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /absolute/ckpts/global_step_100/actor \
  --target_dir /absolute/models/trained-merged

EVAL_MODEL_PATH=/absolute/models/trained-merged \
  bash verl/rl-kl/eval_a100_40g_4gpu.sh
```

该入口使用 TP=1、DP=4，每题采样 16 次，最长生成 8192 token，每个 worker 每批处理 1 题，并复用原有 `eval/eval.py` 的 Math-Verify 评分和 pass@n/mean@n 统计。默认保存详细 JSONL 和 summary JSON 至 `eval/results/`，无需将六个数据集都塞进训练验证配置。评测不重新安装环境，不创建训练任务，也不自动上传独立评测结果到训练的 SwanLab run。

可以使用 `EVAL_N=1` 降低首次检查的采样量，或修改 `EVAL_MAX_RESPONSE_LENGTH`、`EVAL_MAX_MODEL_LEN`、`EVAL_MAX_NUM_SEQS`、`EVAL_GENERATE_BATCH_SIZE`、`EVAL_OUTPUT_DIR`。这些是建议起步值，尚未实测 GPU 显存。修改长度和采样数后，指标不应直接当作原始完整评测配置的结果。传给 EVAL_MODEL_PATH 的必须是包含 config、tokenizer 和完整权重的 HF 目录。

## 9. 保存环境，下次继续使用

安装与试跑成功后执行：

```bash
python -m pip freeze > "$REPO_ROOT/environment-records/requirements-conda-pip-resolved.txt"
conda env export --no-builds > "$REPO_ROOT/environment-records/conda-environment.yml"
conda env export --from-history > "$REPO_ROOT/environment-records/conda-python-history.yml"
git -C "$REPO_ROOT" rev-parse HEAD > "$REPO_ROOT/environment-records/code-commit.txt"
nvidia-smi > "$REPO_ROOT/environment-records/nvidia-smi.txt"
nvcc --version > "$REPO_ROOT/environment-records/nvcc.txt"
```

完整 environment.yml 与 pip freeze 用于记录实际安装情况；freeze 中 editable 的本地绝对路径在另一台机器上要调整，并重新安装那台机器的本地源码。仅使用 `--from-history` 导出的文件无法还原 pip 计算栈，因此两类记录都保留。conda 导出方式见[官方环境管理文档](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)。

下次打开终端：

```bash
conda activate kl-rl
export PYTHONNOUSERSITE=1
cd /你的实际路径/KL-in-LLM-RL
export REPO_ROOT="$PWD"
export KL_CONSTRAINTS="$REPO_ROOT/environment-records/constraints-conda-pip.txt"
cd verl
```

DATA_ROOT、MODEL_PATH 及实验参数需要重新设置，环境激活不会自动恢复上一终端的 export。后续新增 pip 包也应带 `-c "$KL_CONSTRAINTS"`，安装后执行 `python -m pip check`，避免核心计算栈被替换。

## 10. 出错时从哪里查

| 报错或现象 | 排查与处理 |
| --- | --- |
| `conda activate` 不可用 | 初始化当前 Bash，重新打开终端，确认 conda 安装路径 |
| pip 安装到了其他环境 | 对比 `which python` 与 `python -m pip --version`；不要用 sudo pip 或 `--user` |
| torch CUDA 不可用 | 先检查驱动、作业分配、GPU 可见性及实际 torch wheel |
| FlashAttention 找不到 nvcc | 配置 CUDA toolkit 开发环境、CUDA_HOME；pip 的运行时包不能替代编译器 |
| `undefined symbol` | 检查 torch/CUDA/Python/ABI 与 wheel 是否匹配，在正确环境重新安装匹配包 |
| `No matching distribution` | 检查 Python、CPU 架构和 Linux/glibc；不要仅为绕过错误升级整套栈 |
| pip `ResolutionImpossible` | 读取具体冲突包，核对约束；特别检查是否误装 SGLang extra |
| 数学奖励始终为 0 | 先运行第 7 节标准答案测试，排除缺包和 spawn 错误 |
| trainer 等待 GPU | 默认 32 卡配置是否已覆盖，检查 `ray status` |
| 找不到数据或 chat template | 检查绝对路径、验证集覆盖、tokenizer 模板和实际 Parquet schema |
| 环境能导入但训练 OOM | 兼容性与显存容量是两件事；先降低长度和并发，再逐步扩大配置 |

多机训练时，每个节点都要激活兼容的 conda 环境，并核对包版本、代码路径和数据路径；具体 Ray 启动流程见[训练环境配置指南第 8 节](训练环境配置指南.md)。

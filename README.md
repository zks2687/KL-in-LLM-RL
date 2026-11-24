# Future-KL Regularized GRPO: Process-Level Credit Assignment from $f$-Divergence Regularization
This is the codebase for paper "Future-KL Regularized GRPO: Process-Level Credit Assignment from $f$-Divergence Regularization".

## 中文文档

- [代码实现与训练说明](docs/代码实现与训练说明.md)：训练流程、三种训练路径、参数、loss 与 future-KL 实现、运行及评测方式。
- [训练环境配置指南](docs/训练环境配置指南.md)：结合本地依赖与 verl 官方资料的环境配置、数据准备、单机试跑和多机启动步骤。
- [Conda + pip 安装教程](docs/Conda与pip训练环境安装教程.md)：使用 conda 创建环境，通过 pip 安装匹配的训练依赖，完成环境检查与两步训练试跑。

## Data Preparation
Training data: [FlippyDora/dapo-filter](https://huggingface.co/datasets/FlippyDora/dapo-filter)

Evaluation benchmarks:
1. [CorrectKLinRL/olympiadbench](https://huggingface.co/datasets/CorrectKLinRL/olympiadbench)
2. [CorrectKLinRL/minerva_math](https://huggingface.co/datasets/CorrectKLinRL/minerva_math)
3. [CorrectKLinRL/amc23](https://huggingface.co/datasets/CorrectKLinRL/amc23)
4. [CorrectKLinRL/aime25](https://huggingface.co/datasets/CorrectKLinRL/aime25)
5. [CorrectKLinRL/aime24](https://huggingface.co/datasets/CorrectKLinRL/aime24)
6. [CorrectKLinRL/math500](https://huggingface.co/datasets/CorrectKLinRL/math500)

一键安装脚本默认准备以上全部六个评测集，保存为 `verl/data/<benchmark>/test.parquet`，并记录 `verl/benchmark-assets.json`；训练期间默认仍以 math500 做轻量验证。仅下载评测数据可运行：

```bash
python verl/rl-kl/prepare_a100_training.py --benchmarks-only
```

训练结束后，在同一 conda 环境中先把 FSDP checkpoint 合并为 HF 模型，再用四卡评测全部六个基准：

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir /absolute/ckpts/global_step_100/actor \
  --target_dir /absolute/models/trained-merged
EVAL_MODEL_PATH=/absolute/models/trained-merged \
  bash verl/rl-kl/eval_a100_40g_4gpu.sh
```

四卡评测默认 TP=1、DP=4，每题采样 16 次、最长生成 8192 token，输出结果 JSONL 和 summary JSON 到 `eval/results/`。这套评测参数尚未做真实 GPU 验证；`EVAL_N=1` 可减少首次连通测试的采样量。长回答评测模板应与训练一致。

## Training
Use the scripts under `verl/rl-kl`.
```bash
cd verl
bash rl-kl/train.sh
```

单机 4 张 A100 40GB 可使用以下入口（在仓库根目录执行）：

如果还没安装 Python 训练环境，Linux x86_64 服务器已有 NVIDIA 驱动和 CUDA 12.8 开发工具时，可一键安装并启动训练：

```bash
bash verl/rl-kl/install_conda_and_train_a100.sh
```

脚本会复用 conda（没有时安装 Miniforge），创建 Python 3.12 环境，用 pip 安装固定计算栈与本地 verl，检查四卡 CUDA/FlashAttention 和数学奖励，准备默认模型与数据，登录 SwanLab 后启动四卡训练。首次需要下载模型和数据。Base tokenizer 若缺少聊天模板，默认会在本地模型副本写入纯文本模板并备份原配置；正式复现实验请设置 `CHAT_TEMPLATE_FILE`，或用 `ALLOW_PLAIN_CHAT_TEMPLATE=0` 要求已有模板。

```bash
# 一键安装后先跑 10 步
TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=10 \
  bash verl/rl-kl/install_conda_and_train_a100.sh

# 已准备好环境：跳过 pip 安装，但继续检查、准备资产与启动训练
SKIP_INSTALL=1 bash verl/rl-kl/install_conda_and_train_a100.sh
```

已有 conda 训练环境时，也可以直接运行下面的独立训练入口：

```bash
conda activate kl-rl
python -m pip install swanlab
swanlab login
bash verl/rl-kl/train_a100_40g_4gpu.sh
```

默认训练 Qwen3-1.7B-Base，使用 GRPO、2048/2048 prompt/response 长度、32 prompt batch，并将指标上传 SwanLab；完整终端日志保存在 `verl/logs/`。模型优先从 `verl/models/Qwen3-1.7B-Base` 加载，目录不存在时使用 Hugging Face ID。训练集和验证集默认在 `verl/data/` 下，也可通过 `MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE` 指定绝对路径。此配置尚未做实机 GPU 验证。

```bash
# 先用 10 步验证训练、保存和验证链路
TOTAL_TRAINING_STEPS=10 SAVE_FREQ=10 TEST_FREQ=10 \
  bash verl/rl-kl/train_a100_40g_4gpu.sh

# 另外两种实验；每次默认生成独立的实验名和 checkpoint 目录
TRAIN_MODE=prm bash verl/rl-kl/train_a100_40g_4gpu.sh
TRAIN_MODE=prl_current_kl bash verl/rl-kl/train_a100_40g_4gpu.sh
```

三个原训练入口也默认使用 SwanLab；可设 `SWANLAB_LOGIN=1` 在启动时交互登录，或复用已保存的凭证/环境中的 `SWANLAB_API_KEY`。无需写入密钥。设置 `TRAINER_LOGGER='[console]'` 可关闭云端记录，`SWANLAB_MODE=offline` 可使用离线日志。更多设置见 [Conda + pip 教程](docs/Conda与pip训练环境安装教程.md)。

## Citation
```
@article{yao2026prl,
  title={PRL: Process Reward Learning Improves LLMs' Reasoning Ability and Broadens the Reasoning Boundary},
  author={Yao, Jiarui and Wang, Ruida and Zhang, Tong},
  journal={arXiv preprint arXiv:2601.10201},
  year={2026}
}
```

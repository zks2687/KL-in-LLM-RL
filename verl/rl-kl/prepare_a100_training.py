"""Check the GPU stack and prepare local assets for the one-command launcher."""

import argparse
import json
import os
from pathlib import Path

BENCHMARKS = (
    "olympiadbench", "minerva_math", "amc23", "aime25", "aime24", "math500",
)


def normalize_row(row, index, source, split):
    """Accept verl-format data, or explicitly recognized raw math columns."""
    if "prompt" in row and isinstance(row.get("reward_model"), dict):
        prompt = row["prompt"]
        gold = row["reward_model"].get("ground_truth")
    elif "answer" in row and ("problem" in row or "question" in row):
        question = row.get("problem", row.get("question"))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Empty or non-text question at row {index}")
        prompt = [{"role": "user", "content": question.strip()
                   + "\n\nLet's think step by step and output the final answer within \\boxed{}."}]
        gold = row["answer"]
    else:
        raise ValueError(f"Unsupported dataset schema: {sorted(row)}. "
                         "Expected prompt + reward_model, or problem/question + answer.")
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"prompt must be a non-empty message list at row {index}")
    for message in prompt:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str) \
                or not isinstance(message.get("content"), str):
            raise ValueError(f"Invalid text-only message at row {index}")
    if gold is None or isinstance(gold, (dict, list, bool)) or not str(gold).strip():
        raise ValueError(f"Invalid scalar ground_truth at row {index}")
    return {"data_source": source, "prompt": prompt, "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(gold)},
            "extra_info": {"index": index, "split": split}}


def prepare_dataset(repo_id, split, output, source, revision=None):
    from datasets import load_dataset

    output = Path(output).expanduser().resolve()
    if output.is_file():
        ds = load_dataset("parquet", data_files=str(output), split="train")
        if not len(ds):
            raise ValueError(f"Empty local dataset: {output}")
        for i in range(min(32, len(ds))):
            row = ds[i]
            if "prompt" not in row or not isinstance(row.get("reward_model"), dict) \
                    or not isinstance(row["reward_model"].get("ground_truth"), str):
                raise ValueError(f"Existing {output} must already use verl-format prompt and "
                                 "reward_model.ground_truth (string); raw local files are not overwritten")
            normalize_row(row, i, source, split)
            if row.get("data_source") != source:
                raise ValueError(f"Existing {output}: expected data_source={source!r}; "
                                 "normalize your local data explicitly before training.")
        print(f"Using existing dataset: {output} ({len(ds)} rows)", flush=True)
        return
    print(f"Downloading {repo_id}, split={split}, revision={revision or 'main'}", flush=True)
    ds = load_dataset(repo_id, split=split, revision=revision)
    if not len(ds):
        raise ValueError(f"Empty remote dataset: {repo_id}/{split}")
    normalize_row(ds[0], 0, source, split)
    ds = ds.map(lambda row, idx: normalize_row(row, idx, source, split),
                with_indices=True, remove_columns=ds.column_names)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    ds.to_parquet(str(temporary))
    temporary.replace(output)
    print(f"Saved {output} ({len(ds)} rows)", flush=True)


def prepare_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    model_path = Path(os.environ["MODEL_PATH"]).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        snapshot_download(os.environ["MODEL_ID"], local_dir=str(model_path),
                          revision=os.environ.get("MODEL_REVISION"))
    if not any(model_path.glob("*.safetensors")) and not any(model_path.glob("pytorch_model*.bin")):
        raise ValueError(f"No model weights in {model_path}; complete the model download first")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    template_file = os.environ.get("CHAT_TEMPLATE_FILE")
    if template_file or not tokenizer.chat_template:
        if template_file:
            template = Path(template_file).expanduser().read_text(encoding="utf-8")
        elif os.environ.get("ALLOW_PLAIN_CHAT_TEMPLATE", "1") == "1":
            template = "{% for m in messages %}{{ m['content'] }}{{ '\\n' }}{% endfor %}"
            print("NOTE: Base tokenizer has no chat template. Installing a plain-text template "
                  "in the local model copy. This is a runnable baseline, not a verified paper template. "
                  "Set CHAT_TEMPLATE_FILE for your experimental template, or "
                  "ALLOW_PLAIN_CHAT_TEMPLATE=0 to require an existing template.", flush=True)
        else:
            raise ValueError("Missing chat template. Set CHAT_TEMPLATE_FILE to a Jinja template file.")
        if not template.strip():
            raise ValueError("Chat template is empty")
        config_file = model_path / "tokenizer_config.json"
        backup = model_path / "tokenizer_config.before-bootstrap.json"
        if config_file.exists() and not backup.exists():
            backup.write_bytes(config_file.read_bytes())
        tokenizer.chat_template = template
        tokenizer.save_pretrained(str(model_path))
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": "1+1=?"}],
                                             tokenize=False, add_generation_prompt=True)
    if not rendered:
        raise ValueError("Chat template rendered an empty prompt")
    print(f"Model and tokenizer ready: {model_path}", flush=True)


def prepare_benchmarks():
    """Use the six datasets named in the repository README, never a train split fallback."""
    root = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
    paths = {}
    for name in BENCHMARKS:
        if name == "math500":
            repo = os.environ.get("VAL_DATASET", "CorrectKLinRL/math500")
            split = os.environ.get("VAL_SPLIT", "test")
            path = os.environ.get("VAL_FILE", str(root / "data" / name / "test.parquet"))
            revision = os.environ.get("VAL_DATASET_REVISION")
        else:
            repo = f"CorrectKLinRL/{name}"
            split = os.environ.get(f"{name.upper()}_SPLIT", "test")
            path = str(root / "data" / name / "test.parquet")
            revision = os.environ.get(f"{name.upper()}_DATASET_REVISION")
        prepare_dataset(repo, split, path, name, revision)
        paths[name] = {"dataset": repo, "split": split, "revision": revision,
                       "path": str(Path(path).expanduser().resolve())}
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark-assets.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    return paths


def check_environment():
    import importlib.metadata as metadata
    import torch
    import verl
    from flash_attn import flash_attn_func
    from verl.utils.reward_score import default_compute_score

    expected = Path(os.environ["KL_REPO_ROOT"]) / "verl" / "verl" / "__init__.py"
    if Path(verl.__file__).resolve() != expected.resolve():
        raise RuntimeError(f"Wrong verl source: {verl.__file__}")
    for name in ("torch", "vllm", "transformers", "flash-attn", "flashinfer-python", "swanlab"):
        print(f"{name}: {metadata.version(name)}", flush=True)
    if torch.version.cuda != "12.8":
        raise RuntimeError(f"Expected cu128 PyTorch, got CUDA {torch.version.cuda}")
    if torch.cuda.device_count() < 4:
        raise RuntimeError("Need at least 4 visible CUDA GPUs; check the job allocation/CUDA_VISIBLE_DEVICES")
    for device in range(4):
        print(f"GPU {device}: {torch.cuda.get_device_name(device)}", flush=True)
        q = torch.randn(1, 64, 4, 64, device=f"cuda:{device}", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        out = flash_attn_func(q, k, v, causal=True)
        out.float().square().mean().backward()
        torch.cuda.synchronize(device)
        if not torch.isfinite(out).all().item() or not torch.isfinite(q.grad).all().item():
            raise RuntimeError(f"Non-finite FlashAttention result on GPU {device}")
    if default_compute_score("dapo_filter", r"\boxed{2}", "2") != 1.0:
        raise RuntimeError("Math-Verify reward check failed; inspect subprocess/dependency errors")
    print("GPU kernels and math reward checks passed.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--benchmarks-only", action="store_true",
                      help="Prepare all six evaluation datasets without GPU or model checks")
    args = parser.parse_args()
    if args.benchmarks_only:
        prepare_benchmarks()
        return
    check_environment()
    if args.check_only:
        return
    prepare_model()
    prepare_dataset(os.environ.get("TRAIN_DATASET", "FlippyDora/dapo-filter"),
                    os.environ.get("TRAIN_SPLIT", "train"), os.environ["TRAIN_FILE"], "dapo_filter",
                    os.environ.get("TRAIN_DATASET_REVISION"))
    if os.environ.get("DOWNLOAD_BENCHMARKS", "1") == "1":
        prepare_benchmarks()
    else:
        prepare_dataset(os.environ.get("VAL_DATASET", "CorrectKLinRL/math500"),
                        os.environ.get("VAL_SPLIT", "test"), os.environ["VAL_FILE"], "math500",
                        os.environ.get("VAL_DATASET_REVISION"))
    record = Path(os.environ["DATA_ROOT"]).expanduser() / "bootstrap-assets.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    # Explicit allowlist: never serialize all environment variables or credentials.
    record.write_text(json.dumps({key: os.environ.get(key) for key in (
        "MODEL_ID", "MODEL_PATH", "MODEL_REVISION", "TRAIN_FILE", "VAL_FILE",
        "TRAIN_DATASET", "TRAIN_SPLIT", "TRAIN_DATASET_REVISION", "VAL_DATASET", "VAL_SPLIT",
        "VAL_DATASET_REVISION", "CHAT_TEMPLATE_FILE", "ALLOW_PLAIN_CHAT_TEMPLATE")}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

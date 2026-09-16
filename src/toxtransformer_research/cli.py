"""Training, held-out evaluation, and local prediction without hosted services."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from cvae import MultitaskEncoder, MultitaskEncoderConfig
from .data import (
    collate,
    encode_records,
    encode_smiles,
    fit_tokenizer,
    read_records,
    sha256,
    write_json,
)
from .evaluation import evaluate


def configure(seed, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def train(args):
    config = json.loads(Path(args.config).read_text())
    seed = config["seed"]
    configure(seed, args.device)
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if config["learning_rate"] <= 0 or config["max_properties"] < 1:
        raise ValueError("learning_rate and max_properties must be positive")
    records = read_records(args.data)
    tokenizer, property_names = fit_tokenizer(records)
    partitions = encode_records(
        records, tokenizer, property_names, config["max_selfies"], config["max_properties"]
    )
    if not partitions["validation"]:
        raise ValueError("A separate validation partition is required for checkpoint selection")
    model_config = MultitaskEncoderConfig(**config["model"])
    if model_config.output_size not in (None, 2):
        raise ValueError("This workflow supports binary properties only")
    if config["max_selfies"] + 2 * config["max_properties"] > model_config.max_seq_len:
        raise ValueError("max_seq_len must cover SELFIES tokens plus two tokens per property")
    if model_config.hdim % model_config.nhead or (model_config.hdim // model_config.nhead) % 2:
        raise ValueError("Attention head dimensions must be even for rotary embeddings")
    if not 0 < model_config.mean_span_length or (
        model_config.use_span_masking and model_config.mean_span_length <= 1
    ):
        raise ValueError("Span masking requires mean_span_length > 1")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    model = MultitaskEncoder(tokenizer, model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    metadata = {
        "workflow": "research-reference-v1",
        "historical_training_recipe": False,
        "config": config,
        "resolved_model_config": model.config.__dict__,
        "data_sha256": sha256(args.data),
        "config_sha256": sha256(args.config),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "device": args.device,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "numpy", "selfies", "rdkit", "scikit-learn")
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "split_compound_counts": {key: len(value) for key, value in partitions.items()},
        "parameters": model.get_num_parameters(),
    }
    write_json(output / "run.json", metadata)
    write_json(output / "properties.json", property_names)
    write_json(
        output / "splits.json",
        {key: [row["smiles"] for row in rows] for key, rows in partitions.items()},
    )
    best_loss = float("inf")
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(partitions["train"]), generator=generator).tolist()
        loss_sum, n_values = 0.0, 0
        for start in range(0, len(order), config["batch_size"]):
            rows = [partitions["train"][idx] for idx in order[start : start + config["batch_size"]]]
            batch = collate(rows, args.device, generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(*batch)
            loss = F.cross_entropy(logits[batch[3]], batch[2][batch[3]])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            count = int(batch[3].sum())
            loss_sum += float(loss.detach()) * count
            n_values += count
        # Checkpoint selection always uses structure-only validation, never the test partition.
        validation, _ = evaluate(
            model,
            partitions["validation"],
            property_names,
            args.device,
            config["batch_size"],
            context_count=0,
        )
        step = {"epoch": epoch + 1, "train_log_loss": loss_sum / n_values, "validation": validation}
        history.append(step)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_log_loss": step["train_log_loss"],
                    "validation_log_loss": validation["overall"]["log_loss"],
                }
            ),
            flush=True,
        )
        if validation["overall"]["log_loss"] < best_loss:
            best_loss = validation["overall"]["log_loss"]
            model.save(output / "best", metadata={"epoch": epoch + 1, "seed": seed})
        write_json(output / "history.json", history)
    metadata["checkpoint_sha256"] = sha256(output / "best/multitask_encoder.pt")
    write_json(output / "run.json", metadata)


def load_run(run, device):
    run = Path(run)
    metadata = json.loads((run / "run.json").read_text())
    configure(metadata["config"]["seed"], device)
    if sha256(run / "best/multitask_encoder.pt") != metadata["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum differs from run.json")
    model = MultitaskEncoder.load(run / "best").to(device)
    properties = json.loads((run / "properties.json").read_text())
    return model, properties, metadata


def evaluate_command(args):
    model, properties, metadata = load_run(args.run, args.device)
    if sha256(args.data) != metadata["data_sha256"]:
        raise ValueError(
            "Evaluation CSV must match the recorded training input, including split assignments"
        )
    config = metadata["config"]
    partitions = encode_records(
        read_records(args.data),
        model.tokenizer,
        properties,
        config["max_selfies"],
        config["max_properties"],
    )
    metrics, predictions = evaluate(
        model, partitions[args.split], properties, args.device, config["batch_size"], args.context
    )
    result = {
        "split": args.split,
        "data_sha256": metadata["data_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics": metrics,
        "predictions": predictions,
    }
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(metrics, indent=2))


@torch.no_grad()
def predict(args):
    model, properties, metadata = load_run(args.run, args.device)
    if args.property_id not in properties:
        raise ValueError(f"Unknown property_id: {args.property_id}")
    encoded = encode_smiles(args.smiles, model.tokenizer, metadata["config"]["max_selfies"])
    prop = properties.index(args.property_id)
    batch = collate([{"selfies": encoded, "pairs": [(prop, 0)]}], args.device)
    probability = float(model(*batch).softmax(-1)[0, 0, 1])
    print(
        json.dumps(
            {
                "smiles": args.smiles,
                "property_id": args.property_id,
                "probability": probability,
                "n_context": 0,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    trainer = sub.add_parser(
        "train", help="Train the architecture using an explicitly partitioned CSV"
    )
    trainer.add_argument("--data", required=True)
    trainer.add_argument("--config", required=True)
    trainer.add_argument("--output", required=True)
    trainer.add_argument("--device", default="cpu")
    trainer.set_defaults(func=train)
    evaluator = sub.add_parser("evaluate", help="Evaluate a saved run on a held-out partition")
    evaluator.add_argument("--run", required=True)
    evaluator.add_argument("--data", required=True)
    evaluator.add_argument("--split", choices=["validation", "test"], default="test")
    evaluator.add_argument("--context", type=int, default=0)
    evaluator.add_argument("--device", default="cpu")
    evaluator.add_argument("--output", required=True)
    evaluator.set_defaults(func=evaluate_command)
    predictor = sub.add_parser(
        "predict", help="Predict one property locally from molecular structure"
    )
    predictor.add_argument("--run", required=True)
    predictor.add_argument("--smiles", required=True)
    predictor.add_argument("--property-id", required=True)
    predictor.add_argument("--device", default="cpu")
    predictor.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

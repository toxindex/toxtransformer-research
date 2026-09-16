"""Inspect or query an existing checkpoint without a hosted prediction service."""

import argparse
import json
from pathlib import Path

import torch

from cvae import MultitaskEncoder
from .data import collate, encode_smiles, sha256


def verify_manifest(directory, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    directory = Path(directory).resolve()
    for artifact in manifest["files"]:
        path = (directory / artifact["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Artifact path escapes checkpoint directory")
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"Checksum mismatch: {artifact['path']}")
    return manifest


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--manifest", help="Verify the supplied checkpoint against artifact checksums"
    )
    parser.add_argument("--smiles", help="Optionally predict one property from structure")
    parser.add_argument(
        "--property-index", type=int, help="Zero-based model index, not a database ID"
    )
    parser.add_argument("--max-selfies", type=int, default=120)
    args = parser.parse_args()
    if (args.smiles is None) != (args.property_index is None):
        parser.error("--smiles and --property-index must be supplied together")
    torch.set_num_threads(1)
    try:
        if args.manifest:
            verify_manifest(args.checkpoint, args.manifest)
        model = MultitaskEncoder.load(args.checkpoint)
        result = {
            "config": model.config.__dict__,
            "parameters": model.get_num_parameters(),
            "num_properties": model.tokenizer.num_assays,
            "num_classes": model.tokenizer.num_vals,
            "checkpoint_sha256": sha256(Path(args.checkpoint) / "multitask_encoder.pt"),
        }
        if args.smiles is not None:
            if not 0 <= args.property_index < model.tokenizer.num_assays:
                raise ValueError("Property index is outside the checkpoint vocabulary")
            if args.max_selfies + 2 > model.config.max_seq_len:
                raise ValueError("SELFIES and query tokens exceed model max_seq_len")
            encoded = encode_smiles(args.smiles, model.tokenizer, args.max_selfies)
            batch = collate([{"selfies": encoded, "pairs": [(args.property_index, 0)]}], "cpu")
            result["prediction"] = {
                "smiles": args.smiles,
                "property_index": args.property_index,
                "probability": float(model(*batch).softmax(-1)[0, 0, 1]),
                "n_context": 0,
            }
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

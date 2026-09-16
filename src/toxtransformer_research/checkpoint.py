"""Inspect or query an existing checkpoint without a hosted prediction service."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import torch

from cvae import MultitaskEncoder
from .data import collate, encode_smiles, sha256


def require_weights(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing weights: {path}. Run git lfs install and git lfs pull.")
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Weights are an LFS pointer. Run git lfs install and git lfs pull.")


def verify_manifest(directory, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    directory = Path(directory).resolve()
    for artifact in manifest["files"]:
        path = (directory / artifact["path"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Artifact path escapes checkpoint directory")
        if path.suffix == ".pt":
            require_weights(path)
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"Checksum mismatch: {artifact['path']}")
    return manifest


def load_checkpoint(directory, device="cpu"):
    require_weights(Path(directory) / "multitask_encoder.pt")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation")
    # Keep stdout available for machine-readable results.
    with redirect_stdout(sys.stderr):
        return MultitaskEncoder.load(directory).to(device)


def load_properties(directory):
    path = Path(directory) / "properties.json"
    if not path.exists():
        return None
    properties = json.loads(path.read_text())
    if [p["index"] for p in properties] != list(range(len(properties))):
        raise ValueError("Property mapping must contain contiguous zero-based indices")
    if len({p["property_id"] for p in properties}) != len(properties):
        raise ValueError("Property identifiers must be unique")
    return properties


@torch.no_grad()
def predict_checkpoint(model, smiles, property_index, max_selfies=120, context=()):
    if not 0 <= property_index < model.tokenizer.num_assays:
        raise ValueError("Property index is outside the checkpoint vocabulary")
    if max_selfies < 3:
        raise ValueError("max_selfies must allow at least start, molecule, and end tokens")
    pairs = []
    seen = {property_index}
    for prop, value in context:
        if not isinstance(prop, int) or not 0 <= prop < model.tokenizer.num_assays:
            raise ValueError("Context property index is outside the checkpoint vocabulary")
        if value not in (0, 1) or prop in seen:
            raise ValueError(
                "Context requires binary values and cannot repeat or include the target"
            )
        seen.add(prop)
        pairs.append((prop, value))
    # Match upstream single-molecule inference: there is no right-padding gap
    # between the molecule's end token and the first property query.
    encoded = encode_smiles(smiles, model.tokenizer, max_selfies)
    encoded = [token for token in encoded if token != model.token_pad_idx]
    pairs.append((property_index, 0))
    if len(encoded) + 2 * len(pairs) > model.config.max_seq_len:
        raise ValueError("Molecule and context exceed model max_seq_len")
    batch = collate([{"selfies": encoded, "pairs": pairs}], next(model.parameters()).device)
    model.eval()
    probability = float(model(*batch).float().softmax(-1)[0, -1, 1])
    return {
        "smiles": smiles,
        "property_index": property_index,
        "probability": probability,
        "n_context": len(context),
        "selfies_tokens": len(encoded),
        "selfies_padding": "none",
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/toxtransformer")
    parser.add_argument(
        "--manifest", help="Verify the supplied checkpoint against artifact checksums"
    )
    parser.add_argument("--smiles", help="Optionally predict one property from structure")
    parser.add_argument(
        "--property-index", type=int, help="Zero-based model index, not a database ID"
    )
    parser.add_argument("--property-id", help="Stable identifier from the bundled property mapping")
    parser.add_argument("--list-properties", action="store_true")
    parser.add_argument(
        "--search", default="", help="Filter property titles, identifiers, and sources"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-selfies", type=int, default=120)
    args = parser.parse_args()
    if args.property_id is not None and args.property_index is not None:
        parser.error("Use either --property-id or --property-index")
    has_property = args.property_index is not None or args.property_id is not None
    if (args.smiles is not None) != has_property:
        parser.error("Supply --smiles together with --property-index or --property-id")
    torch.set_num_threads(1)
    try:
        if args.manifest:
            verify_manifest(args.checkpoint, args.manifest)
        properties = load_properties(args.checkpoint)
        if args.list_properties:
            if properties is None:
                raise ValueError("This checkpoint has no property mapping")
            query = args.search.lower()
            print(json.dumps([p for p in properties if query in json.dumps(p).lower()], indent=2))
            return
        if args.property_id is not None:
            if properties is None:
                raise ValueError("This checkpoint has no property mapping")
            matches = [p for p in properties if p["property_id"] == args.property_id]
            if not matches:
                raise ValueError(f"Unknown property_id: {args.property_id}")
            args.property_index = matches[0]["index"]
        model = load_checkpoint(args.checkpoint, args.device)
        if properties is not None and len(properties) != model.tokenizer.num_assays:
            raise ValueError("Property mapping size differs from checkpoint vocabulary")
        result = {
            "config": model.config.__dict__,
            "parameters": model.get_num_parameters(),
            "num_properties": model.tokenizer.num_assays,
            "num_classes": model.tokenizer.num_vals,
            "checkpoint_sha256": sha256(Path(args.checkpoint) / "multitask_encoder.pt"),
        }
        if args.smiles is not None:
            result["prediction"] = predict_checkpoint(
                model, args.smiles, args.property_index, args.max_selfies
            )
            if properties is not None:
                result["prediction"]["property"] = properties[args.property_index]
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

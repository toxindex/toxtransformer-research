"""Verify the distributed artifact bundle and reproduce stored CPU predictions."""

import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint, load_properties, predict_checkpoint, verify_manifest


def reproduce(directory, manifest_path, examples_path):
    torch.set_num_threads(1)
    manifest = verify_manifest(directory, manifest_path)
    model = load_checkpoint(directory)
    properties = load_properties(directory)
    if properties is None or len(properties) != model.tokenizer.num_assays:
        raise ValueError("A complete property mapping is required")
    if model.get_num_parameters()["total"] != manifest["parameters"]:
        raise ValueError("Parameter count does not match the manifest")
    reference = json.loads(Path(examples_path).read_text())
    results = []
    for case in reference["cases"]:
        result = predict_checkpoint(
            model, case["smiles"], case["property_index"], context=case.get("context", [])
        )
        delta = abs(result["probability"] - case["probability"])
        if not math.isfinite(delta) or delta > reference["absolute_tolerance"]:
            raise ValueError(f"Prediction differs for {case['name']}: absolute error {delta}")
        results.append(
            {"name": case["name"], "probability": result["probability"], "absolute_error": delta}
        )
    return {
        "checksums": "passed",
        "strict_load": "passed",
        "properties": len(properties),
        "parameters": model.get_num_parameters()["total"],
        "predictions": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/toxtransformer")
    parser.add_argument("--manifest", default="artifacts/upstream-checkpoint.json")
    parser.add_argument("--examples", default="artifacts/reference-predictions.json")
    args = parser.parse_args()
    try:
        print(json.dumps(reproduce(args.checkpoint, args.manifest, args.examples), indent=2))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

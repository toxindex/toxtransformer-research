import json
from pathlib import Path

import pytest
import torch

from cvae import MultitaskEncoder
from toxtransformer_research.checkpoint import (
    load_properties,
    predict_checkpoint,
    require_weights,
    verify_manifest,
)
from toxtransformer_research.data import collate, encode_smiles, sha256

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_rejects_modified_artifact(tmp_path):
    data = tmp_path / "artifact"
    data.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "artifact", "sha256": sha256(data)}]}))
    verify_manifest(tmp_path, manifest)
    data.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_manifest(tmp_path, manifest)


def test_manifest_rejects_path_escape(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"path": "../outside", "sha256": "unused"}]}))
    with pytest.raises(ValueError, match="escapes"):
        verify_manifest(tmp_path, manifest)


def test_inspected_upstream_checkpoint():
    torch.set_num_threads(1)
    directory = ROOT / "models/toxtransformer"
    verify_manifest(directory, ROOT / "artifacts/upstream-checkpoint.json")
    model = MultitaskEncoder.load(directory)
    assert model.get_num_parameters()["total"] == 59_218_946
    assert model.tokenizer.num_assays == 6647
    encoded = encode_smiles("CCO", model.tokenizer, 120)
    encoded = [index for index in encoded if index != model.token_pad_idx]
    with torch.no_grad():
        batch = collate([{"selfies": encoded, "pairs": [(0, 0)]}], "cpu")
        probability = model(*batch).softmax(-1)[0, 0, 1].item()
    # Regression value for this exact artifact and the CPU environment, not an accuracy claim.
    assert probability == pytest.approx(0.0865037218, abs=1e-5)
    local = predict_checkpoint(model, "CCO", 0)
    assert local["selfies_tokens"] == 5
    assert local["probability"] == pytest.approx(probability, abs=1e-7)
    with pytest.raises(ValueError, match="cannot repeat or include the target"):
        predict_checkpoint(model, "CCO", 0, context=[(0, 1)])
    properties = load_properties(directory)
    assert len(properties) == 6647
    assert properties[0]["index"] == 0


def test_lfs_pointer_has_actionable_error(tmp_path):
    weights = tmp_path / "multitask_encoder.pt"
    weights.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:example\nsize 100\n")
    with pytest.raises(ValueError, match="git lfs pull"):
        require_weights(weights)


def test_missing_weights_has_actionable_error(tmp_path):
    with pytest.raises(ValueError, match="git lfs pull"):
        require_weights(tmp_path / "missing.pt")


def test_property_mapping_rejects_reordered_indices(tmp_path):
    (tmp_path / "properties.json").write_text(
        json.dumps(
            [
                {"index": 1, "property_id": "a"},
                {"index": 0, "property_id": "b"},
            ]
        )
    )
    with pytest.raises(ValueError, match="contiguous"):
        load_properties(tmp_path)

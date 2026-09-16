import json
from pathlib import Path

import pytest
import torch

from cvae import MultitaskEncoder
from toxtransformer_research.checkpoint import verify_manifest
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


@pytest.mark.skipif(
    not (ROOT / "assets/upstream/multitask_encoder.pt").is_file(),
    reason="Separately distributed upstream checkpoint is not installed",
)
def test_inspected_upstream_checkpoint():
    torch.set_num_threads(1)
    directory = ROOT / "assets/upstream"
    verify_manifest(directory, ROOT / "artifacts/upstream-checkpoint.json")
    model = MultitaskEncoder.load(directory)
    assert model.get_num_parameters()["total"] == 59_218_946
    assert model.tokenizer.num_assays == 6647
    encoded = encode_smiles("CCO", model.tokenizer, 120)
    with torch.no_grad():
        batch = collate([{"selfies": encoded, "pairs": [(0, 0)]}], "cpu")
        probability = model(*batch).softmax(-1)[0, 0, 1].item()
    # Regression value for this exact artifact and the CPU environment, not an accuracy claim.
    assert probability == pytest.approx(0.0752025321, abs=1e-5)

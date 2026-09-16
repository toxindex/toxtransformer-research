import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cvae import MultitaskEncoder, MultitaskEncoderConfig
from toxtransformer_research.cli import train
from toxtransformer_research.data import (
    collate,
    encode_records,
    encode_smiles,
    fit_tokenizer,
    read_records,
)
from toxtransformer_research.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_and_rows():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    records = read_records(ROOT / "examples/toy.csv")
    tokenizer, names = fit_tokenizer(records)
    rows = encode_records(records, tokenizer, names, 32, 3)
    model = MultitaskEncoder(
        tokenizer,
        MultitaskEncoderConfig(
            hdim=32,
            nhead=4,
            num_layers=2,
            ff_mult=2,
            max_seq_len=64,
            layer_dropout=0,
            attention_dropout=0,
            use_span_masking=False,
        ),
    )
    return model, rows, names


@pytest.mark.parametrize("flash", [True, False])
def test_target_and_future_values_cannot_leak(model_and_rows, flash):
    model, rows, _ = model_and_rows
    for layer in model.decoder.layers:
        layer.attn.use_flash = flash
    model.eval()
    x, properties, values, mask = collate(rows["train"][:3], "cpu")
    with torch.no_grad():
        before = model(x, properties, values, mask)
        changed = values.clone()
        changed[:, 1:] = 1 - changed[:, 1:]
        after = model(x, properties, changed, mask)
    # Property query 1 must not see its own answer or any later answer.
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-6)


def test_context_changes_predictions(model_and_rows):
    model, rows, _ = model_and_rows
    model.eval()
    x, properties, values, mask = collate(rows["train"][:3], "cpu")
    with torch.no_grad():
        before = model(x, properties, values, mask)
        changed = values.clone()
        changed[:, 0] = 1 - changed[:, 0]
        after = model(x, properties, changed, mask)
    torch.testing.assert_close(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1:], after[:, 1:])


def test_padding_and_attention_implementations_agree(model_and_rows):
    model, rows, _ = model_and_rows
    model.eval()
    first, second = rows["train"][:2]
    second = {**second, "pairs": second["pairs"][:1]}
    batch = collate([first, second], "cpu")
    with torch.no_grad():
        flash = model(*batch)
        for layer in model.decoder.layers:
            layer.attn.use_flash = False
        manual = model(*batch)
    assert torch.isfinite(flash).all()
    assert torch.isfinite(manual).all()
    torch.testing.assert_close(flash[batch[3]], manual[batch[3]], atol=2e-6, rtol=2e-5)


def test_checkpoint_roundtrip_and_strict_loading(model_and_rows, tmp_path):
    model, rows, _ = model_and_rows
    model.eval()
    model.save(tmp_path / "checkpoint")
    loaded = MultitaskEncoder.load(tmp_path / "checkpoint")
    batch = collate(rows["train"][:2], "cpu")
    with torch.no_grad():
        torch.testing.assert_close(model(*batch), loaded(*batch), atol=0, rtol=0)
    path = tmp_path / "checkpoint/multitask_encoder.pt"
    payload = torch.load(path, weights_only=True)
    del payload["state_dict"]["classifier.weight"]
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        MultitaskEncoder.load(tmp_path / "checkpoint")


def test_gradient_checkpointing_with_span_masking(model_and_rows):
    model, rows, _ = model_and_rows
    model.train()
    model.decoder.use_checkpoint = True
    model.use_span_masking = True
    batch = collate(rows["train"][:2], "cpu")
    logits = model(*batch)
    loss = torch.nn.functional.cross_entropy(logits[batch[3]], batch[2][batch[3]])
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize(
    "body,match",
    [
        ("CCO,p,1,train\nOCC,p,1,test\n", "multiple splits"),
        ("CCO,p,1,train\nCCO,p,0,train\n", "Conflicting"),
        ("CCO,p,2,train\n", "Values must"),
    ],
)
def test_invalid_dataset_is_rejected(tmp_path, body, match):
    path = tmp_path / "input.csv"
    path.write_text("smiles,property_id,value,split\n" + body)
    with pytest.raises(ValueError, match=match):
        read_records(path)


def test_unknown_symbols_and_truncation_are_rejected(model_and_rows):
    model, _, _ = model_and_rows
    with pytest.raises(ValueError, match="absent from training"):
        encode_smiles("CCl", model.tokenizer, 32)
    with pytest.raises(ValueError, match="max_selfies"):
        encode_smiles("CCCCCC", model.tokenizer, 3)


def test_single_class_metrics_and_context_counts(model_and_rows):
    model, rows, names = model_and_rows
    metrics, predictions = evaluate(model, rows["test"][:1], names, "cpu", 2, context_count=1)
    assert metrics["macro_roc_auc"] is None
    assert metrics["properties_without_auc"] == 3
    assert all(item["n_context"] == 1 for item in predictions)
    assert metrics["overall"]["n"] == 3


def test_seeded_training_repeats_and_refuses_overwrite(tmp_path):
    arguments = dict(
        data=str(ROOT / "examples/toy.csv"), config=str(ROOT / "configs/smoke.json"), device="cpu"
    )
    for name in ("one", "two"):
        train(SimpleNamespace(**arguments, output=str(tmp_path / name)))
    first = json.loads((tmp_path / "one/history.json").read_text())
    second = json.loads((tmp_path / "two/history.json").read_text())
    assert first == second
    a = MultitaskEncoder.load(tmp_path / "one/best")
    b = MultitaskEncoder.load(tmp_path / "two/best")
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        train(SimpleNamespace(**arguments, output=str(tmp_path / "one")))

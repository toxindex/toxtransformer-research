"""Explicit compound splits and deterministic vocabulary construction."""

import csv
import hashlib
import json
from pathlib import Path

import selfies as sf
import torch
from rdkit import Chem

from cvae.tokenizer import SelfiesPropertyValTokenizer, SelfiesTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


def read_records(path):
    """Reject split overlap, conflicting labels, and unknown partitions."""
    compounds = {}
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"smiles", "property_id", "value", "split"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"CSV requires columns: {sorted(required)}")
        for row in reader:
            smiles = canonicalize(row["smiles"])
            split = row["split"]
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"Unknown split: {split}")
            if row["value"] not in {"0", "1"} or not row["property_id"]:
                raise ValueError("Values must be 0 or 1 and property_id must be nonempty")
            compound = compounds.setdefault(
                smiles,
                {"smiles": smiles, "selfies": sf.encoder(smiles), "split": split, "labels": {}},
            )
            if compound["split"] != split:
                raise ValueError(f"Compound occurs in multiple splits: {smiles}")
            prop, value = row["property_id"], int(row["value"])
            if prop in compound["labels"] and compound["labels"][prop] != value:
                raise ValueError(f"Conflicting labels for {smiles}, {prop}")
            compound["labels"][prop] = value
    if not compounds:
        raise ValueError("CSV contains no compounds")
    return [compounds[key] for key in sorted(compounds)]


def fit_tokenizer(records):
    training = [row for row in records if row["split"] == "train"]
    if not training:
        raise ValueError("Training partition is empty")
    tokenizer = SelfiesTokenizer()
    symbols = sorted({symbol for row in training for symbol in sf.split_selfies(row["selfies"])})
    for symbol in symbols:
        if symbol not in tokenizer.symbol_to_index:
            tokenizer.symbol_to_index[symbol] = len(tokenizer.symbol_to_index)
    tokenizer.index_to_symbol = {idx: symbol for symbol, idx in tokenizer.symbol_to_index.items()}
    properties = sorted({prop for row in training for prop in row["labels"]})
    return SelfiesPropertyValTokenizer(tokenizer, len(properties), 2), properties


def encode_smiles(smiles, tokenizer, max_selfies):
    molecule = sf.encoder(canonicalize(smiles))
    vocab = tokenizer.selfies_tokenizer
    missing = set(sf.split_selfies(molecule)) - set(vocab.symbol_to_index)
    if missing:
        raise ValueError(f"SELFIES tokens absent from training vocabulary: {sorted(missing)}")
    encoded = vocab.selfies_to_indices(molecule)
    if len(encoded) > max_selfies:
        raise ValueError(f"Molecule needs {len(encoded)} tokens; max_selfies is {max_selfies}")
    return encoded + [tokenizer.PAD_IDX] * (max_selfies - len(encoded))


def encode_records(records, tokenizer, properties, max_selfies, max_properties):
    mapping = {prop: idx for idx, prop in enumerate(properties)}
    partitions = {name: [] for name in ("train", "validation", "test")}
    for row in records:
        unknown = set(row["labels"]) - mapping.keys()
        if unknown:
            raise ValueError(f"Properties absent from training: {sorted(unknown)}")
        pairs = sorted((mapping[prop], value) for prop, value in row["labels"].items())
        if len(pairs) > max_properties:
            raise ValueError("Too many properties for a compound; increase max_properties")
        partitions[row["split"]].append(
            {
                "smiles": row["smiles"],
                "selfies": encode_smiles(row["smiles"], tokenizer, max_selfies),
                "pairs": pairs,
            }
        )
    return partitions


def collate(rows, device, generator=None):
    """Shuffle observed pairs only during training; pad with safe embedding indices."""
    length = max(len(row["pairs"]) for row in rows)
    properties = torch.zeros((len(rows), length), dtype=torch.long)
    values = torch.zeros_like(properties)
    mask = torch.zeros_like(properties, dtype=torch.bool)
    for index, row in enumerate(rows):
        pairs = row["pairs"]
        if generator is not None:
            pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=generator).tolist()]
        for position, (prop, value) in enumerate(pairs):
            properties[index, position] = prop
            values[index, position] = value
            mask[index, position] = True
    return tuple(
        t.to(device)
        for t in (
            torch.tensor([row["selfies"] for row in rows], dtype=torch.long),
            properties,
            values,
            mask,
        )
    )


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

"""Filesystem helper required by the upstream checkpoint writer."""

import shutil
from pathlib import Path


def mk_empty_directory(path, overwrite=False):
    path = Path(path)
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

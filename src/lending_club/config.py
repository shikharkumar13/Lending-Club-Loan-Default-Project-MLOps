"""Loads params.yaml and resolves project paths.

Every module reads its settings through here, so there is exactly one place
where configuration lives and no magic numbers are buried in the code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# config.py -> lending_club -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


@lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Return params.yaml as a dict (cached: the file is read once per process)."""
    params_path = path or PARAMS_PATH
    with open(params_path) as fh:
        return yaml.safe_load(fh)


def resolve(relative_path: str) -> Path:
    """Turn a path from params.yaml into an absolute path.

    Keeps the code working no matter which directory it is run from.
    """
    return PROJECT_ROOT / relative_path


def path_of(key: str) -> Path:
    """Absolute path for an entry under `paths:` in params.yaml."""
    return resolve(load_params()["paths"][key])

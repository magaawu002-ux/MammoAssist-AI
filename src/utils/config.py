"""Load the project configuration without executing YAML objects."""

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import CONFIG_DIR


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Read a YAML mapping; propagate missing-file and YAML syntax errors."""
    config_path = CONFIG_DIR / "project.yaml" if path is None else Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Project configuration must be a YAML mapping.")
    return config

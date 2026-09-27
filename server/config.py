from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str) -> Path:
    """Chemin relatif à la racine du projet."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p

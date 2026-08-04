from pathlib import Path
import sys
import yaml


def ensure_repo_root(path: Path | str | None = None) -> Path:
    """Ensure the repository root is on sys.path and return the resolved root."""
    if path is None:
        path = Path(__file__).resolve().parent
    else:
        path = Path(path)

    repo_root = path.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def load_config(config_name: str = "config.yaml", repo_root: Path | str | None = None):
    """Load a YAML configuration file from the repository root."""
    root = ensure_repo_root(repo_root)
    config_path = root / config_name
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

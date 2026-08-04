from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enable_repo_root import load_config


def test_load_config_reads_repository_yaml():
    cfg = load_config("config.yaml")

    assert cfg["parameters"]["risk_free_rate"] == 0.04
    assert cfg["defaults"]["benchmark"] == "SPY"

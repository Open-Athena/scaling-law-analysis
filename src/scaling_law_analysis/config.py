"""Global configuration for paths and Chinchilla parameters."""

from pathlib import Path

# Project-level directories
# config.py is at src/scaling_law_simulation/config.py, so we need parents[2] to get project root
PROJECT_ROOT = Path(__file__).parents[2].resolve()
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

"""Global configuration for paths and Chinchilla parameters."""

import shutil
from pathlib import Path

# Project-level directories
# config.py is at src/scaling_law_simulation/config.py, so we need parents[2] to get project root
PROJECT_ROOT = Path(__file__).parents[2].resolve()
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def prepare_output_dir(output_dir: Path) -> Path:
    """Clear and recreate an output directory for results.

    Removes all existing contents before recreating the directory,
    ensuring a clean slate for each run.

    Args:
        output_dir: Path to the output directory

    Returns:
        The output directory path
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

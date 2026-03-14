"""Shared paths for the isoflop data package."""

from scaling_law_analysis import config

DATA_DIR = config.PROJECT_ROOT / "data" / "isoflops"
EXTRACT_DIR = DATA_DIR / "extract"
TRANSFORM_DIR = DATA_DIR / "transform"
ISOFLOPS_CSV = TRANSFORM_DIR / "isoflops.csv"

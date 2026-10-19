"""Execute the notebook for every dataset and save the executed copies.

Usage:
    python run_all.py                 # steel, plastic and ldr700
    python run_all.py steel plastic   # selected datasets

Each run writes outputs/<dataset>/fan_ml_framework_<dataset>.ipynb plus the figures and
tables the notebook saves. The notebook in the repository root is left untouched.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import settings as S

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "fan_ml_framework.ipynb"


def run(dataset: str, variant: str = "recorded", timeout: int = 1200) -> Path:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    params = [c for c in nb.cells if "parameters" in c.get("metadata", {}).get("tags", [])]
    if not params:
        raise RuntimeError("No cell tagged 'parameters' in the notebook.")
    params[0].source = f'DATASET = "{dataset}"\nVARIANT = "{variant}"'
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3",
                            resources={"metadata": {"path": str(ROOT)}})
    client.execute()
    out_dir = S.OUTPUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fan_ml_framework_{dataset}.ipynb"
    nbformat.write(nb, out)
    return out


def main(argv: list[str]) -> int:
    datasets = argv or list(S.DATASETS)
    unknown = [d for d in datasets if d not in S.DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {', '.join(unknown)}. Choose from: {', '.join(S.DATASETS)}")
        return 2
    for d in datasets:
        t0 = time.time()
        path = run(d)
        print(f"{d:8s} done in {time.time() - t0:5.1f} s -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

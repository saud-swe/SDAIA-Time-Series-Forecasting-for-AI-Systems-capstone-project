"""Execute the capstone notebook in place so the committed .ipynb carries
real captured output for every cell, then verify that it does."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
NB = ROOT / "capstone_forecasting_report.ipynb"

nb = nbformat.read(NB, as_version=4)
t0 = time.time()
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.write(nb, NB)
print(f"executed in {time.time() - t0:.1f}s")

# --- verification ------------------------------------------------------
nb = json.loads(NB.read_text(encoding="utf-8"))
bad_exec, bad_md, errors = [], [], []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        if c.get("execution_count") is None:
            bad_exec.append(i)
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                errors.append((i, o.get("ename"), o.get("evalue")))
    if c["cell_type"] == "markdown":
        src = c["source"]
        if isinstance(src, str):
            bad_md.append((i, "source is a flat string, must be a list of lines"))
        elif src and src[0].lstrip().startswith("---"):
            bad_md.append((i, "markdown cell starts with a bare '---'"))

print(f"cells: {len(nb['cells'])}  "
      f"code: {sum(c['cell_type'] == 'code' for c in nb['cells'])}")
print(f"code cells with null execution_count: {bad_exec or 'none'}")
print(f"markdown cell problems: {bad_md or 'none'}")
print(f"error outputs: {errors or 'none'}")
sys.exit(1 if (bad_exec or bad_md or errors) else 0)

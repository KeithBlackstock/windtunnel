"""Condense an OpenFOAM run in case/ into docs/results.json for the Pages site.

Reads the forceCoeffs history, the simpleFoam log and the checkMesh log. Missing
pieces are reported as null rather than failing, so a partial run still publishes.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re

ROOT = Path(__file__).resolve().parent.parent
CASE = ROOT / "case"
OUT = ROOT / "docs" / "results.json"


def last_force_coeffs():
    """Last row of the newest postProcessing/coefficients/<time>/*.dat, by column name."""
    files = sorted((CASE / "postProcessing" / "coefficients").glob("*/*.dat"))
    if not files:
        return None
    cols, last = None, None
    for line in files[-1].read_text().splitlines():
        if line.startswith("#"):
            if "Time" in line:
                cols = line.lstrip("# ").split()
        elif line.strip():
            last = line.split()
    if cols is None or last is None:
        return None
    return {c: float(v) for c, v in zip(cols, last)}


def read(name):
    path = CASE / name
    return path.read_text(errors="replace") if path.exists() else ""


def main():
    solve = read("log.simpleFoam")
    check = read("log.checkMesh")
    converged = re.search(r"SIMPLE solution converged in (\d+) iterations", solve)
    times = re.findall(r"^Time = (\d+)", solve, re.M)
    cells = re.search(r"^\s*cells:\s+(\d+)", check, re.M)

    results = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": os.environ.get("GITHUB_SHA"),
        "run_url": os.environ.get("RUN_URL"),
        "mesh_cells": int(cells.group(1)) if cells else None,
        "mesh_ok": "Mesh OK." in check if check else None,
        "iterations": int(times[-1]) if times else None,
        "converged": bool(converged),
        "coefficients": last_force_coeffs(),
    }
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

'\nWrite MeshLab ALN alignment files. Each entry stores a model filename and its 4x4 model-to-scene rigid transformation with scale.\n'

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def write_aln(path, entries: List[Tuple[str, np.ndarray]]):
    """entries: list of (model_filename, 4x4 model-to-scene matrix)."""
    lines = [str(len(entries))]
    for name, M in entries:
        lines.append(name)
        lines.append("#")
        for r in range(4):
            lines.append(" ".join(f"{M[r, c]:.6f}" for c in range(4)))
    lines.append("0")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_aln(path):
    'Tolerant ALN reader returning (name, 4x4 matrix) pairs.'
    raw = [ln.strip() for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()]
    raw = [ln for ln in raw if ln != ""]
    out = []
    i = 0
    if raw and raw[0].isdigit():
        i = 1
    while i < len(raw):
        if raw[i] == "0":
            break
        name = raw[i]; i += 1
        while i < len(raw) and raw[i].startswith("#"):
            i += 1
        nums = []
        while i < len(raw) and len(nums) < 16:
            toks = raw[i].split()
            try:
                nums.extend(float(t) for t in toks)
            except ValueError:
                break
            i += 1
        if len(nums) >= 16:
            out.append((name, np.array(nums[:16]).reshape(4, 4)))
    return out

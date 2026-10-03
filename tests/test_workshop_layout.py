"""Source-layout checks for the image-matching workshop notebook (no overlong cell lines)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Image_Matching_Workshop.ipynb"


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def test_no_workshop_cell_line_exceeds_2000_characters():
    for cell in _cells():
        longest = max(len(line) for line in "".join(cell["source"]).split("\n"))
        assert longest <= 2000, f"{cell.get('id')}: line of {longest} characters"


def test_split_literals_are_plain_strings():
    found = {}
    for cell in _cells():
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in {
                "PHOTO_MANIFEST_B64",
                "MATCH_RUNNER",
            }:
                found[node.targets[0].id] = ast.literal_eval(node.value)
    assert set(found) == {"PHOTO_MANIFEST_B64", "MATCH_RUNNER"}
    assert all(isinstance(value, str) and value for value in found.values())
    assert "\n" not in found["PHOTO_MANIFEST_B64"]

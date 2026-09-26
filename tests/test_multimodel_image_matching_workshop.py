# ruff: noqa: E501,I001
"""Static contract tests for the comparative image-matching workshop."""
from __future__ import annotations
import ast
import base64
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

from lightglue_pipeline import samples

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Matching_Workshop.ipynb"

PINNED_ASSETS = {
    "aliked_lightglue.safetensors": "9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6",
    "aliked-n16.safetensors": "3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a",
    "xoftr_640.safetensors": "4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69",
}


def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells():
    return ["".join(cell["source"]) for cell in load()["cells"] if cell["cell_type"] == "code"]


def body():
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def as_python(cell):
    return "\n".join(("pass #" + line) if line.lstrip().startswith(("!", "%")) else line for line in cell.splitlines())


def embedded_rows():
    text = body()
    payload = re.search(r'PHOTO_MANIFEST_B64 = "([^"]+)"', text).group(1)
    digest = re.search(r'PHOTO_MANIFEST_SHA256 = "([0-9a-f]{64})"', text).group(1)
    raw = gzip.decompress(base64.b64decode(payload))
    assert hashlib.sha256(raw).hexdigest() == digest
    return json.loads(raw)


def notebook_function(name):
    for cell in code_cells():
        for node in ast.parse(as_python(cell)).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                namespace = {"random": random}
                exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), namespace)
                return namespace[name]
    raise AssertionError(f"{name} not defined in the notebook")


def test_generator_parity():
    subprocess.run([sys.executable, str(REPO / "tools" / "build_multimodel_image_matching_workshop.py"), "--check"], cwd=REPO, check=True)


def test_metadata():
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "TASK-INFERENCE"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["clean_runtime_evidence"] == "pending"
    assert meta["dimer_checkpoint_parity"] == "qualification-required"


def test_code_cells_compile():
    for cell in code_cells():
        compile(as_python(cell), "cell", "exec")


def test_models_tiers_and_evaluation_present():
    text = body()
    for literal in [
        "cvg/LightGlue", "vismatch/xoftr", "vismatch==1.3.2",
        '"easy"', '"moderate"', '"rotation_stress"',
        "USE_BYOD = False",
        "precision_3px", "homography_acc_3px", "homography_acc_5px",
        "frozen_experiment.json", "experiment_manifest.json", "workshop_summary.json",
        "symlink ZIP member refused", "not-measurable",
    ]:
        assert literal in text
    for name, digest in PINNED_ASSETS.items():
        assert name in text and digest in text


def test_embedded_manifest_matches_the_repository_corpus():
    assert [tuple(row) for row in embedded_rows()] == list(samples.SAMPLE_RECORDS)


def test_photo_split_matches_the_repository_split(monkeypatch):
    # The repository draws photos, then renders its own pairs; compare only the photo draw.
    monkeypatch.setattr(samples, "make_pairs", lambda records, seed: [dict(r) for r in records])
    rows = embedded_rows()
    split = notebook_function("builtin_split")([{"source_id": row[0], "species": row[1]} for row in rows], seed=samples.SAMPLE_SEED)
    expected = samples.build_sample_dataset([{"id": row[0], "species": row[1]} for row in samples.SAMPLE_RECORDS])
    for name in ("train", "validation", "test"):
        assert [r["source_id"] for r in split[name]] == [p["source_id"] for p in expected[name]]
    assert {name: len(split[name]) for name in split} == {"train": 216, "validation": 48, "test": 96}


def test_no_runtime_repo_dependency():
    text = body()
    for forbidden in ["git clone ", "pip install -e", "raw.githubusercontent.com/kurtvalcorza", "dimer-backend"]:
        assert forbidden not in text


def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []

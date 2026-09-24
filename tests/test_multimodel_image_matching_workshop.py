# ruff: noqa: E501,I001
"""Static contract tests for the comparative image-matching workshop."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
NOTEBOOK=REPO/"tutorials"/"DIMER_MultiModel_Image_Matching_Workshop.ipynb"

def load():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def body():
    return "\n".join("".join(cell.get("source",[])) for cell in load()["cells"])

def test_generator_parity():
    subprocess.run([sys.executable,str(REPO/"tools"/"build_multimodel_image_matching_workshop.py"),"--check"],cwd=REPO,check=True)

def test_metadata():
    meta=load()["metadata"]["dimer"]
    assert meta["notebook_spec"]=="2.1"
    assert meta["notebook_profile"]=="TASK-INFERENCE"
    assert meta["notebook_mode"]=="WORKSHOP"
    assert meta["standalone"] is True
    assert meta["clean_runtime_evidence"]=="pending"

def test_models_tiers_and_byod():
    text=body()
    for literal in [
        "cvg/LightGlue","vismatch/xoftr",
        "extreme_rotation","standard_hard",
        'USE_BYOD = False  # @param','BYOD_ZIP_PATH = ""  # @param',
        "precision_3px","homography_acc_3px",
        "multimodel_image_matching_provenance.json",
    ]:
        assert literal in text

def test_embedded_source_is_pinned_and_no_runtime_clone():
    text=body()
    assert "86996ba0671c8cdcf1c922d95d9f53ca156f3130" in text
    assert "bb795eb064bf12742beb2e842e2a1aa7dc4093d8" in text
    for forbidden in ["git clone ","pip install -e","dimer-backend"]:
        assert forbidden not in text

def test_clean_notebook():
    for cell in load()["cells"]:
        if cell["cell_type"]=="code":
            assert cell["execution_count"] is None
            assert cell["outputs"]==[]

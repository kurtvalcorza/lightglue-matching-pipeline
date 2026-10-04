# ruff: noqa: E501
"""The image-matching workshop builds its matcher environment the fleet-standard uv way (2026-10-04).

Nothing is installed into the notebook kernel: a pinned uv wheel (size + SHA-256) builds a managed
CPython 3.12.12 venv from a carried hash lock, installed with --require-hashes --only-binary :all:,
and every matcher stage runs with that venv's interpreter. Static checks only (CI has no network).
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import build_multimodel_image_matching_workshop as gen  # noqa: E402

NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Matching_Workshop.ipynb"
LOCK = REPO / "tools" / "multimodel-image-matching-workshop-requirements.lock"
REQUIREMENTS_IN = REPO / "tools" / "multimodel-image-matching-workshop-requirements.in"
DIRECT_PINS = ["vismatch==1.3.2", "torch==2.14.0", "torchvision==0.29.0"]


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def cell(cell_id):
    hits = ["".join(c["source"]) for c in cells() if c.get("id") == cell_id]
    assert len(hits) == 1, cell_id
    return hits[0]


def code():
    return ["".join(c["source"]) for c in cells() if c["cell_type"] == "code"]


def test_nothing_is_installed_into_the_kernel():
    env_id = "dimer-matching-workshop-24"
    for c in cells():
        if c["cell_type"] != "code" or c.get("id") == "mat-uv-lock":  # the lock cell is data (its header quotes the uv command)
            continue
        source = "".join(c["source"])
        assert "pip install" not in source and '"-m","pip"' not in source and "%pip" not in source and "!pip" not in source
        assert source.count('"install"') == (1 if c.get("id") == env_id else 0), c.get("id")
        assert not re.search(r"restart (the )?(session|runtime|kernel)", source, re.I), c.get("id")
    env = cell(env_id)
    assert "sys.executable" not in env and "ensure_uv" not in env
    assert '[UV,"pip","install","--python",ENV_PYTHON,' in env


def test_pinned_uv_wheel_is_size_and_sha256_checked():
    env = cell("dimer-matching-workshop-24")
    assert 'UV_VERSION="0.12.15"' in env and "UV_BYTES=20081404" in env
    assert 'UV_SHA256="aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60"' in env
    assert "uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" in env
    assert "len(wheel)!=UV_BYTES or hashlib.sha256(wheel).hexdigest()!=UV_SHA256" in env


def test_managed_python_and_hash_locked_wheels_only():
    env = cell("dimer-matching-workshop-24")
    assert 'MODEL_ENV_PYTHON="3.12.12"' in env
    assert '[UV,"venv","--managed-python","--python",MODEL_ENV_PYTHON,ENV_ROOT]' in env
    assert '"--require-hashes","--only-binary",":all:"' in env
    assert '"-r",LOCK_PATH' in env
    assert 'platform.system()!="Linux" or platform.machine()!="x86_64"' in env


def test_child_environment_is_clean():
    env = cell("dimer-matching-workshop-24")
    assert 'for name in ("PYTHONPATH","PYTHONHOME","PYTHONSTARTUP"): CHILD_ENV.pop(name,None)' in env
    assert 'CHILD_ENV["MPLBACKEND"]="Agg"' in env
    assert "env=CHILD_ENV" in env


def test_every_matcher_stage_runs_on_the_venv_python():
    stages = [s for s in code() if "run_checked([ENV_PYTHON,runner_path" in s]
    assert len(stages) == 3  # validation, test, unlabelled pair
    for source in code():
        for call in re.findall(r"run_checked\(\[([^,\]]+)", source):
            assert call in {"ENV_PYTHON", "UV"}, call


def test_lock_has_hashes_and_keeps_the_direct_pins():
    text = LOCK.read_text(encoding="utf-8")
    assert "\r" not in text
    blocks = re.split(r"\n(?=[A-Za-z])", text)
    pinned = [b for b in blocks if re.match(r"[A-Za-z0-9]", b)]
    assert len(pinned) >= 100
    for block in pinned:
        assert re.match(r"[A-Za-z0-9._-]+==\S+ \\\n", block), block[:60]
        assert "--hash=sha256:" in block, block[:60]
    for pin in DIRECT_PINS:
        assert re.search(rf"^{re.escape(pin)} \\$", text, re.M), pin
    assert "--python-platform x86_64-manylinux_2_28" in text and "--generate-hashes" in text
    assert [line for line in REQUIREMENTS_IN.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")] == DIRECT_PINS
    env = cell("dimer-matching-workshop-24")
    assert 'PINS=["vismatch==1.3.2","torch==2.14.0","torchvision==0.29.0"]' in env


def test_carried_lock_equals_the_lock_file_and_its_digest():
    source = cell("mat-uv-lock")
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in {"MATCH_ENV_LOCK", "MATCH_ENV_LOCK_SHA256"}:
            values[node.targets[0].id] = ast.literal_eval(node.value)
    text = LOCK.read_bytes().decode("utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert values == {"MATCH_ENV_LOCK": text, "MATCH_ENV_LOCK_SHA256": digest}
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["carried_files"]["requirements.lock.txt"] == {"source": "tools/multimodel-image-matching-workshop-requirements.lock", "sha256": digest}
    ids = [c.get("id") for c in cells()]
    assert ids.index("mat-uv-lock") + 1 == ids.index("dimer-matching-workshop-24")


def test_lock_pieces_are_short_and_rejoin_exactly():
    text = LOCK.read_text(encoding="utf-8")
    parts = gen.pieces(text)
    assert "".join(parts) == text and max(len(p) for p in parts) <= gen.PIECE_LIMIT
    assert gen.pieces("a" * 2500 + "\nb\n", 1000) == ["a" * 1000, "a" * 1000, "a" * 500 + "\nb\n"]


def test_uv_revision_is_logged_in_metadata():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    latest = meta["review_revisions"][-1]
    assert latest["date"] == "2026-10-04" and "uv 0.12.15" in latest["change"] and "--require-hashes" in latest["change"]
    assert meta["clean_runtime_evidence"] == "pending" and meta["release_status"] == "candidate"

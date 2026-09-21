# ruff: noqa: E501
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from lightglue_pipeline import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    build_provenance,
    write_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_SPEC = ROOT / "examples" / "sample-data"


def _load_generator():
    path = SAMPLE_SPEC / "generate_samples.py"
    spec = importlib.util.spec_from_file_location("lightglue_sample_generator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_parity_script_passes() -> None:
    subprocess.run([sys.executable, "scripts/check_lock.py"], cwd=ROOT, check=True)


def test_sample_generator_matches_manifest_and_pillow(tmp_path: Path) -> None:
    expected = {}
    for line in (SAMPLE_SPEC / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        expected[name] = digest

    generator = _load_generator()
    written = generator.generate(tmp_path)

    assert {path.name for path in written} == set(expected)
    for path in written:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == (256, 192)


def test_provenance_identity_and_semantics() -> None:
    record = build_provenance(include_runtime=False)
    assert record["model"]["id"] == MODEL_ID
    assert record["model"]["revision"] == MODEL_REVISION
    assert record["model"]["weight_sha256"] == MODEL_SHA256
    assert record["preprocessing"]["rgb"] is True and record["preprocessing"]["side_range_px"] == [64, 1024]
    assert record["inference"]["match_confidence_semantics"].startswith("assignment_scores")
    assert record["model"]["vendored_code"]["commit"] == "eb42fee2d71449efb0aa5c10549752b5d75384d8"
    assert record["model"]["extractor"]["commit"] == "683d7c65197395c0b3f01ebe76e1084a27e73a65"
    assert record["model"]["revision_kind"] == "github-release-tag"
    assert record["adapter"] is None


def test_provenance_records_actual_pipeline_metadata(tmp_path: Path) -> None:
    fake_pipeline = SimpleNamespace(
        checkpoint_path=tmp_path / "custom_weights",
        checkpoint_source="explicit_path",
        manifest_verified=True,
        weight_sha256="fake_sha256_digest",
        weight_size_bytes=1234567,
        device="cuda:0",
    )

    out_file = tmp_path / "provenance.json"
    write_provenance(out_file, pipeline=fake_pipeline)
    data = json.loads(out_file.read_text(encoding="utf-8"))

    assert data["model"]["checkpoint_source"] == "explicit_path"
    assert data["model"]["checkpoint_path"] == str(tmp_path / "custom_weights")
    assert data["model"]["manifest_verified"] is True
    assert data["model"]["weight_sha256"] == "fake_sha256_digest"
    assert data["model"]["weight_size_bytes"] == 1234567
    assert data["inference"]["device"] == "cuda:0"


def test_colab_notebook_is_json_and_python_cells_compile() -> None:
    path = ROOT / "tutorials" / "lightglue_matching_colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    for index, cell in enumerate(code_cells):
        source = "".join(cell.get("source", []))
        compile(source, f"{path}#cell-{index}", "exec")

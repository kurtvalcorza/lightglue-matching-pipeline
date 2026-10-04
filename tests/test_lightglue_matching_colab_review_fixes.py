"""Regression tests for the Notebook Review Framework v1 findings on `tutorials/lightglue_matching_colab.ipynb`
(review PR #11: LGC-M1..M3, LGC-m1..m7).

The notebook's own cells are executed from the committed JSON in a namespace of the package's public API and inert
stand-ins (a stub pipeline, a fake `google.colab`, small PIL images). Model-level checks use the randomly
initialised vendored networks (`build_models`), never the pinned checkpoints, so the file runs under CI's install line.
"""
# ruff: noqa: E501  -- assertion messages and cell sources are kept on one line

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

# Windows conda trap (fleet note, bioclip2 row 6): import torch before any NumPy linear algebra in this process.
with contextlib.suppress(ImportError):
    import torch

# The carried network needs torchvision (deform_conv2d); CI installs it, a torch-only env skips this file.
pytest.importorskip("torchvision")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import lightglue_pipeline as lg  # noqa: E402
from conftest import textured_image  # noqa: E402
from lightglue_pipeline import (  # noqa: E402
    LightGluePipeline,
    build_models,
    byod_record_limits,
    distinct_images,
    load_byod_dataset,
    make_pairs,
    split_dataset,
    split_minimums,
    upscaling_report,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "lightglue_matching_colab.ipynb"
METRICS = ("precision_3px", "precision_1px", "matches_per_pair", "inliers_per_pair", "median_error_px", "homography_acc_3px", "homography_acc_5px")


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _code_after(heading: str) -> str:
    cells = _cells()
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _source(cell):
            for nxt in cells[i + 1 :]:
                if nxt["cell_type"] == "code":
                    return _source(nxt)
    raise AssertionError(f"no code cell after {heading!r}")


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def _photos(root: Path, n: int, size=(160, 120)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        textured_image(i, size).save(root / f"p{i:03d}.png")
    return root


def _zip(tmp_path: Path, name: str, n: int, size=(160, 120)) -> Path:
    folder = _photos(tmp_path / f"{name}_src", n, size)
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(folder.iterdir()):
            archive.write(file, f"photos/{file.name}")
    return path


def _namespace() -> dict:
    ns: dict = {name: getattr(lg, name) for name in lg.__all__}
    ns.update({"os": __import__("os"), "Path": Path, "json": json})
    return ns


def _fake_colab(monkeypatch, uploads: list[dict]):
    queue = list(uploads)
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: queue.pop(0)
    colab = types.ModuleType("google.colab")
    colab.files = files
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _section4(monkeypatch, tmp_path, *, byod_path: str = "") -> dict:
    """Execute Section 4 verbatim with USE_BYOD = True (form literals substituted) in a namespace of the package API."""
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. iNaturalist photographs, homography pairs and split")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {byod_path!r}", 1)
    ns = _namespace()
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _pipe() -> LightGluePipeline:
    torch.manual_seed(0)
    extractor, matcher = build_models()
    return LightGluePipeline(extractor, matcher, device="cpu")


# --- LGC-m1: the BYOD minimum, refusals that name the split, upscaling, BYOD_PATH ----------------------------------


def test_byod_minimum_is_twelve_and_matches_the_prose():
    assert split_minimums() == {"train": 4, "validation": 4, "test": 4}
    assert byod_record_limits() == (12, 7693)
    markdown = _markdown()
    assert "**at least 12 distinct photographs**" in markdown and "at least eight" not in markdown


def test_twelve_photographs_split_four_four_four_and_eleven_are_refused_naming_the_split():
    images = [{"id": f"img{i}", "image": textured_image(i, (160, 120))} for i in range(12)]
    splits = split_dataset(images, seed=42)
    assert {k: len(v) for k, v in splits.items()} == {"test": 4, "validation": 4, "train": 4}
    with pytest.raises(ValueError) as refused:
        split_dataset(images[:11], seed=42)
    message = str(refused.value)
    assert "the train split would hold 3 photographs (at least 4 are required)" in message
    assert "3/4/4" in message and "needs 12..7693 distinct photographs" in message and "Add photographs" in message


def test_section4_runs_twelve_through_byod_path_and_refuses_eleven(monkeypatch, tmp_path, capsys):
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "twelve", 12)))
    assert ns["disjoint"] == {"test": 4, "validation": 4, "train": 4} and ns["byod"]["distinct"] == 12
    out = capsys.readouterr().out
    assert "'byod_minimum_distinct_photographs': 12" in out and "'needed': '12..7693 distinct photographs'" in out
    with pytest.raises(ValueError, match="the train split would hold 3 photographs"):
        _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "eleven", 11)))


def test_section4_reads_a_folder_and_reports_duplicates_and_upscaling(monkeypatch, tmp_path, capsys):
    folder = _photos(tmp_path / "folder", 12, size=(48, 36))
    textured_image(0, (48, 36)).save(folder / "zz_copy.png")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(folder))
    out = capsys.readouterr().out
    assert ns["byod"]["duplicates_dropped"] == ["zz_copy"] and "'pixel_duplicates_dropped': 1" in out
    upscaling = ns["byod"]["upscaling"]
    assert upscaling["upscaled"] == 12 and len(upscaling["under_min_side"]) == 12 and upscaling["largest_factor"] == round(640 / 48, 1)
    assert "'upscaled_to_working_size': 12" in out and "measure the interpolation" in out


def test_upscaling_report_and_original_size():
    records = [{"id": "big", "image": Image.new("RGB", (800, 600)), "original_size": [800, 600]}, {"id": "small", "image": Image.new("RGB", (320, 240)), "original_size": [320, 240]}, {"id": "tiny", "image": Image.new("RGB", (40, 30)), "original_size": [40, 30]}, {"id": "sample", "image": Image.new("RGB", (40, 30))}]
    report = upscaling_report(records)
    assert report["upscaled"] == 2 and report["under_min_side"] == ["tiny"] and report["largest_factor"] == 16.0


def test_byod_records_carry_their_original_size(tmp_path):
    records = load_byod_dataset(_zip(tmp_path, "sizes", 2, size=(90, 70)))
    assert [r["original_size"] for r in records] == [[90, 70], [90, 70]]
    unique, dropped = distinct_images([*records, {**records[0], "id": "again"}])
    assert len(unique) == 2 and dropped == ["again"]


def test_cancelled_upload_no_colab_and_missing_path_are_actionable(monkeypatch, tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist in this runtime"):
        _section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(RuntimeError, match="set BYOD_PATH to its path"):
        _section4(monkeypatch, tmp_path)
    _fake_colab(monkeypatch, [{}])
    with pytest.raises(ValueError, match="Upload exactly one .zip file"):
        _section4(monkeypatch, tmp_path)


def test_an_uploaded_zip_goes_through_section4(monkeypatch, tmp_path):
    payload = _zip(tmp_path, "upload", 12).read_bytes()
    _fake_colab(monkeypatch, [{"mine.zip": payload}])
    ns = _section4(monkeypatch, tmp_path)
    assert ns["data_source"] == "BYOD (mine.zip)" and ns["byod"]["zip_sha256"] and ns["disjoint"]["train"] == 4


# --- LGC-M2 / LGC-M3: training starts from the pinned base -----------------------------------------------------


def test_adapt_refuses_an_already_adapted_pipeline_and_records_its_start(tmp_path):
    pipe = _pipe()
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (128, 96))} for i in range(4)], tier="easy")
    result = pipe.adapt(records, None, epochs=1, lr=1e-4, batch_size=2, trainable_layers=1)
    assert result["started_from"] == "pinned base (no earlier adaptation)"
    manifest = json.loads((pipe.save_artifact(tmp_path / "a") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"]["started_from"].startswith("pinned base")
    before = {k: v.clone() for k, v in pipe.matcher.state_dict().items()}
    with pytest.raises(ValueError, match="already holds an adapted matcher"):
        pipe.adapt(records, None, epochs=1)
    assert all(torch.equal(v, before[k]) for k, v in pipe.matcher.state_dict().items())


def test_sections_5_to_7_reset_and_section_6_refuses_an_adapted_matcher():
    for heading in ("## 5. Match through the inference contract", "## 6. Baselines and the frozen matcher", "## 7. Bounded fine-tuning"):
        source = _code_after(heading)
        reset = source.index("\nreset_to_pretrained()\n")
        first_use = min(source.index(m) for m in ("pipe.match(", "pipe.evaluate", "pipe.adapt(") if m in source)
        assert reset < first_use, heading
    assert "if frozen_test['adapted']:" in _code_after("## 6. Baselines and the frozen matcher")


def test_reset_to_pretrained_reloads_only_an_adapted_pipeline():
    source = _code_after("## 5. Match through the inference contract")
    helper = source[source.index("def reset_to_pretrained():") : source.index("\nreset_to_pretrained()\n")]
    loads = []

    class Loader:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return types.SimpleNamespace(adapter=None)

    ns = {"gc": __import__("gc"), "LightGluePipeline": Loader, "WEIGHTS_DIR": "w", "pipe": types.SimpleNamespace(adapter=None)}
    exec(compile(helper, "<reset>", "exec"), ns)
    ns["reset_to_pretrained"]()
    assert loads == []
    ns["pipe"] = types.SimpleNamespace(adapter={"best_epoch": 2})
    ns["reset_to_pretrained"]()
    assert loads == ["w"] and ns["pipe"].adapter is None


def test_byod_and_experiments_name_their_scope():
    markdown = _markdown()
    assert "re-run from that cell" not in markdown and "**Runtime → Run after**" in markdown
    assert "**Every pass starts from the pinned base.**" in markdown
    assert "select Section 7 and choose **Run after**" in markdown


# --- LGC-m6: the off-by-default experiment on a fresh base --------------------------------------------------------


def test_experiment_cell_is_off_by_default_and_trains_a_fresh_base(capsys):
    source = _code_after("## 10. Your turn")
    assert re.search(r"^RUN_EXPERIMENT = False  # @param", source, re.M)
    assert "experiment_pipe = LightGluePipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in source and "pipe.adapt(" not in source.replace("experiment_pipe.adapt(", "")
    exec(compile(source, "<section 10>", "exec"), {})
    assert "not run (RUN_EXPERIMENT = False)" in capsys.readouterr().out


# --- LGC-m5: outcomes recorded, never asserted ---------------------------------------------------------------------


def _result(precision: float, *, adapted: bool = False) -> dict:
    tier = {m: precision for m in METRICS}
    return {**{m: precision for m in METRICS}, "by_tier": {"easy": dict(tier), "hard": dict(tier)}, "per_pair": [], "adapted": adapted, "n": 4}


def test_section8_reports_a_drop_and_still_writes_the_report(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    stub = types.SimpleNamespace(evaluate=lambda records, progress=None, stage="model": _result(0.4, adapted=True))
    ns = _namespace()
    ns.update({"np": np, "pipe": stub, "METRICS": METRICS, "short": lambda r: {k: r[k] for k in METRICS}, "show_progress": lambda e: None, "test_records": [], "val_records": [], "baselines": {n: _result(0.1) for n in ("identity", "patch_neighbour", "descriptor_nn")}, "frozen_test": _result(0.6), "frozen_beats": {"identity": True}, "data_source": "BYOD (x.zip)", "byod": {"file": "x.zip"}, "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {"test": 4}, "adapt_seconds": 1.0, "adapt_result": {"best_epoch": 1, "trainable_layers": 2, "epochs": 3, "lr": 1e-4, "history": [{"epoch": 0, "val": {"precision_3px": 0.5}}], "trainable_names": []}})
    exec(compile(_code_after("## 8. Held-out evaluation"), "<section 8>", "exec"), ns)
    assert ns["outcomes"]["adapted_beats_frozen"] is False and ns["outcomes"]["precision_3px"] == "down"
    assert "not above the frozen one" in capsys.readouterr().out
    report = json.loads((tmp_path / "outputs" / "lightglue_matching_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["outcomes"]["precision_3px_by_tier"] == {"easy": "down", "hard": "down"} and len(report["run_history"]) == 1


def test_no_learner_cell_asserts_a_result():
    for cell in _cells():
        source = _source(cell)
        if cell["cell_type"] != "code" or "dimer" in cell.get("metadata", {}) or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:120]


# --- LGC-m7: progress for the long CPU stages ----------------------------------------------------------------------


def test_evaluate_and_baselines_report_progress_per_pair():
    pipe = _pipe()
    records = make_pairs([{"id": f"r{i}", "image": textured_image(i, (128, 96))} for i in range(3)], tier="easy")
    events = []
    pipe.evaluate(records, progress=events.append, stage="frozen model")
    assert [(e["stage"], e["done"], e["of"]) for e in events] == [("frozen model", 1, 3), ("frozen model", 2, 3), ("frozen model", 3, 3)]
    events.clear()
    pipe.evaluate_baselines(records, progress=events.append)
    assert [e["stage"] for e in events if e["done"] == e["of"]] == ["identity", "patch_neighbour", "descriptor_nn"]


def test_show_progress_prints_at_most_every_30_seconds_and_at_the_end(capsys):
    source = _code_after("## 5. Match through the inference contract")
    helper = source[source.index("PROGRESS_EVERY_S = 30") : source.index("def reset_to_pretrained():")]
    ns: dict = {}
    exec(compile(helper, "<progress>", "exec"), ns)
    for done, seconds in ((1, 5.0), (2, 20.0), (3, 31.0), (4, 40.0), (5, 70.0), (6, 72.0)):
        ns["show_progress"]({"stage": "patch_neighbour", "done": done, "of": 6, "seconds": seconds})
    lines = [line for line in capsys.readouterr().out.splitlines() if "progress" in line]
    assert len(lines) == 3 and "'3/6'" in lines[0] and "'5/6'" in lines[1] and "'6/6'" in lines[2]


# --- LGC-m2 / LGC-m3 / LGC-m4: text -------------------------------------------------------------------------------


def test_identity_and_access_text_is_per_source():
    markdown = _markdown()
    assert "v0.1_arxiv…" not in markdown and "No GitHub access" not in markdown
    assert "(`Shiaoming/ALIKED`, the extractor) at the immutable release tag" not in markdown
    assert "the two checkpoints are fetched from GitHub" in markdown
    build = _load_build()
    assert build.short_revision("v0.1_arxiv") == "v0.1_arxiv" and build.short_revision("a" * 40) == "a" * 12 + "…"


def test_worked_answers_quote_the_recorded_t4_run_and_state_the_direction_difference():
    markdown = _markdown()
    assert "**Run-to-run spread.**" in markdown and "**0.768 on the recorded Kaggle Tesla T4 run**" in markdown
    assert "0.708 → 0.698 (down) on the CPU and 0.708 → 0.719 (up) on the T4" in markdown
    assert "rose 0.750 → 0.768 (+0.018)" in markdown and "0.500 → 0.538" in markdown
    assert "the build record measured 0.750 → 0.765" not in markdown


def test_runtime_statement_names_the_isolated_python_and_spec():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    markdown = _markdown()
    assert "Section 1 builds its own **Python 3.12.12** environment" in markdown and "Python 3.13" in markdown


# --- LGC-M1: isolated runtime, lock, status ---------------------------------------------------------------------


def test_exactly_two_kernel_cells_and_no_restart_text():
    kernel = [c for c in _cells() if c["cell_type"] == "code" and "# dimer: kernel cell" in _source(c)]
    assert len(kernel) == 2
    install = _source(kernel[0])
    assert "--require-hashes" in install and "--managed-python" in install and "LOCK_SHA256" in install and "MANAGED_PYTHON = '3.12.12'" in install
    markdown = _markdown()
    assert "Restart the runtime" not in markdown and "its restart" not in markdown
    status = (ROOT / "STATUS.md").read_text(encoding="utf-8")
    assert "Current status: **Candidate**" in status
    verification = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "**Completed in two passes, not promotion evidence**" in verification and "is expected where the runtime" not in verification


def test_lock_pins_every_direct_dependency_with_hashes():
    build = _load_build()
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT), lock)
    assert len(build.lock_packages(lock)) == 33 and "transformers" not in build.lock_packages(lock)


def _load_build():
    spec = importlib.util.spec_from_file_location("build_notebook_under_test", ROOT / "tools" / "build_notebook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: a library calling importlib.util.find_spec("google.colab") raised on a spec-less stub."""
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)
            assert spec is not None and spec.name == name
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before


def test_guided_layer_and_infrastructure_labels():
    markdown = _markdown()
    for marker, least in (("**Predict before running:**", 7), ("**What to notice:**", 7), ("<summary>Check your reasoning</summary>", 7)):
        assert markdown.count(marker) >= least, marker
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "**Predict → Change one thing → Run → Observe → Explain**"):
        assert marker in markdown
    setup = [c for c in _cells() if c["cell_type"] == "code"][:11]
    assert all(c["metadata"].get("cellView") == "form" and _source(c).startswith("# @title Infrastructure: ") for c in setup)
    section6 = _code_after("## 6. Baselines and the frozen matcher")
    assert "figure.save('outputs/lightglue_matching_correspondences.png')" in section6

# ruff: noqa: E501
"""Regression tests for the 2026-10-02 notebook review findings (MAT-M1..M3, MAT-m1..m6).

They need only NumPy and Pillow (CI's lock has no pandas or matplotlib): notebook cells are executed
from the generator source with inert stand-ins for plotting and tables.
"""
from __future__ import annotations

import ast
import io
import json
import math
import re
import sys
import types
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from multimodel_image_matching_workshop_source import CELLS  # noqa: E402

NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Matching_Workshop.ipynb"
TIERS = ("easy", "moderate", "rotation_stress")


def nb_cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def src(cell_id):
    for cell in CELLS:
        if cell.get("id") == cell_id:
            return cell["source"]
    raise AssertionError(cell_id)


def by_title(title):
    hits = [c["source"] for c in CELLS if c["kind"] == "code" and c["source"].startswith(f"# @title {title}")]
    assert len(hits) == 1, title
    return hits[0]


def markdown():
    return [c["source"] for c in CELLS if c["kind"] == "markdown"]


def functions(source, *names, ns=None):
    ns = {} if ns is None else ns
    module = ast.parse(source)
    module.body = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom)) and (not isinstance(n, ast.FunctionDef) or not names or n.name in names)]
    exec(compile(module, "cell", "exec"), ns)
    return ns


def metrics_ns():
    ns = {"np": np, "math": math}
    exec(compile(by_title("Common geometry metrics"), "metrics", "exec"), ns)
    return ns


def fake_matplotlib(calls):
    pyplot = types.ModuleType("matplotlib.pyplot")

    def subplots(nrows=1, ncols=1, figsize=None, squeeze=True):
        calls.append({"nrows": nrows, "ncols": ncols, "figsize": figsize, "squeeze": squeeze})
        axes = np.empty((nrows, ncols), dtype=object)
        for index in np.ndindex(axes.shape):
            axes[index] = mock.MagicMock()
        if squeeze:
            return mock.MagicMock(), axes[0, 0] if axes.size == 1 else axes.squeeze()
        return mock.MagicMock(), axes

    pyplot.subplots = subplots
    for name in ("tight_layout", "savefig", "show", "close"):
        setattr(pyplot, name, lambda *a, **k: None)
    package = types.ModuleType("matplotlib")
    package.pyplot = pyplot
    return {"matplotlib": package, "matplotlib.pyplot": pyplot}


def pair_files(tmp_path, tiers):
    rows = []
    for i, tier in enumerate(tiers):
        p0, p1 = tmp_path / f"p{i}_0.png", tmp_path / f"p{i}_1.png"
        Image.new("RGB", (64, 48), (i * 30, 10, 10)).save(p0)
        Image.new("RGB", (64, 48), (10, i * 30, 10)).save(p1)
        rows.append({"pair_id": f"validation-{i:03d}", "split": "validation", "tier": tier, "image0": str(p0), "image1": str(p1),
                     "homography": np.eye(3).tolist(), "width": 64, "height": 48})
    return rows


# ------------------------------------------------------------------ cell identity
def test_original_cell_ids_are_stable_and_new_cells_are_named():
    ids = [c["id"] for c in nb_cells()]
    original = [i for i in ids if i.startswith("dimer-matching-workshop-")]
    assert sorted(original) == [f"dimer-matching-workshop-{n:02d}" for n in range(58)]
    assert "dimer-matching-workshop-58" not in ids
    assert ids[-1] == "dimer-matching-workshop-52"  # the one troubleshooting table closes the notebook
    assert {i for i in ids if not i.startswith("dimer-matching-workshop-")} == {
        "mat-notice-09", "mat-notice-12", "mat-notice-14", "mat-notice-15", "mat-notice-16", "mat-notice-18", "mat-activity", "mat-activity-answer",
        "mat-uv-lock"}  # 2026-10-04: generated carrier of the matcher environment's hash lock
    assert len(ids) == len(set(ids))


# ------------------------------------------------------------------ MAT-M1
def test_aggregate_metrics_reports_empty_tier_as_n_zero_with_undefined_metrics():
    ns = metrics_ns()
    row = {"n_matches": 3, "n_inliers": 2, "inlier_errors": [0.5, 1.0], "corner_error_px": 2.0,
           "precision_1px": 1 / 3, "precision_3px": 2 / 3, "precision_5px": 2 / 3}
    full = ns["aggregate_metrics"]([row])
    empty = ns["aggregate_metrics"]([])
    assert list(empty) == list(full)
    assert empty["n"] == 0
    assert all(math.isnan(v) for k, v in empty.items() if k != "n")


@pytest.mark.parametrize("tiers", [("easy",), ("easy", "moderate"), TIERS])
def test_tier_figure_skips_missing_tiers(tmp_path, monkeypatch, tiers):
    calls = []
    for name, module in fake_matplotlib(calls).items():
        monkeypatch.setitem(sys.modules, name, module)
    ns = {"pair_rows": pair_files(tmp_path, tiers), "TIERS": TIERS, "Image": Image, "OUTPUT_ROOT": tmp_path}
    (tmp_path / "figures").mkdir()
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(by_title("Visualize the exact three-tier pair bytes"), "tiers", "exec"), ns)
    assert calls[0]["nrows"] == len(tiers) and calls[0]["squeeze"] is False
    missing = [t for t in TIERS if t not in tiers]
    if missing:
        assert "No validation pair in tier(s)" in out.getvalue() and all(t in out.getvalue() for t in missing)
    else:
        assert calls[0]["figsize"] == (12, 13.0) and out.getvalue() == ""


def test_correspondence_plots_skip_missing_test_tiers(tmp_path, monkeypatch):
    calls = []
    for name, module in fake_matplotlib(calls).items():
        monkeypatch.setitem(sys.modules, name, module)
    ns = metrics_ns()
    ns.update({"plt": sys.modules["matplotlib.pyplot"], "Image": Image, "OUTPUT_ROOT": tmp_path, "TIERS": TIERS,
               "MODEL_KEYS": ["lightglue", "xoftr"], "MODEL_SPECS": {"lightglue": {"display_name": "A"}, "xoftr": {"display_name": "B"}},
               "test_pair_rows": pair_files(tmp_path, ("easy",)),
               "load_model_match": lambda root, pid: {"kpts0": np.zeros((1, 2)), "kpts1": np.zeros((1, 2)), "confidence": np.ones(1)}})
    (tmp_path / "figures").mkdir()
    out = io.StringIO()
    with redirect_stdout(out):
        exec(compile(by_title("Visualize first test pair from each tier for both models"), "plots", "exec"), ns)
    assert "No test pair in tier 'moderate'" in out.getvalue() and "rotation_stress" in out.getvalue()
    assert len(calls) == 2  # easy tier only, both models


def test_byod_prose_states_the_thirteen_photo_tier_condition():
    byod = next(m for m in markdown() if m.startswith("## 20. BYOD"))
    assert "fewer than 13 photos" in byod and "n = 0" in byod


# ------------------------------------------------------------------ MAT-M2
def test_metrics_are_defined_before_the_first_scored_table():
    section8 = next(m for m in markdown() if m.startswith("## 8. Common geometric evaluator"))
    for column in ("precision_3px", "matches_per_pair", "inliers_per_pair", "median_error_px", "homography_acc_3px", "corner_error_px"):
        assert f"`{column}" in section8, column
    assert re.search(r"homography_acc_3px.*corner", section8)
    section0 = next(m for m in markdown() if m.startswith("## 0. What image matching produces"))
    assert "**homography**" in section0 and "**DLT**" in section0 and "**RANSAC**" in section0


def test_what_to_notice_follows_each_principal_result():
    ids = [c["id"] for c in nb_cells()]
    after = {"mat-notice-09": "dimer-matching-workshop-22", "mat-notice-12": "dimer-matching-workshop-29",
             "mat-notice-14": "dimer-matching-workshop-35", "mat-notice-15": "dimer-matching-workshop-37",
             "mat-notice-16": "dimer-matching-workshop-39", "mat-notice-18": "dimer-matching-workshop-43"}
    for notice, result in after.items():
        assert ids.index(notice) == ids.index(result) + 1
        assert src(notice).startswith("> **What to notice.**")
    assert "Nothing was tuned" in src("dimer-matching-workshop-30")


def test_runtime_section_is_a_lesson_and_parity_is_a_table():
    section18 = src("dimer-matching-workshop-42")
    assert "Infrastructure" not in section18 and "Compute cost is part of the central question" in section18
    cell = src("dimer-matching-workshop-43")
    assert "[:5000]" not in cell and "parity_rows" in cell
    cell39 = src("dimer-matching-workshop-39")
    assert "LightGlue + ALIKED is most ahead" in cell39 and "XoFTR is most ahead" in cell39


# ------------------------------------------------------------------ MAT-M3
def test_canonical_bundle_excludes_byod_runs(tmp_path):
    ns = functions(by_title("Final provenance"), "write_report_bundle", ns={"zipfile": zipfile, "Path": Path})
    root = tmp_path / "outputs"
    (root / "test").mkdir(parents=True)
    (root / "test" / "aggregate_metrics.csv").write_text("method\n", encoding="utf-8")
    (root / "byod" / "run-abc" / "figures").mkdir(parents=True)
    (root / "byod" / "run-abc" / "figures" / "difficulty_tiers.png").write_bytes(b"user photo derivative")
    canonical = ns["write_report_bundle"](root, tmp_path / "canonical.zip", ("byod",))
    names = zipfile.ZipFile(canonical).namelist()
    assert "test/aggregate_metrics.csv" in names and not any(n.startswith("byod") for n in names)
    byod_run = ns["write_report_bundle"](root / "byod" / "run-abc", tmp_path / "byod.zip", ())
    assert [n for n in zipfile.ZipFile(byod_run).namelist() if not n.endswith("/")] == ["figures/difficulty_tiers.png"]
    cell = by_title("Final provenance")
    assert 'excluded=() if USE_BYOD else ("byod",)' in cell and "make_archive" not in cell


def test_byod_prose_describes_the_canonical_bundle():
    byod = next(m for m in markdown() if m.startswith("## 20. BYOD"))
    assert "canonical outputs remain separate" not in byod
    assert "the canonical report ZIP leaves the `byod/` folder out" in byod


# ------------------------------------------------------------------ MAT-m1
def loader(tmp_path):
    ns = functions(src("dimer-matching-workshop-11"), ns={"WORK_ROOT": tmp_path, "Path": Path})
    functions(src("dimer-matching-workshop-13"), ns=ns)
    return ns


def photos(root, n):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (32, 24), (i * 20 % 255, 7, 3)).save(root / f"photo_{i}.png")
    return root


def test_count_refusal_names_the_count(tmp_path):
    with pytest.raises(ValueError, match=r"found 3 in"):
        loader(tmp_path)["load_byod_photos"](photos(tmp_path / "three", 3))


def test_duplicate_refusal_names_both_files(tmp_path):
    root = photos(tmp_path / "dup", 5)
    with Image.open(root / "photo_0.png") as image:
        image.save(root / "copy.png", compress_level=0)
    with pytest.raises(ValueError, match=r"Duplicate decoded pixels: (copy|photo_0)\.png shows the same image as (photo_0|copy)\.png"):
        loader(tmp_path)["load_byod_photos"](root)


def test_tiny_refusal_names_file_and_size(tmp_path):
    root = photos(tmp_path / "tiny", 5)
    Image.new("RGB", (10, 40)).save(root / "tiny_one.png")
    with pytest.raises(ValueError, match=r"tiny_one\.png: BYOD image sides must be at least 16 pixels; got 10x40"):
        loader(tmp_path)["load_byod_photos"](root)


def test_non_zip_bytes_are_refused_before_extraction(tmp_path):
    fake = tmp_path / "uploaded-byod.zip"
    Image.new("RGB", (32, 32)).save(fake, format="PNG")
    with pytest.raises(ValueError, match="not a readable ZIP archive"):
        loader(tmp_path)["load_byod_photos"](fake)
    acquire = src("dimer-matching-workshop-11")
    assert 'if not name.lower().endswith(".zip"):' in acquire


def test_byod_contract_is_stated_with_the_controls():
    controls_md = src("dimer-matching-workshop-03")
    for fact in ("4–360 JPEG/PNG photographs", "16–4096 pixels", "1000 members", "2 GiB"):
        assert fact in controls_md
    byod = next(m for m in markdown() if m.startswith("## 20. BYOD"))
    assert "assumes your photographs are independent" in byod


# ------------------------------------------------------------------ MAT-m2
def run_controls(tmp_path, monkeypatch, **overrides):
    monkeypatch.chdir(tmp_path)
    source = src("dimer-matching-workshop-04")
    for name, value in overrides.items():
        source, n = re.subn(rf"^({name}\s*=\s*)(.*?)(\s*# @param.*)$", lambda m, v=value: m.group(1) + repr(v) + m.group(3), source, flags=re.M)
        assert n == 1, name
    ns = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(source, "controls", "exec"), ns)
    return ns


def test_byod_runs_share_the_matcher_environment(tmp_path, monkeypatch):
    default = run_controls(tmp_path, monkeypatch)
    byod = run_controls(tmp_path, monkeypatch, USE_BYOD=True)
    assert default["SHARED_WORK_ROOT"] == byod["SHARED_WORK_ROOT"] == Path("work")
    assert byod["WORK_ROOT"] != byod["SHARED_WORK_ROOT"]
    env_cell = src("dimer-matching-workshop-24")
    assert 'ENV_ROOT=SHARED_WORK_ROOT/"envs"/"vismatch";' in env_cell
    assigned = {t.id for node in ast.walk(ast.parse(env_cell)) if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    assert {"ENV_ROOT", "ENV_PYTHON", "PINS", "signature", "marker"} <= assigned


# ------------------------------------------------------------------ MAT-m3
def test_runtime_disclosure_and_cpu_warning():
    opening = src("dimer-matching-workshop-01")
    for fact in ("37.4 MiB", "47.6 MB", "44.4 MB", "0.17 s per pair", "0.29 s per pair", "2026-09-26", "an estimate"):
        assert fact in opening
    assert "Warning: No CUDA GPU was used" in src("dimer-matching-workshop-28")


# ------------------------------------------------------------------ MAT-m4
def test_learning_objectives_replace_template_text():
    opening = src("dimer-matching-workshop-01")
    assert "### Learning objectives" in opening
    objectives = opening.split("### Learning objectives", 1)[1].split("###", 1)[0]
    assert len(re.findall(r"^\d\. \*\*", objectives, re.M)) == 6
    for phrase in ("where applicable", "where the capability supports them", "Run the model or model comparison"):
        assert phrase not in opening


# ------------------------------------------------------------------ MAT-m5
def activity_ns(tmp_path, enabled, strict=1.0):
    rng = np.random.default_rng(0)
    rows, out = [], tmp_path / "outputs"
    for key in ("lightglue", "xoftr"):
        (out / "validation" / "matches" / key).mkdir(parents=True)
    for i, tier in enumerate(TIERS * 2):
        pid = f"validation-{i:03d}"
        rows.append({"pair_id": pid, "tier": tier, "homography": np.eye(3).tolist()})
        k0 = rng.uniform(0, 100, (40, 2))
        for key, shift in (("lightglue", 0.5), ("xoftr", 2.0)):  # every match 0.5 px vs 2 px from the reference
            np.savez_compressed(out / "validation" / "matches" / key / f"{pid}.npz", kpts0=k0, kpts1=k0 + [shift, 0.0], confidence=np.ones(40))
    ns = metrics_ns()
    tables = []
    ns.update({"RUN_TOLERANCE_ACTIVITY": enabled, "ACTIVITY_STRICT_TOLERANCE_PX": strict, "OUTPUT_ROOT": out, "TIERS": TIERS,
               "MODEL_KEYS": ["lightglue", "xoftr"], "MODEL_SPECS": {"lightglue": {"display_name": "LightGlue + ALIKED"}, "xoftr": {"display_name": "XoFTR"}},
               "validation_pair_rows": rows,
               "pd": types.SimpleNamespace(DataFrame=lambda r: tables.append(r) or types.SimpleNamespace(to_string=lambda index=False: json.dumps(r)))})
    loader_cell = by_title("Common validation scoring")
    functions(loader_cell, "load_model_match", ns=ns)
    ns["Path"] = Path
    return ns, tables, out


def snapshot(root):
    return {str(p): p.read_bytes() for p in sorted(Path(root).rglob("*")) if p.is_file()}


def test_activity_is_off_by_default_and_runs_without_writing(tmp_path):
    controls = src("dimer-matching-workshop-04")
    assert re.search(r"^RUN_TOLERANCE_ACTIVITY = False\s+# @param", controls, re.M)
    ns, tables, out = activity_ns(tmp_path, enabled=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        exec(compile(src("mat-activity"), "activity", "exec"), ns)
    assert "Activity not run" in buf.getvalue() and tables == []
    ns, tables, out = activity_ns(tmp_path / "on", enabled=True)
    before = snapshot(out)
    with redirect_stdout(io.StringIO()):
        exec(compile(src("mat-activity"), "activity", "exec"), ns)
    assert snapshot(out) == before
    rows = {(r["model"], r["tier"]): r for r in tables[0]}
    assert len(rows) == 6
    assert rows[("LightGlue + ALIKED", "easy")]["retained_fraction"] == 1.0
    assert rows[("XoFTR", "easy")]["precision_3px"] == 1.0 and rows[("XoFTR", "easy")]["retained_fraction"] == 0.0


@pytest.mark.parametrize("bad", [0, 3, 4.5, -1])
def test_activity_tolerance_is_bounded(tmp_path, monkeypatch, bad):
    with pytest.raises(ValueError, match="ACTIVITY_STRICT_TOLERANCE_PX"):
        run_controls(tmp_path, monkeypatch, ACTIVITY_STRICT_TOLERANCE_PX=bad)


def test_activity_prompt_does_not_give_the_answer_and_exercises_have_guidance():
    prompt = src("dimer-matching-workshop-53")
    assert "cannot increase the fraction" not in prompt and "**Predict.**" in prompt
    assert "<details>" in src("mat-activity-answer") and "<details>" in src("dimer-matching-workshop-49")


# ------------------------------------------------------------------ MAT-m6
def test_information_architecture():
    texts = markdown()
    assert sum(bool(re.search(r"^#+ (\d+\. )?Troubleshooting", m, re.M)) for m in texts) == 1
    numbered = [line for m in texts for line in m.splitlines() if re.match(r"#+ \d+\. ", line)]
    assert numbered and all(line.startswith("## ") for line in numbered)
    assert "Optional model-native adaptation" not in "\n".join(texts)
    troubleshooting = src("dimer-matching-workshop-52")
    assert "release/unload steps" not in troubleshooting and "n = 0" in troubleshooting


def test_review_revision_is_recorded_in_metadata():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    revision = [entry for entry in meta["review_revisions"] if "review" in entry][-1]
    assert revision["date"] == "2026-10-02" and revision["review"].endswith("DIMER_MultiModel_Image_Matching_Workshop_Review.md")
    assert meta["clean_runtime_evidence"] == "pending" and meta["release_status"] == "candidate"


def test_activity_refuses_clearly_without_saved_matches(tmp_path):
    ns, _, out = activity_ns(tmp_path, enabled=True)
    ns["OUTPUT_ROOT"] = tmp_path / "fresh-byod-run"
    with pytest.raises(RuntimeError, match="complete a Run all first"):
        exec(compile(src("mat-activity"), "activity", "exec"), ns)
    ns["OUTPUT_ROOT"], ns["ACTIVITY_STRICT_TOLERANCE_PX"] = out, 3
    with pytest.raises(ValueError, match="ACTIVITY_STRICT_TOLERANCE_PX"):
        exec(compile(src("mat-activity"), "activity", "exec"), ns)
    assert "Do not re-run section 1" in src("dimer-matching-workshop-53")

# ruff: noqa: E501
"""Regression tests for the 2026-10-12 capstone-round review of the workshop (MAT2-M1, MAT2-m1..m4, S1, S2, S4).

NumPy and Pillow only (CI's lock has no pandas or matplotlib). Notebook cells are executed from the
generator source on generated images; this is CPU evidence, not a hosted run.
"""
from __future__ import annotations

import ast
import io
import json
import math
import random
import re
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from multimodel_image_matching_workshop_source import CELLS  # noqa: E402

NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Image_Matching_Workshop.ipynb"
TIERS = ("easy", "moderate", "rotation_stress")


def src(cell_id):
    return next(c["source"] for c in CELLS if c.get("id") == cell_id)


def markdown_before(cell_id):
    out = []
    for cell in CELLS:
        if cell.get("id") == cell_id:
            return "\n".join(out)
        if cell["kind"] == "markdown":
            out.append(cell["source"])
    raise AssertionError(cell_id)


# ------------------------------------------------------------------ MAT2-M1
def test_central_question_names_the_two_configured_systems():
    opening = src("dimer-matching-workshop-00")
    question = re.search(r"central question is:\n\n>\s*([^\n]+)", opening).group(1)
    assert "LightGlue + ALIKED" in question and "XoFTR" in question and "vismatch 1.3.2" in question
    assert "sparse and detector-free matchers differ" not in question


def test_scope_statement_and_xoftr_design_precede_the_test():
    before = markdown_before("dimer-matching-workshop-32")  # section 14
    assert "### Scope of the comparison" in before
    assert "whole pipeline" in before and "one model" in before and "backend default settings" in before
    assert "thermal" in before and "2404.09692" in before
    limitations = src("dimer-matching-workshop-48")
    assert "Two configured systems, not two families" in limitations and "thermal" in limitations


def test_exercise_guidance_no_longer_credits_the_architecture_class():
    exercises = src("dimer-matching-workshop-49")
    assert "so it usually returns more correspondences" not in exercises
    assert "belongs to these two configured systems" in exercises
    assert "these two configured systems" in src("dimer-matching-workshop-55")


def _runner_namespace():
    cell = src("dimer-matching-workshop-26")
    runner = ast.literal_eval(cell.split("MATCH_RUNNER = ", 1)[1].split("\nrunner_path=", 1)[0])
    module = ast.parse(runner)
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "backend_settings"]
    ns = {}
    exec(compile(module, "runner", "exec"), ns)
    return runner, ns["backend_settings"]


def test_runner_records_effective_backend_settings():
    runner, settings = _runner_namespace()
    assert '"backend_settings":backend_settings(args.model,matcher)' in runner
    lg = types.SimpleNamespace(
        extractor=types.SimpleNamespace(conf=types.SimpleNamespace(model_name="aliked-n16", max_num_keypoints=2048, detection_threshold=0.2, nms_radius=2)),
        matcher=types.SimpleNamespace(conf=types.SimpleNamespace(filter_threshold=0.1, depth_confidence=-1, width_confidence=-1)))
    out = settings("lightglue", lg)
    assert out["max_num_keypoints"] == 2048 and out["detection_threshold"] == 0.2 and out["lightglue_filter_threshold"] == 0.1
    xf = types.SimpleNamespace(pretrained_size=640, matcher=types.SimpleNamespace(config={"match_coarse": {"thr": 0.3, "border_rm": 2}, "fine": {"thr": 0.1, "denser": False}}))
    out = settings("xoftr", xf)
    assert out == {"status": "read from the constructed backend", "pretrained_size": 640, "coarse_threshold": 0.3, "coarse_border_rm": 2, "fine_threshold": 0.1, "denser": False,
                   "image_input": "640-px PNG via load_image(resize=None)", "scored_output": "matched_kpts0/1 before the backend RANSAC"}
    # A backend whose internals moved must not break matching, but the failure is labelled, never hidden.
    failed = settings("xoftr", types.SimpleNamespace())
    assert failed["status"] == "unavailable" and "AttributeError" in failed["error"]
    assert "does NOT record the configuration" in src("dimer-matching-workshop-28")
    assert "backend_settings" in src("dimer-matching-workshop-28") and '"backend_settings":backend_settings' in src("dimer-matching-workshop-31")


# ------------------------------------------------------------------ MAT2-m1
def test_degradation_reports_retained_fractions_for_counts():
    tiers = {
        "lightglue": {"easy": {"precision_3px": .9, "inliers_per_pair": 1162., "matches_per_pair": 1200., "homography_acc_3px": 1.}, "moderate": {"precision_3px": .8, "inliers_per_pair": 800., "matches_per_pair": 900., "homography_acc_3px": .9}, "rotation_stress": {"precision_3px": .5, "inliers_per_pair": 264., "matches_per_pair": 500., "homography_acc_3px": .5}},
        "xoftr": {"easy": {"precision_3px": .9, "inliers_per_pair": 3921., "matches_per_pair": 4000., "homography_acc_3px": 1.}, "moderate": {"precision_3px": .7, "inliers_per_pair": 0., "matches_per_pair": 10., "homography_acc_3px": .9}, "rotation_stress": {"precision_3px": .3, "inliers_per_pair": 330., "matches_per_pair": 1500., "homography_acc_3px": .2}},
    }
    frames = []
    fake_pd = types.SimpleNamespace(DataFrame=lambda rows: frames.append(rows) or types.SimpleNamespace(to_csv=lambda *a, **k: None))
    ns = {"MODEL_KEYS": ["lightglue", "xoftr"], "MODEL_SPECS": {"lightglue": {"display_name": "LG"}, "xoftr": {"display_name": "XF"}},
          "test_tiers": tiers, "pd": fake_pd, "display": lambda *a: None, "OUTPUT_ROOT": Path("."), "math": math,
          "finite_csv": lambda frame, path: None}
    exec(compile(src("dimer-matching-workshop-37"), "degradation", "exec"), ns)
    rows = {(r["model"], r["tier"]): r for r in frames[0]}
    assert rows[("XF", "rotation_stress")]["inliers_per_pair_drop"] == pytest.approx(3591.)
    assert rows[("XF", "rotation_stress")]["inliers_per_pair_retained"] == pytest.approx(330 / 3921)
    assert rows[("LG", "rotation_stress")]["inliers_per_pair_retained"] == pytest.approx(264 / 1162)
    assert rows[("XF", "moderate")]["inliers_per_pair_retained"] == 0.0
    note = src("mat-notice-15")
    assert "absolute drops of counts scale with density" in note and "inliers_per_pair_retained" in note


# ------------------------------------------------------------------ MAT2-m2 (REL13)
def test_runtime_quote_names_its_revision_and_wall_time():
    opening = src("dimer-matching-workshop-01")
    runtime = next(line for line in opening.splitlines() if line.startswith("**Runtime.**"))
    assert "`e2ee0f4`" in runtime and "`4991047a0590`" in runtime and "896.9 s" in runtime
    assert "Recorded, not re-run by this revision" in runtime
    assert "wall time was not recorded" not in opening


# ------------------------------------------------------------------ MAT2-m3 (DAT20)
def _photos(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(n)
    for k in range(n):
        Image.fromarray((rng.random((60, 80, 3)) * 255).astype(np.uint8)).save(folder / f"photo_{k:03d}.png")


def _run_byod_cells(tmp_path, monkeypatch, byod_path):
    monkeypatch.chdir(tmp_path)
    controls = src("dimer-matching-workshop-04").replace("USE_BYOD = False", "USE_BYOD = True").replace('BYOD_PATH = ""', f"BYOD_PATH = {str(byod_path)!r}")
    ns = {"__name__": "__main__"}
    with redirect_stdout(io.StringIO()) as out:
        exec(compile(controls, "controls", "exec"), ns)
        exec(compile(src("dimer-matching-workshop-11"), "acquire", "exec"), ns)
    return ns, out.getvalue()


def test_byod_refusal_lists_every_bad_file_and_leaves_no_run_folder(tmp_path, monkeypatch):
    photos = tmp_path / "mine"
    _photos(photos, 6)
    (photos / "zz_dup.png").write_bytes((photos / "photo_000.png").read_bytes())
    Image.new("RGB", (10, 10)).save(photos / "zz_tiny.png")
    with pytest.raises(ValueError) as caught:
        _run_byod_cells(tmp_path, monkeypatch, photos)
    message = str(caught.value)
    assert type(caught.value).__name__ == "BYODRefused"
    assert "2 of 8 photographs were refused" in message and "zz_dup.png" in message and "zz_tiny.png" in message
    assert "run the notebook again from section 1" in message
    rendered = "\n".join(caught.value._render_traceback_())
    assert "Traceback" not in rendered and "zz_tiny.png" in rendered
    assert not (tmp_path / "outputs" / "byod").exists() or not any((tmp_path / "outputs" / "byod").iterdir())
    assert not any((tmp_path / "work" / "byod").iterdir())  # the refused run's work folder is removed too


def test_accepted_byod_creates_its_run_folder_after_validation(tmp_path, monkeypatch):
    photos = tmp_path / "mine"
    _photos(photos, 5)
    ns, out = _run_byod_cells(tmp_path, monkeypatch, photos)
    assert len(ns["photo_records"]) == 5 and ns["dataset_source"] == "BYOD"
    runs = list((tmp_path / "outputs" / "byod").iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("run-") and Path(ns["OUTPUT_DIR"]).resolve() == runs[0].resolve()
    assert all((runs[0] / sub).is_dir() for sub in ("data", "validation", "frozen", "test", "figures", "provenance"))
    assert "BYOD input accepted" in out


def test_default_path_creates_the_same_folders_as_before(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with redirect_stdout(io.StringIO()):
        exec(compile(src("dimer-matching-workshop-04"), "controls", "exec"), {})
    made = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_dir())
    assert made == sorted([
        "outputs", "outputs/data", "outputs/validation", "outputs/frozen", "outputs/test", "outputs/test/matches",
        "outputs/test/matches/lightglue", "outputs/test/matches/xoftr", "outputs/figures", "outputs/future",
        "outputs/provenance", "work", "work/photos", "work/pairs", "work/pairs/validation", "work/pairs/test",
        "work/envs", "work/runners"])


# ------------------------------------------------------------------ MAT2-m4
def _byod_split():
    module = ast.parse(src("dimer-matching-workshop-13"))
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == "byod_split"]
    ns = {"random": random}
    exec(compile(module, "split", "exec"), ns)
    return ns["byod_split"]


@pytest.mark.parametrize("n,val,test", [(4, 2, 2), (7, 3, 4), (12, 5, 7), (13, 5, 8), (360, 144, 216)])
def test_byod_uses_every_photo(n, val, test):
    split = _byod_split()([{"source_id": f"p{i}", "sha256": str(i)} for i in range(n)], 42)
    assert {k: len(v) for k, v in split.items()} == {"train": 0, "validation": val, "test": test}


def _coverage_output(tiers, use_byod=True):
    body = src("dimer-matching-workshop-33")
    body = body[body.index("test_per_tier="):]  # the coverage warning and low-count note are what is under test
    ns = {"USE_BYOD": use_byod, "TIERS": TIERS, "BYOD_MIN_TEST_PAIRS_PER_TIER": 1, "test_pair_rows": [{"tier": t} for t in tiers]}
    printed = io.StringIO()
    with redirect_stdout(printed):
        exec(compile(body, "test-pairs", "exec"), ns)
    return printed.getvalue()


def test_small_byod_test_split_warns_with_the_counts():
    out = _coverage_output(("easy", "moderate"))
    assert "Tier-coverage warning: only 2 test pair(s)" in out and "'rotation_stress': 0" in out
    assert "Low test-pair note" in out and "easy 1.00" in out
    # Seven photos cover every tier (no coverage warning) but still get the dynamic low-count note.
    out = _coverage_output(("easy", "moderate", "rotation_stress", "easy"))
    assert "Tier-coverage warning" not in out and "Low test-pair note" in out and "easy 0.50" in out
    # The built-in corpus (32 test pairs per tier) prints neither.
    assert _coverage_output(TIERS * 32, use_byod=False) == ""
    assert "BYOD_MIN_TEST_PAIRS_PER_TIER = 1" in src("dimer-matching-workshop-13")


# ------------------------------------------------------------------ MAT2-S1, S2, S4
def test_spec_declared_as_2_5_everywhere():
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5"
    assert "**Notebook specification:** `2.5`" in src("dimer-matching-workshop-00")
    for cell_id in ("dimer-matching-workshop-31", "dimer-matching-workshop-51", "dimer-matching-workshop-57"):
        assert '"notebook_spec":"2.5"' in src(cell_id) and '"notebook_spec":"2.1"' not in src(cell_id)
    assert meta["review_revisions"][-1]["date"] == "2026-10-13"


def test_paired_mcnemar_statement_matches_an_exact_binomial():
    cell = src("dimer-matching-workshop-39")
    line = next(x for x in cell.splitlines() if x.startswith("mcnemar_p="))
    only_lg, only_xf = 14, 4
    discordant = only_lg + only_xf
    ns = {"only_lg": only_lg, "only_xf": only_xf, "discordant": discordant, "math": math}
    exec(line, ns)
    expected = min(1.0, 2 * sum(math.comb(18, k) for k in range(5)) / 2 ** 18)
    assert ns["mcnemar_p"] == pytest.approx(expected) and 0.03 < expected < 0.04


def test_self_check_sample_answer_addresses_its_four_questions():
    answer = src("dimer-matching-workshop-54").split("<summary>", 1)[1]
    for marker in ("1. **Task.**", "2. **Reference.**", "3. **Failure mode or tradeoff.**", "4. **Before applying it elsewhere.**"):
        assert marker in answer
    assert "patch-neighbour" in answer and "If your run shows no clear failure tier" in answer
    assert "falls most" not in answer and "only if" not in answer  # conditional on the learner's evidence (peer review)


# ================================================================== peer review of the MAT2 fixes (round 2)
def _generator_ns():
    ns = {"Image": Image, "random": random, "SPLIT_SEED": 42}
    from PIL import ImageOps
    ns["ImageOps"] = ImageOps
    exec(compile(src("dimer-matching-workshop-15"), "generator", "exec"), ns)
    return ns


def _about_centre(m, w, h):
    t = np.array([[1, 0, (w - 1) / 2], [0, 1, (h - 1) / 2], [0, 0, 1.]])
    return t @ np.asarray(m, float) @ np.linalg.inv(t)


@pytest.mark.parametrize("name", ["identity", "translation", "scale", "rotation_180", "perspective"])
def test_rendered_warp_matches_the_scored_homography(name):
    """#1: coordinate-ramp fiducials. Each pixel of a ramp image stores its own integer-centred x (or y); after
    warp_image, pixel q must hold the coordinates of H^-1(q). Bilinear reproduces a ramp exactly, so it must agree
    to < 0.05 px everywhere; the notebook's bicubic warp must be unbiased (mean displacement < 0.05 px)."""
    ns = _generator_ns()
    w, h = 160, 120
    c, s = math.cos(math.pi), math.sin(math.pi)
    H = {"identity": np.eye(3), "translation": np.array([[1, 0, 7.], [0, 1, -4.], [0, 0, 1.]]),
         "scale": _about_centre(np.diag([1.3, 1.3, 1.]), w, h),
         "rotation_180": _about_centre([[c, -s, 0], [s, c, 0], [0, 0, 1.]], w, h),
         "perspective": _about_centre([[1.05, .04, 0], [-.03, .97, 0], [2e-4, -1.5e-4, 1.]], w, h)}[name]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = np.linalg.inv(H) @ np.stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    p = p[:2] / p[2]
    inside = (p[0] > 3) & (p[0] < w - 4) & (p[1] > 3) & (p[1] < h - 4)
    coeffs = ns["perspective_coefficients"](H)
    for interp, bound in ((Image.BILINEAR, "max"), (Image.BICUBIC, "mean")):
        warped = [np.asarray(Image.fromarray(ramp).transform((w, h), Image.PERSPECTIVE, coeffs, interp)).ravel()[inside] for ramp in (xx, yy)]
        dx, dy = warped[0] - p[0][inside], warped[1] - p[1][inside]
        err = float(np.hypot(dx, dy).max()) if bound == "max" else float(np.hypot(dx.mean(), dy.mean()))
        assert err < 0.05, (name, interp, err)
    # warp_image itself is the bicubic path checked above
    assert "perspective_coefficients(H),Image.BICUBIC" in src("dimer-matching-workshop-15")


def test_dlt_rejects_degenerate_minimal_samples_and_fits_overdetermined():
    ns = _generator_ns()
    dlt = ns["dlt_homography"]
    H = np.array([[1.1, .05, 3.], [-.02, .95, -2.], [1e-4, 2e-4, 1.]])
    pts = np.array([[10., 10.], [200., 15.], [190., 150.], [20., 140.]])
    est = dlt(pts, ns["warp_image"].__globals__["np"].asarray([(H @ [x, y, 1])[:2] / (H @ [x, y, 1])[2] for x, y in pts]))
    assert np.allclose(est, H / H[2, 2], atol=1e-6)
    collinear = np.array([[0., 0.], [10., 10.], [20., 20.], [5., 40.]])
    assert dlt(collinear, collinear + 1) is None
    duplicate = np.array([[0., 0.], [0., 0.], [30., 5.], [5., 40.]])
    assert dlt(duplicate, duplicate) is None
    assert dlt(pts, np.array([[1., 1.], [np.nan, 2.], [3., 3.], [4., 9.]])) is None
    rng = np.random.default_rng(0)
    many = rng.uniform(0, 300, (2000, 2))
    proj = np.array([(H @ [x, y, 1])[:2] / (H @ [x, y, 1])[2] for x, y in many])
    assert np.allclose(dlt(many, proj), H / H[2, 2], atol=1e-6)


def test_pairs_record_realised_warp_and_overlap(tmp_path):
    ns = _generator_ns()
    path = tmp_path / "p.png"
    Image.fromarray((np.random.default_rng(1).random((90, 120, 3)) * 255).astype(np.uint8)).save(path)
    im0, im1, H, seed, warp = ns["make_pair"]({"path": str(path)}, 0, "test", "rotation_stress")
    assert set(warp) == {"rotation_deg", "scale", "brightness", "contrast", "gamma", "blur_radius", "noise_sigma", "overlap_fraction"}
    assert abs(warp["rotation_deg"]) <= 150 and 0.6 <= warp["scale"] <= 1.4 and 0 < warp["overlap_fraction"] <= 1
    again = ns["make_pair"]({"path": str(path)}, 0, "test", "rotation_stress")
    assert np.array_equal(np.asarray(im1), np.asarray(again[1])) and again[4] == warp
    assert '"warp":warp' in src("dimer-matching-workshop-17") and '"overlap_fraction":overlap_by_pair' in src("dimer-matching-workshop-35")


def test_byod_first_run_creates_the_shared_environment_folder(tmp_path, monkeypatch):
    """#11: a BYOD-first run rebinds WORK_ROOT, so the shared envs folder must be created before the lock is written."""
    monkeypatch.chdir(tmp_path)
    controls = src("dimer-matching-workshop-04").replace("USE_BYOD = False", "USE_BYOD = True")
    ns = {}
    with redirect_stdout(io.StringIO()):
        exec(compile(controls, "controls", "exec"), ns)
    assert not (Path("work") / "envs").exists()  # the condition that broke the original cell
    env = src("dimer-matching-workshop-24")
    part = env[env.index("ENV_ROOT=SHARED_WORK_ROOT"):env.index("signature=")]
    import hashlib
    lock = "pkg==1.0 --hash=sha256:00\n"
    ns.update({"hashlib": hashlib, "MATCH_ENV_LOCK": lock, "MATCH_ENV_LOCK_SHA256": hashlib.sha256(lock.encode()).hexdigest()})
    exec(compile(part, "env", "exec"), ns)
    assert (Path("work") / "envs" / "vismatch-requirements.lock.txt").read_text(encoding="utf-8") == lock


def test_freeze_is_rechecked_and_exports_write_null():
    freeze, opening = src("dimer-matching-workshop-31"), src("dimer-matching-workshop-33")
    assert "def live_config()" in freeze and '"live_config":live_config()' in freeze and '"test_digest"' in freeze
    assert "drift=[key for key,value in live_config().items()" in opening and "changed_files=" in opening
    assert "auditable snapshot" in src("dimer-matching-workshop-30") and "cannot be changed" not in src("dimer-matching-workshop-30")
    test_eval = src("dimer-matching-workshop-35")
    assert "def finite_csv" in test_eval and "test_baseline_tiers" in test_eval and "Backend settings differ between the validation and test runs" in test_eval
    assert "finite_csv(degradation" in src("dimer-matching-workshop-37")
    assert 'compare.replace([np.inf,-np.inf],np.nan).to_csv' in src("dimer-matching-workshop-39")
    assert "for method in validation_baselines:" in src("dimer-matching-workshop-29")


def test_wording_matches_the_design():
    assert "## 14. Held-out test" in src("dimer-matching-workshop-32")
    assert "never used before" not in src("mat-notice-14") and "not guaranteed to be independent scenes" in src("mat-notice-14")
    assert "different photographs with independently sampled warps" in src("dimer-matching-workshop-14")
    assert "500 random four-match samples" in src("dimer-matching-workshop-19") and "extrapolates" in src("dimer-matching-workshop-19")
    assert "cannot spoil it" not in src("dimer-matching-workshop-02")
    assert "it does not show that the two are equivalent" in src("mat-notice-16")
    assert "paired-bootstrap interval" in src("dimer-matching-workshop-39")
    assert '"accelerator":torch.cuda.get_device_name(0)' in src("dimer-matching-workshop-26")
    assert "so high precision wins" not in src("dimer-matching-workshop-49")
    assert "Retention on its own does not make a method better" in src("mat-notice-15")

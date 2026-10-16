"""Regression tests for the 2026-10-15 review of the tutorial notebook (LGT-M1, LGT-m1..m6).

LGT-M1: Pillow's PERSPECTIVE transform works in pixel-edge coordinates (pixel i spans [i, i + 1)), while the
reference `H`, the metrics and the matcher's keypoints use integer pixel centres. Rendering with `H` directly drew
C⁻¹·H·C (C = +0.5 px shift): the offset is (A - I)(0.5, 0.5) for the linear part A, so 1.0 px at 90° and
√2 ≈ 1.41 px at 180° for a unit-scale rotation. `warp_image` now renders C·H·C⁻¹. Two tolerances:
  * coefficient geometry, measured with coordinate ramps under bilinear sampling (exact for a ramp): < 0.05 px for the
    fixed cases below (identity, translation, scale, centred and off-centre rotations, perspective);
  * end to end through warp_image's bicubic sampling, a Gaussian fiducial lands on H(p) within 0.15 px, because Pillow's
    bicubic filter itself moves a feature by up to ~0.1 px depending on the sub-pixel phase; the same tolerance is used
    for a set of sampled hard-tier homographies, and the number of fiducials actually checked is asserted to be nonzero.
NumPy and Pillow only (CI dependencies), no model, no network.
"""
# ruff: noqa: E501  -- prose docstrings and comments at the fleet's 120-column width; this repo lints at 100

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lightglue_pipeline import samples
from lightglue_pipeline.metrics import dlt_homography

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "lightglue_matching_colab.ipynb"
SIZE = (640, 480)
CX, CY = (SIZE[0] - 1) / 2.0, (SIZE[1] - 1) / 2.0
POINTS = ((300.0, 200.0), (250.0, 260.0), (380.0, 230.0))
TOLERANCE_PX = 0.05
BACKGROUND = 60.0


def _similarity(deg: float, scale: float = 1.0, cx: float = CX + 0.3, cy: float = CY + 0.2) -> np.ndarray:
    # About a point just off the image centre: an exact 180° turn about the half-integer centre puts every sample on
    # a pixel boundary, where Pillow's bicubic rounding differs by one grey level on about half the pixels (measured
    # 2026-10-15) and moves the centroid ~0.05 px. That is quantisation, not the half-pixel convention under test.
    a = math.radians(deg)
    c, s = math.cos(a) * scale, math.sin(a) * scale
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0.0, 0.0, 1.0]])


def _perspective() -> np.ndarray:
    corners = np.array([[0.0, 0.0], [SIZE[0] - 1.0, 0.0], [SIZE[0] - 1.0, SIZE[1] - 1.0], [0.0, SIZE[1] - 1.0]])
    moved = corners + np.array([[18.0, -11.0], [-25.0, 14.0], [9.0, 21.0], [-16.0, -8.0]])
    return dlt_homography(corners, moved) @ _similarity(150.0, 1.1)


WARPS = {
    "identity": np.eye(3),
    "translation": np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]]),
    "scale_1.3": _similarity(0.0, 1.3),
    "rotation_90": _similarity(90.0),
    "rotation_90_centred": _similarity(90.0, cx=CX, cy=CY),
    "rotation_180_centred": _similarity(180.0, cx=CX, cy=CY),
    "rotation_150": _similarity(150.0),
    "rotation_180": _similarity(180.0),
    "rotation_150_scale_1.4": _similarity(150.0, 1.4),
    "perspective": _perspective(),
}


def _fiducial_error(H: np.ndarray, p: tuple[float, float], sigma: float = 3.0) -> float:
    ys, xs = np.mgrid[0 : SIZE[1], 0 : SIZE[0]]
    g = np.exp(-((xs - p[0]) ** 2 + (ys - p[1]) ** 2) / (2.0 * sigma**2))
    # On a mid-grey background, so bicubic overshoot below the background is kept rather than clipped at 0 (a
    # clipped undershoot biases the centroid by a few hundredths of a pixel, depending on the sampling phase).
    image = Image.fromarray((BACKGROUND + 150.0 * g).round().astype(np.uint8)).convert("RGB")
    ref = H @ [p[0], p[1], 1.0]
    ref = ref[:2] / ref[2]
    warped = np.asarray(samples.warp_image(image, H).convert("L"), dtype=np.float64)
    gx, gy, r = int(round(ref[0])), int(round(ref[1])), 16
    window = warped[gy - r : gy + r + 1, gx - r : gx + r + 1] - BACKGROUND
    wy, wx = np.mgrid[gy - r : gy + r + 1, gx - r : gx + r + 1]
    observed = np.array([(window * wx).sum() / window.sum(), (window * wy).sum() / window.sum()])
    return float(np.linalg.norm(observed - ref))


@pytest.mark.parametrize("name", sorted(WARPS))
def test_render_coefficients_place_every_pixel_exactly_on_h(name):
    """Geometry of the render, free of interpolation error: coordinate ramps (pixel value = its own integer-centre x
    or y) warped with warp_image's coefficients under bilinear sampling, which reproduces a ramp exactly. Pixel q of
    the result must hold H⁻¹(q). Before the fix it held (C⁻¹·H·C)⁻¹(q): 1.0 px off at 90°, √2 ≈ 1.41 px at 180°."""
    H = WARPS[name]
    w, h = SIZE
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    coeffs = samples._perspective_coefficients(H)
    got = [
        np.asarray(Image.fromarray(ramp).transform(SIZE, Image.PERSPECTIVE, coeffs, Image.BILINEAR), dtype=np.float64)
        for ramp in (xs, ys)
    ]
    qy, qx = np.mgrid[0:h, 0:w].astype(np.float64)
    src = np.linalg.inv(H) @ np.stack([qx.ravel(), qy.ravel(), np.ones(qx.size)])
    sx, sy = (src[0] / src[2]).reshape(h, w), (src[1] / src[2]).reshape(h, w)
    inside = (sx > 1) & (sx < w - 2) & (sy > 1) & (sy < h - 2)
    assert inside.sum() > 0.25 * w * h
    error = np.hypot(got[0] - sx, got[1] - sy)[inside]
    print({name: round(float(error.max()), 6)})
    assert float(error.max()) < TOLERANCE_PX, {name: float(error.max())}


@pytest.mark.parametrize("name", sorted(WARPS))
def test_warp_image_fiducials_land_on_h(name):
    """End to end through warp_image (bicubic): a Gaussian fiducial lands on H(p). Pillow's bicubic transform itself
    moves a feature by up to ~0.1 px depending on the sub-pixel phase (0 at phase 0 or 0.5, 0.09 px at 0.25, also for a
    pure fractional translation; measured 2026-10-15 with Pillow 11.3.0), so this tolerance is 0.15 px; the
    half-pixel defect it guards against was 1.0-1.8 px."""
    errors = [_fiducial_error(WARPS[name], p) for p in POINTS]
    print({name: [round(e, 4) for e in errors]})
    assert max(errors) < 0.15, {name: errors}


def test_the_uncorrected_render_was_off_by_the_half_pixel_term():
    """The old coefficients (H used directly) drew C⁻¹·H·C: √2 ≈ 1.41 px from H(p) at 180°, the defect LGT-M1 fixed."""
    H = WARPS["rotation_180"]
    shift = samples.PIXEL_CENTRE_SHIFT
    np.testing.assert_allclose(samples.pillow_homography(H), shift @ H @ np.linalg.inv(shift))
    drawn_before = np.linalg.inv(shift) @ H @ shift
    p = np.array([300.0, 200.0, 1.0])
    offset = np.linalg.norm((drawn_before @ p)[:2] - (H @ p)[:2])
    assert offset == pytest.approx(math.sqrt(2.0), abs=1e-9)  # (A - I)(0.5, 0.5) with A = -I


def test_sampled_hard_homographies_render_on_h():
    """Sampled hard-tier homographies (seed 44): a fiducial at the integer pixel whose image is the canvas centre."""
    rng = random.Random(44)
    errors = []
    for _ in range(8):
        H = samples.sample_homography(SIZE, rng, samples.TIER_PARAMS["hard"])
        centre = np.linalg.inv(H) @ [CX, CY, 1.0]
        p = (float(round(centre[0] / centre[2])), float(round(centre[1] / centre[2])))
        if 40 <= p[0] < SIZE[0] - 40 and 40 <= p[1] < SIZE[1] - 40:
            errors.append(_fiducial_error(H, p))
    print({"checked_fiducials": len(errors), "max_px": round(max(errors), 4) if errors else None})
    assert len(errors) > 0
    assert max(errors) < 0.15


# --- LGT-m4: one-message BYOD refusals -------------------------------------------------------------------------


def test_byod_refusal_renders_one_line_and_the_sample_reraises():
    with pytest.raises(samples.BYODRefused) as caught:
        samples.refuse_byod(ValueError("too few photographs for the test split"))
    lines = caught.value._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("BYOD refused: too few photographs") and "Run after" in lines[0]
    assert isinstance(caught.value, ValueError) and caught.value.dimer_message_only
    with pytest.raises(KeyError):
        samples.refuse_byod(KeyError("sample defect"), byod=False)


def _cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _source(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def test_isolated_runtime_sends_refusals_without_a_traceback():
    router = [_source(c) for c in _cells() if c["cell_type"] == "code"][1]
    assert 'if getattr(exc, "dimer_message_only", False):' in router
    assert "class IsolatedRefusal(IsolatedCellError):" in router
    assert "raise (IsolatedCellError if message[1] else IsolatedRefusal)(message[2]) from None" in router


# --- LGT-m1, m2, m3, m5, m6: notebook text and metadata ----------------------------------------------------------


def _markdown() -> str:
    return "\n".join(_source(c) for c in _cells() if c["cell_type"] == "markdown")


def test_quoted_runs_name_their_revision_and_the_refresh_is_pending():
    markdown = _markdown()
    assert "previous notebook version" not in markdown
    for para in re.split(r"\n\s*\n|<details>", markdown):
        if re.search(r"recorded (Kaggle )?(Tesla )?T4 run|build workstation|T4 probe", para):
            assert re.search(r"`(d5e562ea|22f03c5|36a85f7)`", para), para[:160]
    assert "used inconsistent rendering and reference coordinates" in markdown and "## Historical note" in markdown
    assert "the worked answers will quote this version's hosted run once it exists" in markdown.replace("\n", " ")


def test_pretraining_overlap_rotation_wording_notice_and_spec():
    markdown = _markdown()
    assert "**Pretraining overlap.**" in markdown and "MegaDepth" in markdown and "Pretraining overlap was not checked." in markdown
    assert "The rotation is the point" not in markdown and "likeliest" not in markdown and "plateau" not in markdown
    assert "a plausible contributor" in markdown and "**exploratory**" in markdown and "conditional inlier error" in markdown
    assert "changes rotation, perspective jitter, scale and photometry together" in markdown
    first_code = next(i for i, c in enumerate(_cells()) if c["cell_type"] == "code")
    assert any("> **⚠️ Notice.**" in _source(c) for c in _cells()[:first_code])
    for name in ("DIMER_ISOLATED_ENV", "LIGHTGLUE_WEIGHTS_DIR"):
        assert name in markdown
    references = markdown.split("## References")[-1]
    assert re.search(r"^\[1\] P\. Lindenberger", references, re.M) and "[9] DIMER" in references
    meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"


def _section4_source() -> str:
    cells = _cells()
    i = next(k for k, c in enumerate(cells) if c["cell_type"] == "markdown" and "## 4. iNaturalist photographs" in _source(c))
    return next(_source(c) for c in cells[i + 1 :] if c["cell_type"] == "code")


def test_byod_expected_errors_become_one_message_and_unexpected_ones_keep_their_traceback(monkeypatch, tmp_path):
    """Codex follow-up #5: only input, decoding and archive errors are refusals; anything else propagates unchanged."""
    import lightglue_pipeline as lg

    monkeypatch.chdir(tmp_path)
    bad_zip = tmp_path / "broken.zip"
    bad_zip.write_bytes(b"not a zip archive")
    source = _section4_source().replace("USE_BYOD = False", "USE_BYOD = True", 1)

    def run(byod_path, **overrides):
        ns = {name: getattr(lg, name) for name in lg.__all__}
        ns.update({"os": __import__("os"), "Path": Path, "json": json, **overrides})
        code = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {str(byod_path)!r}", 1)
        exec(compile(code, "<section 4>", "exec"), ns)

    with pytest.raises(lg.BYODRefused):
        run(bad_zip)

    def broken_loader(path):
        raise KeyError("a defect, not a refusal")

    real_exec_ns_loader = lg.load_byod_dataset
    monkeypatch.setattr(lg, "load_byod_dataset", broken_loader)
    try:
        with pytest.raises(KeyError):
            run(bad_zip)
    finally:
        monkeypatch.setattr(lg, "load_byod_dataset", real_exec_ns_loader)

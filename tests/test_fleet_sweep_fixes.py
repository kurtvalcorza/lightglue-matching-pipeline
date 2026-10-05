"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['lightglue_matching_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'lightglue_matching_colab': 5}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


NB = NOTEBOOKS[0]


def _learner_code() -> str:
    return "\n".join(c["source"] for c in _code_cells(_nb(NB)) if not c["metadata"].get("dimer", {}).get("embedded_module"))


def _samples():
    """`samples.py` (and `config.py`) imported torch-free under an alias package."""
    import importlib

    name = "_lightglue_sweep_alias"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(ROOT / "src" / "lightglue_pipeline")]
        sys.modules[name] = pkg
    return importlib.import_module(f"{name}.samples")


# --- SWP-A: quality outcomes are recorded verdicts, never asserts -------------------------------------------------


def test_swp_a_no_quality_assert_remains():
    code = _learner_code()
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"precision|homography_acc|frozen_test|adapted_test", line)]
    assert quality == []
    assert "assert parity['identical_pairs'] == parity['of']" in code  # contract integrity stays a hard check


def test_swp_a_negative_results_are_recorded_and_do_not_stop_the_notebook():
    s6 = _cell(_nb(NB), "frozen_test = pipe.evaluate(test_records)")
    frozen_block = s6[s6.index("frozen_beats = ") : s6.index("print({'frozen_vs_baselines'")]
    s8 = _cell(_nb(NB), "adapted_test = pipe.evaluate(test_records)")
    verdict_block = s8[s8.index("delta_precision = ") : s8.index("comparison['by_tier'] = ")]
    cases = ((0.30, 0.40, 0.25, "worse", False), (0.75, 0.35, 0.755, "flat", True), (0.75, 0.35, 0.768, "improved", True))
    for frozen, patch, adapted, expected, beats in cases:
        ns = {
            "frozen_test": {"precision_3px": frozen, "homography_acc_3px": 0.7},
            "baselines": {"patch_neighbour": {"precision_3px": patch}, "identity": {"homography_acc_3px": 0.0}},
            "adapted_test": {"precision_3px": adapted},
            "comparison": {},
        }
        exec(frozen_block, ns)
        exec(verdict_block, ns)
        assert ns["comparison"]["verdicts"]["adapted_vs_frozen_precision_3px"] == expected
        assert ns["comparison"]["verdicts"]["frozen_vs_baselines"].startswith("above both" if beats else "not above both")
    assert s8.index("comparison['verdicts'] = ") < s8.index("json.dump(evaluation_report_payload")


# --- SWP-F: Sections 6 and 7 always start from the frozen matcher -------------------------------------------------


def test_swp_f_frozen_pipeline_reloads_an_adapted_pipeline(capsys):
    s6 = _cell(_nb(NB), "def frozen_pipeline():")
    helper = s6[s6.index("def frozen_pipeline():") : s6.index("\n\n\nfrozen_pipeline()")]
    loads = []

    class Stand:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return types.SimpleNamespace(adapter=None)

    ns = {"pipe": types.SimpleNamespace(adapter={"best_epoch": 2}), "LightGluePipeline": Stand, "WEIGHTS_DIR": "w"}
    exec(helper, ns)
    ns["frozen_pipeline"]()
    assert loads == ["w"] and ns["pipe"].adapter is None
    assert "Reloaded the frozen matcher" in capsys.readouterr().out
    ns["frozen_pipeline"]()
    assert loads == ["w"]


def test_swp_f_scoring_and_training_cells_call_frozen_pipeline_first():
    s6 = _cell(_nb(NB), "frozen_test = pipe.evaluate(test_records)")
    assert s6.index("\nfrozen_pipeline()\n") < s6.index("baselines = pipe.evaluate_baselines(")
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("frozen_pipeline()") < s7.index("adapt_result = pipe.adapt(")


# --- SWP-B: BYOD path, guarded upload, refusals that name the source ----------------------------------------------


def _byod(monkeypatch, path: str, tmp_path) -> dict:
    source = _cell(_nb(NB), "BYOD_PATH = ''")
    block = source[source.index("def byod_source(") : source.index("    splits = split_dataset(records, seed=SPLIT_SEED)")]
    block = block.replace("os.makedirs('outputs', exist_ok=True)\n", "")
    monkeypatch.chdir(tmp_path)
    ns = {"Path": Path, "USE_BYOD": True, "BYOD_PATH": path, "load_byod_dataset": _samples().load_byod_dataset}
    exec(compile(block, "<section 4 BYOD>", "exec"), ns)
    return ns


def _png(path: Path, seed: int) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (80, 96, 3), dtype=np.uint8)).save(path)


def test_swp_b_byod_path_folder_and_zip_work_outside_colab(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    folder = tmp_path / "photos"
    folder.mkdir()
    for i in range(3):
        _png(folder / f"p{i}.png", i)
    ns = _byod(monkeypatch, str(folder), tmp_path)
    assert ns["file_name"] == "photos" and [r["id"] for r in ns["records"]] == ["p0", "p1", "p2"]
    import zipfile

    with zipfile.ZipFile(tmp_path / "photos.zip", "w") as archive:
        for i in range(3):
            archive.write(folder / f"p{i}.png", f"p{i}.png")
    ns = _byod(monkeypatch, str(tmp_path / "photos.zip"), tmp_path)
    assert len(ns["records"]) == 3


def test_swp_b_refusals_name_the_source_and_the_rule(monkeypatch, tmp_path):
    _no_colab(monkeypatch)
    with pytest.raises(FileNotFoundError, match="BYOD_PATH .*missing.* does not exist"):
        _byod(monkeypatch, str(tmp_path / "missing"), tmp_path)
    with pytest.raises(RuntimeError, match="BYOD_PATH is empty and this runtime has no Colab upload dialog"):
        _byod(monkeypatch, "", tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match=r"^empty: BYOD dataset holds no JPEG / PNG image files"):
        _byod(monkeypatch, str(empty), tmp_path)
    notes = tmp_path / "notes.txt"
    notes.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match=r"^notes\.txt: BYOD datasets must be a directory or a \.zip"):
        _byod(monkeypatch, str(notes), tmp_path)


def test_swp_b_cancelled_or_multiple_uploads_are_refused(monkeypatch, tmp_path):
    _colab(monkeypatch, lambda: {})
    with pytest.raises(RuntimeError, match="got 0 .*upload cancelled or empty"):
        _byod(monkeypatch, "", tmp_path)
    _colab(monkeypatch, lambda: {"a.zip": b"", "b.zip": b""})
    with pytest.raises(RuntimeError, match="got 2"):
        _byod(monkeypatch, "", tmp_path)


def test_swp_g_checkpoint_answers_agree_with_the_recorded_run():
    """Numbers quoted in the worked answers are the recorded 2026-09-21 Kaggle T4 run's (docs/release-verification.md)."""
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    assert "precision_3px: {identity: 0.006, patch_neighbour: 0.346, descriptor_nn: 0.624, frozen: 0.75, adapted: 0.768}" in record
    assert "homography_acc_3px: {identity: 0.0, patch_neighbour: 0.458, descriptor_nn: 0.646, frozen: 0.708, adapted: 0.719}" in record
    assert "reload parity {identical_pairs: 4, of: 4}" in record
    for quoted in ("identity 0.006, patch neighbour 0.346, descriptor neighbour 0.624, frozen 0.75", "0.75 → 0.768", "0.708 → \n0.719".replace(" \n", " "), "4 of 4"):
        assert quoted.replace("\n", "") in markdown.replace("\n", " "), quoted

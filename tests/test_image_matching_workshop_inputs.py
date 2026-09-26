import ast
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from multimodel_image_matching_workshop_source import CELLS


def helpers(tmp_path):
    ns = {"WORK_ROOT": tmp_path, "Path": Path}
    module = ast.parse(CELLS[11]["source"])
    module.body = [
        n for n in module.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
    ]
    exec(compile(module, "loader", "exec"), ns)
    module = ast.parse(CELLS[13]["source"])
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    exec(compile(module, "split", "exec"), ns)
    return ns


def photos(tmp_path):
    root = tmp_path / "photos"
    root.mkdir(exist_ok=True)
    for i in range(4):
        Image.new("RGB", (32, 32), (i * 40, 2, 3)).save(root / f"{i}.png")
    return root


def test_valid_local_photos_split_and_hash(tmp_path):
    ns = helpers(tmp_path)
    records = ns["load_byod_photos"](photos(tmp_path))
    split = ns["byod_split"](records)
    assert {k: len(v) for k, v in split.items()} == {"train": 2, "validation": 1, "test": 1}
    assert all(len(r["pixel_sha256"]) == 64 for r in records)


def test_reencoded_duplicate_pixels_rejected(tmp_path):
    ns = helpers(tmp_path)
    root = photos(tmp_path)
    with Image.open(root / "0.png") as image:
        image.save(root / "different.png", compress_level=0)
    assert (root / "0.png").read_bytes() != (root / "different.png").read_bytes()
    with pytest.raises(ValueError, match="Duplicate decoded"):
        ns["load_byod_photos"](root)


def test_byod_tiny_photo_rejected(tmp_path):
    ns = helpers(tmp_path)
    root = photos(tmp_path)
    Image.new("RGB", (1, 1)).save(root / "0.png")
    with pytest.raises(ValueError, match="at least 16"):
        ns["load_byod_photos"](root)


def test_zip_isolation_preserves_existing(tmp_path):
    ns = helpers(tmp_path)
    root = photos(tmp_path)
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for p in root.iterdir():
            z.write(p, p.name)
    old = root / "byod"
    old.mkdir()
    (old / "sentinel").write_text("keep")
    assert len(ns["load_byod_photos"](archive)) == 4
    assert len(ns["load_byod_photos"](archive)) == 4
    assert (old / "sentinel").read_text() == "keep"
    assert len(list(old.glob("dataset-*"))) == 2


@pytest.mark.parametrize("name", ["../escape", "C:/escape"])
def test_zip_path_rejection(tmp_path, name):
    ns = helpers(tmp_path)
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "bad")
    with pytest.raises(ValueError):
        ns["safe_extract_zip"](archive, tmp_path / "out")


def test_zip_symlink_rejected_before_extraction(tmp_path):
    ns = helpers(tmp_path)
    archive = tmp_path / "symlink.zip"
    entry = zipfile.ZipInfo("linked.png")
    entry.create_system = 3
    entry.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(entry, "../outside.png")
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="(?i)symlink"):
        ns["safe_extract_zip"](archive, destination)
    assert not list(destination.iterdir())


def runner_helper():
    node = ast.parse(CELLS[26]["source"]).body[0]
    runner = ast.literal_eval(node.value)
    module = ast.parse(runner)
    module.body = [
        n
        for n in module.body
        if isinstance(n, ast.FunctionDef) and n.name == "normalize_match_result"
    ]
    ns = {"np": np}
    exec(compile(module, "runner", "exec"), ns)
    return ns["normalize_match_result"]


def test_runner_missing_confidence_is_explicit_uniform_fallback():
    a, b, c = runner_helper()({"matched_kpts0": [[1, 2]], "matched_kpts1": [[2, 3]]})
    assert a.shape == b.shape == (1, 2) and c.tolist() == [1]


@pytest.mark.parametrize(
    "result",
    [
        {"matched_kpts0": [[1, 2]], "matched_kpts1": []},
        {"matched_kpts0": [[1, 2]], "matched_kpts1": [[2, 3]], "matched_confidences": []},
        {"matched_kpts0": [[float("nan"), 2]], "matched_kpts1": [[2, 3]]},
        {"matched_kpts0": [[1, 2, 3, 4]], "matched_kpts1": [[1, 2], [3, 4]]},
    ],
)
def test_runner_rejects_malformed_correspondences(result):
    with pytest.raises(ValueError):
        runner_helper()(result)


def test_disabled_baselines_do_not_run(tmp_path):
    ns = {
        "RUN_IDENTITY_BASELINE": False,
        "RUN_PATCH_BASELINE": False,
        "display": lambda x: None,
        "pd": type("Frames", (), {"DataFrame": staticmethod(lambda x: x)}),
        "MODEL_KEYS": [],
    }
    exec(CELLS[22]["source"], ns)
    exec(CELLS[34]["source"], ns)
    assert ns["validation_baselines"] == ns["test_baselines"] == {}


def test_controls_isolate_byod_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = CELLS[4]["source"].replace("USE_BYOD = False", "USE_BYOD = True")
    ns = {}
    exec(source, ns)
    first = ns["OUTPUT_ROOT"]
    assert first.is_dir()
    ns = {}
    exec(source, ns)
    assert ns["OUTPUT_ROOT"] != first
    assert not ns["WORK_ROOT"].is_relative_to(ns["OUTPUT_ROOT"])
    assert ns["WORK_ROOT"].parent == (Path("work") / "byod").resolve()


def test_source_code_parses():
    for cell in CELLS:
        if cell["kind"] == "code":
            compile(cell["source"], "cell", "exec")


def runner_namespace():
    runner = ast.literal_eval(ast.parse(CELLS[26]["source"]).body[0].value)
    ns = {"__name__": "runner_test"}
    exec(runner, ns)
    return ns


def test_checkpoint_hook_verifies_before_load_and_restores(tmp_path, monkeypatch):
    import types

    ns = runner_namespace()
    payload = b"verified fixture"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    for name in ["aliked-n16.pth", "aliked_lightglue.pth"]:
        ns["KNOWN"]["lightglue"]["assets"][name] = [len(payload), digest]
        (tmp_path / name).write_bytes(payload)
    calls = []

    def original(*a, **k):
        return None

    torch = types.SimpleNamespace(
        hub=types.SimpleNamespace(load_state_dict_from_url=original),
        load=lambda p, **kw: calls.append(kw) or {},
    )
    with ns["verified_checkpoint_loaders"]("lightglue", torch, tmp_path) as evidence:
        for url in ns["URL_REQUESTS"]:
            torch.hub.load_state_dict_from_url(url)
    assert len(evidence) == 2 and all(c["weights_only"] is True for c in calls)
    assert torch.hub.load_state_dict_from_url is original
    (tmp_path / "aliked-n16.pth").write_bytes(b"corrupt")
    calls.clear()
    with pytest.raises(ValueError, match="integrity"):
        with ns["verified_checkpoint_loaders"]("lightglue", torch, tmp_path):
            torch.hub.load_state_dict_from_url(
                "https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth"
            )
    assert calls == [] and torch.hub.load_state_dict_from_url is original


def test_checkpoint_hook_rejects_unknown_url(tmp_path):
    import types

    ns = runner_namespace()

    def original(*a, **k):
        return None

    torch = types.SimpleNamespace(hub=types.SimpleNamespace(load_state_dict_from_url=original))
    with pytest.raises(ValueError, match="Unapproved"):
        with ns["verified_checkpoint_loaders"]("lightglue", torch, tmp_path):
            torch.hub.load_state_dict_from_url("https://unapproved.invalid/model.pth")
    assert torch.hub.load_state_dict_from_url is original


def test_xoftr_exact_file_snapshot_and_safe_load(tmp_path, monkeypatch):
    import types

    ns = runner_namespace()
    payload = b"safetensor fixture"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    name = "xoftr_640.safetensors"
    (tmp_path / name).write_bytes(payload)
    ns["KNOWN"]["xoftr"]["assets"][name] = [len(payload), digest]
    calls = []

    def original_snapshot(*a, **k):
        return None

    def original_safe(p, **k):
        calls.append(p)
        return {}

    module = types.SimpleNamespace(snapshot_download=original_snapshot, load_file=original_safe)
    monkeypatch.setattr(ns["importlib"], "import_module", lambda name: module)
    torch = types.SimpleNamespace(
        hub=types.SimpleNamespace(load_state_dict_from_url=lambda *a, **k: None)
    )
    with ns["verified_checkpoint_loaders"]("xoftr", torch, tmp_path) as evidence:
        root = module.snapshot_download("vismatch/xoftr")
        module.load_file(str(Path(root) / name))
    assert len(calls) == len(evidence) == 1 and module.snapshot_download is original_snapshot
    assert module.load_file is original_safe
    with pytest.raises(ValueError, match="Unapproved"):
        with ns["verified_checkpoint_loaders"]("xoftr", torch, tmp_path):
            module.snapshot_download("other/repo")
    calls.clear()
    (tmp_path / name).write_bytes(b"bad")
    with pytest.raises(ValueError, match="integrity"):
        with ns["verified_checkpoint_loaders"]("xoftr", torch, tmp_path):
            pass
    assert calls == [] and module.load_file is original_safe


def test_download_digest_failure_never_promotes_cache(tmp_path, monkeypatch):
    import io

    ns = runner_namespace()
    monkeypatch.setattr(ns["urllib"].request, "urlopen", lambda *a, **k: io.BytesIO(b"wrong"))
    with pytest.raises(ValueError, match="integrity"):
        ns["stage_asset"]("aliked-n16.pth", tmp_path)
    assert not (tmp_path / "aliked-n16.pth").exists()


def test_strict_json_preserves_undefined_metrics_as_null():
    import math

    module = ast.parse(CELLS[4]["source"])
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    ns = {"json": json, "math": math}
    exec(compile(module, "json", "exec"), ns)
    result = ns["strict_json_dumps"]({"corner_error": math.inf, "nested": [float("nan"), 0.0]})
    assert json.loads(result) == {"corner_error": None, "nested": [None, 0.0]}
    assert "Infinity" not in result and "NaN" not in result

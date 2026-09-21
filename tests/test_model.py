# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from lightglue_pipeline import (
    ALIKED,
    CKPT_ALLOWED_GLOBALS,
    CONVERTED_FILENAMES,
    DEFAULT_MODEL_KEY,
    EXTRACTOR_COMMIT,
    EXTRACTOR_FILENAME,
    EXTRACTOR_PARAMETER_COUNT,
    EXTRACTOR_SOURCE_FILENAME,
    EXTRACTOR_SOURCE_SHA256,
    EXTRACTOR_SOURCE_URL,
    EXTRACTOR_STATE_TENSORS,
    MANIFEST_NAME,
    MATCHER_FILENAME,
    MATCHER_LAYERS,
    MATCHER_PARAMETER_COUNT,
    MATCHER_SOURCE_FILENAME,
    MATCHER_SOURCE_SHA256,
    MATCHER_SOURCE_URL,
    MATCHER_STATE_TENSORS,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_SHA256,
    SOURCE_FILENAMES,
    LightGlue,
    LightGluePipeline,
    audit_pickle,
    build_models,
    stage_missing_files,
    verify_snapshot,
)
from lightglue_pipeline import model as model_mod
from lightglue_pipeline.modeling import UPSTREAM_COMMIT

ROOT = Path(__file__).resolve().parents[1]


def test_pins_and_vendored_identity():
    assert MODEL_ID == "cvg/LightGlue" and MODEL_REVISION == "v0.1_arxiv" and len(UPSTREAM_COMMIT) == 40
    assert MATCHER_SOURCE_URL == f"https://github.com/cvg/LightGlue/releases/download/{MODEL_REVISION}/{MATCHER_SOURCE_FILENAME}"
    assert EXTRACTOR_SOURCE_URL == f"https://raw.githubusercontent.com/Shiaoming/ALIKED/{EXTRACTOR_COMMIT}/models/{EXTRACTOR_SOURCE_FILENAME}"
    assert "main" not in EXTRACTOR_SOURCE_URL and len(EXTRACTOR_COMMIT) == 40
    assert MATCHER_FILENAME == "aliked_lightglue.safetensors" and MODEL_SHA256.startswith("9c630a38")
    manifest = json.loads((ROOT / "weights" / DEFAULT_MODEL_KEY / MANIFEST_NAME).read_text(encoding="utf-8"))
    by_path = {e["path"]: e for e in manifest["files"]}
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    assert set(by_path) == set(SOURCE_FILENAMES)  # the manifest lists the two sources; the converted files are pinned in config
    assert by_path[MATCHER_SOURCE_FILENAME]["sha256"] == MATCHER_SOURCE_SHA256 and by_path[MATCHER_SOURCE_FILENAME]["sourceUrl"] == MATCHER_SOURCE_URL
    assert by_path[EXTRACTOR_SOURCE_FILENAME]["sha256"] == EXTRACTOR_SOURCE_SHA256 and by_path[EXTRACTOR_SOURCE_FILENAME]["sourceUrl"] == EXTRACTOR_SOURCE_URL
    assert manifest["upstreamCodeRevision"] == UPSTREAM_COMMIT
    for name in by_path:
        assert by_path[name]["convertsTo"]["path"] in CONVERTED_FILENAMES


def test_vendored_networks_shape():
    extractor, matcher = build_models()
    assert isinstance(extractor, ALIKED) and isinstance(matcher, LightGlue)
    assert sum(p.numel() for p in extractor.parameters()) == EXTRACTOR_PARAMETER_COUNT
    assert sum(p.numel() for p in matcher.parameters()) == MATCHER_PARAMETER_COUNT
    assert len(extractor.state_dict()) == EXTRACTOR_STATE_TENSORS and len(matcher.state_dict()) == MATCHER_STATE_TENSORS
    assert "confidence_thresholds" not in matcher.state_dict()  # non-persistent: absent from the checkpoint, so the load is strict
    assert len(matcher.transformers) == MATCHER_LAYERS and matcher.conf.depth_confidence < 0 and matcher.conf.width_confidence < 0
    assert extractor.dkd.n_limit == 2048 and extractor.dkd.scores_th == 0.2


def test_random_init_forward_returns_the_match_keys():
    torch.manual_seed(0)
    extractor, matcher = build_models()
    extractor.eval()
    matcher.eval()
    image = torch.rand(1, 3, 96, 128)
    with torch.inference_mode():
        feats = extractor({"image": image})
        assert set(feats) == {"keypoints", "descriptors", "keypoint_scores"} and feats["descriptors"].shape[-1] == 128
        feats["image_size"] = torch.tensor([[128.0, 96.0]])
        out = matcher({"image0": feats, "image1": feats})
    assert out["stop"] == MATCHER_LAYERS and out["matches"][0].shape[1] == 2
    assert out["matching_scores0"].shape == (1, feats["keypoints"].shape[1])


def test_ground_truth_and_assignment_loss():
    kpts0 = np.array([[10.0, 10.0], [50.0, 20.0], [90.0, 60.0], [30.0, 80.0]])
    shift = np.array([[1.0, 0.0, 4.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kpts1 = np.array([[14.0, 10.0], [54.0, 21.0], [200.0, 200.0]])  # third keypoint of image0 has no partner; third of image1 has none
    gt = LightGluePipeline.match_ground_truth(kpts0, kpts1, shift)
    assert gt["positives"].tolist() == [[0, 0], [1, 1]]
    assert gt["unmatched0"].tolist() == [False, False, True, True] and gt["unmatched1"].tolist() == [False, False, True]
    scores = torch.full((1, 5, 4), -3.0)
    loss = LightGluePipeline.assignment_loss(scores, gt)
    assert torch.isclose(loss, torch.tensor(3.0 + 0.5 * (3.0 + 3.0)))
    peaked = scores.clone()
    peaked[0, 0, 0] = peaked[0, 1, 1] = 0.0
    peaked[0, 2, 3] = peaked[0, 3, 3] = peaked[0, 4, 2] = 0.0
    assert LightGluePipeline.assignment_loss(peaked, gt) < loss
    empty = LightGluePipeline.match_ground_truth(np.zeros((0, 2)), kpts1, shift)
    assert len(empty["positives"]) == 0 and empty["unmatched1"].all()


def test_trainable_scope_counts():
    _extractor, matcher = build_models()
    pipe = LightGluePipeline(_extractor, matcher, device="cpu")
    two = pipe._trainable_names(2)
    assert all(n.startswith(("transformers.7.", "transformers.8.", "log_assignment.8.")) for n in two)
    counts = dict(pipe.matcher.named_parameters())
    assert sum(counts[n].numel() for n in two) == 2_567_169
    assert sum(counts[n].numel() for n in pipe._trainable_names(1)) == 1_250_560 + 66_049
    with pytest.raises(ValueError, match="1..9"):
        pipe._trainable_names(10)
    with pytest.raises(ValueError, match="1..9"):
        pipe._trainable_names(True)


# --- pickle audit ------------------------------------------------------------------------------------


def test_audit_pickle_lists_globals_and_refuses_outsiders(tmp_path):
    state = {"w": torch.zeros(2)}
    torch.save(state, tmp_path / "ok.pth")
    summary = audit_pickle(tmp_path / "ok.pth")
    assert summary["torch_archive"] and set(summary["globals"]) <= CKPT_ALLOWED_GLOBALS and summary["violations"] == []
    assert len(summary["audit_sha256"]) == 64
    (tmp_path / "bad.pkl").write_bytes(pickle.dumps({"f": print}))
    with pytest.raises(ValueError, match="outside the allow-list"):
        audit_pickle(tmp_path / "bad.pkl")
    assert "builtins.print" in audit_pickle(tmp_path / "bad.pkl", allowed=frozenset({"builtins.print"}))["globals"]


# --- snapshot verification with stand-ins ----------------------------------------------------------


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    payloads = {MATCHER_SOURCE_FILENAME: b"stand-in-matcher-pickle", EXTRACTOR_SOURCE_FILENAME: b"stand-in-extractor-pickle"}
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)
    manifest = {
        "format": "dimer_release_snapshot",
        "formatVersion": 1,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": n, "bytes": len(p), "sha256": hashlib.sha256(p).hexdigest()} for n, p in payloads.items()],
        "totalBytes": sum(len(p) for p in payloads.values()),
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    converted = {MATCHER_FILENAME: b"stand-in-matcher-safetensors", EXTRACTOR_FILENAME: b"stand-in-extractor-safetensors"}
    monkeypatch.setattr(model_mod, "CONVERTED", {n: (hashlib.sha256(p).hexdigest(), len(p), 0) for n, p in converted.items()})
    return tmp_path, converted


def test_verify_snapshot_before_and_after_conversion(stand_in):
    root, converted = stand_in
    result = verify_snapshot(root)
    assert result["modelId"] == MODEL_ID and len(result["files"]) == 2 and result["converted"] is False
    for name, payload in converted.items():
        (root / name).write_bytes(payload)
    assert verify_snapshot(root)["converted"] is True
    (root / "extra.bin").write_bytes(b"pickle")
    with pytest.raises(RuntimeError, match="unsafe"):
        verify_snapshot(root)
    (root / "extra.bin").unlink()
    (root / MATCHER_SOURCE_FILENAME).unlink()  # converted-only shape: the sources are not required
    assert verify_snapshot(root)["converted"] is True
    (root / EXTRACTOR_FILENAME).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256|size"):
        verify_snapshot(root)


def test_stage_fetches_only_the_absent_sources(stand_in):
    root, converted = stand_in
    (root / MATCHER_SOURCE_FILENAME).unlink()
    calls = []

    def downloader(relative_path, dest):
        calls.append(relative_path)
        (dest / relative_path).write_bytes(b"stand-in-matcher-pickle")

    with pytest.raises(FileNotFoundError, match="allow_download"):
        stage_missing_files(root)
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [MATCHER_SOURCE_FILENAME]
    assert calls == [MATCHER_SOURCE_FILENAME] and stage_missing_files(root, allow_download=True, downloader=downloader) == []
    verify_snapshot(root)
    # once a converted file exists its source is no longer staged
    (root / MATCHER_SOURCE_FILENAME).unlink()
    (root / MATCHER_FILENAME).write_bytes(converted[MATCHER_FILENAME])
    assert stage_missing_files(root, allow_download=True, downloader=downloader) == [] and calls == [MATCHER_SOURCE_FILENAME]
    (root / MANIFEST_NAME).write_text(json.dumps({"modelId": "other/x", "revision": MODEL_REVISION, "files": [{"path": "a"}]}))
    with pytest.raises(ValueError, match="refusing"):
        stage_missing_files(root, allow_download=True, downloader=downloader)


def test_release_download_refuses_unknown_entries(tmp_path):
    with pytest.raises(ValueError, match="not a downloadable"):
        model_mod._release_download("something-else.pth", tmp_path)


def test_convert_sources_rejects_a_tampered_source(tmp_path):
    _extractor, matcher = build_models()
    torch.save(matcher.state_dict(), tmp_path / MATCHER_SOURCE_FILENAME)  # right shape, wrong bytes
    with pytest.raises(ValueError, match="size|sha256"):
        model_mod._check_pinned_source(tmp_path, MATCHER_SOURCE_FILENAME)
    with pytest.raises(FileNotFoundError):
        model_mod._check_pinned_source(tmp_path, EXTRACTOR_SOURCE_FILENAME)

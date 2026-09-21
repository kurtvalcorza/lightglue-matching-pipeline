# ruff: noqa: E501
"""Regressions against the real pinned checkpoints: skipped unless the converted snapshot is staged under
weights/lightglue-aliked/. The build ran these CPU-only; the CUDA variant runs only where a device is visible."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import textured_image
from lightglue_pipeline import (
    DEFAULT_MODEL_KEY,
    EXTRACTOR_FILENAME,
    EXTRACTOR_PARAMETER_COUNT,
    MATCHER_FILENAME,
    MATCHER_PARAMETER_COUNT,
    MATCHER_STATE_TENSORS,
    LightGluePipeline,
    make_pair,
    make_pairs,
    reprojection_errors,
)

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / DEFAULT_MODEL_KEY
pytestmark = pytest.mark.skipif(
    not ((WEIGHTS / MATCHER_FILENAME).is_file() and (WEIGHTS / EXTRACTOR_FILENAME).is_file()),
    reason="pinned checkpoints not staged",
)


@pytest.fixture(scope="module")
def pipe():
    return LightGluePipeline.from_pretrained(weights_dir=WEIGHTS, device="cpu")


def test_strict_load_and_identity(pipe):
    assert pipe.manifest_verified and pipe.checkpoint_source == "explicit_path"
    assert sum(p.numel() for p in pipe.matcher.parameters()) == MATCHER_PARAMETER_COUNT
    assert sum(p.numel() for p in pipe.extractor.parameters()) == EXTRACTOR_PARAMETER_COUNT
    assert len(pipe.matcher.state_dict()) == MATCHER_STATE_TENSORS
    assert not any(p.requires_grad for p in pipe.matcher.parameters())
    assert pipe.conversion is None  # already converted on disk: nothing was unpickled by this load


def test_frozen_matcher_is_accurate_on_a_warped_photograph(pipe):
    pair = make_pair(textured_image(11, (800, 600)), seed=3, tier="easy", record_id="p")
    result = pipe.match(pair["image0"], pair["image1"])
    # ALIKED finds a few dozen keypoints on the synthetic texture (44 / 56 on the build workstation), not the
    # hundreds a dense matcher returns: the count floor is sparse-matcher sized, the precision floor is not.
    assert len(result["kpts0"]) > 20 and result["size0"] == [640, 480] and result["layers"] == 9
    errors = reprojection_errors(result["kpts0"], result["kpts1"], np.asarray(pair["homography"]))
    assert float((errors < 3.0).mean()) > 0.9 and float(np.median(errors)) < 1.0
    easy = pipe.evaluate([pair, make_pair(textured_image(12, (800, 600)), seed=4, tier="easy", record_id="q")])
    assert easy["n"] == 2 and easy["precision_3px"] > 0.7 and easy["homography_acc_5px"] > 0.0
    # The hard tier rotates up to 150 degrees, which neither ALIKED's descriptors nor the matcher's positional
    # encoding survive: a rotated pair must score below the easy pair, and that gap is what `adapt` works on.
    hard = pipe.evaluate([make_pair(textured_image(12, (800, 600)), seed=4, tier="hard", record_id="r")])
    assert hard["precision_3px"] < easy["precision_3px"]
    nn = pipe.descriptor_nn_match(pair["image0"], pair["image1"])
    nn_errors = reprojection_errors(nn["kpts0"], nn["kpts1"], np.asarray(pair["homography"]))
    assert len(nn["kpts0"]) > 20 and float((nn_errors < 3.0).mean()) < float((errors < 3.0).mean()) + 0.05


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, tmp_path):
    records = make_pairs([{"id": f"r{i}", "image": textured_image(20 + i, (480, 360))} for i in range(6)], tier="easy")
    before = {k: v.clone() for k, v in pipe.matcher.state_dict().items()}
    result = pipe.adapt(records[:4], records[4:], epochs=1, lr=1e-5, batch_size=2, trainable_layers=1)
    assert result["n_trainable"] == 1_250_560 + 66_049 and result["history"][0]["val"]["precision_3px"] > 0.5
    assert result["ground_truth"]["positive_pairs"] > 100
    assert all(name.startswith(("transformers.8.", "log_assignment.8.")) for name in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "model-backed"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"]["best_epoch"] == result["best_epoch"] and len(manifest["tensors"]) == 26
    reloaded = LightGluePipeline.from_artifact(artifact, weights_dir=WEIGHTS, device="cpu")
    a = pipe.match(records[0]["image0"], records[0]["image1"])
    b = reloaded.match(records[0]["image0"], records[0]["image1"])
    assert len(a["kpts0"]) == len(b["kpts0"]) and np.allclose(a["kpts1"], b["kpts1"], atol=1e-4)
    for k, v in pipe.matcher.state_dict().items():  # restore the shared fixture's base weights
        if not torch.equal(v, before[k]):
            v.copy_(before[k])
    pipe.adapter = None


def test_transactional_restore_on_failure(pipe):
    records = make_pairs([{"id": f"r{i}", "image": textured_image(30 + i, (480, 360))} for i in range(4)], tier="easy")
    before = {k: v.clone() for k, v in pipe.matcher.state_dict().items()}
    bad = [*records, {**records[0], "id": "bad", "homography": [[1, 0, 0], [1, 0, 0], [0, 0, 1]]}]
    with pytest.raises(ValueError, match="singular"):
        pipe.adapt(bad, epochs=1)
    assert all(torch.equal(v, before[k]) for k, v in pipe.matcher.state_dict().items()) and pipe.adapter is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_cuda_path_matches_cpu(pipe):
    pair = make_pair(textured_image(11, (800, 600)), seed=3, tier="easy", record_id="p")
    cuda = LightGluePipeline.from_pretrained(weights_dir=WEIGHTS, device="cuda")
    a, b = pipe.match(pair["image0"], pair["image1"]), cuda.match(pair["image0"], pair["image1"])
    assert abs(len(a["kpts0"]) - len(b["kpts0"])) < 0.05 * max(1, len(a["kpts0"]))

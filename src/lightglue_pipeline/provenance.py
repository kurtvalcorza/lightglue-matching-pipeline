from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import (
    DEPTH_CONFIDENCE,
    DETECTION_THRESHOLD,
    EXTRACTOR_COMMIT,
    EXTRACTOR_FILENAME,
    EXTRACTOR_REPOSITORY,
    EXTRACTOR_SHA256,
    EXTRACTOR_SIZE_BYTES,
    FILTER_THRESHOLD,
    MATCHER_FILENAME,
    MATCHER_SHA256,
    MATCHER_SIZE_BYTES,
    MAX_KEYPOINTS,
    MAX_SIDE,
    MIN_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    MODEL_REVISION_KIND,
    NMS_RADIUS,
    WIDTH_CONFIDENCE,
)
from .modeling import UPSTREAM_COMMIT, UPSTREAM_REPOSITORY

_RUNTIME_PACKAGES = (
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "torchvision",
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def build_provenance(
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> dict[str, Any]:
    checkpoint_source = None
    manifest_verified = False
    weight_sha256 = MATCHER_SHA256
    weight_size = MATCHER_SIZE_BYTES
    extractor_sha256 = EXTRACTOR_SHA256
    device_str = None
    resolved_checkpoint_path = None
    adapter = None

    if pipeline is not None:
        if getattr(pipeline, "checkpoint_path", None) is not None:
            resolved_checkpoint_path = str(pipeline.checkpoint_path)
        checkpoint_source = getattr(pipeline, "checkpoint_source", None)
        manifest_verified = bool(getattr(pipeline, "manifest_verified", False))
        if getattr(pipeline, "weight_sha256", None):
            weight_sha256 = pipeline.weight_sha256
        if getattr(pipeline, "weight_size_bytes", None):
            weight_size = pipeline.weight_size_bytes
        if getattr(pipeline, "extractor_sha256", None):
            extractor_sha256 = pipeline.extractor_sha256
        if getattr(pipeline, "device", None) is not None:
            device_str = str(pipeline.device)
        if getattr(pipeline, "adapter", None):
            skip = ("history", "trainable_names")
            adapter = {k: v for k, v in pipeline.adapter.items() if k not in skip}
    if checkpoint_path is not None:
        resolved_checkpoint_path = str(checkpoint_path)

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "revision_kind": MODEL_REVISION_KIND,
            "weight_file": MATCHER_FILENAME,
            "weight_sha256": weight_sha256,
            "weight_size_bytes": weight_size,
            "weight_format": "safetensors converted once from the audited release pickle",
            "extractor": {
                "repository": EXTRACTOR_REPOSITORY,
                "commit": EXTRACTOR_COMMIT,
                "weight_file": EXTRACTOR_FILENAME,
                "weight_sha256": extractor_sha256,
                "weight_size_bytes": EXTRACTOR_SIZE_BYTES,
            },
            "checkpoint_source": checkpoint_source,
            "checkpoint_path": resolved_checkpoint_path,
            "manifest_verified": manifest_verified,
            "vendored_code": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
        },
        "preprocessing": {
            "rgb": True,
            "resize": "none (the sample pairs are built at 640 px on the long side)",
            "side_range_px": [MIN_SIDE, MAX_SIDE],
            "padding": "inside ALIKED to a multiple of 32; keypoints reported in the input frame",
        },
        "inference": {
            "max_keypoints": MAX_KEYPOINTS,
            "detection_threshold": DETECTION_THRESHOLD,
            "nms_radius": NMS_RADIUS,
            "match_threshold": FILTER_THRESHOLD,
            "depth_confidence": DEPTH_CONFIDENCE,
            "width_confidence": WIDTH_CONFIDENCE,
            "match_confidence_semantics": "assignment_scores_not_calibrated_probability",
            "geometric_verification": (
                "none in the pipeline; the evaluation's RANSAC-DLT is a metric, not a filter"
            ),
            "device": device_str,
        },
        "adapter": adapter,
    }
    if include_runtime:
        provenance["runtime"] = {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": sys.platform,
            "packages": {name: _package_version(name) for name in _RUNTIME_PACKAGES},
        }
    return provenance


def write_provenance(
    path: str | Path,
    *,
    pipeline: Any | None = None,
    checkpoint_path: str | Path | None = None,
    include_runtime: bool = True,
) -> Path:
    record = build_provenance(
        pipeline=pipeline, checkpoint_path=checkpoint_path, include_runtime=include_runtime
    )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return out

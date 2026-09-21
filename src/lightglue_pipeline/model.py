from __future__ import annotations

import hashlib
import io
import json
import os
import pickletools
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .config import (
    CKPT_ALLOWED_GLOBALS,
    CONVERTED_FILENAMES,
    DEFAULT_MODEL_KEY,
    DEPTH_CONFIDENCE,
    DETECTION_THRESHOLD,
    EXTRACTOR_FILENAME,
    EXTRACTOR_PARAMETER_COUNT,
    EXTRACTOR_PICKLE_AUDIT_SHA256,
    EXTRACTOR_SHA256,
    EXTRACTOR_SIZE_BYTES,
    EXTRACTOR_SOURCE_FILENAME,
    EXTRACTOR_SOURCE_SHA256,
    EXTRACTOR_SOURCE_SIZE_BYTES,
    EXTRACTOR_SOURCE_URL,
    EXTRACTOR_STATE_TENSORS,
    FILTER_THRESHOLD,
    MATCHER_FILENAME,
    MATCHER_PARAMETER_COUNT,
    MATCHER_PICKLE_AUDIT_SHA256,
    MATCHER_SHA256,
    MATCHER_SIZE_BYTES,
    MATCHER_SOURCE_FILENAME,
    MATCHER_SOURCE_SHA256,
    MATCHER_SOURCE_SIZE_BYTES,
    MATCHER_SOURCE_URL,
    MATCHER_STATE_TENSORS,
    MAX_KEYPOINTS,
    MODEL_ID,
    MODEL_REVISION,
    NMS_RADIUS,
    SOURCE_FILENAMES,
    UNSAFE_WEIGHT_EXTENSIONS,
    WIDTH_CONFIDENCE,
)

MANIFEST_NAME = "dimer-base-manifest.json"
#: Fleet snapshot scheme (DIMER NOTEBOOK_SPEC 1.1 MOD13): the pinned files live in a repository-
#: local snapshot directory named by the model key and described by the committed manifest; a
#: standalone notebook carries that manifest inline and stages/verifies a working-directory copy.
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / DEFAULT_MODEL_KEY

#: The two pinned sources (name -> (url, sha256, bytes, pickle-audit digest)) and the two converted
#: files they become (name -> (sha256, bytes, state tensors)).
SOURCES: dict[str, tuple[str, str, int, str]] = {
    MATCHER_SOURCE_FILENAME: (
        MATCHER_SOURCE_URL,
        MATCHER_SOURCE_SHA256,
        MATCHER_SOURCE_SIZE_BYTES,
        MATCHER_PICKLE_AUDIT_SHA256,
    ),
    EXTRACTOR_SOURCE_FILENAME: (
        EXTRACTOR_SOURCE_URL,
        EXTRACTOR_SOURCE_SHA256,
        EXTRACTOR_SOURCE_SIZE_BYTES,
        EXTRACTOR_PICKLE_AUDIT_SHA256,
    ),
}
CONVERTED: dict[str, tuple[str, int, int]] = {
    MATCHER_FILENAME: (MATCHER_SHA256, MATCHER_SIZE_BYTES, MATCHER_STATE_TENSORS),
    EXTRACTOR_FILENAME: (EXTRACTOR_SHA256, EXTRACTOR_SIZE_BYTES, EXTRACTOR_STATE_TENSORS),
}
SOURCE_OF: dict[str, str] = {
    MATCHER_FILENAME: MATCHER_SOURCE_FILENAME,
    EXTRACTOR_FILENAME: EXTRACTOR_SOURCE_FILENAME,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(
    snapshot_path: str | Path,
    *,
    require_configs: bool = False,
    return_manifest_verified: bool = False,
) -> Path | tuple[Path, bool]:
    """Assert the two converted safetensors files against their pinned byte counts and digests,
    refuse any weight file in an unsafe format other than the two pinned (digest-checked, never
    loaded) sources, and size/digest-check every manifest entry that is present."""
    root = Path(snapshot_path)
    if not root.is_dir():
        raise RuntimeError(f"Checkpoint directory does not exist: {root}")

    for name in CONVERTED_FILENAMES:
        if not (root / name).is_file():
            raise RuntimeError(
                f"Pinned checkpoint is missing {name}; run convert_sources() on the audited "
                f"{SOURCE_OF[name]} first"
            )

    unsafe = sorted(
        p.name
        for p in root.iterdir()
        if p.is_file()
        and p.suffix.lower() in UNSAFE_WEIGHT_EXTENSIONS
        and p.name not in SOURCE_FILENAMES
    )
    if unsafe:
        raise RuntimeError(f"Refusing unsafe weight files: {unsafe}")

    manifest_path = root / MANIFEST_NAME
    manifest_verified = False

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc

        files = manifest.get("files") or []
        if not files:
            raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")

        for entry in files:
            rel_path = entry.get("path")
            if not rel_path:
                continue
            target = root / rel_path
            if not target.is_file():
                if rel_path in SOURCE_FILENAMES:
                    continue  # converted-only (DIMER-hosted) shape: the sources are not required
                raise RuntimeError(f"Manifest file missing: {rel_path}")
            exp_bytes = entry.get("bytes")
            if exp_bytes is not None and target.stat().st_size != exp_bytes:
                raise RuntimeError(
                    f"Size mismatch for {rel_path}: {target.stat().st_size} != {exp_bytes}"
                )
            exp_sha = entry.get("sha256")
            if exp_sha is not None and _sha256(target) != exp_sha:
                raise RuntimeError(f"SHA-256 mismatch for {rel_path}")

        manifest_verified = True

    for name, (sha, size_bytes, _tensors) in CONVERTED.items():
        weight_path = root / name
        size = weight_path.stat().st_size
        if size != size_bytes:
            raise RuntimeError(f"Unexpected {name} size: {size}; expected {size_bytes}")
        digest = _sha256(weight_path)
        if digest != sha:
            raise RuntimeError(f"Unexpected {name} SHA-256: {digest}; expected {sha}")

    # The networks' configurations are carried in code (modeling.ALIKED.cfgs and
    # modeling.LightGlue.features); with require_configs there is nothing else to require.

    if return_manifest_verified:
        return root, manifest_verified
    return root


def _read_manifest(root: Path) -> dict[str, Any]:
    """Load and identity-check ``<root>/dimer-base-manifest.json``."""
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise RuntimeError(f"Corrupt manifest {MANIFEST_NAME}: {exc}") from exc
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing"
        )
    if not manifest.get("files"):
        raise RuntimeError(f"Manifest {MANIFEST_NAME} contains no files")
    return manifest


def _check_manifest_files(root: Path, manifest: dict[str, Any]) -> None:
    for entry in manifest["files"]:
        target = root / entry["path"]
        if not target.is_file():
            raise RuntimeError(f"Manifest file missing: {entry['path']}")
        if target.stat().st_size != entry.get("bytes"):
            raise RuntimeError(f"Size mismatch for {entry['path']}")
        if _sha256(target) != entry.get("sha256"):
            raise RuntimeError(f"SHA-256 mismatch for {entry['path']}")


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Manifest-driven verification of a fleet snapshot directory; raise on the first mismatch.

    The identity in the manifest must be the pinned one. Before conversion the two source pickles
    are size- and SHA-256-checked (never unpickled here); once both converted files exist,
    :func:`verify_checkpoint` asserts their pinned digests and byte counts and checks every manifest
    entry still present. Returns ``{"path": ..., **manifest, "converted": bool}``.
    """
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    converted = all((root / name).is_file() for name in CONVERTED_FILENAMES)
    if not converted:
        _check_manifest_files(root, manifest)
        return {"path": str(root), **manifest, "converted": False}
    _, manifest_verified = verify_checkpoint(
        root, require_configs=True, return_manifest_verified=True
    )
    if not manifest_verified:
        raise RuntimeError(f"manifest at {root} was not verified")  # pragma: no cover
    return {"path": str(root), **manifest, "converted": True}


def _release_download(relative_path: str, root: Path) -> None:
    """Fetch one pinned source from its immutable URL: the LightGlue release asset or the ALIKED
    file at the pinned commit (never a branch). The digest is re-checked by ``verify_snapshot`` and
    again by the static audit before anything is unpickled: a substituted file is caught first."""
    if relative_path not in SOURCES:
        raise ValueError(f"{relative_path} is not a downloadable manifest entry")
    url = SOURCES[relative_path][0]
    root.mkdir(parents=True, exist_ok=True)
    target = root / relative_path
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, open(tmp, "wb") as fh:  # noqa: S310
        while chunk := response.read(1 << 20):
            fh.write(chunk)
    tmp.replace(target)


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed sources that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). A source is not needed when its converted file is already present.
    Returns the relative paths fetched; :func:`verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    for converted_name, source_name in SOURCE_OF.items():
        if (root / converted_name).is_file() and source_name in missing:
            missing.remove(source_name)  # converted-only shape: the source pickle is not needed
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them from the {MODEL_REVISION} release assets"
        )
    fetch = downloader or _release_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _pickle_globals(data: bytes) -> dict[str, int]:
    """Every global a pickle stream would import, collected with `pickletools.genops` (no
    execution)."""
    found: dict[str, int] = {}
    stack: list[Any] = []
    for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
        if op.name == "GLOBAL":  # pickletools renders the (module, name) pair space-separated
            key = arg.replace("\n", " ").replace(" ", ".", 1)
            found[key] = found.get(key, 0) + 1
        elif op.name == "STACK_GLOBAL":
            key = f"{stack[-2]}.{stack[-1]}"
            found[key] = found.get(key, 0) + 1
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
            stack.append(arg)
        elif op.name in ("MEMOIZE", "BINPUT", "LONG_BINPUT", "PUT"):
            pass
        else:
            stack.append(None)
    return found


def audit_pickle(
    path: str | Path, *, allowed: frozenset[str] = CKPT_ALLOWED_GLOBALS
) -> dict[str, Any]:
    """Statically list the globals a pickle (plain, or inside a torch zip archive) would import and
    refuse any outside `allowed`. Executes nothing. Returns the sorted globals and their digest."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"file not found: {file_path}")
    data = file_path.read_bytes()
    found: dict[str, int] = {}
    nested = 0
    if data[:4] == b"PK\x03\x04":
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.endswith(".pkl"):
                nested += 1
                for key, count in _pickle_globals(archive.read(name)).items():
                    found[key] = found.get(key, 0) + count
    else:
        found = _pickle_globals(data)
    violations = sorted(name for name in found if name not in allowed)
    summary = {
        "file": file_path.name,
        "torch_archive": data[:4] == b"PK\x03\x04",
        "pickles": nested if nested else 1,
        "globals": sorted(found),
        "violations": violations,
        "audit_sha256": hashlib.sha256("\n".join(sorted(found)).encode("utf-8")).hexdigest(),
    }
    if violations:
        raise ValueError(
            f"{file_path.name}: pickle audit failed, globals outside the allow-list: {violations}"
        )
    return summary


def _check_pinned_source(root: Path, name: str) -> dict[str, Any]:
    _url, sha, size_bytes, audit_sha = SOURCES[name]
    source = root / name
    if not source.is_file():
        raise FileNotFoundError(f"source file not found: {source}")
    size = source.stat().st_size
    if size != size_bytes:
        raise ValueError(f"{name}: size {size} != pinned {size_bytes}")
    digest = _sha256(source)
    if digest != sha:
        raise ValueError(f"{name}: sha256 {digest} != pinned {sha}")
    audit = audit_pickle(source)
    if audit["audit_sha256"] != audit_sha:
        raise ValueError(
            f"{name}: pickle audit digest {audit['audit_sha256']} != pinned {audit_sha}"
        )
    return {"path": name, "bytes": size, "sha256": digest, "audit": audit}


def build_models(
    *,
    max_keypoints: int = MAX_KEYPOINTS,
    detection_threshold: float = DETECTION_THRESHOLD,
    filter_threshold: float = FILTER_THRESHOLD,
) -> tuple[Any, Any]:
    """The vendored ALIKED-N(16) extractor and the ALIKED-feature LightGlue matcher at random
    initialisation, in the inference configuration of this contract (nine layers, no pruning)."""
    from .modeling import ALIKED, LightGlue

    extractor = ALIKED(
        max_num_keypoints=max_keypoints,
        detection_threshold=detection_threshold,
        nms_radius=NMS_RADIUS,
    )
    matcher = LightGlue(
        features="aliked",
        filter_threshold=filter_threshold,
        depth_confidence=DEPTH_CONFIDENCE,
        width_confidence=WIDTH_CONFIDENCE,
    )
    return extractor, matcher


def convert_sources(path: str | Path | None = None) -> dict[str, Any]:
    """Convert the two pinned pickles into their safetensors files, deterministically, after size,
    digest and static-audit checks: torch's weights-only unpickler, a strict load into the vendored
    network, and the network's own state dict saved (sorted keys, contiguous tensors). Each pickle
    is unpickled exactly once, here. Converted files that already exist are left as they are."""
    from safetensors.torch import save_file

    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    started = time.perf_counter()
    extractor, matcher = build_models()
    report: dict[str, Any] = {"converted": [], "skipped": []}
    for converted_name, module in ((MATCHER_FILENAME, matcher), (EXTRACTOR_FILENAME, extractor)):
        target = root / converted_name
        sha, size_bytes, n_tensors = CONVERTED[converted_name]
        if target.is_file():
            if target.stat().st_size != size_bytes or _sha256(target) != sha:
                raise ValueError(
                    f"{converted_name}: existing file does not match its pinned digest"
                )
            report["skipped"].append(converted_name)
            continue
        source_name = SOURCE_OF[converted_name]
        source = _check_pinned_source(root, source_name)
        state = torch.load(root / source_name, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or any(
            not isinstance(v, torch.Tensor) for v in state.values()
        ):
            raise ValueError(f"{source_name} did not unpickle to a state dict of tensors")
        if len(state) != n_tensors:
            raise ValueError(
                f"{source_name}: state dict has {len(state)} tensors, expected {n_tensors}"
            )
        module.load_state_dict(state, strict=True)
        canonical = {k: v.contiguous() for k, v in module.state_dict().items()}
        if len(canonical) != n_tensors:
            raise ValueError(
                f"converted state dict has {len(canonical)} tensors; expected {n_tensors}"
            )
        save_file(canonical, str(target), metadata={"format": "pt"})
        size = target.stat().st_size
        digest = _sha256(target)
        if size != size_bytes or digest != sha:
            target.unlink()
            raise ValueError(
                f"{converted_name}: converted file {size} B / {digest} != pinned "
                f"{size_bytes} B / {sha}"
            )
        report["converted"].append(
            {
                "source": {k: v for k, v in source.items() if k != "audit"},
                "audit": source["audit"],
                "state_dict_tensors": len(state),
                "parameters": sum(p.numel() for p in module.parameters()),
                "converted": {"path": converted_name, "bytes": size, "sha256": digest},
            }
        )
    report["seconds"] = round(time.perf_counter() - started, 2)
    return report


def resolve_weights_path(
    weights_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[Path, str]:
    """Resolve weights path with precedence:

    1. Explicit argument `weights_path` -> 'explicit_path'
    2. Environment variable `LIGHTGLUE_WEIGHTS_DIR` -> 'env_var'
    3. Source checkout convention `weights/lightglue-aliked` -> 'repo_offline'
       (only if pyproject.toml exists at repo root and the directory holds the manifest)

    There is no Hub fallback: the checkpoints are GitHub-hosted files staged by
    :func:`stage_missing_files` into a manifest-described snapshot directory.
    """
    if weights_path is not None:
        return Path(weights_path), "explicit_path"

    env_dir = os.environ.get("LIGHTGLUE_WEIGHTS_DIR")
    if env_dir:
        return Path(env_dir), "env_var"

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "pyproject.toml").is_file():
        repo_weights = repo_root / "weights" / DEFAULT_MODEL_KEY
        if (repo_weights / MANIFEST_NAME).is_file():
            return repo_weights, "repo_offline"

    raise FileNotFoundError(
        "no weights directory: pass weights_path / weights_dir, set LIGHTGLUE_WEIGHTS_DIR, or run "
        f"from a checkout holding weights/{DEFAULT_MODEL_KEY}/{MANIFEST_NAME}"
    )


_resolve_weights_path = resolve_weights_path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_components(
    *,
    device: str | torch.device | None = None,
    cache_dir: str | Path | None = None,
    weights_path: str | Path | None = None,
    return_metadata: bool = False,
    max_keypoints: int = MAX_KEYPOINTS,
    detection_threshold: float = DETECTION_THRESHOLD,
    filter_threshold: float = FILTER_THRESHOLD,
) -> tuple[Any, Any, torch.device, Path] | tuple[Any, Any, torch.device, Path, dict[str, Any]]:
    """Verify and load the two converted checkpoints into the vendored networks (strict state-dict
    loads from safetensors; no pickle is opened here)."""
    from safetensors.torch import load_file

    candidate_path, source = resolve_weights_path(
        weights_path=weights_path,
        cache_dir=cache_dir,
    )

    verified, manifest_verified = verify_checkpoint(
        candidate_path,
        require_configs=True,
        return_manifest_verified=True,
    )
    target_device = _resolve_device(device)

    extractor, matcher = build_models(
        max_keypoints=max_keypoints,
        detection_threshold=detection_threshold,
        filter_threshold=filter_threshold,
    )
    loaded: dict[str, Any] = {}
    for name, module, n_tensors, n_params in (
        (EXTRACTOR_FILENAME, extractor, EXTRACTOR_STATE_TENSORS, EXTRACTOR_PARAMETER_COUNT),
        (MATCHER_FILENAME, matcher, MATCHER_STATE_TENSORS, MATCHER_PARAMETER_COUNT),
    ):
        state = load_file(str(verified / name))
        if len(state) != n_tensors:
            raise RuntimeError(f"{name}: {len(state)} tensors, expected {n_tensors}")
        module.load_state_dict(state, strict=True)
        count = sum(p.numel() for p in module.parameters())
        if count != n_params:
            raise RuntimeError(f"{name}: network has {count} parameters, expected {n_params}")
        module.eval().to(target_device)
        loaded[name] = {"state_tensors": len(state), "parameters": count}

    weight_file = verified / MATCHER_FILENAME
    metadata: dict[str, Any] = {
        "checkpoint_path": verified,
        "checkpoint_source": source,
        "manifest_verified": manifest_verified,
        "weight_sha256": _sha256(weight_file),
        "weight_size_bytes": weight_file.stat().st_size,
        "extractor_sha256": _sha256(verified / EXTRACTOR_FILENAME),
        "extractor_size_bytes": (verified / EXTRACTOR_FILENAME).stat().st_size,
        "device": str(target_device),
        "state_tensors": loaded[MATCHER_FILENAME]["state_tensors"],
        "parameters": loaded[MATCHER_FILENAME]["parameters"],
        "extractor_state_tensors": loaded[EXTRACTOR_FILENAME]["state_tensors"],
        "extractor_parameters": loaded[EXTRACTOR_FILENAME]["parameters"],
    }

    if return_metadata:
        return extractor, matcher, target_device, verified, metadata
    return extractor, matcher, target_device, verified

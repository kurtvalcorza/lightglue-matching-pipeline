# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_weights import (  # noqa: E402
    DEFAULT_DEST_DIR,
    DIMER_FILES,
    MANIFEST_FORMAT,
    MANIFEST_NAME,
    format_bytes,
    package_dimer_zip,
    stage_and_convert,
)

from lightglue_pipeline import model as model_mod  # noqa: E402
from lightglue_pipeline.config import (  # noqa: E402
    CONVERTED_FILENAMES,
    DEFAULT_MODEL_KEY,
    EXTRACTOR_FILENAME,
    EXTRACTOR_SOURCE_FILENAME,
    MATCHER_FILENAME,
    MATCHER_SOURCE_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_FILENAMES,
)


def test_format_bytes():
    assert format_bytes(512) == "512 B" and format_bytes(47_564_948) == "45.4 MB"


def test_committed_manifest_names_the_pinned_sources_only():
    manifest = json.loads((DEFAULT_DEST_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["format"] == MANIFEST_FORMAT and manifest["modelKey"] == DEFAULT_MODEL_KEY
    assert (manifest["modelId"], manifest["revision"]) == (MODEL_ID, MODEL_REVISION)
    assert {e["path"] for e in manifest["files"]} == set(SOURCE_FILENAMES)
    assert manifest["totalBytes"] == sum(e["bytes"] for e in manifest["files"])
    assert all(e["sourceUrl"].startswith("https://") and "/main/" not in e["sourceUrl"] for e in manifest["files"])
    assert DIMER_FILES == (*CONVERTED_FILENAMES, MANIFEST_NAME)


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A snapshot with stand-in bytes; the digests in `model` are patched so no real network is built."""
    sources = {MATCHER_SOURCE_FILENAME: b"matcher-pickle", EXTRACTOR_SOURCE_FILENAME: b"extractor-pickle"}
    converted = {MATCHER_FILENAME: b"matcher-safetensors", EXTRACTOR_FILENAME: b"extractor-safetensors"}
    manifest = {
        "format": MANIFEST_FORMAT,
        "formatVersion": 1,
        "modelKey": DEFAULT_MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": n, "bytes": len(p), "sha256": hashlib.sha256(p).hexdigest()} for n, p in sources.items()],
        "totalBytes": sum(len(p) for p in sources.values()),
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(model_mod, "CONVERTED", {n: (hashlib.sha256(p).hexdigest(), len(p), 0) for n, p in converted.items()})

    def fake_convert(path):
        for name, payload in converted.items():
            (Path(path) / name).write_bytes(payload)
        return {"converted": [], "skipped": [], "seconds": 0.0}

    monkeypatch.setattr("scripts.fetch_weights.convert_sources", fake_convert)
    return tmp_path, sources, converted


def test_stage_and_convert_fetches_verifies_converts(stand_in):
    root, sources, converted = stand_in
    calls = []

    def downloader(relative_path, dest):
        calls.append(relative_path)
        (dest / relative_path).write_bytes(sources[relative_path])

    with pytest.raises(FileNotFoundError, match="allow_download"):
        stage_and_convert(root, allow_download=False)
    report = stage_and_convert(root, allow_download=True, downloader=downloader)
    assert sorted(report["fetched"]) == sorted(sources) and sorted(calls) == sorted(sources)
    assert report["conversion"] is not None and report["snapshot"]["converted"] is True
    for name, payload in converted.items():
        assert (root / name).read_bytes() == payload
    again = stage_and_convert(root, allow_download=False)
    assert again["fetched"] == [] and again["conversion"] is None


def test_stage_and_convert_refuses_a_tampered_download(stand_in):
    root, sources, _converted = stand_in

    def tampered(relative_path, dest):
        (dest / relative_path).write_bytes(b"not the pinned bytes")

    with pytest.raises(RuntimeError, match="SHA-256|Size"):
        stage_and_convert(root, allow_download=True, downloader=tampered)
    assert not any((root / name).is_file() for name in CONVERTED_FILENAMES)


def test_package_dimer_zip_holds_the_converted_files_only(stand_in, tmp_path):
    root, sources, converted = stand_in
    for name, payload in {**sources, **converted}.items():
        (root / name).write_bytes(payload)
    out = tmp_path / "dimer.zip"
    names = package_dimer_zip(root, out)
    assert names == [f"{DEFAULT_MODEL_KEY}/{n}" for n in DIMER_FILES]
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == sorted(names)
        assert not any(n.endswith(".pth") for n in zf.namelist())
    (root / MATCHER_FILENAME).unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        package_dimer_zip(root, tmp_path / "again.zip")

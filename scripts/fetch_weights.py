#!/usr/bin/env python
"""Stage, verify and convert the pinned LightGlue + ALIKED checkpoints (a thin CLI over the package).

Default: fetch the two pinned source pickles that are absent from ``weights/lightglue-aliked/`` (the
LightGlue release asset and the ALIKED file at its pinned commit), size- and digest-check every manifest
entry, statically audit and convert each pickle once to safetensors (``convert_sources``), and verify the
converted files against their pinned digests. ``--verify-only`` skips the download; ``--zip`` packages the
converted-only snapshot (the two safetensors files plus the manifest) for a DIMER upload.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lightglue_pipeline.config import (  # noqa: E402
    CONVERTED_FILENAMES,
    DEFAULT_MODEL_KEY,
    MODEL_ID,
    MODEL_REVISION,
)
from lightglue_pipeline.model import (  # noqa: E402
    MANIFEST_NAME,
    convert_sources,
    stage_missing_files,
    verify_snapshot,
)

DEFAULT_DEST_DIR = ROOT / "weights" / DEFAULT_MODEL_KEY
MANIFEST_FORMAT = "dimer_release_snapshot"
MANIFEST_FORMAT_VERSION = 1
DIMER_FILES = (*CONVERTED_FILENAMES, MANIFEST_NAME)


def format_bytes(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover


def package_dimer_zip(model_dir: Path, zip_dest: Path) -> list[str]:
    """Package the converted-only snapshot (safetensors + manifest; never the source pickles) for DIMER."""
    names = []
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in DIMER_FILES:
            path = model_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"{name} is missing from {model_dir}; run the conversion first")
            arcname = f"{DEFAULT_MODEL_KEY}/{name}"
            zf.write(path, arcname=arcname)
            names.append(arcname)
    print(f"Archive created: {zip_dest} ({zip_dest.stat().st_size:,} bytes; {len(names)} files)")
    return names


def stage_and_convert(dest: Path, *, allow_download: bool, downloader=None) -> dict:
    """Stage absent sources, verify, convert when needed, verify again; return the final snapshot record."""
    fetched = stage_missing_files(dest, allow_download=allow_download, downloader=downloader)
    snapshot = verify_snapshot(dest)
    report = {"fetched": fetched, "conversion": None}
    if not snapshot["converted"]:
        report["conversion"] = convert_sources(dest)
        snapshot = verify_snapshot(dest)
    report["snapshot"] = snapshot
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST_DIR, help=f"snapshot directory (default: {DEFAULT_DEST_DIR})")
    parser.add_argument("--verify-only", action="store_true", help="verify (and convert if the sources are present) without downloading")
    parser.add_argument("--zip", type=Path, default=None, help="write a DIMER zip of the converted-only snapshot")
    args = parser.parse_args()
    dest: Path = args.dest
    if dest.name == "weights" and (dest / DEFAULT_MODEL_KEY).is_dir():
        dest = dest / DEFAULT_MODEL_KEY

    print(f"{MODEL_ID} @ {MODEL_REVISION} -> {dest}")
    try:
        report = stage_and_convert(dest, allow_download=not args.verify_only)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1
    snapshot = report["snapshot"]
    print(f"fetched: {report['fetched'] or 'nothing'}")
    if report["conversion"]:
        for entry in report["conversion"]["converted"]:
            print(f"converted {entry['source']['path']} -> {entry['converted']['path']} ({format_bytes(entry['converted']['bytes'])}; audit {entry['audit']['audit_sha256'][:12]})")
    print(f"OK: snapshot verified ({len(snapshot['files'])} manifest entries, converted={snapshot['converted']})")
    if args.zip:
        package_dimer_zip(dest, args.zip)
    print(json.dumps({"path": snapshot["path"], "modelId": snapshot["modelId"], "revision": snapshot["revision"], "converted": snapshot["converted"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

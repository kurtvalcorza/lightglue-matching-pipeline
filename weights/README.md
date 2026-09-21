# Base Model Weights Cache

This directory holds the offline checkpoints, their manifest and the run-time data cache for the LightGlue + ALIKED matching pipeline.

```
weights/
├── lightglue-aliked/
│   ├── dimer-base-manifest.json
│   ├── aliked_lightglue.pth          (excluded from Git; the pinned matcher source — a release asset of cvg/LightGlue v0.1_arxiv)
│   ├── aliked-n16.pth                (excluded from Git; the pinned extractor source — Shiaoming/ALIKED at commit 683d7c65)
│   ├── aliked_lightglue.safetensors  (excluded from Git; the audited conversion — the file to upload to DIMER)
│   └── aliked-n16.safetensors        (excluded from Git; the audited conversion — the file to upload to DIMER)
└── inat-birds/                       (excluded from Git; the 360 pinned photographs, fetched at run time)
```

## Available Base Model Snapshots

- [**`lightglue-aliked`**](lightglue-aliked/): Dedicated snapshot for LightGlue on ALIKED features (`cvg/LightGlue` release `v0.1_arxiv`, Apache-2.0; Lindenberger, Sarlin and Pollefeys, ICCV 2023) with the ALIKED-N(16) extractor (`Shiaoming/ALIKED` @ `683d7c65`, BSD-3-Clause; Zhao et al., IEEE TIM 2023). The matcher is 47,632,827 bytes as a pickle and 47,564,948 bytes as safetensors (253 tensors, 11,884,625 parameters); the extractor 2,738,091 / 2,719,928 bytes (76 tensors, 677,356 parameters).
  - [**Manifest**](lightglue-aliked/dimer-base-manifest.json): Cryptographic record of byte counts and SHA-256 hashes for the two source files, their pickle-audit digests, and the byte counts and SHA-256 hashes of the two conversions.

## DIMER Architecture & Git Tracking Strategy

1. **The sources** (`*.pth`, 50 MB) are excluded from Git and fetched once from the pinned release tag and commit (`stage_missing_files(..., allow_download=True)`); each is verified by size and digest, audited statically (only the fleet's four allowed pickle globals), unpickled exactly once with `torch.load(weights_only=True)`, loaded strictly, and exported as safetensors.
2. **The conversions** (`*.safetensors`, 50 MB) are excluded from Git (`weights/**/*.safetensors`) and are the only files the networks are ever loaded from — **upload these two to DIMER**, not the pickles. A snapshot that holds them never opens a pickle again, and `stage_missing_files` does not fetch a source whose conversion exists.
3. **The networks** are not downloaded: they are vendored as `src/lightglue_pipeline/modeling.py` from `cvg/LightGlue` @ `eb42fee2` with their inference configuration in code, so an offline container needs the package and the two safetensors files only.
4. **The manifest** is version-controlled so it can be asserted before any download.

## Management & Verification Tooling

```bash
# Fetch the two sources into weights/lightglue-aliked, audit, convert and verify them:
python scripts/fetch_weights.py

# Verify the existing snapshot (sources and conversions that are present):
python scripts/fetch_weights.py --verify-only
```

`LightGluePipeline.from_pretrained(weights_dir=..., allow_download=True)` performs the same staging, verification and one-time conversion itself before the strict load.

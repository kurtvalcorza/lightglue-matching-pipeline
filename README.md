# LightGlue + ALIKED Matching Pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/lightglue-matching-pipeline/blob/main/tutorials/lightglue_matching_colab.ipynb)

DIMER-oriented inference and bounded fine-tuning wrapper for **one immutable pair of open-weight checkpoints** — Lindenberger, Sarlin and Pollefeys' LightGlue sparse matcher (*LightGlue: Local Feature Matching at Light Speed*, ICCV 2023) on Zhao et al.'s ALIKED-N(16) keypoints and descriptors — with a homography-supervised evaluation contract and a bounded adaptation contract for labelled image pairs.

- model: `cvg/LightGlue` (GitHub; LightGlue publishes no Hugging Face repository) at the immutable release tag `v0.1_arxiv` (2023-06-26)
- matcher file: `aliked_lightglue.pth` — a release asset, `47,632,827` bytes, SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`; a plain PyTorch pickle, audited and converted once to `aliked_lightglue.safetensors` (`47,564,948` bytes, SHA-256 `9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6`, 253 tensors, 11,884,625 parameters)
- extractor file: `models/aliked-n16.pth` of `Shiaoming/ALIKED` @ `683d7c65197395c0b3f01ebe76e1084a27e73a65` — `2,738,091` bytes, SHA-256 `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2`; converted to `aliked-n16.safetensors` (`2,719,928` bytes, SHA-256 `3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a`, 76 tensors, 677,356 parameters)
- vendored code: `cvg/LightGlue` @ `eb42fee2d71449efb0aa5c10549752b5d75384d8` (the matcher, the authors' ALIKED port, the extractor base) as `src/lightglue_pipeline/modeling.py`
- upstream model licenses: Apache-2.0 (LightGlue code and weights), BSD-3-Clause (ALIKED port and weights)

The wrapper code in this repository is MIT licensed. The model weights and the vendored networks retain their upstream licenses.

## Status

**Candidate.** The inference contract, the homography-supervised evaluation, the adaptation contract, the pickle audit and conversion and the real pinned checkpoints have been exercised on the build workstation's CPU only (the unit and model-backed suites, and the default tutorial path through the package API — `MODEL_CARD.md` item 9). No GPU has run this repository yet and no clean hosted runtime has executed the committed notebook blob; both are what promotion to Release-grade requires (`docs/release-verification.md`). Production HTTP serving / DIMER worker packaging remains out of scope.

## Three things to know before you start

**The networks are carried, not installed.** `modeling.py` is the upstream LightGlue matcher and its ALIKED port at the pinned commit; the checkpoints are two GitHub files, each a plain PyTorch pickle. `from_pretrained` stages them, verifies their sizes and digests, audits their pickle streams statically (only the fleet's four allowed globals), unpickles each exactly once with `torch.load(weights_only=True)`, loads it strictly into the vendored network and saves a safetensors file that is itself digest-verified; every later load reads the safetensors files only. The upstream adaptive early exit and point pruning are switched off, so results do not depend on a confidence schedule.

**The labels are exact.** The tutorial's pairs are photographs and their own copies under seeded homographies with seeded re-lighting, so every returned match has a reprojection error against the reference and every reported precision is against ground truth, not a human judgement. Real pairs of a scene need depth, pose or a fitted homography before they can be scored.

**The frozen matcher fails on rotation, and that is the tutorial's task.** ALIKED descriptors and LightGlue's positional encoding are not rotation-invariant. The build record measured precision at 3 px of @P:FROZEN_P3@ over the 96 test pairs — @P:FROZEN_EASY_P3@ on the easy tier, **@P:FROZEN_HARD_P3@ on the tier rotated up to ±150°** — against @P:DNN_P3@ for the same keypoints matched by descriptor nearest neighbour, @P:PNN_P3@ for a patch nearest neighbour and @P:ID_P3@ for the identity guess. A bounded fine-tuning of the matcher's last two layers and assignment head on 216 pairs (half rotated) @P:GAIN_SHORT@; the adaptation is selected on validation precision and a run that keeps epoch 0 is a valid outcome the notebook anticipates.

## Quick start

```python
from lightglue_pipeline import LightGluePipeline, build_sample_dataset, fetch_corpus, ransac_homography, read_corpus

pipe = LightGluePipeline.from_pretrained(weights_dir="weights/lightglue-aliked", allow_download=True)  # stage + verify + audit + convert once, strict load
result = pipe.match("view_a.jpg", "view_b.jpg")                                        # kpts0, kpts1 (M, 2), confidence (M,)
homography, inliers = ransac_homography(result["kpts0"], result["kpts1"])              # the caller's verification (a helper, not part of match)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)                    # 216 / 48 / 96 homography pairs from 360 CC0 photographs
print(pipe.evaluate_baselines(splits["test"])["descriptor_nn"]["precision_3px"])       # the same keypoints without the learned matcher
print(pipe.evaluate(splits["test"])["precision_3px"])                                  # frozen
pipe.adapt(splits["train"], splits["validation"], epochs=@P:EPOCHS@, lr=@P:LR@, batch_size=4)  # bounded matcher fine-tuning, epoch selected on validation precision
print(pipe.evaluate(splits["test"])["precision_3px"])                                  # adapted, same pairs
pipe.save_artifact("outputs/adapter")
```

`match()` takes two images (local paths, bytes or PIL images; any mode, converted to RGB; sides in [64, 1024] px, passed at their own size) and returns correspondences between ALIKED keypoints in the input frames with confidences in (0, 1] — thresholded assignment scores, not probabilities of correctness, and no geometric verification. `extract()` returns one image's keypoints, descriptors and scores. `evaluate()` and `adapt()` take `{id, image0, image1, homography}` records with a finite, non-singular 3 × 3 reference mapping image0 pixels to image1 pixels. Validation is structural (sizes, decodability, id pattern, homography rank), never semantic: a wrong reference is scored without complaint.

## Adaptation contract

- `validate_dataset(records)` checks the record shape, the image sizes, the id pattern and uniqueness and the homography's shape and rank (4..5,000 records) and returns a manifest with a dataset digest; `make_pair` / `make_pairs` synthesise pairs from photographs (tiers `easy` / `hard` — the hard tier rotates up to ±150°); `build_sample_dataset` splits the pinned corpus by photograph, stratified per species; `split_dataset` does the same for BYOD photographs after pixel-digest de-duplication; `check_split_disjoint` asserts no photograph is shared; `observer_overlap` reports observers in more than one split.
- `evaluate(records)` matches every pair and scores it with `metrics.pair_metrics` / `matching_metrics`, per tier and per pair; `evaluate_baselines(records)` scores the identity guess, the patch nearest neighbour and the descriptor nearest neighbour (the pipeline's own ALIKED keypoints matched by mutual nearest neighbour, no learned matcher) with the same code.
- `adapt(train, val=None, *, epochs=3, lr=1e-4, batch_size=4, trainable_layers=2, seed=0, progress=None)` caches the frozen extractor's features once, then trains only the last `trainable_layers` of LightGlue's nine transformer layers and the final assignment head (2,567,169 of 11,884,625 parameters by default) with the LightGlue assignment loss against the homography's ground-truth assignment (pairs under 3 px positive, keypoints with no partner within 5 px unmatched); AdamW, pairs accumulated per step, gradient clipping at 1.0, seeded order, no scheduler; the epoch with the highest validation precision at 3 px is kept (ties: homography accuracy); transactional restore on any failure.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` (about @P:ADAPTER_MB@ MB) plus a `manifest.json` (format `org.valcorza.lightglue.adapter.v1`: base id, revision and the two weight digests, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the matcher, and overlays the tensors onto a freshly loaded base.

## Live tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/lightglue-matching-pipeline/blob/main/tutorials/lightglue_matching_colab.ipynb)

`tutorials/lightglue_matching_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's seven modules (the vendored networks among them), the model identity (`cvg/LightGlue` at the immutable release tag `v0.1_arxiv`, the ALIKED file at its pinned commit), the manifest with both source and converted digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and the validator). It writes:

- `lightglue_matching_train.csv`
- `lightglue_matching_input_manifest.json`
- `lightglue_matching_evaluation_report.json`
- `lightglue_matching_shapes.json`
- `lightglue_matching_adapter/` (`adapter.safetensors`, `manifest.json`)
- `lightglue_matching_result.json`
- `provenance.json`

The default path runs on CPU and uses CUDA automatically when present (about @P:CPU_TOTAL_MIN@ minutes on the build workstation's CPU after the downloads; a hosted T4 finishes in minutes). The metrics it prints are one seeded split of one 360-pair sample under synthetic warps — evidence that the adaptation contract works, not a matching benchmark or production-fitness evidence. The `main` integration workflow runs the real-checkpoint tests and the notebook against the pinned files.

## Release status

**Candidate** — not yet executed in a clean hosted runtime. The `E2E` notebook is generated, parity-checked and unit-tested, and the default path has been run on the build workstation's CPU through the package API, but static and unit checks — including the standalone generator parity checks — are necessary, not the evidence; the hosted run is. The status becomes Release-grade when a clean Kaggle / Colab execution of the committed notebook blob is recorded in `docs/release-verification.md`; a later change to the carried modules or the notebook returns it to Candidate.

## Weights layout

```
weights/lightglue-aliked/  aliked_lightglue.pth, aliked-n16.pth               (git-ignored; the pinned source pickles, fetched once)
                           aliked_lightglue.safetensors, aliked-n16.safetensors  (git-ignored; the audited conversions — the files to upload to DIMER)
                           dimer-base-manifest.json
weights/inat-birds/        <photo id>.jpg × 360                             (git-ignored; the pinned photographs fetched at run time)
```

`from_pretrained(weights_dir=...)` calls `stage_missing_files()` (fetches only absent manifest entries, only from the pinned release tag and commit, only with `allow_download=True`; a source is not needed once its conversion exists), `verify_snapshot()` (byte size + SHA-256 of every present manifest entry), `convert_sources()` when a conversion is absent (static audit, one weights-only unpickle, strict load, safetensors export, digest check), and then loads the safetensors files strictly into the vendored networks, refusing on the first mismatch. `docs/WEIGHTS.md` records the provenance, the vendored code, the data pins and the DIMER hosting notes.

## Sample data

`fetch_corpus()` fetches the 360 pinned photographs (about 39 MB) from the iNaturalist open-data bucket, each verified by byte size and SHA-256 and cached under `weights/inat-birds/`; `read_corpus` decodes them with their observation page, observer and species; `build_sample_dataset(seed=42)` draws 36 / 8 / 16 photographs per species into train / validation / test and turns each into one pair at 640 px on the long side with a seeded homography (tiers alternating `easy` and `hard`) and seeded photometric changes. `examples/sample-data/generate_samples.py` renders the drawn scene the tutorial matches through the inference contract (digest-pinned in `SHA256SUMS`).

## Reproducible reference environment

Python 3.12 is the supported runtime. The repository keeps exact direct pins in `pyproject.toml` (torch, torchvision, numpy, pillow, safetensors) and a fully version-pinned Linux/CPU reference graph in `requirements.lock.txt`.

```bash
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python scripts/check_lock.py
```

`requirements.lock.txt` records the exact dependency versions proven by the real-checkpoint `main` CI path, including the official CPU PyTorch wheel. It is a version lock, not a cryptographic hash lock.

## Tests

```bash
ruff check .
pytest -m "not integration"
```

Tests are offline and run on a CPU in about a minute: the vendored networks at random initialisation (shapes, the strict load with the non-persistent threshold buffer, the ground-truth assignment and loss, the trainable scope), the pickle audit against crafted streams, the metrics (DLT recovers a known homography, RANSAC ignores outliers, the baselines on a pure translation), the pair synthesis and validation, stand-in snapshots and adapter manifests; `tests/test_model_backed.py` runs the real checkpoints when they are staged (conversion, strict load, a warped pair matched, a short adaptation, artifact parity).

Real-checkpoint integration:

```bash
RUN_INTEGRATION=1 pytest -m integration -q
python tools/run_notebook.py tutorials/lightglue_matching_colab.ipynb
```

## Scope boundaries

This repository does **not** claim to provide:

- homography, pose or depth estimation as a product (the evaluation's RANSAC-DLT is a metric and a helper, not a verified solver);
- object detection, semantic segmentation, OCR or captioning;
- calibrated match probabilities or universal thresholds;
- geometric verification inside `match`;
- other extractors or LightGlue checkpoints (SuperPoint, DISK, SIFT), dense matching, or the upstream adaptive early exit and point pruning;
- fine-tuning of the extractor, the input projection, the positional encoding or the earlier matcher layers, or any adaptation beyond the matcher's last layers and assignment head;
- production HTTP serving or DIMER worker packaging.

Those require separate solvers, data, calibration, or serving work.

## AI Assistance Disclosure

This repository's code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.

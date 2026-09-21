# Release verification

`tutorials/lightglue_matching_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`config.py`, `modeling.py`, `model.py`, `metrics.py`, `samples.py`,
  `pipeline.py`, `provenance.py`, in dependency order), each equal to its source after the generator's documented
  rewrites (the `DEFAULT_WEIGHTS_DIR` rule, the `resolve_weights_path` checkout-convenience line, and the removal of
  package-relative imports); the inline `MANIFEST` equal to the committed 2-entry snapshot manifest (each source with
  its conversion) and the inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on
  LF) to `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cells (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision an immutable GitHub release tag, the ALIKED
  commit and the vendored LightGlue commit the only 40-hex revisions a document may cite, and the same identity string
  in `README.md` and `MODEL_CARD.md`;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `LightGluePipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, the digest-asserted synthetic scene and its `warp_image` partner, `validate_inputs` with the
  remote-URL refusal probe, `match` with the sanity checks, `reprojection_errors` and the per-pair
  `evaluation_report` on the drawn pair, `evaluate_baselines`, `pipe.evaluate` on the frozen matcher with the
  baseline assertion, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the validation and test
  splits after adaptation with the not-worse assertion, `match` and `evaluation_report` on the drawn pair after
  adaptation, `pipe.save_artifact`, `LightGluePipeline.from_artifact` and the reload-parity assertion,
  `write_provenance`, and the result fields `weight_file` / `weight_format` / `weight_sha256` / `extractor_file` /
  `extractor_sha256` / `vendored_code` and the `corpus` block), the seven expected `outputs/` paths, the
  learner-facing statements (Apache-2.0 weights, uncalibrated match confidences, adaptation with labelled pairs, the
  CC0 corpus, reprojection error, precision at 3 px, homography accuracy, the three baselines including the
  descriptor nearest neighbour, no dispersion estimate, the leakage and reference guidance, the excluded tasks, the
  snapshot note) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` /
  repository import on the primary path, a mutable branch revision, direct `from huggingface_hub import` /
  `snapshot_download` / `safetensors` imports / `urllib.request` / `torch.optim` / `.backward(` / `requires_grad` /
  `pipe.model` / `pipe.matcher` / `pipe.extractor` / `log_assignment` / `extractall(` use **outside the carried
  module cells**, `trust_remote_code=True`, `pickle.load`, a `torch.load(` without `weights_only=True`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  checkpoint-invariants section.

CI also installs the frozen CPU reference environment (`requirements.lock.txt`), runs `ruff`, `scripts/check_lock.py`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/`, including `test_metrics.py`,
`test_samples.py`, `test_model.py`, `test_pipeline.py`, `test_fetch_weights.py`, `test_notebook_parity.py`; no
weights, injected downloader and photo fetcher — `tests/test_model_backed.py` is skipped without the snapshot). These
are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Repository CI integration job (`tools/run_notebook.py`, manual `workflow_dispatch` or push to `main`) | GitHub-hosted Ubuntu runner, the frozen CPU reference environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights; a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/lightglue-aliked/` or the photograph cache `weights/inat-birds/` (the standalone path writes the
   manifest itself, fetches the two source pickles from the release tag and the pinned commit, audits and converts
   them, and fetches the pinned photographs from the iNaturalist open-data bucket, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = @P:EPOCHS@`, `LEARNING_RATE = @P:LR@`, `BATCH_SIZE = 4`,
   `TRAINABLE_LAYERS = @P:LAYERS@`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0` (an interpreter restart after the install is expected where the runtime's preinstalled torch
   or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the seven carried module cells execute (defining `ALIKED`, `LightGlue`, `LightGluePipeline`,
     `verify_snapshot`, `stage_missing_files`, `audit_pickle`, `convert_sources`, `validate_inputs`,
     `evaluation_report`, `pair_metrics`, `matching_metrics`, `identity_baseline`, `patch_neighbour_baseline`,
     `mutual_nn_matches`, `fetch_corpus`, `read_corpus`, `build_sample_dataset`, `make_pair`, `warp_image`,
     `validate_dataset`, `check_split_disjoint`, `observer_overlap`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, `write_provenance` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting both sources fetched (the 48 MB `aliked_lightglue.pth` from the `v0.1_arxiv`
     release, the 3 MB `aliked-n16.pth` from the pinned ALIKED commit) on a clean runtime, `verify_snapshot`
     returning its dict, and `from_pretrained(weights_dir=WEIGHTS_DIR)` auditing and converting each source once
     (`checkpoint_source` reporting the conversion) and loading the two safetensors files strictly into the vendored
     networks (253 matcher tensors / 11,884,625 parameters; 76 extractor tensors / 677,356 parameters);
   - Section 4: `fetch_corpus` fetching the 360 pinned photographs with every byte count and SHA-256 matching; the
     seeded split into 216 / 48 / 96 (36 / 8 / 16 per species) and one pair per photograph at 640 px with tiers
     alternating (108 / 108, 24 / 24, 48 / 48; the hard tier's `rotation` 150.0), `check_split_disjoint` reporting
     no shared photograph, the observer overlap and the three dataset digests printed; `outputs/…_train.csv`
     written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_SIDE` 64, `MAX_SIDE` 1024, `DIVISIBLE_BY` 8, `MAX_IMAGE_SIDE` 4096,
     `MIN_RECORDS` 4, `MAX_RECORDS` 5000, at most 2,048 keypoints at detection threshold 0.2, match threshold 0.1)
     surfaced; the synthetic scene generated in code with SHA-256 `3420b1d3…` (equal to
     `examples/sample-data/SHA256SUMS`) and warped under `SHAPES_H`; `validate_inputs` writing
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the remote-URL
     probe); `match` on the drawn pair with every sanity check `True` and the per-pair `evaluation_report` verdict
     `sample-sanity` (the build record measured @P:SCENE_FROZEN@ frozen and @P:SCENE_ADAPTED@ adapted);
   - Section 6: the identity guess (precision at 3 px ≈ @P:ID_P3@), the patch nearest neighbour (≈ @P:PNN_P3@),
     the descriptor nearest neighbour (≈ @P:DNN_P3@) and the frozen matcher's test score (precision at 3 px ≈
     @P:FROZEN_P3@ — ≈ @P:FROZEN_EASY_P3@ easy / @P:FROZEN_HARD_P3@ hard —, ≈ @P:FROZEN_MPP@ matches per pair,
     homography accuracy at 3 px ≈ @P:FROZEN_HACC@ on the CPU build record) with the per-tier breakdown and the
     weakest pairs, and the cell's assertion that the frozen matcher is above the two non-neural baselines;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 2,567,169 trainable of 11,884,625 parameters,
     the ground-truth rule (3 px positive, 5 px negative) and an epoch history with the validation precision
     selecting the epoch (`best_epoch` @P:BEST_EPOCH@ in the build record — an epoch-0 result is a valid outcome);
   - Section 8: `pipe.evaluate` on the validation and test splits with the five-way comparison, the per-tier
     breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test precision at 3 px
     is not below the frozen one by more than 0.01 — @P:ADAPTED_P3@ versus @P:FROZEN_P3@ in the build record);
   - Section 9: the drawn pair re-matched by the adapted model with the `sample-sanity` report,
     `outputs/…_shapes.json` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors,manifest.json}` (@P:ADAPTER_TENSORS@ tensors, @P:ADAPTER_BYTES@ bytes)
     and `LightGluePipeline.from_artifact` reloading it with 4/4 identical match sets on four test pairs (the cell
     asserts it); `outputs/provenance.json` and `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`, `extractor_file`,
     `extractor_sha256`, `vendored_code`), the `corpus` block, the inference-contract reports, the comparison, the
     artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, device), the model identifier and
   immutable revision, whether the model cache, the weights directory and the photograph cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `lightglue_matching_colab.ipynb` (`E2E`) | — | — | — | **not yet executed** in a clean supported runtime; the first clean-room execution is queued on the workspace's Kaggle serial suite and will be recorded here |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/lightglue_matching_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/lightglue_matching_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | package API, not the notebook (source at the revision that generated the first committed blob) | Build workstation CPU (`CUDA_VISIBLE_DEVICES=-1`, Python 3.12, torch 2.14.0; snapshot converted and the 360 photographs pre-staged) | The notebook's default path replayed cell by cell through the package API (`build_sample_dataset(seed=42)` → 216 / 48 / 96 pairs, `validate_dataset` per split, `check_split_disjoint`, `evaluate_baselines`, `pipe.evaluate` frozen, `pipe.adapt` at the defaults, `pipe.evaluate` adapted, `save_artifact`, `from_artifact` with match parity): @P:BUILD_RECORD@ | @P:CPU_TOTAL_S@ s | PASS — pre-flight only |
| 2026-09-20 | package API at the working tree of 2026-09-20 (the XoFTR row's ±35° hard tier, before this row's ±150° tier) | Build workstation CPU | The same replay with the earlier tier: frozen precision at 3 px 0.981 (homography accuracy 0.969, 720 matches per pair), descriptor nearest neighbour 0.884, patch neighbour 0.381, identity 0.007; `adapt(epochs=3, lr=1e-4, trainable_layers=2)` kept epoch 0 (validation 0.981 → 0.976 / 0.976 / 0.979) so adapted = frozen — the finding that made this row rotate its hard tier | 2061.1 s | PASS — pre-flight only; superseded tier |
| 2026-09-20 | vendored networks and conversion smoke (source before the package existed) | Build workstation CPU | both pickles audited (globals = the fleet's set), converted and loaded strictly into the stitched `modeling.py`; one photograph matched against its warp; a training forward + backward through the matcher | — | PASS — pre-flight only |

## Current status

**Candidate.** No clean-runtime execution of the committed `E2E` notebook blob has been recorded yet. The build
workstation's CPU pre-flight above shows the default path completing through the package API with the numbers the
notebook prose quotes; it is not REL1/REL10 supported-runtime evidence, and a GPU has not run this repository at all
(GPU execution is deferred to the hosted Kaggle / Colab run by the workspace's standing rule). The registry moves to
**Release-grade** when the committed blob runs top-to-bottom in a clean Kaggle or Colab runtime and that run is
recorded here; any later change to the carried modules or to the notebook produces a new blob and returns the
registry to **Candidate** until a clean run of that blob is recorded.

Facts a reviewer should still weigh: @P:REVIEWER_FACTS@ The synthetic warps exercise no viewpoint change and no
occlusion; the 48-pair validation split that picks the epoch is small and the 96-pair test split gives no dispersion
estimate; the drawn pair re-matched after adaptation is one image of evidence about behaviour outside the corpus, not
a measurement.

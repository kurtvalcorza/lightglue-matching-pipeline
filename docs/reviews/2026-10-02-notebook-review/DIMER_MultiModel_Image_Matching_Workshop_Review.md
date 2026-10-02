# DIMER Image Matching Comparison Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/lightglue-matching-pipeline`  
**Notebook:** `tutorials/DIMER_MultiModel_Image_Matching_Workshop.ipynb`  
**Reviewed commit:** `024e801760878f967359eef2120f8459fafbd46d` (`main`, confirmed equal to `gh api repos/kurtvalcorza/lightglue-matching-pipeline/commits/main`)  
**Notebook Git blob:** `a9a59bd5642d487061a9e439c0559aa988bf2fd3`  
**Finding prefix:** `MAT`

## Executive assessment

The notebook runs a well-controlled experiment: two frozen matchers (LightGlue + ALIKED, XoFTR) and two
model-neutral baselines receive the same digest-pinned photographs, the same seeded photo split, the same rendered
pair bytes and the same exact reference homographies, and one notebook-side evaluator scores everything. Validation
is frozen before the test pairs are opened, undefined metrics are exported as null rather than zero, checkpoint
acquisition is digest-verified before deserialization, and the default path has two passing hosted Colab T4 runs on
code identical to this revision.

It is not yet a complete guided learning experience, and its BYOD branch does not deliver its documented range. A
BYOD run with 4–12 photographs (the notebook promises 4–360) stops with a bare `StopIteration` before any model
runs. The headline metric, homography accuracy, is never defined, and the result tables arrive without guidance on
how to read them. The canonical report ZIP silently includes every earlier BYOD run's outputs, contradicting the
notebook's own statement that canonical outputs remain separate. Six Minor findings cover BYOD refusal messages, a
per-run rebuild of the multi-GB matcher environment, runtime disclosure, orientation boilerplate, a weak activity and
information architecture.

No finding here says the recorded default-path numbers are wrong. The default journey is the strongest part of the
notebook.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `WORKSHOP`; notebook specification `2.1` (metadata and opening cell) |
| Specification baseline | Declared 2.1; requirements checked against fleet `NOTEBOOK_SPEC.md` **2.2** (2026-09-26) at `ml-worker` `origin/main` `b1cfe13` |
| Stated learner | "Learners who can run Python cells in Colab/Jupyter and are new to local features, image correspondence, or geometric matching" |
| Prerequisites | Not stated beyond the learner sentence; no Python/NumPy level given |
| Supported runtime | "NVIDIA Tesla T4 or equivalent" (recommended); matchers run in a separate Python 3.12 `uv` environment with `vismatch==1.3.2`, `torch==2.14.0`, `torchvision==0.29.0` |
| Promised outcomes | One shared geometric experiment answering how sparse and detector-free matchers differ in correspondence accuracy, match density, geometric robustness and compute cost; standalone path (no clone, no DIMER source, no worker, no credentials, no upload); BYOD from a folder or ZIP of 4–360 photos through the same downstream stages; one controlled change activity |
| Default data | 360 CC0 iNaturalist photographs (6 species × 60), embedded gzip+base64 manifest, byte size + SHA-256 per photo; seeded 216 / 48 / 96 photo split; 48 validation and 96 test pairs over three tiers |
| Optional paths | BYOD (`USE_BYOD`, `BYOD_PATH` or Colab upload); baseline toggles; the "Try it yourself" scratch-cell activity (markdown only) |
| Design document | `docs/multimodel-image-matching-workshop-spec.md` (repository design spec, read for intent) |

Scope: the whole learner-facing artifact — 59 cells (26 code, 33 markdown), controls, outputs, figures, exports, the
report bundle, the activity, exercises and failure messages. Not a diff review.

### Existing execution evidence and what it covers

`docs/release-verification.md` records two passing Google Colab T4 runs of the **default path** on 2026-09-26:
blob `6623fdf1f180` (`4387c12`) and blob `90d0c883b7f0` (`5da6653`). The reviewed blob `a9a59bd5` differs from
`90d0c883` only in cell 0 (markdown: the AI Use Disclosure); every code cell and cell id is identical (checked
cell by cell in this review). The second run therefore stands as documented evidence for the default code path of
this revision: test homography accuracy @3 px LightGlue + ALIKED 0.823, XoFTR 0.719, patch neighbour 0.333,
identity 0.000. Its executed notebook is not archived in the repository; this review inspected the record, **not**
the executed notebook. No recorded run covers BYOD, the activity, the canonical-after-BYOD sequence or an invalid
input (REL12 open). `docs/execution-evidence/` does not exist in this repository.

### Evidence obtained in this review

- **Source inspection** of every cell at the reviewed commit, the generator
  (`tools/build_multimodel_image_matching_workshop.py` + `tools/multimodel_image_matching_workshop_source.py`), the
  design spec, the release record, `tutorials/README.md`, the tests and CI.
- **Direct execution, CPU only, labelled.** `run_probes.py` executes the notebook's own code cells top to bottom on
  a Windows workstation (Python 3.13.9, `CUDA_VISIBLE_DEVICES=-1`). Real: photo download and digest verification
  (all 360 pinned photos), split, pair generator, both baselines, evaluator, freeze, exports, report bundle, BYOD
  loader and split. **Stand-ins:** the matcher environment is not built (its `subprocess` calls are faked) and the
  matcher runner is replaced by a deterministic stand-in that warps random points through each pair's recorded
  reference homography with noise and 10% outliers. Nothing here is evidence about the real models, a GPU or Colab.
- **Not verified:** real-model behaviour, GPU memory, Colab upload dialog, learner observation.

Probe results are in `DIMER_MultiModel_Image_Matching_Workshop_Review_Probes.zip` (`run_probes.py`,
`results.json`, `source_manifest.json`).

## 2. Separate judgments

**Technical correctness — default path sound, BYOD path broken for small sets.** The default path is
deterministic and well-guarded (manifest digest, per-photo digest, decoded-pixel duplicate rejection, ZIP safety,
nonfinite-correspondence rejection, strict JSON). The CPU replay (P10, 878 s, all 360 pinned photos downloaded and digest-verified) completed 26/26 code cells, gave the 216 / 48 / 96 split, validation-pair digest `4d10dac9…` and test-pair digest `e3d1b550…`, and reproduced the hosted test baselines: homography accuracy @3 px identity 0.000 and patch neighbour 0.333 (hosted 2026-09-26: 0.000 / 0.333). The stand-in matcher numbers are not model evidence.
Defects are in the optional branches: BYOD with fewer than 13 photos crashes (MAT-M1), the canonical bundle mixes in
BYOD outputs (MAT-M3), and each BYOD run rebuilds the matcher environment (MAT-m2).

**Promise fulfilment — mostly delivered on the default path; BYOD and "canonical outputs remain separate" not.**
Accuracy, density, robustness by tier and compute cost are all measured on the same pairs. The BYOD range (4–360)
and the separation of canonical outputs are promised but not delivered (MAT-M1, MAT-M3).

**Learner experience — the experiment is there, the guidance is not.** The learner meets eight metric columns and
roughly ten result tables with one generic "What to notice" note, an undefined headline metric, two unlabelled
disagreement tables, and the compute-cost section labelled as infrastructure (MAT-M2). The opening has no learning
objectives and template wording (MAT-m4); the only activity is a scratch-cell snippet whose answer is stated in the
prompt (MAT-m5).

**Spec conformance (2.2) — open MUSTs:** EVAL3 (principal metrics explained: homography accuracy is not), UX1
(learning objectives), DAT12 (limits before the user selects data: stated only in section 20, after the controls),
SPL3 (random BYOD split assumption), REL12 (BYOD verification) and REL1 for any changed blob. SHOULD gaps: GDL1, GDL5,
GDL6, GDL8, GDL9, GDL10, GDL13 duplication, UX10, UX12, RUN11 wording. The declared spec version is 2.1 (S1).

## 3. Findings

### MAT-M1 — Major: BYOD with 4–12 photographs stops with a bare `StopIteration`

- **Cell/section:** §7 "Visualize the exact three-tier pair bytes" (cell `dimer-matching-workshop-18`); §17
  "Visualize first test pair from each tier" (cell `-41`); `aggregate_metrics` in §8 (cell `-20`); §20 BYOD prose.
- **Observed issue:** tiers are assigned in turn within each split (`TIERS[index % 3]`), and `byod_split` gives
  validation `max(1, round(0.2 n))` photos. With fewer than 13 photos at least one tier has no validation pair, and
  `next(r for r in validation_pair_rows if r["tier"] == tier)` raises `StopIteration`. With fewer than 9 photos the
  §17 plot cell would fail the same way on the test split. Empty tiers also reach `np.mean([])` in
  `aggregate_metrics` (RuntimeWarning, NaN).
- **Consequence:** the documented BYOD range "4–360 JPEG/PNG photographs" fails for 4–12 photos before either
  matcher runs, with an error that names nothing. A learner trying the smallest documented input, the natural first
  try, cannot reach the promised downstream stages (DAT15, REL12).
- **Evidence (direct execution, stand-in matchers):** probe P01 — n = 4, 8, 12: stop at "Visualize the exact
  three-tier pair bytes" with `StopIteration`, 8 of 26 code cells completed; n = 13 and 15: 26/26 cells complete.
- **Recommended correction:** keep the documented 4–360 range; make the tier figures skip tiers without pairs and
  say so; make `aggregate_metrics` return `n = 0` with undefined (null) metrics for an empty tier; tell the learner in
  §20 that fewer than 13 photos leave some tiers empty.
- **Acceptance check:** BYOD runs with 4, 8, 12 and 13 valid photos complete every code cell (P01 all `completed`);
  the output names the empty tiers; per-tier rows for empty tiers have `n = 0`; default-path pair digests and exported
  metric CSVs are byte-identical before and after the fix (P10).

### MAT-M2 — Major: The headline metric is undefined and the results arrive without reading guidance

- **Cell/section:** §0 (cell `-02`), §8 (cell `-19`), §12–§18 result cells (`-22`, `-29`, `-35`, `-37`, `-39`,
  `-43`), §13 note (`-30`), §18 heading (`-42`), Glossary (`-56`).
- **Observed issue:** (a) `homography_acc_3px/5px`, the number the release record leads with, is introduced only as
  "seeded RANSAC-DLT homography accuracy@3/5px"; that it is the fraction of pairs whose estimated homography moves the
  image corners within 3/5 px of the reference (mean corner error) is never said, nor are `inliers_per_pair`,
  `median_error_px` or `n`. (b) Homography, RANSAC and DLT appear in §0 and §8 but are only defined in the Glossary at
  the end. (c) Of roughly ten result tables only one generic "What to notice" note exists, placed at the freeze
  (§13), where there is no result to notice. (d) §16 shows two `head()` tables with no label saying which shows
  LightGlue ahead and which XoFTR ahead. (e) §18, the only place compute cost — part of the central question — is
  shown, is labelled "Infrastructure … not a learning objective", and ends by printing two raw JSON dumps truncated at
  5000 characters.
- **Consequence:** the learner cannot tell what the principal comparison measures or how to read it, so the
  conclusion template at the end asks for evidence the notebook never taught them to read. EVAL3 (MUST) is unmet;
  GDL6, GDL8, UX4 are not followed.
- **Evidence (source inspection; static probe P05):** no corner-error definition before the first scored table;
  1 "What to notice" note after validation scoring; §18 labelled Infrastructure; `[:5000]` JSON dump present;
  disagreement tables unlabelled.
- **Recommended correction:** define every column in §8 (a table: what it measures, why it matters), including
  homography accuracy as mean corner error of the RANSAC-DLT estimate; define homography, DLT and RANSAC in §0; add a
  "What to notice" note after the baseline, validation, test, degradation, disagreement and density/runtime outputs
  without hard-coding a winner; label the two disagreement tables; replace the §18 Infrastructure label with an
  explanation of the runtime columns and replace the JSON dump with a compact parity table; give the §13 note a
  freeze-specific purpose.
- **Acceptance check:** P05 — homography accuracy defined with "corner" before the first scored table, inliers per
  pair defined there, RANSAC explained before §8 code; ≥5 "What to notice" notes after the validation scoring cell;
  §18 not labelled Infrastructure; no `[:5000]`; both disagreement tables preceded by a printed label naming the
  matcher that is ahead.

### MAT-M3 — Major: The canonical report ZIP silently includes every earlier BYOD run

- **Cell/section:** §1 controls (cell `-04`), §24 "Final provenance" (cell `-51`), §20 prose (cell `-46`).
- **Observed issue:** a BYOD run writes to `outputs/byod/run-*`, inside the canonical `outputs/` folder. The export
  cell zips `OUTPUT_DIR` whole with `shutil.make_archive`, so a default run after a BYOD run puts the BYOD run's pair
  manifests, match files, metrics and figures of the user's photographs, and its own report ZIP, into the canonical
  report. §20 states "canonical outputs remain separate".
- **Consequence:** the canonical report no longer belongs to the run that produced it, and a learner who shares it
  can share derivatives of photographs they were told stay in the runtime and should not be uploaded without
  authorisation (DAT17/DAT18 spirit; framework clean-default criterion "exports belong to this run").
- **Evidence (direct execution, stand-in matchers):** probe P02 — default run, BYOD run (15 photos), default run
  again in the same runtime (RUN_PATCH_BASELINE off for the last run): the canonical report ZIP held **64 members under `byod/`**, including the BYOD run's pair manifests, figures, metrics and its own report ZIP (`byod/run-*_DIMER_Image_Matching_Workshop_Report.zip`).
- **Recommended correction:** build the canonical bundle so it excludes the `byod/` folder (and say so when one
  exists); keep each BYOD bundle limited to its run folder; correct the §20 sentence.
- **Acceptance check:** after the default → BYOD → default sequence, the canonical report ZIP has no member under
  `byod/` (P02 `canonical_bundle_byod_member_count == 0`), the BYOD bundle still contains its run, and §20 says what
  the canonical ZIP contains.

### MAT-m1 — Minor: BYOD limits arrive late and refusals do not name the offending file

- **Cell/section:** §1 controls (`-03`/`-04`), §4 (`-10`, `-11`), §20 (`-46`).
- **Observed issue:** the 4–360 count, 16–4096 px size and duplicate rules are stated only in §20, after the
  controls the learner sets in §1 (DAT12). Refusals do not say what failed where: "BYOD requires 4..360 JPEG/PNG
  photographs" (no count found), "Duplicate decoded pixels would leak across image splits" (no file names), "BYOD
  image sides must be at least 16 pixels" (no file or size). A non-ZIP file uploaded in Colab is saved as
  `uploaded-byod.zip` and fails with `BadZipFile: File is not a zip file`. The random BYOD split's independence
  assumption is not stated (SPL3).
- **Consequence:** a learner with a folder of 200 photos and one tiny or duplicated image has to bisect the folder.
- **Evidence (direct execution; P04):** messages recorded verbatim in `results.json`; all four "names the
  file/count" checks false; limits not stated by the controls cell.
- **Recommended correction:** state the BYOD contract next to the controls; include the found count, the file name
  and the measured size or duplicate partner in each refusal; check the upload's name and the ZIP signature; state
  the split assumption in §20.
- **Acceptance check:** P04 all true (count, file names, tiny-file name, no `BadZipFile`); limits (360, 4096) stated
  in markdown at or before the controls cell; §20 states the independence assumption.

### MAT-m2 — Minor: Every BYOD run rebuilds the multi-GB matcher environment

- **Cell/section:** §10 (cell `-24`) with §1 (`-04`).
- **Observed issue:** `ENV_ROOT = WORK_ROOT/"envs"/"vismatch"`, and a BYOD run rebinds `WORK_ROOT` to a fresh
  `work/byod/run-*` folder, so the readiness marker is never found and `uv` + PyTorch + `vismatch` are installed again
  for each BYOD run.
- **Consequence:** minutes of waiting and several GB of extra disk per BYOD run in a hosted session; repeated BYOD
  attempts can exhaust Colab disk.
- **Evidence (source inspection, P03; direct execution, P02):** the default run used `work/envs/vismatch`; the
  following BYOD run resolved `work/byod/run-*/envs/vismatch` and invoked environment creation again
  (`byod_run_rebuilds_model_environment: true`; the `uv`/`pip` calls were faked, so time and size were not measured).
- **Recommended correction:** anchor the environment at the shared work root so default and BYOD runs reuse it.
- **Acceptance check:** P02 `env_root_default == env_root_byod` and `byod_run_rebuilds_model_environment` is false.

### MAT-m3 — Minor: Runtime cost and the CPU fallback are not disclosed

- **Cell/section:** opening (`-00`, `-01`), §12 (`-28`).
- **Observed issue:** "Use the documented GPU runtime" gives no download sizes (photos, checkpoints, PyTorch
  environment) or timings; the recorded T4 per-pair timings exist but are not quoted; the runner silently uses the CPU
  when CUDA is absent (only a `device` column shows it). UX12, RUN11, ENV4.
- **Consequence:** a learner cannot plan the session or tell that a slow run is a CPU run.
- **Evidence (source inspection; P09).**
- **Recommended correction:** state the GPU requirement, the sizes and the measured T4 per-pair timings with their
  date (labelled; total wall time was not recorded), and print a warning when a matcher ran without CUDA.
- **Acceptance check:** P09 all true (sizes and measured per-pair timing in the opening, a "No CUDA/GPU" warning in
  code); the CPU harness run prints the warning.

### MAT-m4 — Minor: No learning objectives; orientation is template text

- **Cell/section:** "How to use this notebook" (`-01`).
- **Observed issue:** no learning objectives (UX1 MUST, GDL5); the roadmap and "What successful execution looks
  like" are generic ("Establish the simple reference/baseline where applicable", "Run the model or model comparison",
  "where the capability supports them"); prerequisites beyond running cells are not stated (GDL1).
- **Consequence:** the learner cannot tell what they should be able to do afterwards or what a finished run
  produces.
- **Evidence (source inspection; P06):** no objectives heading; four template phrases present.
- **Recommended correction:** observable objectives tied to sections; a roadmap of this notebook's sections; a
  completion description naming the actual output files.
- **Acceptance check:** P06 `learning_objectives_heading` true and `template_phrases` empty.

### MAT-m5 — Minor: The only activity is a pasted snippet whose answer is in the prompt

- **Cell/section:** "Try it yourself" (`-53`), §23 Exercises (`-49`).
- **Observed issue:** the activity asks the learner to paste code into a scratch cell, uses one pair and one model,
  and states the conclusion ("explain why a stricter tolerance cannot increase the fraction") before the learner
  predicts. The five exercises have no guidance (GDL9).
- **Consequence:** the Predict → Change → Observe → Explain cycle (GDL10, UX6) is reduced to confirming a stated
  tautology; nothing compares the two matchers.
- **Evidence (source inspection; P08).**
- **Recommended correction:** an executable, off-by-default activity cell controlled from §1, re-scoring saved
  validation correspondences for both matchers and every tier at 3 px and a learner-chosen stricter tolerance, with a
  prediction prompt that does not give the answer and a collapsible sample interpretation; guidance for the
  exercises.
- **Acceptance check:** P08 — executable activity cell present, control default off, answer not in the prompt,
  exercise guidance present; with the control on the activity runs and the output tree is unchanged
  (`output_tree_unchanged` true).

### MAT-m6 — Minor: Information architecture: duplicate troubleshooting, mixed heading levels, an empty "optional" section

- **Cell/section:** `-52`, `-58`, headings of §12, §14, §19–§23, `-47`.
- **Observed issue:** two Troubleshooting sections; the second is generic template text (it mentions "labels",
  "release/unload steps" and "too many large models", none of which exist here); eight numbered sections use `#`
  while the rest use `##`; "21. Optional model-native adaptation" contains no optional step.
- **Consequence:** the table of contents is misleading and the learner reads advice that does not apply.
- **Evidence (source inspection; P07).**
- **Recommended correction:** one notebook-specific troubleshooting table at the end; numbered sections at `##`;
  retitle §21.
- **Acceptance check:** P07 — one troubleshooting section, no generic template rows, no numbered heading above
  level 2, §21 no longer titled "Optional model-native adaptation".

### MAT-S1 — Suggestion: Declare NOTEBOOK_SPEC 2.2

Not a defect; the notebook declares 2.1 and the repository validator pins 2.1.

### MAT-S2 — Suggestion: Record the test-pair digest in the freeze

The freeze fixes the validation-pair digest; the test-pair digest appears only in the final manifest. Recording it in
`frozen_experiment.json` before test inference would make the freeze self-contained.

### MAT-S3 — Suggestion: Align BYOD size limits with the design spec

`docs/multimodel-image-matching-workshop-spec.md` §34 asks for sides of 64–1024 px after preparation; the notebook
accepts 16–4096 px source images, so an extreme aspect ratio can prepare to an 8-px side. Decide which contract is
intended.

## 4. Promise-to-evidence matrix

| Claim | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Same pairs, geometry, thresholds for both models | §6–§7 generator, one `pair_manifest.json` | pair digests in freeze/manifest | explained in §6 | Delivered (P10, hosted) |
| Model-neutral baselines | §8–§9 | baseline table | no reading guidance | Delivered; guidance missing (M2) |
| Accuracy / density / robustness / compute cost | §12–§18 | ~10 tables, 3 figure sets | one generic note; metric undefined | Partly (M2) |
| Freeze before test | §13–§14 | `frozen_experiment.json`, digest assert | explained | Delivered |
| New-pair inference without ground truth | §19 | match counts, "not-measurable" | explained | Delivered |
| BYOD: 4–360 photos through the same stages | §4, §5, §20 | crash for 4–12 | — | Not delivered (M1) |
| Canonical outputs separate from BYOD | §1, §24 | canonical ZIP includes `byod/` | stated as separate | Not delivered (M3) |
| One controlled change | "Try it yourself" | scratch snippet | answer stated | Weak (m5) |
| Standalone, credential-free, no upload by default | whole notebook | hosted run | stated | Delivered (hosted 2026-09-26) |

| Objective (implied; none stated) | Learner activity | Evidence exercised |
|---|---|---|
| Compare sparse vs detector-free matching | prediction prompts §9, §12, §14 | prompts only; no guided reading (M2) |
| Read precision together with density | Exercise 2 | no guidance (m5) |
| Diagnose rotation robustness | Exercise 3, §15 table | table present; drop sign unexplained (M2) |
| Apply to own data | BYOD | fails for small sets (M1) |

## 5. Journeys

| Journey | Basis | Verdict |
|---|---|---|
| First-time learner | Source inspection | Needs revision — M2, m4, m5, m6 |
| Clean default | Documented hosted execution (2026-09-26, code-identical blob) + CPU replay with stand-in matchers (P10) | Passes for the code; real-model re-run of this blob not available |
| Active learning | Source inspection (no executable activity) | Weak — m5 |
| Reuse and recovery | Direct execution (P01, P02, P04), stand-in matchers | Fails — M1, M3, m1, m2 |

## 6. Readiness

**Needs revision.** Three Major findings are open (M1, M3 on the BYOD/export journey; M2 on the learner journey),
and EVAL3, UX1, DAT12, SPL3 and REL12 are unmet. After fixes the notebook blob changes, so the 2026-09-26 hosted runs
no longer cover it: a fresh Colab T4 **Run all** of the fixed head, a BYOD run (including fewer than 13 photos and
one refused input) and the activity are needed before readiness can move past **Verification pending**. Status stays
**Candidate**; checkpoint parity remains `qualification-required`.

## 7. Verified versus inferred

- **Verified by direct CPU execution (stand-in matchers):** M1 crash thresholds; M3 bundle contamination; m2 BYOD
  environment path; m1 messages; the default path's split, digests and baseline metrics (P10).
- **Verified by source inspection:** M2, m3, m4, m5, m6; code identity with the hosted-run blob.
- **Inferred:** the size of the PyTorch environment rebuilt per BYOD run (several GB) and its wall time — not
  measured here; that real matchers behave on small BYOD sets as the stand-ins do after the tier fix.
- **Only the maintainer can confirm:** a hosted Colab run of the fixed head, including the upload dialog.
- **Finding most likely to be wrong:** MAT-M3's severity. It needs the specific default → BYOD → default sequence in
  one runtime; if learners always use a fresh runtime for BYOD, it is closer to Minor.

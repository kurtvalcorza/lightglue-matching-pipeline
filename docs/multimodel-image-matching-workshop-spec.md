# DIMER Workshop Specification: Comparing Image Matching Models

**Status:** Proposed  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_MultiModel_Image_Matching_Workshop.ipynb`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

# 1. Purpose

This workshop compares two live DIMER image-matching systems on the **same geometrically controlled image pairs**:

| Model | Matching strategy | Core idea |
|---|---|---|
| **LightGlue + ALIKED** | sparse keypoint matching | detect local features first, then match them |
| **XoFTR** | detector-free semi-dense matching | infer correspondences directly from image feature grids |

The workshop asks:

> Given exactly the same pair of images and the same known geometric transformation, how do sparse and detector-free matchers differ in correspondence quality, robustness, match density, geometric accuracy, and computational cost?

The notebook MUST NOT present either model as universally superior.

---

# 2. Notebook profile

The canonical notebook SHALL use:

**Profile:** `TASK-INFERENCE`  
**Mode:** `WORKSHOP`

The core comparison performs no training or parameter adaptation.

Canonical workflow:

```text
source photographs
→ deterministic homography pairs
→ validate
→ common baselines
→ LightGlue matching
→ XoFTR matching
→ common geometric evaluation
→ difficulty analysis
→ runtime comparison
→ new-pair inference
→ export
```

The individual DIMER model notebooks remain the authoritative `E2E` adaptation tutorials.

---

# 3. Core models

## 3.1 LightGlue + ALIKED

**Model identity:** `cvg/LightGlue`  
**Release:** `v0.1_arxiv`  
**Upstream LightGlue code revision:**  
`eb42fee2d71449efb0aa5c10549752b5d75384d8`

Feature extractor:

**ALIKED-N(16)**  
Source commit:  
`683d7c65197395c0b3f01ebe76e1084a27e73a65`

Served DIMER assets:

```text
aliked_lightglue.safetensors
47,564,948 bytes
SHA-256:
9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6
```

```text
aliked-n16.safetensors
2,719,928 bytes
SHA-256:
3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a
```

Approximate parameters:

- ALIKED: 677K
- LightGlue: 11.88M

Input semantics:

- RGB;
- sides 64–1024 px;
- up to 2,048 ALIKED keypoints;
- 128-d local descriptors;
- LightGlue assignment threshold 0.1.

The workshop MUST use the converted SafeTensors assets and MUST NOT deserialize the original pickle checkpoints during normal execution.

---

## 3.2 XoFTR

**Model:** `vismatch/xoftr`  
**Revision:**  
`d8ee7d89be3c9e5c157db3886db1c0f0e038b321`

Checkpoint:

```text
xoftr_640.safetensors
44,419,304 bytes
SHA-256:
4d5ed62e8b41f862ecc5c660e31f1c450402966623d6a28e85acf7fbd794cc69
```

Approximate parameters:

**11.09M**

Input semantics:

- grayscale internally;
- sides 64–1024 px;
- dimensions cropped to multiples of 8;
- checkpoint targeted at approximately 640-px imagery;
- detector-free coarse-to-fine matching;
- coarse threshold 0.3;
- fine threshold 0.1.

The workshop SHALL use only the pinned 640-px checkpoint.

---

# 4. Shared runtime

Both live DIMER implementations are compatible with:

```text
Python 3.12
torch==2.14.0
numpy==2.5.3
pillow==11.3.0
safetensors==0.8.0
```

LightGlue additionally requires:

```text
torchvision==0.29.0
```

The workshop SHOULD therefore use one common tested environment rather than separate virtual environments unless clean-runtime qualification identifies a conflict.

---

# 5. Canonical dataset

Reuse the shared image-matching corpus already used by both pipelines:

**360 CC0 iNaturalist photographs**

Structure:

- 6 bird species;
- 60 photographs per species;
- one photograph per observer;
- each image individually pinned by photo ID, byte size, and SHA-256;
- fetched at runtime from iNaturalist Open Data;
- no photographs committed to the repository.

Total pinned corpus size:

approximately **39.2 MB**.

The workshop SHOULD carry the existing image manifest inside the standalone notebook.

---

# 6. Canonical image split

Reuse the existing image-disjoint split:

| Split | Per species | Total photographs |
|---|---:|---:|
| Train / development | 36 | 216 |
| Validation | 8 | 48 |
| Test | 16 | 96 |

Seed:

```text
42
```

No source photograph may appear in more than one split.

Even though the canonical workshop performs no training, the existing split SHOULD be preserved so:

- validation can be used to inspect behaviour before the independent test;
- the test set remains untouched;
- the optional model-native adaptation extension can reuse the exact same train/validation/test boundary.

---

# 7. Common pair-generation contract

The individual LightGlue and XoFTR tutorials currently use different definitions of their hardest tier.

The comparative workshop MUST replace those with **one shared pair generator**.

Every photograph is:

1. resized so its long side is 640 px;
2. converted into `image0`;
3. warped using a seeded projective homography;
4. photometrically modified into `image1`;
5. stored with the exact 3×3 homography mapping `image0 → image1`.

The same pair bytes and same reference homography MUST be supplied to both models.

---

# 8. Three common difficulty tiers

Instead of only `easy` / `hard`, use three common tiers.

## Tier A — Easy

```text
corner jitter: ≤ 6%
rotation: ±10°
scale: 0.9–1.1
brightness: 0.85–1.15
contrast: 0.85–1.15
gamma: 1.0
blur: 0
Gaussian noise: 4
```

Purpose:

> ordinary moderate geometric/photometric variation.

---

## Tier B — Moderate

Derived from the existing XoFTR hard tier:

```text
corner jitter: ≤ 18%
rotation: ±35°
scale: 0.6–1.4
brightness: 0.6–1.4
contrast: 0.6–1.4
gamma: 0.7–1.4
blur: 1.2
Gaussian noise: 10
```

Purpose:

> difficult but still broadly realistic image-registration stress.

---

## Tier C — Rotation stress

Derived from the current LightGlue hard tier:

```text
corner jitter: ≤ 18%
rotation: ±150°
scale: 0.6–1.4
brightness: 0.6–1.4
contrast: 0.6–1.4
gamma: 0.7–1.4
blur: 1.2
Gaussian noise: 10
```

Purpose:

> explicitly test strong in-plane rotation.

This is a stress test, not a claim about normal deployment frequency.

---

# 9. Tier assignment

Within each split, pairs SHOULD be distributed approximately equally across:

```text
easy
moderate
rotation_stress
```

For the 96-image test set:

```text
32 easy
32 moderate
32 rotation-stress
```

For validation:

```text
16 each
```

For training/development:

```text
72 each
```

Tier assignment MUST be deterministic.

Recommended:

```text
tier = TIERS[index % 3]
```

after the existing seeded/shuffled image split.

---

# 10. Pair identity and reproducibility

Each pair MUST record:

```text
id
source_image_id
split
tier
seed
image0_sha256
image1_sha256
homography
width
height
```

Pair generation seed SHOULD follow the existing deterministic pattern:

```text
pair_seed = split_seed * 100003 + pair_index
```

or an equivalent explicitly versioned deterministic rule.

Changing:

- source image;
- warp;
- photometric transformation;
- resize;
- tier assignment;

MUST change the pair identity/digest.

---

# 11. Why homography supervision

Synthetic homographies provide an exact geometric reference.

For a predicted correspondence:

\[
p_0 \leftrightarrow p_1
\]

the known homography \(H\) maps:

\[
\hat p_1 = H p_0
\]

The correspondence error is therefore:

\[
e = \|p_1-\hat p_1\|
\]

No human annotation is required.

The notebook MUST state that synthetic homographies do not reproduce all properties of real matching:

- true 3-D viewpoint change;
- occlusion;
- parallax;
- non-planar scenes;
- moving objects;
- modality differences.

---

# 12. Input fairness

Both models receive the identical generated:

```text
image0
image1
reference homography
```

Model-native preprocessing is preserved.

LightGlue:

```text
RGB
→ ALIKED keypoint detection
→ descriptors
→ LightGlue
```

XoFTR:

```text
RGB source image
→ model-native grayscale conversion
→ crop to supported multiple
→ detector-free matcher
```

Do not pre-convert the entire common corpus to grayscale merely to equalize the inputs.

The preprocessing difference is part of each model's actual deployed architecture.

---

# 13. Common baseline A — identity geometry

The identity baseline assumes:

\[
H = I
\]

This is not a feature matcher.

It answers:

> What if we simply assume the image did not move?

Score it using the same homography/corner-error evaluator.

Expected performance should deteriorate quickly as the synthetic transformation becomes stronger.

---

# 14. Common baseline B — patch nearest neighbour

Use the existing model-neutral patch-nearest-neighbour baseline.

It provides simple image correspondences without a learned matcher.

Evaluate it using the exact same:

- precision thresholds;
- inlier definition;
- RANSAC;
- homography metric.

This is the principal correspondence baseline shared across both models.

---

# 15. LightGlue-only descriptor baseline

The existing LightGlue tutorial also exposes:

**ALIKED descriptor nearest-neighbour matching**

This MAY appear in an optional diagnostic section.

It MUST NOT appear in the main common-baseline table because it uses LightGlue's own feature extractor.

It is useful for answering:

> How much does LightGlue improve over simply nearest-neighbour matching the same ALIKED descriptors?

---

# 16. Common output normalization

Both adapters MUST emit a common table:

```text
pair_id
model
tier
match_index
x0
y0
x1
y1
score
```

Where:

- `(x0, y0)` = point in image0;
- `(x1, y1)` = corresponding point in image1;
- `score` = model-produced matching score/confidence.

Scores from different models MUST NOT be assumed numerically calibrated or directly comparable.

---

# 17. Geometric reference evaluation

For every predicted match:

1. project the image0 point with the reference homography;
2. calculate Euclidean reprojection error;
3. mark inliers at fixed thresholds.

Required thresholds:

```text
1 px
3 px
5 px
```

---

# 18. Common matching metrics

Report:

### Precision@1px

Fraction of returned matches whose reference reprojection error ≤1 px.

### Precision@3px

Primary correspondence metric.

### Precision@5px

More permissive correspondence metric.

### Matches per pair

Average number of correspondences produced.

### Inliers per pair

Average matches satisfying:

```text
reprojection error <= 3 px
```

### Median inlier error

Median reprojection error among 3-px inliers.

Report every metric:

- overall;
- per difficulty tier;
- per pair.

---

# 19. Homography estimation metric

Using the predicted correspondences:

```text
matches
→ seeded RANSAC
→ four-point DLT
→ refit on consensus inliers
→ estimated homography
```

Use:

```text
RANSAC inlier threshold = 3 px
iterations = 500
seed = 0
```

A pair with fewer than four usable correspondences counts as a geometric-estimation failure.

---

# 20. Homography accuracy

Compare the estimated homography to the known synthetic reference by warping image corners.

Report:

### Homography accuracy @3px

Fraction of pairs whose mean/defined corner displacement error is within 3 px according to the existing DIMER metric contract.

### Homography accuracy @5px

Equivalent at 5 px.

The notebook MUST explicitly explain:

> RANSAC-DLT is part of the **evaluation**, not an image-matching output produced by LightGlue or XoFTR.

---

# 21. Why precision and match count must be read together

A matcher can achieve high precision by returning very few easy matches.

Another can return many more matches with slightly lower precision.

Therefore the workshop MUST NOT interpret `precision@3px` alone.

For every model show together:

```text
precision@3px
matches/pair
inliers/pair
homography accuracy
```

---

# 22. Validation stage

Run both frozen models on the 48 validation pairs.

Display:

| Model | Precision@1 | Precision@3 | Precision@5 | Matches/pair | Inliers/pair | Homography acc@3 |
|---|---:|---:|---:|---:|---:|---:|

Also report the same table by tier.

No model parameter, matching threshold, metric threshold, or RANSAC parameter may be changed after validation solely because of the test results.

---

# 23. Freeze-before-test

Write:

```text
outputs/frozen/frozen_experiment.json
```

containing:

- source-corpus manifest digest;
- photo split;
- pair-generator version;
- tier definitions;
- pair seeds;
- model identities;
- checkpoint digests;
- inference thresholds;
- evaluation thresholds;
- RANSAC settings;
- baseline definitions;
- runtime packages.

Only after this record exists may the final 96 test pairs be scored.

---

# 24. Independent test

Run:

- identity baseline;
- patch-NN;
- LightGlue + ALIKED;
- XoFTR

on exactly the same final 96 pairs.

Produce:

## Overall table

| Method | Precision@1 | Precision@3 | Precision@5 | Matches/pair | Inliers/pair | Median error | H acc@3 | H acc@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Per-tier table

Rows:

```text
easy
moderate
rotation_stress
```

for every method.

---

# 25. Primary teaching comparison

The notebook SHOULD emphasize three behavioural dimensions.

### Accuracy

How often are returned correspondences geometrically correct?

### Density

How many usable correspondences are returned?

### Robustness

How quickly does each method degrade as geometry becomes harder?

These are more useful than one global ranking.

---

# 26. Correspondence visualization

For selected pairs, show:

```text
image0 | image1
```

with lines connecting matched keypoints.

Use deterministic examples:

1. first easy pair;
2. first moderate pair;
3. first rotation-stress pair.

Display for both:

- LightGlue;
- XoFTR.

Optionally distinguish:

- inliers;
- outliers

according to the 3-px geometric reference.

---

# 27. Match-density visualization

Because LightGlue and XoFTR produce very different quantities of correspondences, show:

```text
matches per pair
inliers per pair
```

as distributions or paired scatter plots.

Participants SHOULD be able to see that:

> "more matches" and "better matches" are different properties.

---

# 28. Difficulty degradation

For each model calculate:

\[
\Delta P_{3px}
=
P_{easy}-P_{tier}
\]

for:

- moderate;
- rotation stress.

Also calculate degradation in:

- homography accuracy;
- inlier count.

Do not collapse these into one composite robustness score.

---

# 29. Pair-level disagreement analysis

Build:

```text
pair_id
tier
lightglue_precision_3px
xoftr_precision_3px
lightglue_inliers
xoftr_inliers
lightglue_homography_success
xoftr_homography_success
```

Then identify deterministically:

- pair where both perform strongly;
- pair where both fail;
- largest LightGlue advantage;
- largest XoFTR advantage;
- largest disagreement in number of inliers.

This forms the error-analysis gallery.

---

# 30. Runtime and resource comparison

Measure separately:

### Model load

- checkpoint bytes;
- model load seconds.

### Inference

- total test inference seconds;
- seconds per pair;
- matches/sec if meaningful;
- peak CUDA memory where practical.

### Geometric evaluation

Keep RANSAC/evaluation time separate from neural inference time.

Suggested table:

| Model | Weights | Params | Device | sec/pair | Matches/pair | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|
| LightGlue + ALIKED | ~50 MB combined | ~12.6M | … | … | … | … |
| XoFTR | ~44 MB | ~11.1M | … | … | … | … |

No aggregate efficiency score is required.

---

# 31. New-pair inference

After evaluation, run both models on a deterministic drawn or photographic pair without a reference homography.

Output:

```text
model
n_matches
correspondences
scores
```

Because no reference geometry exists:

```text
evaluation status = not measurable
```

Do not invent precision.

This teaches the distinction between:

- inference output; and
- evaluation evidence.

---

# 32. BYOD mode A — photographs → synthetic pairs

The simplest BYOD path SHALL accept:

- directory; or
- ZIP

containing JPEG/PNG photographs.

Then:

```text
photographs
→ validate
→ deterministic split
→ synthetic homography generation
→ exact reference geometry
→ matching
→ measurable evaluation
```

This is the preferred BYOD educational path.

---

# 33. BYOD mode B — supplied real pairs

A later advanced BYOD mode MAY accept:

```text
pairs.csv
image0
image1
optional reference homography
```

If no reference homography is supplied:

- matching runs;
- correspondences can be visualized;
- geometric accuracy MUST be `not-measurable`.

If a valid reference homography is supplied, the common evaluator may score it.

The notebook MUST NOT infer a reference homography and then use the same estimate as "ground truth."

---

# 34. BYOD validation

For every image:

- decode successfully;
- local file only;
- supported JPEG/PNG;
- side length 64–1024 after preparation;
- finite reference homography if provided;
- non-singular 3×3 homography.

Remote image URLs MUST be rejected at the matching API boundary.

ZIP handling MUST reject:

- absolute paths;
- `..` traversal;
- symlinks;
- unreasonable expanded size.

---

# 35. BYOD data guidance

The notebook MUST state:

> User-supplied images are processed inside the selected notebook runtime and are not sent to DIMER workers or APIs. A hosted notebook remains an external compute environment. Do not upload confidential, personal, restricted, biometric, surveillance, security-sensitive, or proprietary imagery unless authorized.

---

# 36. Optional model-native adaptation

The canonical comparison MUST remain frozen-model inference.

An optional advanced section MAY demonstrate the existing E2E adaptation contracts separately.

## LightGlue

Current model-native adaptation:

- ALIKED remains frozen;
- train last LightGlue transformer layers;
- train assignment head;
- assignment loss;
- validation precision@3px selection.

## XoFTR

Current model-native adaptation:

- backbone remains frozen;
- fine stage remains frozen;
- train final coarse transformer layers;
- train coarse projection;
- coarse focal matching loss;
- validation precision@3px selection.

These are materially different optimization problems.

Therefore their adapted metrics MUST NOT replace the frozen common-comparison results.

---

# 37. Why the common test should include ±150° rotation

The live LightGlue tutorial intentionally uses ±150° because its frozen ALIKED/LightGlue combination is known to degrade substantially under strong in-plane rotation.

The current XoFTR tutorial uses only ±35°.

A comparative workshop should expose both systems to the same stronger stress tier.

This allows the workshop to answer empirically:

> Is the strong-rotation weakness specific to sparse ALIKED + LightGlue, or does detector-free XoFTR degrade similarly?

The answer MUST come from the workshop results rather than being assumed in advance.

---

# 38. Why not use only the ±35° tier

On the existing XoFTR-style ±35° fixture, frozen LightGlue is already extremely strong.

That produces little behavioural separation and makes the workshop less informative.

Using:

```text
easy
moderate
rotation_stress
```

provides a useful robustness curve instead of a single easy benchmark.

---

# 39. Workshop exercises

### Exercise A — sparse vs detector-free

Before execution:

> Which matcher do you expect to return more correspondences?

Explain based on architecture.

### Exercise B — precision versus density

Suppose:

```text
Model A: 99% precision, 40 matches
Model B: 95% precision, 2,000 matches
```

Which is better?

The answer depends on the downstream geometry problem.

### Exercise C — rotation

Predict which architecture will degrade more under ±150° rotation.

Then compare with the measured stress-tier result.

### Exercise D — homography

Why can high match precision still fail to recover a good homography?

Potential reasons include:

- too few matches;
- poor spatial distribution;
- near-collinear matches;
- clustered features.

### Exercise E — real-world transfer

What changes when moving from synthetic homographies to:

- drone frames;
- street imagery;
- satellite images;
- thermal/visible image pairs;
- historical photographs?

---

# 40. Interpretation boundaries

The notebook MUST explain:

### Synthetic geometry

The reference is exact, but synthetic homography pairs are not equivalent to real 3-D viewpoint changes.

### Planarity

One homography assumes one planar/projective mapping.

Real scenes with depth variation exhibit parallax.

### Matching confidence

Model match scores are not calibrated probabilities that a match is correct.

### RANSAC

The evaluation RANSAC is not part of the raw model output.

### No-overlap pairs

Both matchers can still return matches when two images have little or no meaningful overlap.

### Texture dependence

Flat or repetitive regions remain difficult.

---

# 41. Output structure

```text
outputs/
├── data/
│   ├── corpus_manifest.json
│   ├── split_manifest.json
│   └── pair_manifest.json
├── validation/
│   ├── lightglue_metrics.json
│   ├── xoftr_metrics.json
│   └── comparison.csv
├── frozen/
│   └── frozen_experiment.json
├── test/
│   ├── matches/
│   │   ├── lightglue/
│   │   └── xoftr/
│   ├── per_pair_metrics.csv
│   ├── aggregate_metrics.csv
│   ├── per_tier_metrics.csv
│   └── disagreements.csv
├── figures/
├── future/
│   └── unlabelled_pair_matches/
├── provenance/
│   ├── model_manifest.json
│   └── experiment_manifest.json
└── workshop_summary.json
```

---

# 42. Provenance

Record:

```text
notebook_spec
notebook_profile
notebook_mode
workshop_revision

corpus:
  manifest_digest
  license
  split_seed
  split_counts

pair_generator:
  version
  working_long_side
  tier_parameters
  seed_policy
  interpolation
  photometric_transforms

models:
  id
  revision
  weight_digests
  code_revision
  model_parameters
  inference_thresholds
  runtime_versions

evaluation:
  precision_thresholds
  inlier_threshold
  ransac_iterations
  ransac_seed
  homography_metric
  baselines
```

---

# 43. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "TASK-INFERENCE",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "multi-model-image-matching",
    "carrier": "comparative homography-supervised image-matching workshop",
    "dataset": "360 CC0 iNaturalist photographs with deterministic synthetic homography pairs",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 44. Release acceptance

| Requirement | Required |
|---|---:|
| Notebook Spec 2.1 | PASS |
| Fresh `Run all` | PASS |
| No Git clone | PASS |
| No DIMER runtime source fetch | PASS |
| No DIMER workers | PASS |
| No credentials | PASS |
| 360-photo manifest integrity | PASS |
| Image-disjoint split | PASS |
| Common pair generator | PASS |
| Three difficulty tiers | PASS |
| Exact reference homographies | PASS |
| Identity baseline | PASS |
| Patch-NN baseline | PASS |
| LightGlue + ALIKED inference | PASS |
| XoFTR inference | PASS |
| Common geometric evaluator | PASS |
| Precision@1/3/5 | PASS |
| Inliers/matches | PASS |
| Homography accuracy@3/5 | PASS |
| Freeze-before-test | PASS |
| Per-tier analysis | PASS |
| Pair-level disagreement analysis | PASS |
| Unlabelled new-pair inference | PASS |
| BYOD synthetic-pair path | PASS |
| BYOD refusal cases | PASS |
| Provenance export | PASS |

---

# 45. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Runtime | Sample | BYOD | Run-all | Status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_Image_Matching_Workshop.ipynb` | `TASK-INFERENCE` | `WORKSHOP` | LightGlue+ALIKED vs XoFTR geometric image matching | T4 | 360 CC0 iNaturalist photographs → deterministic easy/moderate/rotation-stress homography pairs | yes | pending | candidate |
```

---

# 46. Implementation principle

The canonical experiment is:

> **same photograph → same deterministic warp → same exact homography → same evaluation → different matcher**

That is substantially cleaner than combining the two existing E2E notebooks.

The individual model tutorials answer:

> How does this matcher work, adapt, and export?

The comparative workshop answers:

> **How do sparse keypoint matching and detector-free matching behave differently when the geometry, imagery, metrics, and evaluation pipeline are held constant?**
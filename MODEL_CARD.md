---
license: Apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-feature-extraction
task: "Others - Image Matching"
tags:
  - image-matching
  - local-features
  - keypoint-matching
  - homography
  - fine-tuning
base_model: cvg/LightGlue
date_published: "2023-06-26"
date_published_source: "the `v0.1_arxiv` release of cvg/LightGlue on GitHub (2023-06-26, https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv), whose assets are the pinned matcher checkpoints; the LightGlue paper (arXiv 2306.13643) was submitted 2023-06-23; the ALIKED extractor checkpoint is a file of Shiaoming/ALIKED at the pinned commit"
---

# LightGlue + ALIKED — Sparse Local-Feature Matching (Correspondences, Homography-Supervised Evaluation & Bounded Fine-Tuning)

[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-cvg%2FLightGlue-181717?style=flat&logo=github&logoColor=white)](https://github.com/cvg/LightGlue)
[![Extractor](https://img.shields.io/badge/Extractor-Shiaoming%2FALIKED-181717?style=flat&logo=github&logoColor=white)](https://github.com/Shiaoming/ALIKED)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2306.13643-b31b1b.svg)](https://arxiv.org/abs/2306.13643)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream licenses, which control your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the authors and contributors accept no liability for any loss or damage arising from use of this material.

---

## Interactive Colab Tutorials

This repository ships one standalone Google Colab tutorial that exercises its public pipeline API end to end — bootstrap a fresh runtime, stage, audit, convert and verify the two pinned checkpoint pickles into the vendored network, build a labelled pair set with exact references from digest-pinned photographs, measure the frozen matcher against three baselines, run a bounded fine-tuning, evaluate on an image-disjoint split, and export and reload the adapter:

- **E2E Fine-tuning Tutorial**: \
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/lightglue-matching-pipeline/blob/main/tutorials/lightglue_matching_colab.ipynb) [`lightglue_matching_colab.ipynb`](https://github.com/kurtvalcorza/lightglue-matching-pipeline/blob/main/tutorials/lightglue_matching_colab.ipynb) \
  *Sparse matching with the pinned `cvg/LightGlue` matcher over ALIKED keypoints, then bounded supervised fine-tuning of the matcher's last layers and assignment head on 216 homography pairs built from CC0 iNaturalist photographs (half of them rotated up to ±150°, the regime the frozen matcher fails on): the frozen matcher's precision at 3 px, inlier count and homography accuracy beside the identity-guess, patch-nearest-neighbour and descriptor-nearest-neighbour baselines, the LightGlue assignment loss with validation-precision epoch selection, held-out evaluation per difficulty tier, a drawn pair re-matched, and a safetensors adapter that reloads with verified parity.*

> [!NOTE]
> The notebook runs on CPU and uses CUDA automatically when present. Its clean-runtime execution record and the promotion requirements are in [release verification](docs/release-verification.md).

---

#### Description

LightGlue (Lindenberger, Sarlin and Pollefeys, ICCV 2023) is a sparse feature matcher: given two sets of keypoints with descriptors it runs nine transformer layers, each a self-attention block with rotary positional encoding over the keypoint coordinates and a cross-attention block between the two images, then a unified assignment head — a matchability score per keypoint and a dual-softmax over descriptor similarities — from which mutual best matches above a threshold are returned, each with one confidence. The checkpoint carried here is the one trained on ALIKED features (`aliked_lightglue.pth`, an asset of the immutable `v0.1_arxiv` GitHub release, Apache-2.0; 11,884,625 parameters, 253 tensors), paired with ALIKED-N(16) (Zhao et al., IEEE TIM 2023; `aliked-n16.pth` from `Shiaoming/ALIKED` at a pinned commit, BSD-3-Clause; 677,356 parameters), a deformable-convolution keypoint detector and descriptor that returns up to 2,048 keypoints with 128-d descriptors per image. There is no Hugging Face repository for either: both files are plain PyTorch pickles, which this repository audits statically, converts once with the weights-only unpickler to safetensors, and loads strictly into a vendored network (`modeling.py`, the upstream LightGlue module and its ALIKED port at the pinned commit, with the adaptive early exit and point pruning switched off so every pair runs all nine layers on every keypoint).

#### Intended Use and Limitations

The sections below outline the primary machine learning tasks, targeted user cohorts, and explicit capability boundaries established for this pipeline.

###### Primary Intended Uses

The primary intended uses of this pipeline comprise four technical capabilities:
1. Image matching (`LightGluePipeline.match`, `extract`): ALIKED keypoints and descriptors per image, and sparse correspondences between two images — `(x0, y0) ↔ (x1, y1)` pairs with a confidence each — for downstream homography, pose or registration solvers the caller supplies.
2. Homography-supervised evaluation (`LightGluePipeline.evaluate`, `evaluate_baselines`): Scoring `{id, image0, image1, homography}` records with exact references on precision, inlier count, median error and homography accuracy, and the same readings for the identity-guess, patch-nearest-neighbour and descriptor-nearest-neighbour baselines.
3. Bounded supervised fine-tuning (`LightGluePipeline.adapt`, `save_artifact`, `from_artifact`): Adapting the matcher's last layers and assignment head to a labelled pair set with the LightGlue assignment loss and validation-based epoch selection, exporting the adapter, and reloading it with verified parity.
4. Pair synthesis with exact references (`samples.make_pair`, `make_pairs`, `build_sample_dataset`): Turning photographs into homography pairs with seeded geometric and photometric changes, so a matcher can be scored without manual annotation.
Target application domains include image registration and stitching research, evaluation of matchers under controlled warps, teaching material about sparse matching and its failure modes, and reproducible experiments on bounded adaptation.

###### Primary Intended Users

Primary intended users are computer-vision engineers, researchers and data scientists building or studying correspondence, registration or structure-from-motion pipelines. Users are expected to understand what a homography is and when a pair of views is (not) related by one, why a match confidence is a thresholded assignment score rather than a probability of correctness, why geometric verification belongs to the downstream solver, and — for the adaptation contract — why a change measured on synthetic warps of one sample is evidence that the contract works rather than a benchmark, why splits must be image- and scene-disjoint, and why the three baselines are read before the adapted number.

###### Out-of-scope use cases

1. **Capability boundaries:** This model returns correspondences only. It does not estimate a homography, a pose or depth (the evaluation's RANSAC-DLT is a metric, not a product), does not detect, segment or describe objects, and does not tell whether two images show the same scene — any two images produce whatever passes the threshold. It matches only the keypoints ALIKED detects: flat, texture-less or heavily blurred regions yield few or none.
2. **Input boundaries:** Accepts local image paths, raw image bytes, and PIL Image instances; images are converted to RGB and passed at their own size (sides in [64, 1024] px; ALIKED pads to a multiple of 32 internally and reports keypoints in the input frame). Remote URLs (`http://`, `https://`) are strictly rejected at the API boundary.
3. **Adaptation boundaries:** `adapt` trains the matcher's last `trainable_layers` transformer layers and the final assignment head only; ALIKED, the input projection, the positional encoding, the earlier layers and the per-layer token confidences stay as they are, so keypoint detection and description cannot be changed through this contract, and a narrow adaptation can erode the matcher outside its set (the tutorial re-matches a drawn pair as a small look at this, not a measurement). Datasets are validated structurally, never semantically: a wrong reference homography is scored without complaint. Mixed real/synthetic pairs, depth- or pose-supervised references and thresholds are not provided.
4. **Decision boundaries:** Autonomous, unreviewed deployment in safety-critical, legal, or punitive workflows — navigation, surveillance, forensic image comparison, medical registration — is strictly prohibited.

---

#### Factors

This section describes factors influencing model representation and behavior, including demographic categories, capturing instruments, and operational runtime environments.

###### Groups

LightGlue was trained on MegaDepth (internet photographs of landmarks with SfM-derived depth and poses) after a homography pre-training on the Oxford-Paris 1M distractor images; ALIKED was trained on MegaDepth and its own synthetic homography data. The training scenes are landmarks, buildings and streets, not people; the matcher has no notion of identity or demographic attributes and matches texture, corners and structure. Applied to imagery of people it would match faces and bodies like any other texture, which is exactly why surveillance and identification uses are out of scope; no demographic audit exists and none is claimed. The tutorial sample is wildlife photographs with no people.

###### Instrumentation

The training pairs are consumer and web photographs of outdoor scenes with real viewpoint change; the tutorial pairs are consumer photographs at 640 px on the long side, warped in software with bicubic resampling and re-lit with brightness, contrast, gamma, blur and Gaussian noise. Key instrumentation factors are texture (ALIKED detects corners and blobs; flat regions yield no keypoints), image scale, motion blur, compression, exposure differences, **in-plane rotation** (neither ALIKED's descriptors nor LightGlue's positional encoding is rotation-invariant, which the tutorial's hard tier exercises on purpose), and — for real pairs — viewpoint and occlusion, which a synthetic warp never produces.

###### Environment

1. **Operating environment:** Designed to run on Python 3.12 with `torch==2.14.0` and `torchvision==0.29.0` (the pinned reference environment; torchvision supplies the deformable convolution ALIKED uses); the network needs torch, torchvision, numpy and pillow only. Supported hardware includes x86_64 CPUs and NVIDIA GPUs supporting CUDA 12.x. Matching one 640 × 480 pair takes about a second on the build workstation's CPU and needs under 2 GB of memory; float32 is the default and the only precision qualified here.
2. **Data environment:** Assumes textured images of a scene under a change the matcher can bridge — viewpoint, scale within a factor of about two, lighting. Behaviour degrades on texture-less surfaces, repeated patterns, large scale changes, strong in-plane rotation (the build record measured 0.500 precision at 3 px on pairs rotated up to ±150° against 0.999 on the easy tier), and pairs that do not overlap at all, where matches are still returned.

---

#### Metrics

This section details performance metrics, decision thresholds, and uncertainty management applied across pipeline operations.

###### Performance Measures

`match` returns correspondences with confidences in (0, 1] — the assignment score of each mutual match above the match threshold (0.1). `evaluate` scores a pair set against its reference homographies: precision at 1 / 3 / 5 px (fraction of returned matches within the threshold, averaged over pairs), matches and inliers (< 3 px) per pair, the median reprojection error of the inliers, and homography accuracy at 3 / 5 px (fraction of pairs whose RANSAC-DLT homography from the matches moves the image corners by less than the threshold against the reference — the HPatches-style reading; a pair with fewer than four inliers counts as a failure); `identity_baseline`, `patch_neighbour_baseline` and `descriptor_nn_match` are scored by the same `pair_metrics`. Every reading is also given per tier (`by_tier`) and per pair (`per_pair`). The per-pair `evaluation_report` on a synthetic drawing is labelled `sample-sanity`. Nothing here is calibrated: no number is a probability that a match is right.

###### Decision thresholds

The pipeline ships the upstream inference thresholds (ALIKED detection score 0.2 with a 2-px non-maximum-suppression radius and at most 2,048 keypoints; LightGlue match threshold 0.1) and **no geometric verification**: a match above the threshold is returned whether or not it is geometrically consistent, and the confidence is not a probability of correctness. The evaluation's RANSAC (3 px inlier threshold, 500 iterations, four-point DLT samples with a refit on the consensus set) is a metric, not a filter the pipeline applies. Downstream operators own outlier rejection and any acceptance threshold, which must be calibrated against their own references by balancing the costs of a wrong match against a missed one.

###### Approaches to uncertainty and variability

Inference is deterministic on CPU for a given image pair (`model.eval()`, no sampling, no adaptive early exit); variations across runs can arise solely from floating-point kernel differences across hardware or non-deterministic GPU kernels. Adaptation is seeded (`seed=0`: pair order) but not bit-reproducible across devices; every corpus metric the tutorial reports is one value on one seeded split (`build_sample_dataset(seed=42)`) of one 360-pair sample under one seeded set of warps, with a 48-pair validation split that selects the epoch — the build record's epoch history (validation precision at 3 px moved 0.723 → 0.735 → 0.745 → 0.732 over epochs 0–3 and homography accuracy 0.729 → 0.708 → 0.708 → 0.688, differences of a few pairs in 48) is the only view of run-to-run stability, and it is not a dispersion estimate. No confidence intervals are reported anywhere.

---

#### Ethical considerations and biases

This section examines data sensitivity, life-critical implications, implemented mitigations, failure risks, and prohibited uses.

###### Data

The matcher weights were trained by the LightGlue authors on MegaDepth after homography pre-training on Oxford-Paris distractor images, and the extractor by the ALIKED authors on MegaDepth and synthetic homographies; the corpora are internet photographs of landmarks and streets, and no instance-level manifest is provided upstream, so the presence of incidental people, vehicles or private property in training frames cannot be ruled out. This repository distributes only open-source Python code, tests, configuration manifests and the vendored network code; no weight blob and no image is distributed through git. The tutorial's adaptation corpus is 360 research-grade iNaturalist photographs of six common North American birds (60 per species, one per observer; American Goldfinch, Chipping Sparrow, Dark-eyed Junco, House Finch, Song Sparrow, White-throated Sparrow), each released under CC0 1.0 by its observer, pinned by photo id, byte size and SHA-256 and fetched from the iNaturalist open-data bucket at run time — no photograph is redistributed by this repository. The pairs are the photographs and their own warped copies: no annotation was made and no annotator was involved.

###### Human Life

LightGlue is a research matcher and is **not** certified, tested, or approved for life-critical applications or high-stakes decision-making. It must never be the sole basis of a navigation, landing, docking, medical-registration, surveillance or forensic decision; correspondences can be numerous and confident and still wrong, and the pipeline provides no verification. Any secondary deployment in human-adjacent safety workflows demands extensive independent domain verification, geometric verification, redundant sensing and continuous human oversight.

###### Mitigations

This repository enforces concrete, inspectable architectural and supply-chain mitigations:
1. **Cryptographic supply-chain locking:** Pinned to the immutable GitHub release tag `v0.1_arxiv` of `cvg/LightGlue` (`aliked_lightglue.pth`, `47,632,827` bytes, SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`) and the commit `683d7c65197395c0b3f01ebe76e1084a27e73a65` of `Shiaoming/ALIKED` (`models/aliked-n16.pth`, `2,738,091` bytes, SHA-256 `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2` — the upstream code fetches this file from the mutable `main` branch; the pipeline pins the commit), each verified by byte size and digest before any byte is unpickled.
2. **Audited pickles, converted once, then never opened again:** each source pickle's stream is walked statically (`pickletools`) and refused unless its globals are exactly the four permitted globals (`collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2`; the audit digests are pinned), then unpickled exactly once with `torch.load(weights_only=True)`, loaded strictly into the vendored network, and saved as a safetensors file (`aliked_lightglue.safetensors` `47,564,948` bytes SHA-256 `9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6`; `aliked-n16.safetensors` `2,719,928` bytes SHA-256 `3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a`) that is itself digest-verified; a snapshot that already holds the safetensors files never opens a pickle, and every other unsafe format in the snapshot directory is refused. The network is vendored as plain PyTorch from the upstream commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`; no Hub code, no third-party matching framework.
3. **SSRF protection:** Rejects remote `http://` and `https://` image paths at the API boundary, accepting only validated local filesystem paths, in-memory bytes, or PIL images. The public `validate_inputs` helper applies exactly these input checks and returns an input manifest of the schema, ceilings, per-image observations and verdict before the model runs.
4. **Exact references, not annotations:** The tutorial's labels are homographies the pipeline itself applied, so every reported precision is against ground truth, not a human judgement; the notebook states that real pairs need depth, pose or a fitted homography before they can be scored.
5. **Baselines scored by the same code:** the identity guess, the patch nearest neighbour and the descriptor nearest neighbour (the pipeline's own ALIKED keypoints matched without the learned matcher) go through the same `pair_metrics` as the model.
6. **Adaptation integrity:** `adapt` validates the dataset before any tensor is built, caches the frozen extractor's features once, trains only the named matcher tensors with every other parameter's `requires_grad` false, restores the base weights on any exception, and records the configuration and epoch history in the artifact; `from_artifact` re-verifies the base snapshot and checks the manifest's format, base identity and weight digests, the file size and SHA-256 and the exact tensor set **before** deserialising, refuses any tensor outside the matcher, and overlays onto a freshly loaded base.

###### Risks and harms

Key identified risks include:
1. **Automation bias:** Operators may treat hundreds of confident matches as a verified registration; without geometric verification a numerous, confident, wrong correspondence set looks like a good one.
2. **Domain bias:** The checkpoints were made for outdoor landmark photographs; on other modalities, indoor scenes, texture-less or repetitive surfaces their behaviour is untested here, and the tutorial's synthetic warps do not exercise viewpoint change or occlusion.
3. **Rotation failure:** in-plane rotations beyond roughly ±45° collapse both the keypoint descriptors and the matcher's positional reasoning (the build record's hard tier); a pair of rotated views returns few, mostly wrong matches with ordinary-looking confidences.
4. **Adversarial and repetitive-structure failures:** Repeated patterns (windows, tiles, text) and adversarial textures produce plausible but wrong correspondences.
5. **Surveillance misuse:** A matcher can be used to register or track imagery of people and property; such uses are prohibited by this card and unsupported by any validation here.
6. **Adaptation risks:** fine-tuning on a small pair set learns that set's warps and re-lighting; a gain measured on image-disjoint but observer-overlapping synthetic pairs can overstate transfer; and a narrow adaptation can erode matching on scenes it never saw.

###### Use cases

The following use cases are strictly prohibited by policy and developer intent:
1. Mass surveillance, tracking or re-identification of people or private property through image registration.
2. Autonomous navigation, landing, docking or targeting without independent verification.
3. Forensic or legal image comparison presented as evidence of identity or provenance.
4. Medical image registration for diagnosis or treatment.
5. Any application that violates the upstream Apache-2.0 (LightGlue) or BSD-3-Clause (ALIKED) license terms, the iNaturalist community guidelines under which the sample photographs were published, or applicable privacy regulations.

---

## Technical Specifications and Architecture

### Architecture Overview

Two networks, both vendored in `modeling.py`:
- **ALIKED-N(16) extractor:** a small deformable-convolution CNN (channels 16 / 32 / 64 / 128, 128-d descriptors, sparse deformable descriptor head with K = 3, M = 16) producing keypoints by differentiable keypoint detection with 2-px non-maximum suppression, at most 2,048 per image above score 0.2, in the input frame. 677,356 parameters; needs torchvision's `DeformConv2d`.
- **LightGlue matcher:** an input projection of the 128-d descriptors to 256-d, a learnable Fourier positional encoding of the normalised keypoint coordinates applied as rotary embeddings, nine transformer layers each of a self-attention block (4 heads, d = 256, GELU MLP) and a cross-attention block between the two images, and a unified assignment head per layer (matchability and a final projection for the dual softmax); the per-layer token confidences that drive upstream's adaptive early exit and point pruning are present but disabled (`depth_confidence = width_confidence = -1`). 11,884,625 parameters; the upstream `confidence_thresholds` buffer is computed, not loaded.
- **Output:** mutual nearest neighbours of the final assignment above the match threshold (0.1), with the assignment score as confidence.
- **Loss (upstream):** the negative log-likelihood of the ground-truth assignment under the log-assignment matrix (matched pairs, unmatched keypoints in each image); the pipeline's adaptation uses exactly this loss on the final layer, with the homography as the ground-truth source (pairs under 3 px reprojection error positive, keypoints with no partner within 5 px unmatched, the rest ignored).

### Checkpoint Invariants and Loading Controls

The snapshot loader (`lightglue_pipeline.model.load_components`) enforces strict supply-chain controls:
1. Pinned matcher source: `cvg/LightGlue` GitHub release `v0.1_arxiv` (2023-06-26; release assets are immutable), asset `aliked_lightglue.pth` — `47,632,827` bytes, SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`, pickle-audit digest `e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932`
2. Pinned extractor source: `Shiaoming/ALIKED` @ `683d7c65197395c0b3f01ebe76e1084a27e73a65`, file `models/aliked-n16.pth` — `2,738,091` bytes, SHA-256 `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2`, pickle-audit digest `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10`
3. Served files (the deterministic conversions, the only files the network is loaded from): `aliked_lightglue.safetensors` — `47,564,948` bytes, SHA-256 `9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6`, 253 float32 tensors; `aliked-n16.safetensors` — `2,719,928` bytes, SHA-256 `3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a`, 76 tensors (68 float32 parameters and running statistics, 8 int64 BatchNorm counters)
4. Upstream parameters: matcher `11,884,625` F32 parameters (input projection, positional encoding, nine layers of self / cross attention, nine assignment heads and nine token-confidence heads); extractor `677,356`
5. Vendored code: `cvg/LightGlue` @ `eb42fee2d71449efb0aa5c10549752b5d75384d8` (`lightglue/utils.py` — the `Extractor` base only —, `lightglue/aliked.py` and `lightglue/lightglue.py`, concatenated into `src/lightglue_pipeline/modeling.py` with the package-relative imports removed; edits, each marked `# vendored:` and listed in `docs/WEIGHTS.md`: kornia's `grayscale_to_rgb` is a channel `expand`, torchvision's `conv1x1` / `conv3x3` are two one-line helpers (`torchvision.ops.deform_conv2d` is the one torchvision call kept), the optional `flash_attn` import, the download-at-construction code, `ALIKED.describe` and the cv2 / kornia image utilities are not carried, and the `confidence_thresholds` buffer is non-persistent so the checkpoint loads strictly), Apache-2.0; the ALIKED port and weights BSD-3-Clause
6. Execution policy: static pickle audit → one weights-only unpickle → strict `load_state_dict` → safetensors export, once; strict `load_state_dict` from safetensors thereafter; no Hub code, no third-party matching framework; every unsafe format in the snapshot directory is refused; the computed `confidence_thresholds` buffer is non-persistent so the load is strict
7. Adapter artifact format: `org.valcorza.lightglue.adapter.v1` — `adapter.safetensors` (the trained tensors only; 48 tensors for the default two layers and the assignment head, 2,567,169 parameters, 10,273,700 bytes) plus `manifest.json` naming the base id, revision and the two weight digests, the tensor names, the file size and SHA-256, the training configuration and the epoch history
8. Tutorial corpus: 360 iNaturalist photographs (CC0 1.0; six species, 60 each, one per observer), `CORPUS_BYTES = 39,223,447`, each pinned by photo id, byte size and SHA-256 in `lightglue_pipeline/samples.py`; split 216 / 48 / 96 by photograph with `build_sample_dataset(seed=42)`, one seeded homography pair per photograph at 640 px on the long side, tiers alternating `easy` (corner jitter ≤ 6 %, rotation ±10°, scale 0.9–1.1) and `hard` (jitter ≤ 18 %, **rotation ±150°**, scale 0.6–1.4, strong photometry) — the hard tier is this profile's own: the XoFTR profile's ±35° left the frozen matcher at 0.981 with nothing for an adaptation to move
9. Build record (CPU, 2026-09-21): the default tutorial path run through the package API on the build workstation's CPU (`torch 2.14.0`, Python 3.12, `CUDA_VISIBLE_DEVICES=-1`, snapshot and photographs pre-staged) — frozen matcher scored on the 96 test pairs in 154.8 s, the three baselines in 1139.5 s (identity 1 s, patch neighbour 1005 s, descriptor neighbour 131 s), ALIKED features of the 264 training and validation pairs cached in 288 s, 3 epochs of the default recipe (lr 1e-4, batch 4, last 2 layers + head, 2,567,169 of 11,884,625 parameters, ground truth 127,211 positive pairs at 3 / 5 px) in 2096.3 s (validation precision at 3 px 0.723 → 0.735 → 0.745 → 0.732, homography accuracy 0.729 → 0.708 → 0.708 → 0.688, train loss 3.721 → 3.225 → 2.947, epoch 2 kept), the adapter 48 tensors / 10,273,700 bytes with match parity on the first test pair (same count True, max abs difference 0.0); test readings — identity guess precision@3px 0.006 / homography acc@3px 0.000; patch neighbour 0.346 / 0.458 (267 matches per pair); descriptor neighbour 0.624 / 0.646 (626 matches per pair; easy 0.981 / hard 0.268); frozen 0.750 (1 px 0.591, 5 px 0.803) / 0.708 (582 matches, 559 inliers per pair, median inlier error 0.37 px; easy 0.999 / hard 0.500); adapted 0.765 / 0.698 (579 matches per pair; easy 0.998 / hard 0.532); drawn pair 32 matches / precision 1.000 frozen → 32 / 1.000 adapted. A Kaggle Tesla T4 recipe probe (`kurtvalcorza/dimer-probe-lightglue-recipe` v1, 2026-09-21, package API at `36a85f7`, 1521 s; frozen 0.750 / hard 0.500 and the same three baselines as the CPU record) ran four arms from a fresh base — A lr 0.0001 × 6 epochs, last 2 layers: test 0.777 (hard 0.556, homography accuracy 0.698), epoch 5 kept, validation 0.723 → 0.735 → 0.736 → 0.729 → 0.732 → 0.744 → 0.727; B lr 0.0003 × 6 epochs, last 2 layers: test 0.772 (hard 0.546, homography accuracy 0.708), epoch 2 kept, validation 0.723 → 0.748 → 0.748 → 0.742 → 0.729 → 0.738 → 0.735; C lr 0.0001 × 6 epochs, last 4 layers: test 0.771 (hard 0.545, homography accuracy 0.719), epoch 2 kept, validation 0.723 → 0.766 → 0.769 → 0.746 → 0.751 → 0.740 → 0.746; D lr 0.0003 × 8 epochs, last 4 layers: test 0.772 (hard 0.547, homography accuracy 0.688), epoch 1 kept, validation 0.723 → 0.750 → 0.739 → 0.727 → 0.743 → 0.724 → 0.729 → 0.732 → 0.722 — every arm landing between 0.771 and 0.777 with the training loss still falling while validation precision wandered 0.72–0.77: a plateau, not a recipe waiting to be found. The last matcher layers learn a few more rotated correspondences from frozen, rotation-variant descriptors and no more; the CPU default (lr 1e-4 × 3, two layers) is kept because it is the cheapest arm inside that plateau and the one the build record was measured with.
10. Recorded in `docs/release-verification.md` as a pre-flight. Executed 2026-09-21: the **committed notebook blob** (`22f03c5` / `d5e562ea`) run top-to-bottom on a clean Kaggle Tesla T4 kernel (`kurtvalcorza/dimer-nb2-lightglue-matching` v1, `torch 2.14.0+cu130`, Python 3.12.13, `cuda`, empty Hugging Face cache, no repository checkout, blob SHA-1 verified against GitHub before execution): 15/15 ok (1 restart after install cell), 1284.6 s, 365 files, 140 MB fetched (GitHub checkpoints + iNaturalist photographs) and digest-verified inside the notebook; comparison {precision_3px: {identity: 0.006, patch_neighbour: 0.346, descriptor_nn: 0.624, frozen: 0.75, adapted: 0.768}, precision_1px: {identity: 0.0, patch_neighbour: 0.293, descriptor_nn: 0.548, frozen: 0.591, adapted: 0.594}, matches_per_pair: {identity: 267.188, patch_neighbour: 267.188, descriptor_nn: 625.583, frozen: 582.146, adapted: 577.896}, inliers_per_pair: {identity: 1.677, patch_neighbour: 91.656, descriptor_nn: 506.292, frozen: 559.198, adapted: 559.938}, median_error_px: {identity: 2.325, patch_neighbour: 0.534, descriptor_nn: 0.343, frozen: 0.371, adapted: 0.372}, homography_acc_3px: {identity: 0.0, patch_neighbour: 0.458, descriptor_nn: 0.646, frozen: 0.708, adapted: 0.719}, homography_acc_5px: {identity: 0.0, patch_neighbour: 0.49, descriptor_nn: 0.656, frozen: 0.74, adapted: 0.781}, delta_vs_frozen: {precision_3px: 0.018, precision_1px: 0.003, matches_per_pair: -4.25, inliers_per_pair: 0.74, median_error_px: 0.001, homography_acc_3px: 0.01, homography_acc_5px: 0.042}} (easy / rotated tiers in the archived result); reload parity {identical_pairs: 4, of: 4}. Not executed: real viewpoint pairs, other extractors or LightGlue checkpoints, repeated seeds (no dispersion), BYOD, and the adapted model on any pairs but that test split.

### Public Inference API

```python
from lightglue_pipeline import load_pipeline, ransac_homography

pipe = load_pipeline(device="cpu")

# 1. Correspondences between two images (RGB, sides in [64, 1024] px; keypoints in the input frame)
result = pipe.match("view_a.jpg", "view_b.jpg")
kpts0, kpts1, confidence = result["kpts0"], result["kpts1"], result["confidence"]

# 2. Keypoints and descriptors of one image (what the matcher sees)
features = pipe.extract("view_a.jpg")  # keypoints (N, 2), descriptors (N, 128), scores (N,)

# 3. Geometric verification is the caller's (the evaluation's RANSAC-DLT is exposed as a helper)
homography, inliers = ransac_homography(kpts0, kpts1, threshold=3.0)
```

### Public Adaptation API

```python
from lightglue_pipeline import LightGluePipeline, build_sample_dataset, fetch_corpus, read_corpus

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)  # 216 / 48 / 96 homography pairs
pipe = LightGluePipeline.from_pretrained(weights_dir="weights/lightglue-aliked")

baselines = pipe.evaluate_baselines(splits["test"])          # identity guess, patch nearest neighbour, descriptor nearest neighbour
frozen = pipe.evaluate(splits["test"])                       # precision_3px, inliers_per_pair, homography_acc_3px, per_pair, by_tier
result = pipe.adapt(splits["train"], splits["validation"], epochs=3, lr=1e-4, batch_size=4, trainable_layers=2)
adapted = pipe.evaluate(splits["test"])
pipe.save_artifact("outputs/adapter")                        # adapter.safetensors + manifest.json
again = LightGluePipeline.from_artifact("outputs/adapter", weights_dir="weights/lightglue-aliked")
```

Dataset contract (`samples.py`): records `{id, image0, image1, homography}` (`id` matching `[A-Za-z0-9_.:-]{1,64}` and unique; PIL images or decodable files with sides in [64, 1024] px; a finite, non-singular 3 × 3 reference mapping image0 pixels to image1 pixels); `validate_dataset(records, *, min_records=4, max_records=5000)`; `make_pair(image, *, seed, tier)`, `make_pairs(records, *, seed, tier=None)`; `split_dataset(images, *, val_fraction=0.15, test_fraction=0.2, seed=0)` (image-disjoint, then pairs; for BYOD photographs); `check_split_disjoint(splits)`; `observer_overlap(splits)`; `load_byod_dataset(path)` (directory or zip of JPEG / PNG files); `write_dataset_csv(records, path)`; `fetch_corpus(cache_dir=None)`, `read_corpus(files)`, `build_sample_dataset(records, *, seed=42, sizes=SAMPLE_SPLIT)`. Metrics (`metrics.py`): `warp_points`, `reprojection_errors`, `dlt_homography`, `ransac_homography`, `pair_metrics`, `matching_metrics`, `identity_baseline`, `patch_neighbour_baseline`, `mutual_nn_matches`.

### Upstream References and Citations

- **LightGlue Paper:** Lindenberger, Sarlin and Pollefeys, *"LightGlue: Local Feature Matching at Light Speed"*, ICCV 2023, arXiv:2306.13643; code and checkpoints at https://github.com/cvg/LightGlue (Apache-2.0), release `v0.1_arxiv`.
- **ALIKED Paper:** Zhao, Wu, Chen, Ge and Lam, *"ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation"*, IEEE Transactions on Instrumentation and Measurement 2023, arXiv:2304.03608; code and checkpoints at https://github.com/Shiaoming/ALIKED (BSD-3-Clause).
- **SuperGlue:** Sarlin, DeTone, Malisiewicz and Rabinovich, *"SuperGlue: Learning Feature Matching with Graph Neural Networks"*, CVPR 2020 — the learned-matcher family LightGlue streamlines.
- **Tutorial corpus:** iNaturalist open data (CC0 photographs under each observer's own licence), https://www.inaturalist.org/pages/developers — bucket https://inaturalist-open-data.s3.amazonaws.com/ ; the same 360 photographs DIMER's SigLIP pipelines pin.
- **Homography-accuracy reading:** Balntas, Lenc, Vedaldi and Mikolajczyk, *"HPatches: A benchmark and evaluation of handcrafted and learned local descriptors"*, CVPR 2017.

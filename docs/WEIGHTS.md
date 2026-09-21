# Weights, vendored code and data provenance

## The pinned checkpoints

LightGlue publishes no Hugging Face repository. The matcher is an asset of an immutable GitHub release and the extractor a file of the ALIKED repository at a pinned commit; both are plain PyTorch state-dict pickles, which the package audits statically and converts once to safetensors — the only files the networks are ever loaded from.

| Item | Value |
|---|---|
| Matcher source | [`cvg/LightGlue`](https://github.com/cvg/LightGlue) release [`v0.1_arxiv`](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv) (2023-06-26; release assets are immutable), asset `aliked_lightglue.pth` — 47,632,827 B, SHA-256 `d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503`; a torch zip pickle whose stream imports exactly `collections.OrderedDict`, `torch.FloatStorage`, `torch._utils._rebuild_tensor_v2` (audit digest `e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932`); 253 float32 tensors, 11,884,625 parameters |
| Extractor source | [`Shiaoming/ALIKED`](https://github.com/Shiaoming/ALIKED) @ `683d7c65197395c0b3f01ebe76e1084a27e73a65`, file `models/aliked-n16.pth` — 2,738,091 B, SHA-256 `5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2`; the upstream LightGlue code fetches this file from the repository's mutable `main` branch, this package pins the commit; stream imports the fleet's four globals (`collections.OrderedDict`, `torch.FloatStorage`, `torch.LongStorage`, `torch._utils._rebuild_tensor_v2`; audit digest `5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10`); 76 tensors (68 float32 parameters and running statistics, 8 int64 BatchNorm counters), 677,356 parameters |
| Served files | `aliked_lightglue.safetensors` — 47,564,948 B, SHA-256 `9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6`; `aliked-n16.safetensors` — 2,719,928 B, SHA-256 `3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a`. Each is the deterministic export of the network's own state dict after a strict load of the source (sorted keys, contiguous tensors); the manifest pins both the source and its conversion |
| Licences | Apache-2.0 (LightGlue code and checkpoints); BSD-3-Clause (ALIKED code and checkpoints — the vendored `aliked.py` keeps its licence header) |
| Manifest | `weights/lightglue-aliked/dimer-base-manifest.json` (format `dimer_release_snapshot`): the two sources with size, digest, source URL and audit digest, each with its `convertsTo` entry; 50,370,918 B of sources in total |

## Audit and conversion (once)

`from_pretrained` / `convert_sources`: for each source that has no conversion yet — verify byte size and SHA-256 → walk the pickle stream with `pickletools` (a torch zip archive's `data.pkl` or a plain pickle) and refuse any global outside the allowed set → `torch.load(map_location="cpu", weights_only=True)` → strict `load_state_dict` into the vendored network → `safetensors.torch.save_file` of `state_dict()` → verify the conversion's size and digest against the manifest. A snapshot that already holds both safetensors files never opens a pickle; `stage_missing_files` does not fetch a source whose conversion is present.

## The vendored networks

| Item | Value |
|---|---|
| Source | `cvg/LightGlue` @ `eb42fee2d71449efb0aa5c10549752b5d75384d8` (Apache-2.0); `lightglue/aliked.py` is the LightGlue authors' BSD-3-Clause port of ALIKED |
| Files carried | `lightglue/utils.py` (the `Extractor` base only), `lightglue/aliked.py`, `lightglue/lightglue.py` — concatenated in dependency order into `src/lightglue_pipeline/modeling.py` with the package-relative imports removed |
| Not carried | the optional `flash_attn` import, the download-at-construction code of both classes, `ALIKED.describe`, the cv2 / kornia image utilities, the SuperPoint / DISK / SIFT / DoGHardNet extractors and their LightGlue weights, the visualisation code |
| Substitutions (each marked `# vendored:`) | kornia's `grayscale_to_rgb` is a channel `expand`; `torchvision.models.resnet.conv1x1` / `conv3x3` are two one-line helpers (`torchvision.ops.deform_conv2d` is the one torchvision call kept); LightGlue's `confidence_thresholds` buffer is registered `persistent=False` (it is computed from the configuration and absent from the checkpoint) so the checkpoint loads with `strict=True` instead of upstream's `strict=False` |
| Configuration | ALIKED `aliked-n16`, at most 2,048 keypoints, detection threshold 0.2, NMS radius 2; LightGlue `aliked` features (128-d input), nine layers, four heads, match threshold 0.1, `depth_confidence = width_confidence = -1` (the adaptive early exit and point pruning are off — every pair runs every layer on every keypoint) |
| Loading | strict `load_state_dict` of both safetensors files; `load_components` asserts 253 matcher tensors / 11,884,625 parameters and 76 extractor tensors / 677,356 parameters |
| Verified | the audit, the conversion, the strict load and a homography pair matched on the build workstation's CPU (`docs/release-verification.md`) |

## DIMER hosting

Upload the two converted files — `aliked_lightglue.safetensors` (48 MB) and `aliked-n16.safetensors` (3 MB) — with the manifest. The profile's inference needs the package (for the vendored networks) and those two files; the source pickles are not needed once the conversions exist and should not be uploaded.

## Tutorial data

| Item | Value |
|---|---|
| Photographs | 360 research-grade iNaturalist photographs of six North American bird species (60 per species, one per observer), CC0 1.0 — the same sample the fleet's SigLIP rows pin (`samples.py`: photo id, observation id, observer, byte size, SHA-256 of the served `medium` JPEG), fetched from `https://inaturalist-open-data.s3.amazonaws.com/photos/<id>/medium.<ext>` at run time; 39,223,447 B in total |
| Pairs | one per photograph: `image0` = the photograph at 640 px on the long side (bicubic, sides cropped to multiples of 8); `image1` = `image0` warped by a seeded homography (PIL `PERSPECTIVE`, bicubic, black outside) with seeded photometric changes (brightness, contrast, gamma, Gaussian blur, Gaussian noise); the reference `H` maps image0 pixels to image1 pixels |
| Tiers | `easy`: corner jitter ≤ 6 % of the short side, rotation ±10°, scale 0.9–1.1, brightness / contrast 0.85–1.15, noise σ 4/255; `hard`: jitter ≤ 18 %, **rotation ±150°**, scale 0.6–1.4, brightness / contrast 0.6–1.4, gamma 0.7–1.4, blur up to σ 1.2, noise σ 10/255. Tiers alternate per photograph within each split. The hard tier is this row's own: with the XoFTR row's ±35° the frozen matcher scored 0.981 precision at 3 px and a 3-epoch adaptation kept epoch 0 (2026-09-20 sweep) — nothing to move; at ±150° the frozen matcher fails on the rotated half, which is the regime the adaptation is asked to learn |
| Splits | `build_sample_dataset(seed=42)`: 36 / 8 / 16 photographs per species → 216 / 48 / 96 pairs; photograph-disjoint by construction and asserted by `check_split_disjoint` (pixel digest of `image0`) |
| Redistribution | none — the repository commits only the pins; the photographs are cached under `weights/inat-birds/` and the pairs live in memory |

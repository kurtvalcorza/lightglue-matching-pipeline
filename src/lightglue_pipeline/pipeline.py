"""LightGlue + ALIKED matching pipeline: the inference contract (`extract`, `match`), homography-supervised
evaluation with three baselines, a bounded adaptation of the matcher's last layers with the LightGlue
assignment loss, and a digest-manifested safetensors adapter."""
# ruff: noqa: E501  -- docstrings and record literals kept on single lines at the fleet width

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .config import (
    DEFAULT_MODEL_KEY,
    DETECTION_THRESHOLD,
    EXTRACTOR_FILENAME,
    EXTRACTOR_SHA256,
    FILTER_THRESHOLD,
    MATCHER_FILENAME,
    MATCHER_PARAMETER_COUNT,
    MATCHER_SHA256,
    MAX_KEYPOINTS,
    MAX_SIDE,
    MIN_SIDE,
    MODEL_ID,
    MODEL_REVISION,
)
from .metrics import (
    identity_baseline,
    matching_metrics,
    mutual_nn_matches,
    pair_metrics,
    patch_neighbour_baseline,
    warp_points,
)
from .model import convert_sources, load_components, stage_missing_files, verify_snapshot

ImageInput = str | Path | bytes | Image.Image

# --------------------------------------------------------------------------
# Adaptation contract (E2E): bounded fine-tuning of the matcher's last transformer layers and its final
# assignment head with the LightGlue assignment loss, supervised by the pairs' exact homographies.
# --------------------------------------------------------------------------
MATCHER_LAYERS = 9  # LightGlue transformer layers (self + cross attention each)
DEFAULT_TRAINABLE_LAYERS = 2  # the last two layers + the final assignment head (2,567,169 params)
POSITIVE_PX = 3.0  # a keypoint pair is a ground-truth match under this symmetric reprojection error
NEGATIVE_PX = 5.0  # a keypoint with no partner under this error is ground-truth unmatched
MAX_EVAL_RECORDS = 5_000
MIN_SCORED_RECORDS = 30  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.lightglue.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"
PARAMETER_COUNT = MATCHER_PARAMETER_COUNT

INPUT_SCHEMA: dict[str, Any] = {
    "images": (
        "PIL.Image.Image, raw bytes, or a local path decodable by Pillow; any mode, converted to RGB; "
        "remote URLs are refused"
    ),
    "image_size": (
        f"sides in [{MIN_SIDE}, {MAX_SIDE}] px; no resizing or cropping (ALIKED pads to a multiple of 32 "
        "internally and reports keypoints in the input frame)"
    ),
    "keypoints": {"max_per_image": MAX_KEYPOINTS, "detection_threshold": DETECTION_THRESHOLD},
    "thresholds": {"match": FILTER_THRESHOLD},
    "output": (
        "kpts0 / kpts1 (M, 2) float32 pixel coordinates (x, y) of the matched ALIKED keypoints and "
        "confidence (M,) in (0, 1] — the assignment score of each mutual match"
    ),
    "validation": (
        "size and decodability only. Nothing checks that the two images show the same scene: any two "
        "images are matched, and a pair with no overlap still returns whatever passes the threshold"
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coerce_image(value: ImageInput) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        image = Image.open(BytesIO(value))
        image.load()
        return image.convert("RGB")
    if isinstance(value, str | Path):
        text = str(value)
        if text.lower().startswith(("http://", "https://")):
            raise ValueError("remote image URLs are not accepted; pass a local path, bytes or a PIL image")
        path = Path(text)
        if not path.is_file():
            raise ValueError(f"image file not found: {path}")
        image = Image.open(path)
        image.load()
        return image.convert("RGB")
    raise ValueError("image must be a local path, bytes or a PIL.Image.Image")


def _check_size(image: Image.Image, what: str) -> None:
    if min(image.size) < MIN_SIDE or max(image.size) > MAX_SIDE:
        raise ValueError(f"{what}: sides must lie in [{MIN_SIDE}, {MAX_SIDE}] px; got {image.size}")


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """RGB float tensor (1, 3, H, W) in [0, 1] — the extractor's input, at the image's own size."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)[None].contiguous()


def validate_inputs(
    image0: ImageInput, image1: ImageInput, *, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validation stage: exactly the checks `match` applies, reported as an input manifest before the model runs."""
    findings: list[dict[str, Any]] = []
    observations = []
    labels = list(names) if names else ["image0", "image1"]
    for label, value in zip(labels, (image0, image1), strict=True):
        image = _coerce_image(value)
        _check_size(image, label)
        width, height = image.size
        observations.append({"name": label, "size": [width, height], "mode": "RGB after conversion"})
    return {"schema": INPUT_SCHEMA, "images": observations, "findings": findings, "verdict": "accepted"}


class LightGluePipeline:
    def __init__(
        self,
        extractor: Any,
        matcher: Any,
        *,
        device: str | torch.device = "cpu",
        checkpoint_path: Path | str | None = None,
        checkpoint_source: str | None = None,
        manifest_verified: bool = False,
        weight_sha256: str | None = None,
        weight_size_bytes: int | None = None,
        extractor_sha256: str | None = None,
    ) -> None:
        self.extractor = extractor
        self.matcher = matcher
        self.model = matcher  # the adaptable network, under the fleet's attribute name
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.checkpoint_source = checkpoint_source
        self.manifest_verified = manifest_verified
        self.weight_sha256 = weight_sha256
        self.weight_size_bytes = weight_size_bytes
        self.extractor_sha256 = extractor_sha256
        self.adapter: dict[str, Any] | None = None
        self.conversion: dict[str, Any] | None = None
        for module in (self.extractor, self.matcher):
            if hasattr(module, "parameters"):
                module.eval()  # ALIKED's BatchNorm statistics must never drift: a train-mode extractor updates them on every forward
                for param in module.parameters():
                    param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        *,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        weights_path: str | Path | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        max_keypoints: int = MAX_KEYPOINTS,
        detection_threshold: float = DETECTION_THRESHOLD,
        filter_threshold: float = FILTER_THRESHOLD,
        progress: Callable[[Any], None] | None = None,
    ) -> LightGluePipeline:
        """Load the two pinned checkpoints into the vendored networks.

        ``weights_dir`` names a fleet snapshot directory holding ``dimer-base-manifest.json``: absent sources
        are staged with :func:`stage_missing_files` (only when ``allow_download=True``), the directory is
        verified against the manifest by :func:`verify_snapshot`, the two safetensors files are produced by
        :func:`convert_sources` when absent (static pickle audit, one weights-only unpickle each, strict load,
        deterministic save, pinned-digest check), and :func:`load_components` loads them as an explicit path.
        """
        conversion = None
        if weights_dir is not None:
            if weights_path is not None:
                raise ValueError("pass either weights_dir or weights_path, not both")
            stage_missing_files(weights_dir, allow_download=allow_download)
            snapshot = verify_snapshot(weights_dir)
            if not snapshot.get("converted"):
                conversion = convert_sources(weights_dir)  # audit + one-time weights-only unpickle
                if progress is not None:
                    progress({"conversion": conversion})
                verify_snapshot(weights_dir)
            weights_path = weights_dir
        extractor, matcher, target_device, _, metadata = load_components(
            device=device,
            cache_dir=cache_dir,
            weights_path=weights_path,
            return_metadata=True,
            max_keypoints=max_keypoints,
            detection_threshold=detection_threshold,
            filter_threshold=filter_threshold,
        )
        pipe = cls(
            extractor,
            matcher,
            device=target_device,
            checkpoint_path=metadata.get("checkpoint_path"),
            checkpoint_source=metadata.get("checkpoint_source"),
            manifest_verified=metadata.get("manifest_verified", False),
            weight_sha256=metadata.get("weight_sha256"),
            weight_size_bytes=metadata.get("weight_size_bytes"),
            extractor_sha256=metadata.get("extractor_sha256"),
        )
        pipe.conversion = conversion
        return pipe

    # ------------------------------------------------------------------ inference contract

    def _features(self, image: Image.Image) -> dict[str, torch.Tensor]:
        """ALIKED keypoints, descriptors and scores of one image, as batched tensors on the device."""
        tensor = _to_tensor(image).to(self.device)
        with torch.inference_mode():
            feats = self.extractor({"image": tensor})
        feats = {k: v.clone() for k, v in feats.items()}  # leave inference mode: tensors may feed autograd
        feats["image_size"] = torch.tensor([[tensor.shape[-1], tensor.shape[-2]]], dtype=torch.float32, device=self.device)
        return feats

    def extract(self, image: ImageInput) -> dict[str, Any]:
        """The extractor half of the contract: up to `MAX_KEYPOINTS` ALIKED keypoints of one image with their
        128-d L2-normalised descriptors and detection scores, in the image's own pixel frame."""
        img = _coerce_image(image)
        _check_size(img, "image")
        feats = self._features(img)
        return {
            "keypoints": feats["keypoints"][0].cpu().numpy().astype(np.float32),
            "descriptors": feats["descriptors"][0].cpu().numpy().astype(np.float32),
            "scores": feats["keypoint_scores"][0].cpu().numpy().astype(np.float32),
            "size": [int(img.width), int(img.height)],
        }

    def match(self, image0: ImageInput, image1: ImageInput) -> dict[str, Any]:
        """Sparse matches between two images: `{kpts0, kpts1, confidence, size0, size1, n_keypoints0,
        n_keypoints1}` with keypoints as (M, 2) float32 (x, y) pixel coordinates in each image's own frame."""
        img0, img1 = _coerce_image(image0), _coerce_image(image1)
        _check_size(img0, "image0")
        _check_size(img1, "image1")
        feats0, feats1 = self._features(img0), self._features(img1)
        with torch.inference_mode():
            out = self.matcher({"image0": feats0, "image1": feats1})
        pairs = out["matches"][0]
        kpts0 = feats0["keypoints"][0][pairs[:, 0]].cpu().numpy().astype(np.float32)
        kpts1 = feats1["keypoints"][0][pairs[:, 1]].cpu().numpy().astype(np.float32)
        conf = out["scores"][0].cpu().numpy().astype(np.float32)
        return {
            "kpts0": kpts0.reshape(-1, 2),
            "kpts1": kpts1.reshape(-1, 2),
            "confidence": conf.reshape(-1),
            "size0": [int(img0.width), int(img0.height)],
            "size1": [int(img1.width), int(img1.height)],
            "n_keypoints0": int(feats0["keypoints"].shape[1]),
            "n_keypoints1": int(feats1["keypoints"].shape[1]),
            "layers": int(out["stop"]),
        }

    def descriptor_nn_match(self, image0: ImageInput, image1: ImageInput) -> dict[str, Any]:
        """The third baseline: the same ALIKED keypoints and descriptors, paired by mutual nearest neighbour in
        descriptor space — no LightGlue."""
        img0, img1 = _coerce_image(image0), _coerce_image(image1)
        _check_size(img0, "image0")
        _check_size(img1, "image1")
        feats0, feats1 = self._features(img0), self._features(img1)
        pairs, sims = mutual_nn_matches(feats0["descriptors"][0].cpu().numpy(), feats1["descriptors"][0].cpu().numpy())
        kpts0 = feats0["keypoints"][0].cpu().numpy()[pairs[:, 0]]
        kpts1 = feats1["keypoints"][0].cpu().numpy()[pairs[:, 1]]
        return {
            "kpts0": np.asarray(kpts0, dtype=np.float64).reshape(-1, 2),
            "kpts1": np.asarray(kpts1, dtype=np.float64).reshape(-1, 2),
            "confidence": np.asarray(sims, dtype=np.float64),
            "size0": [int(img0.width), int(img0.height)],
            "baseline": "ALIKED descriptors, mutual nearest neighbour (no learned matcher)",
        }

    # ------------------------------------------------------------------ evaluation

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        matcher: Callable[[Image.Image, Image.Image], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Homography-supervised scoring of a validated pair dataset with `metrics.matching_metrics`; `matcher`
        substitutes a baseline for the model (same record structure, same scoring)."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        rows = []
        for record in checked:
            result = (
                matcher(record["image0"], record["image1"])
                if matcher is not None
                else self.match(record["image0"], record["image1"])
            )
            size = tuple(result.get("size0", record["image0"].size))
            row = pair_metrics(result, np.asarray(record["homography"]), (int(size[0]), int(size[1])))
            row.update({"id": record["id"], "tier": record["tier"]})
            rows.append(row)
        out = matching_metrics(rows)
        tiers = sorted({r["tier"] for r in rows})
        out["by_tier"] = {
            tier: {
                k: v
                for k, v in matching_metrics([r for r in rows if r["tier"] == tier]).items()
                if k != "definitions"
            }
            for tier in tiers
        }
        out.update(
            {
                "per_pair": [{k: v for k, v in r.items() if k != "inlier_errors"} for r in rows],
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "matcher": "model" if matcher is None else "baseline",
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return out

    def evaluate_baselines(self, records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
        """The three references scored exactly as the model is: two non-neural (identity guess, patch nearest
        neighbour) and the same keypoints without the learned matcher (descriptor mutual nearest neighbour)."""
        out = {}
        for name, fn in (
            ("identity", identity_baseline),
            ("patch_neighbour", patch_neighbour_baseline),
            ("descriptor_nn", self.descriptor_nn_match),
        ):
            result = self.evaluate(records, matcher=fn)
            result["baseline"] = name
            out[name] = result
        return out

    # ------------------------------------------------------------------ adaptation

    def _assignment_forward(self, feats0: Mapping[str, torch.Tensor], feats1: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The matcher's forward to the final log-assignment matrix with gradients: keypoint normalisation,
        input projection, positional encoding, all nine transformer layers (no early exit, no pruning) and the
        last assignment head. Returns (1, M + 1, N + 1) log scores with the dustbin row and column."""
        from .modeling import normalize_keypoints

        m = self.matcher
        kpts0 = normalize_keypoints(feats0["keypoints"], feats0["image_size"]).clone()
        kpts1 = normalize_keypoints(feats1["keypoints"], feats1["image_size"]).clone()
        desc0 = m.input_proj(feats0["descriptors"].detach().contiguous())
        desc1 = m.input_proj(feats1["descriptors"].detach().contiguous())
        encoding0, encoding1 = m.posenc(kpts0), m.posenc(kpts1)
        for layer in m.transformers:
            desc0, desc1 = layer(desc0, desc1, encoding0, encoding1)
        scores, _sim = m.log_assignment[-1](desc0, desc1)
        return scores

    @staticmethod
    def match_ground_truth(
        kpts0: np.ndarray,
        kpts1: np.ndarray,
        homography: np.ndarray,
        *,
        positive_px: float = POSITIVE_PX,
        negative_px: float = NEGATIVE_PX,
    ) -> dict[str, np.ndarray]:
        """Ground-truth assignment of two keypoint sets under a homography, the LightGlue training rule: the
        symmetric reprojection distance (`H · k0` against `k1` and `H⁻¹ · k1` against `k0`, the larger of the
        two), mutual nearest neighbours under `positive_px` are the positive pairs, keypoints with no partner
        under `negative_px` are unmatched, and the rest are ignored by the loss."""
        k0 = np.asarray(kpts0, dtype=np.float64).reshape(-1, 2)
        k1 = np.asarray(kpts1, dtype=np.float64).reshape(-1, 2)
        m, n = len(k0), len(k1)
        if m == 0 or n == 0:
            return {"positives": np.zeros((0, 2), dtype=np.int64), "unmatched0": np.ones(m, dtype=bool), "unmatched1": np.ones(n, dtype=bool)}
        p01 = warp_points(k0, homography)
        p10 = warp_points(k1, np.linalg.inv(np.asarray(homography, dtype=np.float64)))
        d01 = np.linalg.norm(p01[:, None, :] - k1[None, :, :], axis=2)
        d10 = np.linalg.norm(k0[:, None, :] - p10[None, :, :], axis=2)
        dist = np.maximum(np.nan_to_num(d01, nan=np.inf), np.nan_to_num(d10, nan=np.inf))
        nn0 = dist.argmin(axis=1)
        nn1 = dist.argmin(axis=0)
        i = np.arange(m)
        best0 = dist[i, nn0]
        positive = (nn1[nn0] == i) & (best0 < positive_px)
        positives = np.stack([i[positive], nn0[positive]], axis=1).astype(np.int64)
        unmatched0 = dist.min(axis=1) > negative_px
        unmatched1 = dist.min(axis=0) > negative_px
        return {"positives": positives, "unmatched0": unmatched0, "unmatched1": unmatched1}

    @staticmethod
    def assignment_loss(scores: torch.Tensor, ground_truth: Mapping[str, np.ndarray]) -> torch.Tensor:
        """The LightGlue assignment loss on one pair: the negative log-likelihood of the positive pairs in the
        log-assignment matrix, plus half the mean negative log-likelihood of the dustbin entries of the
        unmatched keypoints of each image (`log(1 − σ)`), each term averaged over its own set."""
        s = scores[0]
        zero = s.sum() * 0.0
        pos = torch.as_tensor(ground_truth["positives"], dtype=torch.long, device=s.device)
        un0 = torch.as_tensor(ground_truth["unmatched0"], dtype=torch.bool, device=s.device)
        un1 = torch.as_tensor(ground_truth["unmatched1"], dtype=torch.bool, device=s.device)
        nll_pos = -s[pos[:, 0], pos[:, 1]].mean() if len(pos) else zero
        nll_neg0 = -s[:-1, -1][un0].mean() if bool(un0.any()) else zero
        nll_neg1 = -s[-1, :-1][un1].mean() if bool(un1.any()) else zero
        return nll_pos + 0.5 * (nll_neg0 + nll_neg1)

    def _trainable_names(self, trainable_layers: int) -> list[str]:
        if (
            isinstance(trainable_layers, bool)
            or not isinstance(trainable_layers, int)
            or not 1 <= trainable_layers <= MATCHER_LAYERS
        ):
            raise ValueError(f"trainable_layers must be an int in 1..{MATCHER_LAYERS}")
        first = MATCHER_LAYERS - trainable_layers
        prefixes = tuple(f"transformers.{k}." for k in range(first, MATCHER_LAYERS)) + (
            f"log_assignment.{MATCHER_LAYERS - 1}.",
        )
        return [name for name, _p in self.matcher.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 1e-4,
        batch_size: int = 4,
        trainable_layers: int = DEFAULT_TRAINABLE_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the matcher on validated pairs.

        Only the last `trainable_layers` transformer layers of LightGlue and its final assignment head
        (`log_assignment.8`: matchability and final projection) train — 2,567,169 of 11,884,625 parameters by
        default; ALIKED, the input projection, the positional encoding, the earlier layers and the token
        confidences stay frozen. ALIKED features of the training pairs are extracted once and cached (the
        extractor is frozen). Each pair runs the matcher to the final log-assignment matrix and is scored with
        the LightGlue assignment loss against the homography's ground-truth assignment; `batch_size` pairs are
        accumulated per AdamW step (pairs have different keypoint counts, so they are not stacked), gradients
        are clipped at 1.0, the order is seeded, no scheduler. Epoch 0 records the frozen model's validation
        metrics; the epoch with the highest validation precision at 3 px is kept (ties broken by homography
        accuracy at 3 px). Transactional: any failure restores the base tensors."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        names = self._trainable_names(trainable_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        model = self.matcher
        torch.manual_seed(seed)
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        cached = []
        for record in train_checked:
            feats0, feats1 = self._features(record["image0"]), self._features(record["image1"])
            gt = self.match_ground_truth(feats0["keypoints"][0].cpu().numpy(), feats1["keypoints"][0].cpu().numpy(), np.asarray(record["homography"]))
            cached.append((feats0, feats1, gt))
        n_positive = int(sum(len(gt["positives"]) for _f0, _f1, gt in cached))
        feature_seconds = round(time.perf_counter() - started, 2)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            result = self.evaluate(val_checked)
            return {k: result[k] for k in ("precision_3px", "homography_acc_3px", "inliers_per_pair", "matches_per_pair", "n")}

        def key(entry: dict[str, Any]) -> tuple[float, float]:
            return (entry["val"]["precision_3px"], entry["val"]["homography_acc_3px"]) if entry["val"] else (-math.inf, -math.inf)

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_key = key(entry)
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                self.extractor.eval()  # the features are cached, but any later _features call must not touch BatchNorm statistics
                order = torch.randperm(len(cached), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    optimiser.zero_grad(set_to_none=True)
                    chunk = order[start : start + batch_size]
                    total = 0.0
                    for i in chunk:
                        feats0, feats1, gt = cached[i]
                        scores = self._assignment_forward(feats0, feats1)
                        loss = self.assignment_loss(scores, gt) / len(chunk)
                        loss.backward()
                        total += float(loss.detach())
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(total)
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                if not entry["val"] or key(entry) > best_key:
                    best_key = key(entry)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_layers": trainable_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation precision at 3 px (ties: homography accuracy at 3 px)"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_val": len(val_checked),
            "ground_truth": {"positive_px": POSITIVE_PX, "negative_px": NEGATIVE_PX, "positive_pairs": n_positive},
            "feature_seconds": feature_seconds,
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ------------------------------------------------------------------ artifacts

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted matcher tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in self.matcher.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": DEFAULT_MODEL_KEY,
                "weight_file": MATCHER_FILENAME,
                "weight_sha256": MATCHER_SHA256,
                "extractor_file": EXTRACTOR_FILENAME,
                "extractor_sha256": EXTRACTOR_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256_file(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"artifact format_version {manifest.get('format_version')!r} != {ARTIFACT_FORMAT_VERSION!r}")
        base = manifest.get("base_model") or {}
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (MODEL_ID, MODEL_REVISION, MATCHER_SHA256):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", MATCHER_FILENAME) != MATCHER_FILENAME or base.get("extractor_sha256", EXTRACTOR_SHA256) != EXTRACTOR_SHA256:
            raise ValueError("artifact was adapted from a different base weight file or extractor")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1 or files[0].get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must list exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / ARTIFACT_WEIGHTS_NAME).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        layers = (manifest.get("adapter") or {}).get("trainable_layers")
        expected = sorted(self._trainable_names(layers))
        if sorted(manifest.get("tensors") or []) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        entry = files[0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256_file(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = self.matcher.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith(("transformers.", "log_assignment.")):
                raise ValueError(f"artifact tensor {key} is not an adaptable matcher tensor of the base")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(f"artifact tensor {key} has shape {tuple(value.shape)}, base has {tuple(state[key].shape)}")
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        self.matcher.load_state_dict(merged, strict=True)
        self.matcher.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": manifest["tensors"], "history": manifest.get("history", [])}
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> LightGluePipeline:
        """A fresh pipeline from the pinned base with an adapter overlaid."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe


def load_pipeline(**kwargs: Any) -> LightGluePipeline:
    return LightGluePipeline.from_pretrained(**kwargs)


def evaluation_report(result: Mapping[str, Any], reference: Mapping[str, Any] | None = None, *, sample_kind: str = "synthetic") -> dict[str, Any]:
    """Evaluation stage for the drawn-shape sanity pair: a machine-readable report even when nothing is
    measurable. With a `reference` homography and image size the report carries the pair metrics with the
    verdict `sample-sanity`; without it the verdict is `not-measurable`."""
    base: dict[str, Any] = {
        "task": "sparse image matching (ALIKED keypoints, LightGlue assignment)",
        "score_semantics": (
            "match confidences are LightGlue assignment scores in (0, 1] (mutual matches over the threshold), "
            "not calibrated probabilities that a match is correct; the decision rule is the upstream match "
            "threshold; no geometric verification ships"
        ),
        "sample_kind": sample_kind,
        "n_matches": int(len(np.asarray(result.get("kpts0", [])).reshape(-1, 2))),
    }
    if not reference:
        return {**base, "metrics": [], "verdict": "not-measurable"}
    row = pair_metrics(result, np.asarray(reference["homography"]), tuple(reference["size"]))
    metrics = [
        {"id": "precision_3px", "value": row["precision_3px"], "definition": "fraction of matches under 3 px reprojection error"},
        {"id": "n_inliers", "value": row["n_inliers"], "definition": "matches under 3 px"},
        {"id": "corner_error_px", "value": row["corner_error_px"], "definition": "mean corner displacement of the RANSAC-DLT homography vs the reference"},
    ]
    return {**base, "metrics": metrics, "verdict": "sample-sanity"}


__all__ = [
    "ARTIFACT_FORMAT",
    "DEFAULT_TRAINABLE_LAYERS",
    "INPUT_SCHEMA",
    "LightGluePipeline",
    "MATCHER_LAYERS",
    "MAX_EVAL_RECORDS",
    "MIN_SCORED_RECORDS",
    "NEGATIVE_PX",
    "PARAMETER_COUNT",
    "POSITIVE_PX",
    "evaluation_report",
    "load_pipeline",
    "validate_inputs",
]

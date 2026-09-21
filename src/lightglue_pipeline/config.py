from __future__ import annotations

# LightGlue (Lindenberger, Sarlin and Pollefeys, ICCV 2023) with ALIKED local features. LightGlue
# publishes no Hugging Face repository: the matcher checkpoint is an asset of the immutable GitHub
# release tag below, and the ALIKED extractor checkpoint is a file of the ALIKED repository at a
# pinned commit. Both are plain PyTorch state-dict pickles, audited statically and converted once to
# safetensors (the only files the network is ever loaded from).
MODEL_ID = "cvg/LightGlue"
# GitHub release tag (2023-06-26); its assets are immutable.
MODEL_REVISION = "v0.1_arxiv"
MODEL_REVISION_KIND = "github-release-tag"
# LightGlue code and weights.
MODEL_LICENSE = "Apache-2.0"
EXTRACTOR_LICENSE = "BSD-3-Clause"  # ALIKED code (the port carried in modeling.py) and weights

# --- the matcher: LightGlue trained on ALIKED features -------------------------------------------
MATCHER_SOURCE_FILENAME = "aliked_lightglue.pth"
MATCHER_SOURCE_URL = (
    f"https://github.com/{MODEL_ID}/releases/download/{MODEL_REVISION}/{MATCHER_SOURCE_FILENAME}"
)
MATCHER_SOURCE_SHA256 = "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503"
MATCHER_SOURCE_SIZE_BYTES = 47_632_827
MATCHER_PICKLE_AUDIT_SHA256 = "e7b998d087a5dcadd37713daf30b63cc571160c3180ebc138500ab662197e932"
MATCHER_FILENAME = "aliked_lightglue.safetensors"  # the deterministic conversion of the source
MATCHER_SHA256 = "9c630a386c74c534428370ce46253e1d0968655db180f97074cb6ad797bd2bc6"
MATCHER_SIZE_BYTES = 47_564_948
MATCHER_STATE_TENSORS = 253  # all float32 parameters; the confidence-threshold buffer is computed
MATCHER_PARAMETER_COUNT = 11_884_625

# --- the extractor: ALIKED-N(16) ----------------------------------------------------------------
EXTRACTOR_REPOSITORY = "Shiaoming/ALIKED"
EXTRACTOR_COMMIT = "683d7c65197395c0b3f01ebe76e1084a27e73a65"
EXTRACTOR_SOURCE_FILENAME = "aliked-n16.pth"
EXTRACTOR_SOURCE_URL = (
    f"https://raw.githubusercontent.com/{EXTRACTOR_REPOSITORY}/{EXTRACTOR_COMMIT}"
    f"/models/{EXTRACTOR_SOURCE_FILENAME}"
)
EXTRACTOR_SOURCE_SHA256 = "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"
EXTRACTOR_SOURCE_SIZE_BYTES = 2_738_091
EXTRACTOR_PICKLE_AUDIT_SHA256 = "5b9f0ba08490293d6c17b9cef219991e1a6edda31609429679f8dca1af5a7b10"
EXTRACTOR_FILENAME = "aliked-n16.safetensors"
EXTRACTOR_SHA256 = "3c8ca40c0c985cd4d641e96e4b408b14d067b5b3521ac17b36590447d49d115a"
EXTRACTOR_SIZE_BYTES = 2_719_928
EXTRACTOR_STATE_TENSORS = 76  # 68 float32 parameter / running-statistic tensors + 8 int64 counters
EXTRACTOR_PARAMETER_COUNT = 677_356

# The globals a checkpoint pickle may import (the fleet's four); anything else fails the audit.
CKPT_ALLOWED_GLOBALS = frozenset(
    {
        "collections.OrderedDict",
        "torch.FloatStorage",
        "torch.LongStorage",
        "torch._utils._rebuild_tensor_v2",
    }
)

# The served (primary) weight file for the fleet's single-file conventions is the matcher.
MODEL_FILENAME = MATCHER_FILENAME
MODEL_SHA256 = MATCHER_SHA256
MODEL_SIZE_BYTES = MATCHER_SIZE_BYTES
SOURCE_FILENAMES = (MATCHER_SOURCE_FILENAME, EXTRACTOR_SOURCE_FILENAME)
CONVERTED_FILENAMES = (MATCHER_FILENAME, EXTRACTOR_FILENAME)

DEFAULT_MODEL_KEY = "lightglue-aliked"
UNSAFE_WEIGHT_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".h5",
    ".msgpack",
)
ALLOWED_CHECKPOINT_FILES = CONVERTED_FILENAMES

# Inference contract.
MAX_KEYPOINTS = 2048  # ALIKED keypoints per image (upstream demo default)
DETECTION_THRESHOLD = 0.2  # ALIKED keypoint score threshold (upstream default)
NMS_RADIUS = 2  # ALIKED non-maximum suppression radius, px (upstream default)
FILTER_THRESHOLD = 0.1  # LightGlue match threshold on the assignment score (upstream default)
DEPTH_CONFIDENCE = -1.0  # upstream's adaptive early exit (0.95) is off: all nine layers always run
WIDTH_CONFIDENCE = -1.0  # upstream's adaptive point pruning (0.99) is off: no keypoint is dropped
DIVISIBLE_BY = 8  # sample pairs are built with sides that are multiples of this (ALIKED pads to 32)
MIN_SIDE = 64
MAX_SIDE = 1024

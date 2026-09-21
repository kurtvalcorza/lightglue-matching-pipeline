from lightglue_pipeline import DEFAULT_WEIGHTS_DIR, LightGluePipeline, ransac_homography


def main() -> None:
    # Stages the two pinned checkpoints on first use (GitHub release asset + pinned ALIKED commit),
    # audits and converts them once to safetensors, then loads strictly.
    pipe = LightGluePipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR, allow_download=True)
    result = pipe.match("view_a.jpg", "view_b.jpg")
    print(f"{result['n_keypoints0']} / {result['n_keypoints1']} keypoints, {len(result['kpts0'])} matches; "
          f"confidence range {result['confidence'].min():.3f}..{result['confidence'].max():.3f}")
    homography, inliers = ransac_homography(result["kpts0"], result["kpts1"], threshold=3.0)
    print(f"RANSAC-DLT homography from {int(inliers.sum())} inliers:\n{homography}")


if __name__ == "__main__":
    main()

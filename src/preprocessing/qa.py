"""Reproducible QA for explicitly selected, validated whole mammograms."""

from pathlib import Path

from src.preprocessing.canonical import preprocess_dicom
from src.preprocessing.manifests import local_path, validated_manifests
from src.utils.config import load_config


def preprocessing_qa(root: Path, image_ids: list[str], destination: Path) -> None:
    """Plot the same images at three stages; selection order is sorted by image ID."""
    if not image_ids or len(set(image_ids)) != len(image_ids):
        raise ValueError("Select at least one unique validated image ID")
    config = load_config(root / "config/project.yaml")
    splits, _ = validated_manifests(root, config)
    lookup = {r["canonical_image_id"]: r for rows in splits.values() for r in rows}
    if any(image_id not in lookup for image_id in image_ids):
        raise ValueError("QA selection contains an unvalidated image")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(image_ids), 3, figsize=(10, 4 * len(image_ids)), squeeze=False)
    try:
        for axes_row, image_id in zip(axes, sorted(image_ids)):
            path = local_path(root, lookup[image_id]["dicom_relative_path"], ("data", "raw", "CBIS-DDSM", "images"))
            result = preprocess_dicom(path, config, keep_stages=True)
            for ax, pixels, title in zip(axes_row, (result.decoded, result.cropped, result.image),
                ("Decoded / linear display", f"Crop: {result.metadata['crop_reason']}",
                 f"Canonical {config['input_size']} x {config['input_size']}")):
                ax.imshow(pixels, cmap="gray", vmin=0, vmax=1)
                ax.set_title(title, fontsize=9)
                ax.set_axis_off()
            axes_row[0].set_title(image_id + "\nDecoded / linear display", fontsize=8)
        fig.tight_layout()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200)
    finally:
        plt.close(fig)

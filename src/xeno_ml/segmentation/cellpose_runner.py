"""
Segmentation wrapper around Cellpose that

1.  Loads one or more embryo images.
2.  Segments them with the pretrained “cyto2” model.
3.  Saves one *.npy* mask per image.
4.  Logs simple metrics to MLflow.
5.  Builds an HTML (and PDF if possible) phenotyping report and
    logs it as an MLflow artifact.

Usage
-----
>>> from pathlib import Path
>>> from xeno_ml.segmentation.cellpose_runner import segment
>>> imgs = Path("data/raw/embryos").glob("*.jpg")
>>> segment(list(imgs), Path("outputs/masks"), gpu=False)
"""

from pathlib import Path
from typing import List

import numpy as np
from cellpose import io, models
import mlflow

from xeno_ml.segmentation.mlflow_utils import new_run
from xeno_ml.segmentation.report import build_report


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────
def segment(image_paths: List[Path], out_dir: Path, gpu: bool = False) -> None:
    """
    Segment each image in *image_paths* and write masks to *out_dir*.

    Parameters
    ----------
    image_paths : list[Path]
        List of PNG/JPG/TIFF files to segment.
    out_dir : Path
        Directory where `<stem>_mask.npy` (and report.*) are written.
    gpu : bool, default False
        If True and CUDA is available, Cellpose will use the GPU.
    """
    if not image_paths:
        raise ValueError("No images supplied to segment().")

    with new_run("cellpose-seg"):
        mlflow.log_param("n_images", len(image_paths))
        mlflow.log_param("gpu", gpu)

        # ── run Cellpose ────────────────────────────────────────────────────
        model = models.Cellpose(model_type="cyto2", gpu=gpu)
        imgs: list[np.ndarray] = [io.imread(str(p)) for p in image_paths]
        results = model.eval(imgs, diameter=None, normalize=True)
        masks: list[np.ndarray] = results[0]  # first element always masks list

        # ── save masks + collect simple stats ──────────────────────────────
        out_dir.mkdir(parents=True, exist_ok=True)
        has_mask = []
        for img_path, mask in zip(image_paths, masks):
            np.save(out_dir / f"{img_path.stem}_mask.npy", mask.astype("uint16"))
            has_mask.append(int((mask > 0).any()))

        mlflow.log_metric("images_with_mask", sum(has_mask))
        mlflow.log_metric("images_no_mask", len(image_paths) - sum(has_mask))

        # ── build & log report (HTML always, PDF if GTK present) ───────────
        report_path = build_report(imgs, masks, out_dir / "report.pdf")
        mlflow.log_artifact(str(report_path))

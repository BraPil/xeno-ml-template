"""
Lightweight Cellpose wrapper.
Segments a list of image files and writes NumPy mask arrays
(one `.npy` per input) to an output directory.

Usage (inside Poetry env):
>>> from pathlib import Path
>>> from xeno_ml.segmentation.cellpose_runner import segment
>>> imgs = list(Path("data/raw/embryos").glob("*.jpg"))
>>> segment(imgs, Path("outputs/masks"), gpu=False)
"""

from pathlib import Path
from typing import List

import numpy as np
from cellpose import models, io

from xeno_ml.segmentation.mlflow_utils import new_run

# ...

def segment(image_paths: List[Path], out_dir: Path, gpu: bool = False) -> None:
    if not image_paths:
        raise ValueError("No images supplied to segment().")

    with new_run("cellpose-seg"):
        mlflow.log_param("n_images", len(image_paths))
        mlflow.log_param("gpu", gpu)

        model = models.Cellpose(model_type="cyto2", gpu=gpu)
        imgs  = [io.imread(str(p)) for p in image_paths]

        results = model.eval(imgs, diameter=None, normalize=True)
        masks   = results[0]

        areas = []
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path, mask in zip(image_paths, masks):
            np.save(out_dir / f"{img_path.stem}_mask.npy", mask.astype("uint16"))
            areas.append(int(mask.sum() > 0))          # 1 if any mask pixels

        mlflow.log_metric("images_with_mask", sum(areas))
        mlflow.log_metric("images_no_mask",   len(image_paths) - sum(areas))


def segment(image_paths: List[Path], out_dir: Path, gpu: bool = False) -> None:
    """Segment every image in `image_paths` with Cellpose cyto2 model.

    Args:
        image_paths: list of file paths (PNG/JPG/TIFF…).
        out_dir: directory to write `<stem>_mask.npy` files.
        gpu: set True if CUDA is available.
    """
    if not image_paths:
        raise ValueError("No images supplied to segment().")

    model = models.Cellpose(model_type="cyto2", gpu=gpu)
    imgs = [io.imread(str(p)) for p in image_paths]

    # --- run inference ----------------------------------------------------
    results = model.eval(imgs, diameter=None, normalize=True)

    # Cellpose returns (masks, flows, styles [, diams])
    masks = results[0]
    # we ignore the rest for now


    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path, mask in zip(image_paths, masks):
        np.save(out_dir / f"{img_path.stem}_mask.npy", mask.astype("uint16"))

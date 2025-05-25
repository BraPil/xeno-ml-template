from pathlib import Path
import numpy as np
from xeno_ml.segmentation.cellpose_runner import segment

def test_segment_runs(tmp_path: Path):
    """Smoke-test: segment one embryo image, mask file appears."""
    sample = next(Path("data/raw/embryos").glob("*.jpg"))
    segment([sample], tmp_path, gpu=False)
    out_file = tmp_path / f"{sample.stem}_mask.npy"
    assert out_file.exists() and np.load(out_file).ndim == 2

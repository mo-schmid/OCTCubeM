from collections.abc import Sequence

import numpy as np
from monai.data.image_reader import ImageReader
from monai.utils import MetaKeys, ensure_tuple
from PIL import Image


class ImageSeriesReader(ImageReader):
    """Loads a list of single-frame grayscale images (one B-scan each) into one (D, H, W) volume.

    Ported from RetiMod. MONAI's default multi-file loading concatenates same-shape
    images along the *channel* axis (meant for stacking co-registered modalities),
    which would merge B-scans into channels instead of a depth axis -- so this reader
    stacks them along a new leading axis itself. A single file is the `D == 1` case.
    Unlike MONAI's `PILReader`, it keeps the `(H, W)` order rather than reversing it.

    Args:
        suffixes: file extensions `verify_suffix` accepts, lowercase and
            leading-dot. The default covers every per-B-scan format in use, so
            datamodules can pass the class itself rather than an instance.
    """

    def __init__(self, suffixes: Sequence[str] = (".png", ".tif", ".tiff", ".jpg", ".jpeg")):
        super().__init__()
        self.suffixes = tuple(suffixes)

    def verify_suffix(self, filename: Sequence[str] | str) -> bool:
        return all(str(name).lower().endswith(self.suffixes) for name in ensure_tuple(filename))

    def read(self, data: Sequence[str] | str, **kwargs) -> np.ndarray:
        slices = [np.asarray(Image.open(name).convert("L")) for name in ensure_tuple(data)]
        return np.stack(slices, axis=0)

    def get_data(self, img: np.ndarray) -> tuple[np.ndarray, dict]:
        meta = {MetaKeys.SPATIAL_SHAPE: np.asarray(img.shape), MetaKeys.ORIGINAL_CHANNEL_DIM: "no_channel"}
        return img, meta

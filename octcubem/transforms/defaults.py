"""Transforms shared by every datamodule: load -> `standardize` -> train/eval augment -> `normalize`.

A datamodule only supplies its loader; `standardize` brings every dataset to the same
per-sample layout, a `(1, D, H, W)` uint8 `image` (plus a `(D, H, W)` long `mask` for
segmentation). Injected train/eval augments therefore see images on the 0-255 scale.

`normalize` scales images to `[0, 1]` once per batch, as the very last step -- in the
dataloader's `collate_fn`, or after the transfer to the GPU, so the batch crosses the worker
IPC and the host->device copy as uint8. Running last in both cases keeps the two paths
numerically identical, whatever augmentations ran before it.
"""

import math
from collections.abc import Sequence

import monai.transforms as mt
import torch
import torch.nn.functional as F
from monai.data import MetaTensor

# torchvision's `Normalize` constants, which OCTCube's 2D branch applies to its RGB-converted B-scans
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def standardize(
    keys: Sequence[str],
    depth: int,
    image_size: int | tuple[int, int],
    mask_key: str | None = None,
    channel_dim: str | int | None = None,
    resize_mode: str = "area",
) -> list[mt.Transform]:
    """Geometry every sample goes through: channel-first, `depth` B-scans, `image_size` in-plane, uint8.

    Args:
        keys: Every spatial key, the image and the mask (if any) alike.
        depth: B-scans per sample; volumes are centre-cropped, or zero-padded, to it.
        image_size: The in-plane side of a square B-scan, or `(H, W)`.
        mask_key: The key among `keys` holding a class-id mask, resized nearest-neighbour
            and returned as a long `(D, H, W)` class-id tensor.
        channel_dim: Forwarded to `EnsureChannelFirstD`, e.g. `"no_channel"` for a raw array
            that carries no channel metadata.
        resize_mode: Interpolation of the in-plane image resize; OCTCube's own
            pipeline uses `"trilinear"`, which with depth untouched is bilinear in-plane.
            `"bicubic"` resizes every B-scan antialiased as PIL does (`InPlaneBicubicResizeD`),
            as OCTCube's 2D branch does.
    """
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    image_keys = [key for key in keys if key != mask_key]

    if resize_mode == "bicubic":
        resize = InPlaneBicubicResizeD(keys=image_keys, size=(height, width))
    else:
        resize = mt.ResizeD(keys=image_keys, spatial_size=(-1, height, width), mode=resize_mode)

    transforms = [
        mt.EnsureChannelFirstD(keys=keys, channel_dim=channel_dim),
        # select the central B-scans as OCTCube does, zero-padding volumes with fewer
        # than `depth` B-scans so every sample has the same size
        mt.ResizeWithPadOrCropD(keys=keys, spatial_size=(depth, -1, -1)),
        resize,
        # `ResizeD` upcasts to float32 -- back to uint8, the dtype the batch ships in
        ToUint8D(keys=image_keys),
    ]
    if mask_key is not None:
        transforms.append(mt.ResizeD(keys=[mask_key], spatial_size=(-1, height, width), mode="nearest"))
    if mask_key is not None:
        # the mask indexes classes rather than carrying features, so it drops the
        # channel axis `EnsureChannelFirstD` added, and is cast to long class ids
        transforms += [
            mt.SqueezeDimD(keys=[mask_key], dim=0),
            mt.CastToTypeD(keys=[mask_key], dtype=torch.long),
        ]
    # plain tensors, not `MetaTensor`s: `PersistentDataset` caches plain tensors, and a
    # batch mixing cached and freshly computed samples would fail to collate
    transforms.append(mt.ToTensorD(keys=keys, track_meta=False))
    return transforms


def normalize(batch: dict, keys: Sequence[str], max_value: float = 255.0) -> dict:
    """Scales the images under `keys` from `[0, max_value]` to float `[0, 1]`; missing keys are skipped."""
    batch = dict(batch)
    for key in keys:
        if key not in batch:
            continue
        image = batch[key].float()
        image = image / max_value
        batch[key] = image.clamp(0.0, 1.0)
    return batch


def octcube_2d_input(image: torch.Tensor, imagenet_normalize: bool = True) -> torch.Tensor:
    """Turns a `[0, 1]` batch of single B-scans `(B, 1, 1, H, W)` into OCTCube's 2D model input `(B, 1, 3, H, W)`.

    OCTCube's 2D branch loads a B-scan as RGB, so `ToTensor` makes 3 identical channels,
    which the model then reads as 3 frames -- one temporal patch. `Normalize` with the
    ImageNet constants follows, which makes the 3 frames differ slightly.
    """
    image = image.expand(-1, -1, 3, -1, -1)
    if not imagenet_normalize:
        return image
    mean = image.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (image - mean) / std


class InPlaneBicubicResizeD(mt.MapTransform):
    """Resizes every B-scan of a `(C, D, H, W)` image to `size`, bicubic and antialiased.

    Matches PIL's `resize(..., BICUBIC)`, which OCTCube's 2D branch uses through
    torchvision's `Resize(interpolation=3)` -- MONAI's `ResizeD` has no antialiased
    bicubic for a 3D image. Returns float; follow it with `ToUint8D`.
    """

    def __init__(self, keys: Sequence[str], size: tuple[int, int]):
        super().__init__(keys)
        self.size = tuple(size)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[key]
            if isinstance(image, MetaTensor):
                image = image.as_tensor()
            image = torch.as_tensor(image).float()
            channels, depth, height, width = image.shape
            image = F.interpolate(
                image.reshape(channels * depth, 1, height, width),
                size=self.size,
                mode="bicubic",
                antialias=True,
                align_corners=False,
            )
            d[key] = image.reshape(channels, depth, *self.size)
        return d


class ToUint8D(mt.MapTransform):
    """Rounds, clips to 0-255 and casts to uint8 -- undoes the float upcast of an interpolating transform."""

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = torch.as_tensor(d[key])
            if image.dtype != torch.uint8:
                image = image.round()
                image = image.clamp(0, 255)
                image = image.to(torch.uint8)
            d[key] = image
        return d


class InPlaneAugment(mt.Compose):
    """Random flip, rotation and translation within the B-scan plane `(H, W)`, image keys only.

    The depth axis is the B-scan sequence order, which augmentation leaves alone. Ends in
    `ToUint8D`, so the sample stays uint8 for `normalize`.

    Args:
        keys: Image keys to augment -- not masks, which would need nearest-neighbour
            resampling and no uint8 cast.
        max_rotation_deg: Largest in-plane rotation, either way.
        max_translation: Largest shift along `H` in voxels.
        prob: Probability of each of the flip and the affine.
    """

    def __init__(
        self,
        keys: Sequence[str] = ("image",),
        max_rotation_deg: float = 10.0,
        max_translation: float = 22.4,
        prob: float = 0.5,
    ):
        super().__init__(
            [
                mt.RandFlipD(keys=keys, spatial_axis=[1, 2], prob=prob),
                mt.RandAffineD(
                    keys=keys,
                    prob=prob,
                    rotate_range=(math.radians(max_rotation_deg), 0, 0),
                    translate_range=(0, max_translation, 0),
                ),
                ToUint8D(keys=keys),
            ]
        )


class FlipAugment(mt.Compose):
    """Random flips along depth and left-right, each drawn independently, image keys only.

    OCTCube's pretraining augmentation (its `create_3d_transforms`). Ends in
    `ToUint8D`, so the sample stays uint8 for `normalize`.
    """

    def __init__(self, keys: Sequence[str] = ("image",), prob: float = 0.5):
        super().__init__(
            [
                mt.RandFlipD(keys=keys, spatial_axis=0, prob=prob),
                mt.RandFlipD(keys=keys, spatial_axis=2, prob=prob),
                ToUint8D(keys=keys),
            ]
        )

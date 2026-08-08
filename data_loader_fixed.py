"""
Fixed data loader for Thalia WebDataset shards.
Fixes applied:
  1. Z-score normalization using correct channel order from statistics.json
  2. RandomMix for balanced pos/neg loading
  3. Correct label extraction from sample.pth

SERVER VERSION (par-sce-ai-02): only the __main__ test block's paths were
changed (Mac -> server). All loading/normalization/label logic is identical
to the original, verified Mac version.
"""
import io
import json
import random
from glob import glob
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset

# Channel order as hardcoded in Thalia/utilities/utils.py
# This is the order channels are stored in image.pth for every timestep
_CHANNELS_PER_TIMESTEP = [
    "insar_difference",
    "insar_coherence",
    "dem",
    "primary_date_total_column_water_vapour",
    "secondary_date_total_column_water_vapour",
    "primary_date_surface_pressure",
    "secondary_date_surface_pressure",
    "primary_date_vertical_integral_of_temperature",
    "secondary_date_vertical_integral_of_temperature",
]
N_CHANNELS_PER_TIMESTEP = len(_CHANNELS_PER_TIMESTEP)  # 9

# "core" channel ablation: keep only the geophysical channels per timestep
# (insar_difference, insar_coherence, dem) and drop the 6 atmospheric ones.
CORE_CHANNEL_IDX = [0, 1, 2]


def _core_flat_idx(timeseries_length):
    """Flattened indices of the core channels across all timesteps.

    e.g. for timeseries_length=3: [0,1,2, 9,10,11, 18,19,20]
    """
    return [
        t * N_CHANNELS_PER_TIMESTEP + c
        for t in range(timeseries_length)
        for c in CORE_CHANNEL_IDX
    ]


def _stats_key(ch_name):
    """Strip primary_date_ / secondary_date_ prefix for statistics.json lookup."""
    for prefix in ("primary_date_", "secondary_date_"):
        if ch_name.startswith(prefix):
            return ch_name[len(prefix):]
    return ch_name


def normalize(image, stats, timeseries_length=3):
    """
    Apply z-score normalization using dataset-level statistics.

    Args:
        image            : FloatTensor [T * 9, H, W]
        stats            : dict loaded from statistics.json
        timeseries_length: T (number of timesteps, default 3)

    Returns:
        Normalized FloatTensor, same shape.
    """
    for t in range(timeseries_length):
        for c, ch_name in enumerate(_CHANNELS_PER_TIMESTEP):
            idx = t * N_CHANNELS_PER_TIMESTEP + c
            key = _stats_key(ch_name)
            if key in stats:
                mean = stats[key]["mean"]
                std  = stats[key]["std"]
                image[idx] = (image[idx] - mean) / (std + 1e-8)

    return torch.nan_to_num(image, nan=0.0, posinf=3.0, neginf=-3.0)


def _make_augment_fn():
    """
    Build an augmentation callable that applies the same spatial transform to
    all T timesteps simultaneously.  Image is [T*C, H, W] float32 (z-scored).

    By stacking all channels into a single [H, W, T*C] array before calling
    albumentations, every timestep gets the identical flip / rotation — which is
    required for InSAR timeseries (you cannot mirror t=0 differently from t=1).

    Transforms are identical to Thalia's active augmentations (augmentation.json):
      - HorizontalFlip      p=0.3
      - VerticalFlip        p=0.3
      - Rotate              p=0.6  (default limit ±90°, matching Thalia's A.Rotate(p=0.6))
      - GaussianBlur        p=0.3  sigma=(0.1,2.0), default blur_limit (3,7)
      - RandomResizedCrop   p=0.3  default scale (0.08,1.0)
    Applied to ALL training samples (pos + neg), identical to Thalia's behaviour.
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.Rotate(p=0.6),                                  # default limit=(-90,90)
        A.GaussianBlur(sigma_limit=(0.1, 2.0), p=0.3),   # default blur_limit=(3,7)
        A.RandomResizedCrop(size=(512, 512), p=0.3),      # default scale=(0.08,1.0)
    ])

    def apply(image):
        # [T*C, H, W] → [H, W, T*C] for albumentations, then back
        arr = image.numpy().transpose(1, 2, 0)
        out = transform(image=arr)["image"]
        return torch.from_numpy(np.ascontiguousarray(out.transpose(2, 0, 1)))

    return apply


def decode_sample(raw, stats, timeseries_length=3, shuffle_frames=False,
                  augment_fn=None, channels="all"):
    """
    Decode one raw WebDataset dict into (image, label, meta).

    Args:
        shuffle_frames: if True, randomly permute the T timesteps.
                        Used for the ablation experiment — proves ordering matters.
        channels      : "all" keeps all 9 channels/timestep (27 total).
                        "core" keeps only [insar_difference, insar_coherence, dem]
                        per timestep (9 total) — ablation testing whether the
                        6 atmospheric channels matter.

    Returns None on error so the pipeline can skip bad samples.
    """
    try:
        image = torch.load(io.BytesIO(raw["image.pth"]),  weights_only=False).float()
        meta  = torch.load(io.BytesIO(raw["sample.pth"]), weights_only=False)

        # Binary classification label: 1 if any timestep has deformation
        raw_label    = meta.get("label", [0])
        binary_label = int(any(raw_label) if isinstance(raw_label, (list, tuple)) else raw_label)

        # Normalize on the full 9-channel layout first so stats indexing stays
        # correct, regardless of whether we slice channels down afterwards.
        image = normalize(image, stats, timeseries_length)

        if augment_fn is not None:
            image = augment_fn(image)

        # Ablation: shuffle timestep order to destroy temporal information
        if shuffle_frames:
            perm   = torch.randperm(timeseries_length)
            chunks = image.reshape(timeseries_length, N_CHANNELS_PER_TIMESTEP, *image.shape[1:])
            image  = chunks[perm].reshape(image.shape)

        if channels == "core":
            image = image[_core_flat_idx(timeseries_length)]

        return image, torch.tensor(binary_label, dtype=torch.long), meta

    except Exception as e:
        print(f"[decode_sample] skipping sample — {e}")
        return None


class _RandomMix(IterableDataset):
    """Interleave two datasets with equal probability (50/50 pos/neg)."""

    def __init__(self, pos_dataset, neg_dataset):
        self.datasets = [pos_dataset, neg_dataset]

    def __iter__(self):
        sources   = [iter(d) for d in self.datasets]
        exhausted = [False, False]

        while not all(exhausted):
            available = [i for i, ex in enumerate(exhausted) if not ex]
            i = random.choice(available)
            try:
                yield next(sources[i])
            except StopIteration:
                exhausted[i] = True


def _collate(batch):
    batch  = [b for b in batch if b is not None]
    if not batch:
        return None
    images = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    metas  = [b[2] for b in batch]
    return images, labels, metas


def create_loaders(
    data_root,
    stats_path,
    timeseries_length=3,
    batch_size=2,
    num_workers=0,
    seed=42,
    shuffle_frames=False,
    augment=False,
    channels="all",
):
    """
    Build train / val / test DataLoaders.

    Args:
        data_root         : path to webdatasets split folder
                            e.g. /SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3
        stats_path        : path to statistics.json
        timeseries_length : T (must match the folder, here 3)
        batch_size        : samples per batch
        num_workers       : 0 on Mac (multiprocessing issues with wds);
                            start with 4 on server, drop to 0 if hangs
        seed              : random seed
        augment           : if True, apply Thalia-identical spatial augmentation
                            to all training samples (pos + neg)
        channels          : "all" (9 ch/timestep, 27 total) or "core"
                            (3 ch/timestep, 9 total — drops atmospheric channels)

    Returns:
        train_loader, val_loader, test_loader
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    with open(stats_path) as f:
        stats = json.load(f)
    data_root = Path(data_root)

    # ── Train: separate pos/neg shards → RandomMix ──────────────────────
    pos_shards = sorted(glob(str(data_root / "train_pos" / "*.tar")))
    neg_shards = sorted(glob(str(data_root / "train_neg" / "*.tar")))

    # Augmentation applied to all training samples (pos + neg) — identical to Thalia.
    aug_fn = _make_augment_fn() if augment else None

    def decode_train(raw):
        return decode_sample(raw, stats, timeseries_length,
                             shuffle_frames=shuffle_frames, augment_fn=aug_fn,
                             channels=channels)

    pos_ds = (
        wds.WebDataset(pos_shards, shardshuffle=100)
        .map(decode_train)
        .select(lambda x: x is not None)
    )
    neg_ds = (
        wds.WebDataset(neg_shards, shardshuffle=100)
        .map(decode_train)
        .select(lambda x: x is not None)
    )

    train_loader = DataLoader(
        _RandomMix(pos_ds, neg_ds),
        batch_size=batch_size,
        collate_fn=_collate,
        num_workers=num_workers,
    )

    # ── Val / Test ───────────────────────────────────────────────────────
    def decode_eval(raw):
        return decode_sample(raw, stats, timeseries_length,
                             shuffle_frames=False, augment_fn=None,
                             channels=channels)

    def make_eval_loader(split):
        shards = sorted(glob(str(data_root / split / "*.tar")))
        ds = (
            wds.WebDataset(shards, shardshuffle=False)
            .map(decode_eval)
            .select(lambda x: x is not None)
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            collate_fn=_collate,
            num_workers=num_workers,
        )

    val_loader  = make_eval_loader("val")
    test_loader = make_eval_loader("test")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    DATA_ROOT  = "/SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3"
    STATS_PATH = "statistics.json"

    train_loader, val_loader, test_loader = create_loaders(
        data_root=DATA_ROOT,
        stats_path=STATS_PATH,
        timeseries_length=3,
        batch_size=2,
        num_workers=0,
        channels="all",
    )

    print("=== Verifying train batch ===")
    batch = next(iter(train_loader))
    images, labels, metas = batch
    print(f"image shape : {images.shape}")   # expect [2, 27, 512, 512]
    print(f"labels      : {labels.tolist()}")
    print(f"frame_ids   : {[m['frame_id'] for m in metas]}")

    print()
    print("=== Per-channel stats after normalization (first sample, T=0) ===")
    names = [
        "insar_diff", "insar_coh", "dem",
        "wv_prim", "wv_sec", "sp_prim", "sp_sec", "temp_prim", "temp_sec",
    ]
    for c, name in enumerate(names):
        ch = images[0, c]
        print(f"  ch{c:02d} ({name:12s}): mean={ch.mean():+.3f}  std={ch.std():.3f}  "
              f"min={ch.min():+.3f}  max={ch.max():+.3f}")

    print()
    print("=== Val batch ===")
    vbatch = next(iter(val_loader))
    vi, vl, _ = vbatch
    print(f"image shape : {vi.shape}")
    print(f"labels      : {vl.tolist()}")

"""
Report exactly what is inside a set of WebDataset shards.

Run this before evaluating on any shards you did not build in this session.
Two folders can look identical from the outside and hold different things:
a classification build stores a scalar label, a segmentation build stores a
[H,W] or [T,H,W] mask, and Thalia's writer produces both from the same config
file with one field changed. Guessing wrong gives you numbers, not errors.

  python inspect_shards.py --path /SCE_Data/gautbib/thalia_spatiotemporal3/webdatasets/spatiotemporal/3

It prints, per split folder: shard count, a decoded sample's key names, every
tensor's shape and dtype, the metadata fields, how the label is stored, and
the date range covered. Nothing is loaded onto a GPU and only a few samples
per split are read, so it is fast even on a large build.
"""

import argparse
import io
import json
import tarfile
from collections import Counter, OrderedDict
from glob import glob
from pathlib import Path

import torch


def describe(obj, indent=6):
    pad = ' ' * indent
    if torch.is_tensor(obj):
        u = torch.unique(obj) if obj.numel() < 5_000_000 else None
        uniq = (f"  unique={u.tolist()}" if u is not None and u.numel() <= 8 else "")
        return (f"{pad}Tensor shape={tuple(obj.shape)} dtype={obj.dtype} "
                f"min={obj.min().item():.4g} max={obj.max().item():.4g}{uniq}")
    if isinstance(obj, dict):
        lines = [f"{pad}dict with {len(obj)} keys:"]
        for k, v in obj.items():
            if torch.is_tensor(v):
                lines.append(f"{pad}  {k!r}: Tensor {tuple(v.shape)} {v.dtype}")
            else:
                s = repr(v)
                lines.append(f"{pad}  {k!r}: {type(v).__name__} = "
                             f"{s if len(s) <= 90 else s[:87] + '...'}")
        return "\n".join(lines)
    s = repr(obj)
    return f"{pad}{type(obj).__name__} = {s if len(s) <= 90 else s[:87] + '...'}"


def read_members(tar_path, max_samples):
    """Yield {extension: deserialised object} per sample key, in shard order."""
    groups = OrderedDict()
    with tarfile.open(tar_path) as tar:
        for m in tar:
            if not m.isfile():
                continue
            name = Path(m.name).name
            key, _, ext = name.partition('.')
            groups.setdefault(key, {})
            raw = tar.extractfile(m).read()
            if ext.endswith('pth'):
                try:
                    obj = torch.load(io.BytesIO(raw), weights_only=False)
                except Exception as e:
                    obj = f"<unreadable: {e}>"
            elif ext in ('json',):
                obj = json.loads(raw)
            else:
                obj = f"<{len(raw)} bytes>"
            groups[key][ext] = obj
            if len(groups) > max_samples:
                groups.popitem()
                return groups
    return groups


def count_samples(tar_path):
    with tarfile.open(tar_path) as tar:
        return len({Path(m.name).name.partition('.')[0]
                    for m in tar if m.isfile()})


def inspect_split(folder, max_samples, count_all):
    shards = sorted(glob(str(folder / "*.tar"))) + sorted(glob(str(folder / "*.tar.gz")))
    print(f"\n{'='*78}\n{folder.name}/   {len(shards)} shard(s)\n{'='*78}")
    if not shards:
        print("  (no shards)")
        return

    if count_all:
        total = sum(count_samples(s) for s in shards)
        print(f"  samples across all shards: {total}")
    else:
        n0 = count_samples(shards[0])
        print(f"  samples in first shard: {n0}   "
              f"(rough total ~{n0 * len(shards)}; use --count-all to be exact)")

    groups = read_members(shards[0], max_samples)
    if not groups:
        print("  (shard decoded to nothing)")
        return

    first_key, first = next(iter(groups.items()))
    print(f"\n  member extensions per sample: {sorted(first.keys())}")
    print(f"  first sample key: {first_key!r}")
    for ext, obj in first.items():
        print(f"\n  --- {ext} ---")
        print(describe(obj))

    # How is the label stored? This is the field that decides whether the
    # classification pipeline can read these shards at all.
    print(f"\n  --- label interpretation ---")
    meta = first.get('sample.pth')
    if isinstance(meta, dict) and 'label' in meta:
        lab = meta['label']
        if torch.is_tensor(lab):
            print(f"      sample.pth['label'] is a Tensor {tuple(lab.shape)} "
                  f"-> SEGMENTATION-style target")
        else:
            print(f"      sample.pth['label'] = {lab!r} -> classification-style, "
                  f"binary = {int(any(lab) if isinstance(lab, (list, tuple)) else lab)}")
    else:
        print("      no 'label' key in sample.pth  -> UNLABELLED for classification")
    if 'labels.pth' in first:
        lab = first['labels.pth']
        if torch.is_tensor(lab):
            print(f"      labels.pth Tensor {tuple(lab.shape)} -> per-pixel mask present")

    # date coverage and class balance over the sampled subset
    dates, labels, frames = [], [], []
    for g in groups.values():
        m = g.get('sample.pth')
        if not isinstance(m, dict):
            continue
        frames.append(m.get('frame_id'))
        for k in ('primary_date', 'primary_dates', 'dates', 'secondary_date'):
            if k in m:
                dates.append((k, m[k]))
                break
        if 'label' in m and not torch.is_tensor(m['label']):
            l = m['label']
            labels.append(int(any(l) if isinstance(l, (list, tuple)) else l))
    if dates:
        print(f"\n  date field {dates[0][0]!r}, e.g. {[d[1] for d in dates[:3]]}")
    if labels:
        c = Counter(labels)
        print(f"  label balance over {len(labels)} sampled: "
              f"{c.get(1,0)} positive / {c.get(0,0)} negative")
    if frames:
        print(f"  frame_ids sampled: {frames[:5]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--path', required=True,
                    help="folder holding split subfolders, or a single split folder")
    ap.add_argument('--splits', nargs='*', default=None,
                    help="which subfolders to look at (default: all found)")
    ap.add_argument('--max_samples', type=int, default=3,
                    help="samples to decode per split (default 3)")
    ap.add_argument('--count-all', action='store_true', dest='count_all',
                    help="open every shard for an exact sample count (slower)")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if glob(str(root / "*.tar")) and not subdirs:
        targets = [root]
    else:
        targets = [p for p in subdirs
                   if args.splits is None or p.name in args.splits]

    print(f"root: {root}")
    print(f"split folders: {[p.name for p in targets]}")
    for t in targets:
        inspect_split(t, args.max_samples, args.count_all)

    print(f"\n{'='*78}")
    print("What to check before using these shards for evaluation:")
    print("  1. Does sample.pth carry a scalar/list 'label'? If it is a mask,")
    print("     these are segmentation shards and the classification pipeline")
    print("     will read every sample as negative.")
    print("  2. Is image.pth shaped [T*9, H, W] with T=3 and H=W=512?")
    print("  3. Do the dates overlap your training range? If the split dates")
    print("     match the build you trained on, 'test' here is the SAME data")
    print("     you already reported - not a held-out set.")
    print('='*78)


if __name__ == '__main__':
    main()

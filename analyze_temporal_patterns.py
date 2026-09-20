"""
Temporal-pattern analysis on existing checkpoints. No training, no GPU hours.

The Thalia sequence label is any(timestep_labels), so a positive sequence can
carry its deformation in any of seven patterns (001, 010, 011, 100, 101, 110,
111). This script asks whether the four architectures differ in WHICH temporal
patterns they detect - the temporal analogue of the spatial finding that the
baseline only fires on large events.

Two methodological constraints are enforced rather than left to the caller:

1. RECALL ONLY inside pattern groups. Patterns partition the positives; false
   positives come from 000 negatives, which belong to no group. Precision and
   F1 are therefore undefined within a group and are never computed here.

2. PAIRED TESTS. Every model scores the identical samples in the identical
   order, so model-vs-model comparisons use McNemar on the paired detection
   outcomes - far more powerful than comparing independent seed means. The
   exact binomial form is used whenever the discordant count is small, with
   the chi-square approximation only when it is safely large.

  python analyze_temporal_patterns.py \
      --checkpoints outputs/best_convlstm_aug_ce_wd0.01_es20_nonelr.pth \
                    outputs/best_convgru_aug_ce_wd0.01_es20_nonelr.pth \
                    outputs/best_cnn_lstm_aug_ce_wd0.01_es20.pth \
                    outputs/best_baseline_aug_ce_wd0.01_es20.pth
"""

import argparse
import json
from collections import Counter, OrderedDict
from glob import glob
from pathlib import Path

import numpy as np
import torch

from data_loader_fixed import N_CHANNELS_PER_TIMESTEP
from evaluate_unseen import (build_model, get_device, infer_channels,
                             infer_model_name, make_loader, unwrap_checkpoint,
                             DEFAULT_DATA_ROOT, DEFAULT_STATS_PATH)


def pattern_of(meta):
    """Per-timestep label as a tuple of 0/1, or None if unavailable."""
    lab = meta.get('label')
    if lab is None:
        return None
    if torch.is_tensor(lab):
        if lab.dim() == 0:
            return None
        lab = lab.tolist()
    if not isinstance(lab, (list, tuple)):
        return None
    return tuple(int(bool(v)) for v in lab)


def mcnemar(a_correct, b_correct):
    """Paired test on two boolean arrays over the same samples.

    Returns (b01, b10, p, method) where b01 = A right & B wrong.
    Uses the exact binomial test unless the discordant total is large enough
    for the chi-square approximation to be trustworthy.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    b01 = int((a & ~b).sum())
    b10 = int((~a & b).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0, 'identical'
    if n < 25:
        try:
            from scipy.stats import binomtest
            p = binomtest(min(b01, b10), n, 0.5).pvalue
        except ImportError:
            from scipy.stats import binom_test
            p = binom_test(min(b01, b10), n, 0.5)
        return b01, b10, float(p), f'exact binomial (n={n})'
    from scipy.stats import chi2
    stat = (abs(b01 - b10) - 1) ** 2 / n          # continuity-corrected
    return b01, b10, float(chi2.sf(stat, 1)), f'chi-square cc (n={n})'


def wilson(k, n, z=1.96):
    """Wilson score interval - behaves sensibly at the small group sizes here."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def run_checkpoint(path, shards, stats, args, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state, _ = unwrap_checkpoint(ckpt)
    name = infer_model_name(state)
    channels = infer_channels(state, name, args.timeseries_length)
    n_ch = N_CHANNELS_PER_TIMESTEP if channels == 'all' else 3
    model = build_model(name, n_ch, args.timeseries_length).to(device)
    from train import _load_into_model
    _load_into_model(model, state)
    model.eval()

    loader = make_loader(shards, stats, args.timeseries_length, channels,
                         args.batch_size, args.num_workers)
    recs = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, labels, metas = batch
            p = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu().numpy()
            for i, m in enumerate(metas):
                recs.append({'frame_id': m.get('frame_id', '?'),
                             'pattern': pattern_of(m),
                             'label': int(labels[i]),
                             'prob': float(p[i])})
    return name, channels, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--stats_path', default=DEFAULT_STATS_PATH)
    ap.add_argument('--timeseries_length', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0,
                    help="0 keeps sample order identical across models (required "
                         "for the paired tests); do not raise this")
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    device = get_device()
    shards = sorted(glob(str(Path(args.data_root) / args.split / "*.tar")))
    if not shards:
        raise SystemExit(f"no shards under {Path(args.data_root) / args.split}")
    with open(args.stats_path) as f:
        stats = json.load(f)

    print(f"split   : {args.split}  ({len(shards)} shards)")
    print(f"device  : {device}\n")

    results = OrderedDict()
    for ck in args.checkpoints:
        name, channels, recs = run_checkpoint(ck, shards, stats, args, device)
        key = f"{name}" + ("_core" if channels == 'core' else "")
        while key in results:
            key += "'"
        results[key] = recs
        print(f"  scored {len(recs):5d} samples  ->  {key:<16} ({Path(ck).name})")

    # ---- pairing check: identical samples, identical order -----------------
    keys = list(results)
    ref = [r['frame_id'] for r in results[keys[0]]]
    for k in keys[1:]:
        ids = [r['frame_id'] for r in results[k]]
        if ids != ref:
            raise SystemExit(
                f"sample order differs between {keys[0]} and {k}. The paired "
                "tests below would be invalid. Re-run with --num_workers 0.")
    print(f"\npairing verified: all {len(keys)} models scored the same "
          f"{len(ref)} samples in the same order")

    recs0 = results[keys[0]]
    have_patterns = sum(r['pattern'] is not None for r in recs0)
    if have_patterns == 0:
        raise SystemExit("no per-timestep labels found in sample.pth['label'] - "
                         "cannot do pattern analysis on this build")

    # ---- 1. distribution ---------------------------------------------------
    pos = [i for i, r in enumerate(recs0) if r['label'] == 1]
    neg = [i for i, r in enumerate(recs0) if r['label'] == 0]
    pats = Counter(recs0[i]['pattern'] for i in pos)
    print(f"\n{'='*70}\n1. TEMPORAL PATTERN DISTRIBUTION\n{'='*70}")
    print(f"  positives {len(pos)}   negatives {len(neg)}   total {len(recs0)}")
    print(f"\n  {'pattern':<12}{'n':>6}{'% of pos':>10}")
    for pat, c in sorted(pats.items(), key=lambda kv: (-kv[1], kv[0])):
        s = ''.join(str(v) for v in pat) if pat else 'unknown'
        print(f"  {s:<12}{c:>6}{100*c/max(len(pos),1):>9.1f}%")

    groups = OrderedDict()
    for i in pos:
        p = recs0[i]['pattern']
        if p is None:
            continue
        groups.setdefault(sum(p), []).append(i)
    print(f"\n  {'n positive timesteps':<24}{'n':>6}")
    for g in sorted(groups):
        print(f"  {g:<24}{len(groups[g]):>6}")
    small = [g for g in groups if len(groups[g]) < 20]
    if small:
        print(f"\n  NOTE: group(s) {small} have fewer than 20 samples. Recall "
              f"there is too noisy to interpret on its own;")
        print("  rely on the paired McNemar comparisons rather than the point estimates.")

    # ---- 2. recall by group ------------------------------------------------
    print(f"\n{'='*70}\n2. RECALL BY NUMBER OF POSITIVE TIMESTEPS\n{'='*70}")
    print("Recall only: patterns partition the positives, so precision and F1")
    print("are undefined within a group (false positives come from 000 negatives).\n")
    hdr = f"  {'model':<16}" + ''.join(f"{'n=' + str(g):>18}" for g in sorted(groups)) + f"{'all pos':>18}"
    print(hdr)
    det = {k: np.array([r['prob'] >= args.threshold for r in results[k]]) for k in keys}
    recall_tbl = {}
    for k in keys:
        row = f"  {k:<16}"
        recall_tbl[k] = {}
        for g in sorted(groups) + ['all']:
            idx = groups[g] if g != 'all' else pos
            hit = int(sum(det[k][i] for i in idx))
            lo, hi = wilson(hit, len(idx))
            recall_tbl[k][str(g)] = {'hit': hit, 'n': len(idx),
                                     'recall': 100 * hit / max(len(idx), 1),
                                     'ci': [lo, hi]}
            row += f"{100*hit/max(len(idx),1):>10.1f} [{lo:.0f},{hi:.0f}]"
        print(row)
    print("\n  [ ] = 95% Wilson interval")

    # ---- 3. paired comparisons --------------------------------------------
    print(f"\n{'='*70}\n3. PAIRED MODEL COMPARISONS (McNemar)\n{'='*70}")
    print("Same samples, same order. b01 = first model detects and second misses.\n")
    pair_out = {}
    for gi, g in enumerate(sorted(groups) + ['all']):
        idx = groups[g] if g != 'all' else pos
        label = f"n={g} positive timesteps" if g != 'all' else "all positives"
        print(f"  --- {label}  (n={len(idx)}) ---")
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                b01, b10, p, meth = mcnemar([det[a][x] for x in idx],
                                            [det[b][x] for x in idx])
                star = '  *' if p < 0.05 else ''
                print(f"    {a:<14} vs {b:<14} b01={b01:<4} b10={b10:<4} "
                      f"p={p:.4f}  [{meth}]{star}")
                pair_out[f"{g}|{a}|{b}"] = {'b01': b01, 'b10': b10,
                                            'p': p, 'method': meth}
        print()

    out = args.out or f"outputs/temporal_patterns_{args.split}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({
            'split': args.split,
            'threshold': args.threshold,
            'n_samples': len(recs0),
            'n_positive': len(pos),
            'pattern_counts': {''.join(map(str, k)): v for k, v in pats.items() if k},
            'group_counts': {str(g): len(v) for g, v in groups.items()},
            'recall': recall_tbl,
            'mcnemar': pair_out,
            'per_sample': {k: [{'frame_id': r['frame_id'],
                                'pattern': ''.join(map(str, r['pattern'])) if r['pattern'] else None,
                                'label': r['label'], 'prob': r['prob']}
                               for r in results[k]] for k in keys},
        }, f, indent=2)
    print(f"report -> {out}")


if __name__ == '__main__':
    main()

"""
Permutation test-time augmentation, with the val/test split enforced by the tool.

The Thalia sequence target is any(timestep_labels), which is permutation-
invariant. A trained ConvLSTM is not: it is a chronological function. Averaging
its prediction over all six orderings symmetrises it over a symmetry the task
actually has. That is the justification for trying this, and it costs inference
only.

LEAKAGE GUARD. The adopt/reject decision is made on validation and written to
outputs/tta_decision.json. Test evaluation refuses to run until that file
exists, and then reports both arms exactly once. You cannot reach the test
number without first committing to a decision on val.

ONE DECISION FOR ALL SEEDS. Pass every seed checkpoint at once. The rule is
applied to the mean over checkpoints, never per-seed, so the choice cannot be
tuned per run.

Stage 1 - decide on validation:
  python permutation_tta.py --checkpoints outputs/best_convlstm_*nonelr*.pth --split val

Stage 2 - apply once to test:
  python permutation_tta.py --checkpoints outputs/best_convlstm_*nonelr*.pth --split test --confirm
"""

import argparse
import itertools
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch

from data_loader_fixed import N_CHANNELS_PER_TIMESTEP
from evaluate_unseen import (build_model, get_device, infer_channels,
                             infer_model_name, make_loader, metrics_at,
                             unwrap_checkpoint, DEFAULT_DATA_ROOT,
                             DEFAULT_STATS_PATH)

DECISION_PATH = Path('outputs/tta_decision.json')
PERMS = list(itertools.permutations(range(3)))          # 6, identity first


def permute_frames(x, perm, T, C):
    """x: (B, T*C, H, W) -> frames reordered by perm."""
    B, _, H, W = x.shape
    return x.view(B, T, C, H, W)[:, list(perm)].reshape(B, T * C, H, W)


def score(path, shards, stats, args, device):
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
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, y, _ = batch
            x = images.to(device)
            per_perm = []
            for p in PERMS:
                xi = x if p == (0, 1, 2) else permute_frames(x, p, args.timeseries_length, n_ch)
                per_perm.append(torch.softmax(model(xi), dim=1)[:, 1].cpu().numpy())
            probs.append(np.stack(per_perm, axis=1))     # (B, 6)
            labels.append(y.numpy())
    return name, np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'test'])
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--stats_path', default=DEFAULT_STATS_PATH)
    ap.add_argument('--timeseries_length', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--confirm', action='store_true',
                    help="required for --split test; refuses without a stored val decision")
    args = ap.parse_args()

    if args.split == 'test':
        if not args.confirm:
            raise SystemExit("--split test requires --confirm (and a stored val decision)")
        if not DECISION_PATH.exists():
            raise SystemExit(
                f"no decision at {DECISION_PATH}. Run --split val first and let it "
                "record adopt/reject. Evaluating test before deciding on val is "
                "exactly the selection effect this guard exists to prevent.")

    device = get_device()
    shards = sorted(glob(str(Path(args.data_root) / args.split / "*.tar")))
    if not shards:
        raise SystemExit(f"no shards under {Path(args.data_root) / args.split}")
    with open(args.stats_path) as f:
        stats = json.load(f)

    ident = PERMS.index((0, 1, 2))
    print(f"split {args.split}  |  {len(shards)} shards  |  {len(PERMS)} permutations "
          f"|  device {device}\n")

    rows, spreads = [], []
    for ck in args.checkpoints:
        name, P, y = score(ck, shards, stats, args, device)
        base = P[:, ident]
        tta = P.mean(axis=1)
        mb = metrics_at(y, base, args.threshold)
        mt = metrics_at(y, tta, args.threshold)
        sd = P.std(axis=1)
        spreads.append(sd)
        rows.append({'checkpoint': ck, 'model': name, 'n': int(len(y)),
                     'base_f1': mb['f1'], 'tta_f1': mt['f1'],
                     'base_recall': mb['recall'], 'tta_recall': mt['recall'],
                     'base_precision': mb['precision'], 'tta_precision': mt['precision'],
                     'mean_perm_sd': float(sd.mean()),
                     'p95_perm_sd': float(np.percentile(sd, 95)),
                     'n_flipped': int(((base >= args.threshold) !=
                                       (tta >= args.threshold)).sum())})
        print(f"  {Path(ck).name}")
        print(f"     base F1 {mb['f1']:6.2f}   TTA F1 {mt['f1']:6.2f}   "
              f"delta {mt['f1']-mb['f1']:+5.2f}   flipped {rows[-1]['n_flipped']}")

    base_mean = float(np.mean([r['base_f1'] for r in rows]))
    tta_mean = float(np.mean([r['tta_f1'] for r in rows]))
    allsd = np.concatenate(spreads)

    print(f"\n{'='*68}\nORDER SENSITIVITY OF THE LEARNED FUNCTION\n{'='*68}")
    print("Spread of p(deformation) across the six orderings, per sample.")
    print(f"  mean SD  {allsd.mean():.4f}     median {np.median(allsd):.4f}"
          f"     95th pct {np.percentile(allsd, 95):.4f}     max {allsd.max():.4f}")
    print("\nThis is the substantive output: a permutation-invariant target, and a")
    print("model whose output still moves by this much when order changes.")

    print(f"\n{'='*68}\nAGGREGATE OVER {len(rows)} CHECKPOINT(S)\n{'='*68}")
    print(f"  base mean F1 {base_mean:6.2f}      TTA mean F1 {tta_mean:6.2f}"
          f"      delta {tta_mean-base_mean:+5.2f}")

    if args.split == 'val':
        adopt = tta_mean > base_mean
        DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        DECISION_PATH.write_text(json.dumps({
            'rule': 'adopt if mean val F1 over all supplied checkpoints improves',
            'decided_on': 'val', 'adopt': bool(adopt),
            'val_base_mean_f1': base_mean, 'val_tta_mean_f1': tta_mean,
            'checkpoints': list(args.checkpoints), 'threshold': args.threshold,
            'rows': rows}, indent=2))
        print(f"\n  DECISION: {'ADOPT' if adopt else 'REJECT'} permutation TTA")
        print(f"  written to {DECISION_PATH} - applies to every seed, not per-run.")
        print("  Do not revisit this after seeing test.")
    else:
        d = json.loads(DECISION_PATH.read_text())
        print(f"\n  val decision was: {'ADOPT' if d['adopt'] else 'REJECT'} "
              f"(val base {d['val_base_mean_f1']:.2f} -> TTA {d['val_tta_mean_f1']:.2f})")
        print(f"  headline test number = "
              f"{tta_mean if d['adopt'] else base_mean:.2f} "
              f"({'TTA' if d['adopt'] else 'base'})")
        print("  the other arm is reported for completeness only.")

    out = f"outputs/tta_{args.split}.json"
    with open(out, 'w') as f:
        json.dump({'split': args.split, 'rows': rows,
                   'base_mean_f1': base_mean, 'tta_mean_f1': tta_mean,
                   'perm_sd_mean': float(allsd.mean()),
                   'perm_sd_p95': float(np.percentile(allsd, 95))}, f, indent=2)
    print(f"\nreport -> {out}")


if __name__ == '__main__':
    main()

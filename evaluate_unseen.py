"""
Evaluate a trained checkpoint on data it has never seen.

Unlike train.py (which only ever touches the fixed train/val/test folders) and
gradcam.py (which is restricted to --split val|test), this script points at an
ARBITRARY set of WebDataset shards. Use it for a geographic hold-out, a later
date range, or any new frames packed into the same shard format.

Typical use
-----------
  # a different build entirely (check it first with inspect_shards.py)
  python evaluate_unseen.py \
      --checkpoint outputs/best_convlstm_aug_ce_wd0.01_es20_nonelr.pth \
      --data_root  /SCE_Data/gautbib/thalia_spatiotemporal3/webdatasets/spatiotemporal/3 \
      --split      test \
      --stats_path /SCE_Data/gautbib/thalia_temporal3/statistics.json

  # shards anywhere on disk
  python evaluate_unseen.py \
      --checkpoint outputs/best_convlstm_aug_ce_wd0.01_es20_nonelr.pth \
      --shards '/SCE_Data/gautbib/new_frames/*.tar'

  # compare every model on the same unseen data
  for m in baseline cnn_lstm convlstm convgru; do
      python evaluate_unseen.py --checkpoint outputs/best_$m*.pth --split leave3out
  done

Three things this script refuses to get wrong
---------------------------------------------
1. NORMALISATION. The unseen data is normalised with the statistics the model
   was TRAINED on, never statistics recomputed from the unseen set itself.
   Recomputing would quietly leak the new distribution into the input scaling
   and make the result incomparable to every number in the thesis. The script
   also measures how well those training statistics actually fit the new data
   and reports it, because on a geographic hold-out that drift is itself a
   finding.

2. CHANNELS. The checkpoint records whether it was trained on 'all' (9 ch per
   frame) or 'core' (3). That value is read from the checkpoint and used; a
   mismatch is an error, not a silent reshape.

3. LABELS. decode_sample() defaults a missing label to 0. On unlabelled data
   that would produce a confident, entirely meaningless F1. This script checks
   whether labels are genuinely present and switches to prediction-only output
   if they are not.
"""

import argparse
import csv
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import webdataset as wds
from torch.utils.data import DataLoader

from data_loader_fixed import decode_sample, _collate, N_CHANNELS_PER_TIMESTEP
from train import (BaselineResNet50, CNNLSTMClassifier, ConvLSTMClassifier,
                   _load_into_model, compute_metrics)

DEFAULT_DATA_ROOT  = "/SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3"
DEFAULT_STATS_PATH = "/SCE_Data/gautbib/thalia_temporal3/statistics.json"


# ---------------------------------------------------------------------------
# model identification
# ---------------------------------------------------------------------------

def infer_model_name(state_dict):
    """Work out which architecture a checkpoint holds from its parameter names.

    Filenames lie (they get renamed, copied, resumed); parameter names do not.
    ConvLSTM and ConvGRU share the submodule name `conv_lstm` so that existing
    checkpoints load unchanged, so they are told apart by gate shape instead:
    the LSTM cell emits 4 gate groups from one conv, the GRU 2.
    """
    keys = set(state_dict.keys())
    if any(k.startswith('temporal_proj.') for k in keys):
        return 'latefusion'          # Arm C: no recurrent cell at all
    if any(k.startswith('conv_lstm.') for k in keys):
        # The GRU cell has a separate candidate conv; the LSTM cell does not.
        # Cross-checked against gate width: LSTM emits 4 groups, GRU 2.
        is_gru = 'conv_lstm.candidate.weight' in keys
        w = state_dict.get('conv_lstm.gates.weight')
        if w is not None:
            n_groups = w.shape[0] // 128
            expected = 2 if is_gru else 4
            if n_groups != expected:
                raise ValueError(
                    f"checkpoint looks like {'ConvGRU' if is_gru else 'ConvLSTM'} "
                    f"(candidate conv {'present' if is_gru else 'absent'}) but "
                    f"conv_lstm.gates emits {n_groups} groups, expected {expected}")
        return 'convgru' if is_gru else 'convlstm'
    if any(k.startswith('lstm.') for k in keys):
        return 'cnn_lstm'
    if any(k.startswith('model.') for k in keys):
        return 'baseline'
    raise ValueError("could not identify architecture from checkpoint keys")


def load_arch_meta(ckpt_path):
    """Read the sidecar written by train.save_arch_meta, if one exists.

    Authoritative when present. Its absence is not an error: every checkpoint
    predating it was trained with aggregate='last'.
    """
    side = Path(str(ckpt_path) + '.meta.json')
    if side.exists():
        with open(side) as f:
            return json.load(f)
    return None


def unwrap_checkpoint(ckpt):
    """best_*.pth files are bare state dicts; resume_*.pth wrap one in a dict
    alongside optimiser state and the channels flag. Accept either."""
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        return ckpt['model_state_dict'], ckpt
    return ckpt, {}


def infer_channels(state_dict, model_name, timeseries_length):
    """Recover 'all' vs 'core' from the first convolution's input width.

    Only resume_*.pth records `channels`; best_*.pth does not. Guessing wrong
    means either a crash or - worse - silently feeding the model the wrong
    channels, so this is derived from the weights rather than the filename.
    """
    key = 'model.conv1.weight' if model_name == 'baseline' else 'cnn.conv1.weight'
    w = state_dict.get(key)
    if w is None:
        raise ValueError(f"cannot infer channels: {key} missing from checkpoint")
    in_ch = w.shape[1]
    per_frame = in_ch // timeseries_length if model_name == 'baseline' else in_ch
    if per_frame == N_CHANNELS_PER_TIMESTEP:
        return 'all'
    if per_frame == 3:
        return 'core'
    raise ValueError(f"unexpected input width {in_ch} in {key} "
                     f"({per_frame} per frame); expected "
                     f"{N_CHANNELS_PER_TIMESTEP} (all) or 3 (core)")


def build_model(model_name, n_ch_per_frame, timeseries_length, num_classes=2):
    if model_name == 'baseline':
        return BaselineResNet50(in_channels=n_ch_per_frame * timeseries_length,
                                num_classes=num_classes)
    if model_name == 'cnn_lstm':
        return CNNLSTMClassifier(backbone='resnet50',
                                 in_channels_per_frame=n_ch_per_frame,
                                 timeseries_len=timeseries_length,
                                 lstm_hidden=256, num_classes=num_classes)
    if model_name in ('convlstm', 'convgru', 'convlstm_max', 'latefusion'):
        cell = {'convgru': 'gru', 'latefusion': 'none'}.get(model_name, 'lstm')
        aggregate = 'max' if model_name in ('convlstm_max', 'latefusion') else 'last'
        return ConvLSTMClassifier(backbone='resnet50',
                                  in_channels_per_frame=n_ch_per_frame,
                                  timeseries_len=timeseries_length,
                                  bottleneck_channels=256, hidden_channels=128,
                                  num_classes=num_classes,
                                  cell=cell, aggregate=aggregate)
    raise ValueError(f"Unknown model: {model_name}")


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def resolve_shards(args):
    if args.shards:
        shards = sorted(glob(args.shards))
        where = args.shards
    else:
        where = str(Path(args.data_root) / args.split / "*.tar")
        shards = sorted(glob(where))
    if not shards:
        raise SystemExit(
            f"No .tar shards matched:\n  {where}\n\n"
            "Point --split at a folder under --data_root, or give --shards a glob.\n"
            f"Folders currently under {args.data_root}:\n  " +
            "\n  ".join(sorted(p.name for p in Path(args.data_root).glob('*') if p.is_dir()))
            if Path(args.data_root).is_dir() else ""
        )
    return shards, where


def make_loader(shards, stats, timeseries_length, channels, batch_size, num_workers):
    def decode(raw):
        return decode_sample(raw, stats, timeseries_length,
                             shuffle_frames=False, augment_fn=None,
                             channels=channels, load_masks=False)
    ds = (wds.WebDataset(shards, shardshuffle=False)
            .map(decode)
            .select(lambda x: x is not None))
    return DataLoader(ds, batch_size=batch_size, collate_fn=_collate,
                      num_workers=num_workers)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def run(model, loader, device, timeseries_length, n_ch_per_frame):
    """Return probabilities, labels, per-sample records and input-drift stats."""
    model.eval()
    probs, labels, records = [], [], []
    labelled = 0
    # Welford accumulators over the normalised input, per channel-within-frame.
    ch_n = np.zeros(n_ch_per_frame)
    ch_mean = np.zeros(n_ch_per_frame)
    ch_m2 = np.zeros(n_ch_per_frame)

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images, batch_labels, metas = batch
            x = images.to(device)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            # input drift: fold every frame's channel c into the same accumulator
            arr = images.numpy()
            b = arr.shape[0]
            arr = arr.reshape(b, timeseries_length, n_ch_per_frame, *arr.shape[-2:])
            for c in range(n_ch_per_frame):
                v = arr[:, :, c, :, :].ravel()
                n_new = v.size
                mu_new, var_new = v.mean(), v.var()
                n_old, mu_old = ch_n[c], ch_mean[c]
                n_tot = n_old + n_new
                delta = mu_new - mu_old
                ch_mean[c] = mu_old + delta * n_new / n_tot
                ch_m2[c] += var_new * n_new + delta ** 2 * n_old * n_new / n_tot
                ch_n[c] = n_tot

            for i, meta in enumerate(metas):
                has_label = 'label' in meta
                labelled += int(has_label)
                records.append({
                    'frame_id': meta.get('frame_id', '?'),
                    'prob': float(p[i]),
                    'label': int(batch_labels[i]) if has_label else None,
                    'raw_label': str(meta.get('label', '')) if has_label else '',
                })
            probs.extend(p.tolist())
            labels.extend(batch_labels.numpy().tolist())

    drift = [{'channel': c,
              'mean': float(ch_mean[c]),
              'std': float(np.sqrt(ch_m2[c] / max(ch_n[c], 1)))}
             for c in range(n_ch_per_frame)]
    return np.array(probs), np.array(labels), records, drift, labelled


def metrics_at(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    rec = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'threshold': threshold, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', default=None,
                    help="folder name under --data_root holding the unseen shards")
    ap.add_argument('--shards', default=None,
                    help="explicit glob of .tar shards; overrides --split")
    ap.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    ap.add_argument('--stats_path', default=DEFAULT_STATS_PATH,
                    help="statistics.json the model was TRAINED with. Do not "
                         "recompute this on the unseen data.")
    ap.add_argument('--model', default='auto',
                    choices=['auto', 'baseline', 'cnn_lstm', 'convlstm', 'convgru',
                             'convlstm_max', 'latefusion'])
    ap.add_argument('--timeseries_length', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--num_workers', type=int, default=0)
    ap.add_argument('--threshold', type=float, default=0.5,
                    help="decision threshold fixed in advance (default 0.5, as "
                         "used throughout the thesis)")
    ap.add_argument('--out', default=None, help="path for the JSON report")
    ap.add_argument('--csv', default=None, help="path for per-sample predictions")
    args = ap.parse_args()

    if not args.split and not args.shards:
        ap.error("give either --split or --shards")

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state, wrapper = unwrap_checkpoint(ckpt)
    meta = load_arch_meta(args.checkpoint)
    model_name = infer_model_name(state) if args.model == 'auto' else args.model
    # Arms A and B are indistinguishable from weights alone; resolve via sidecar,
    # then filename, then the pre-existing default.
    if args.model == 'auto' and model_name in ('convlstm', 'convgru'):
        if meta and meta.get('aggregate') == 'max':
            model_name = 'convlstm_max'
            print("architecture  : max-logit aggregation (from checkpoint sidecar)")
        elif 'maxlogit' in Path(args.checkpoint).name:
            model_name = 'convlstm_max'
            print("architecture  : max-logit aggregation inferred from FILENAME - "
                  "no sidecar found. Verify this is correct.")
    channels = infer_channels(state, model_name, args.timeseries_length)
    if wrapper.get('channels') and wrapper['channels'] != channels:
        raise SystemExit(
            f"checkpoint records channels='{wrapper['channels']}' but its weights "
            f"say '{channels}' - refusing to guess")
    n_ch = N_CHANNELS_PER_TIMESTEP if channels == 'all' else 3

    print(f"checkpoint    : {args.checkpoint}")
    print(f"architecture  : {model_name}" + ("  (inferred)" if args.model == 'auto' else ""))
    print(f"channels      : {channels}  ({n_ch} per frame, "
          f"{n_ch * args.timeseries_length} total)   [read from the weights]")
    if wrapper:
        print(f"trained epochs: {wrapper.get('epoch', '?')}   "
              f"best val F1: {wrapper.get('best_f1', '?')}")
    print(f"device        : {device}")

    model = build_model(model_name, n_ch, args.timeseries_length).to(device)
    _load_into_model(model, state)

    shards, where = resolve_shards(args)
    print(f"stats         : {args.stats_path}  (training statistics - not recomputed)")
    print(f"shards        : {len(shards)} matching {where}")

    with open(args.stats_path) as f:
        stats = json.load(f)

    loader = make_loader(shards, stats, args.timeseries_length, channels,
                         args.batch_size, args.num_workers)

    probs, labels, records, drift, labelled = run(
        model, loader, device, args.timeseries_length, n_ch)
    n = len(probs)
    if n == 0:
        raise SystemExit(
            "No samples decoded from these shards.\n\n"
            "decode_sample() returns None on any error, so every sample was\n"
            "rejected. The usual causes:\n"
            "  - segmentation shards: sample.pth['label'] is a mask tensor, and\n"
            "    any(tensor) raises, so the classification decoder drops them\n"
            f"  - wrong timeseries length: image.pth is not [T*9, H, W] with T={args.timeseries_length}\n"
            "  - a different channel layout than statistics.json describes\n\n"
            f"Run:  python inspect_shards.py --path {Path(where).parent}\n"
            "to see what these shards actually contain.")

    print(f"\nsamples       : {n}   (labels present on {labelled})")

    print("\nINPUT DRIFT - normalised channel stats on the unseen data")
    print("(training statistics fit perfectly => mean 0.00, std 1.00;")
    print(" large departures mean the new data is off-distribution)")
    print(f"  {'ch':>3}  {'mean':>8}  {'std':>8}")
    worst = 0.0
    for d in drift:
        print(f"  {d['channel']:>3}  {d['mean']:>8.3f}  {d['std']:>8.3f}")
        worst = max(worst, abs(d['mean']), abs(d['std'] - 1.0))
    print(f"  worst departure from (0, 1): {worst:.3f}")

    report = {
        'checkpoint': args.checkpoint,
        'model': model_name,
        'channels': channels,
        'source': where,
        'n_shards': len(shards),
        'n_samples': n,
        'n_labelled': labelled,
        'stats_path': args.stats_path,
        'input_drift': drift,
        'worst_drift': worst,
    }

    if labelled < n:
        print(f"\n{'='*66}\nUNLABELLED DATA - no metrics computed.")
        print("decode_sample() defaults a missing label to 0, so any F1 here")
        print("would be an artefact. Writing predictions only.")
        print('='*66)
        report['mode'] = 'predictions_only'
        report['positive_rate_at_threshold'] = float((probs >= args.threshold).mean())
        print(f"\npredicted positive rate at p>={args.threshold}: "
              f"{report['positive_rate_at_threshold']*100:.2f}%")
    else:
        report['mode'] = 'scored'
        pos = int(labels.sum())
        print(f"class balance : {pos} positive / {n - pos} negative "
              f"({100.0*pos/n:.1f}% positive)")

        fixed = metrics_at(labels, probs, args.threshold)
        full = compute_metrics(labels.tolist(), probs.tolist())
        report['metrics'] = {**fixed, 'auroc': full.get('auroc')}

        print(f"\n{'='*66}")
        print(f"RESULT at the pre-fixed threshold {args.threshold}")
        print('='*66)
        print(f"  F1        : {fixed['f1']:6.2f}")
        print(f"  Precision : {fixed['precision']:6.2f}")
        print(f"  Recall    : {fixed['recall']:6.2f}")
        print(f"  AUROC     : {full.get('auroc', float('nan')):6.2f}")
        print(f"  confusion : TP {fixed['tp']}  FP {fixed['fp']}  "
              f"FN {fixed['fn']}  TN {fixed['tn']}")

        # Reported for completeness, flagged so it cannot be quoted as a result.
        best = max((metrics_at(labels, probs, t / 100.0) for t in range(1, 100)),
                   key=lambda m: m['f1'])
        report['oracle_threshold_NOT_A_VALID_RESULT'] = best
        print(f"\n  [oracle] best achievable F1 on this set is {best['f1']:.2f} at "
              f"threshold {best['threshold']:.2f}.")
        print("  This is NOT a valid result - the threshold was chosen using the")
        print("  unseen labels. Quote the number above it. Use this only as an")
        print("  upper bound, or re-select the threshold on val and pass it in.")

    tag = Path(args.checkpoint).stem.replace('best_', '')
    where_tag = args.split or 'shards'
    out = args.out or f"outputs/eval_{tag}__{where_tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nreport  -> {out}")

    csv_path = args.csv or f"outputs/eval_{tag}__{where_tag}.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['frame_id', 'prob', 'label', 'raw_label'])
        w.writeheader()
        w.writerows(records)
    print(f"per-sample -> {csv_path}")


if __name__ == '__main__':
    main()

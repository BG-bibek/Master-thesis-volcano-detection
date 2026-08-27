"""
Threshold sweep — find the F1-maximizing decision threshold on the
validation set for an already-trained checkpoint, then report test-set
metrics at both the default (0.5) and the val-selected threshold.

Why this exists: every metric in train.py's training/eval loop uses a fixed
0.5 cutoff on softmax(logits)[:, 1]. AUROC is threshold-independent and can
look strong (e.g. the convlstm_aug_wd0.01_es10 run: 91.90% test AUROC) while
F1 at 0.5 undersells the model if 0.5 isn't the right operating point for
this class balance (~10% positive). This answers that with zero retraining
— just one forward pass over val and test each.

Threshold is selected on VAL, then applied once (unchanged) to TEST — never
sweep directly on test, that leaks test-set information into the reported
number and the thesis result would no longer be honest.

Usage:
    python threshold_sweep.py --model convlstm \\
        --checkpoint outputs/best_convlstm_aug_wd0.01_es10.pth

    python threshold_sweep.py --model cnn_lstm \\
        --checkpoint outputs/best_cnn_lstm.pth --channels core
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from train import BaselineResNet50, CNNLSTMClassifier, ConvLSTMClassifier, _load_into_model
from data_loader_fixed import create_loaders

DEFAULT_DATA_ROOT  = "/SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3"
DEFAULT_STATS_PATH = "/SCE_Data/gautbib/thalia_temporal3/statistics.json"


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(model_name, n_ch_per_frame, timeseries_length, num_classes=2):
    if model_name == 'baseline':
        return BaselineResNet50(
            in_channels=n_ch_per_frame * timeseries_length, num_classes=num_classes
        )
    elif model_name == 'cnn_lstm':
        return CNNLSTMClassifier(
            backbone='resnet50', in_channels_per_frame=n_ch_per_frame,
            timeseries_len=timeseries_length, lstm_hidden=256, num_classes=num_classes,
        )
    elif model_name == 'convlstm':
        return ConvLSTMClassifier(
            backbone='resnet50', in_channels_per_frame=n_ch_per_frame,
            timeseries_len=timeseries_length, bottleneck_channels=256,
            hidden_channels=128, num_classes=num_classes,
        )
    raise ValueError(f"Unknown model: {model_name}")


@torch.no_grad()
def collect_probs(model, loader, device):
    """Run one inference pass, return (labels, probs) as numpy arrays.
    probs = softmax(logits)[:, 1], same quantity train.py's evaluate() uses."""
    model.eval()
    all_labels, all_probs = [], []
    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())  # labels never left CPU
    return np.array(all_labels), np.array(all_probs)


def metrics_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    return {
        'precision': precision_score(labels, preds, zero_division=0) * 100,
        'recall':    recall_score(labels, preds, zero_division=0) * 100,
        'f1':        f1_score(labels, preds, zero_division=0) * 100,
    }


def sweep_thresholds(labels, probs, start=0.01, stop=0.99, step=0.01):
    """Return (best_threshold, best_metrics) — the threshold in [start, stop]
    that maximizes F1 on the given (labels, probs). Ties keep the first
    (lowest) threshold found."""
    best_threshold, best_f1, best_metrics = 0.5, -1.0, None
    t = start
    while t <= stop + step / 2:
        m = metrics_at_threshold(labels, probs, t)
        if m['f1'] > best_f1:
            best_f1 = m['f1']
            best_threshold = round(float(t), 4)
            best_metrics = m
        t += step
    return best_threshold, best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', required=True, choices=['baseline', 'cnn_lstm', 'convlstm'])
    parser.add_argument('--checkpoint', required=True,
                        help='Path to a best_*.pth checkpoint (raw state dict, as saved by train.py)')
    parser.add_argument('--channels', default='all', choices=['all', 'core'],
                        help='Must match what the checkpoint was trained with')
    parser.add_argument('--data_root',  default=DEFAULT_DATA_ROOT)
    parser.add_argument('--stats_path', default=DEFAULT_STATS_PATH)
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--step', type=float, default=0.01, help='Threshold sweep step size')
    parser.add_argument('--out', default=None,
                        help='Where to save results JSON (default: outputs/threshold_sweep_<checkpoint_stem>.json)')
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    n_ch_per_frame    = 3 if args.channels == 'core' else 9
    timeseries_length = 3

    print(f"\nLoading data (channels={args.channels})...")
    _, val_loader, test_loader = create_loaders(
        data_root=args.data_root,
        stats_path=args.stats_path,
        timeseries_length=timeseries_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_frames=False,
        augment=False,   # never augment for eval
        channels=args.channels,
    )

    print(f"\nBuilding {args.model} and loading checkpoint: {args.checkpoint}")
    model = build_model(args.model, n_ch_per_frame, timeseries_length)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    _load_into_model(model, state)
    model = model.to(device)

    print("\nRunning inference on VAL set...")
    val_labels, val_probs = collect_probs(model, val_loader, device)
    val_auroc   = roc_auc_score(val_labels, val_probs) * 100 if len(np.unique(val_labels)) > 1 else 50.0
    default_val = metrics_at_threshold(val_labels, val_probs, 0.5)

    print(f"Sweeping thresholds on VAL (step={args.step})...")
    best_threshold, best_val_metrics = sweep_thresholds(val_labels, val_probs, step=args.step)

    print("\nRunning inference on TEST set...")
    test_labels, test_probs = collect_probs(model, test_loader, device)
    test_auroc    = roc_auc_score(test_labels, test_probs) * 100 if len(np.unique(test_labels)) > 1 else 50.0
    default_test  = metrics_at_threshold(test_labels, test_probs, 0.5)
    optimal_test  = metrics_at_threshold(test_labels, test_probs, best_threshold)

    print("\n" + "=" * 70)
    print("VAL — threshold selection (AUROC is threshold-independent, unaffected)")
    print("=" * 70)
    print(f"  VAL AUROC             : {val_auroc:6.2f}%")
    print(f"  Default (t=0.50)  F1={default_val['f1']:6.2f}%  P={default_val['precision']:6.2f}%  R={default_val['recall']:6.2f}%")
    print(f"  Best    (t={best_threshold:.2f})  F1={best_val_metrics['f1']:6.2f}%  P={best_val_metrics['precision']:6.2f}%  R={best_val_metrics['recall']:6.2f}%")

    print("\n" + "=" * 70)
    print("TEST — applying the VAL-selected threshold (no test-set leakage)")
    print("=" * 70)
    print(f"  TEST AUROC             : {test_auroc:6.2f}%")
    print(f"  Default (t=0.50)  F1={default_test['f1']:6.2f}%  P={default_test['precision']:6.2f}%  R={default_test['recall']:6.2f}%")
    print(f"  Optimal (t={best_threshold:.2f})  F1={optimal_test['f1']:6.2f}%  P={optimal_test['precision']:6.2f}%  R={optimal_test['recall']:6.2f}%")
    print("=" * 70)

    delta_f1 = optimal_test['f1'] - default_test['f1']
    print(f"\nTest F1 change from threshold tuning: {delta_f1:+.2f}pp")

    out_path = args.out or f"outputs/threshold_sweep_{Path(args.checkpoint).stem}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({
            'checkpoint':         args.checkpoint,
            'model':              args.model,
            'channels':           args.channels,
            'val_auroc':          val_auroc,
            'test_auroc':         test_auroc,
            'default_threshold':  0.5,
            'default_val':        default_val,
            'default_test':       default_test,
            'best_threshold':     best_threshold,
            'best_val':           best_val_metrics,
            'optimal_test':       optimal_test,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")

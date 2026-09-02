"""
Parse all training .log files into one structured results table.

The server logs are the ground truth for every experiment run — this pulls the
config, the per-epoch history, and the final test metrics out of each one so the
results notebook doesn't have to re-parse text.

Usage:
    python parse_logs.py                  # writes results_summary.json + prints table
    python parse_logs.py --log_dir .      # explicit directory
"""
import argparse
import json
import re
from pathlib import Path

# "Epoch 44/90" then the indented metric block that follows it
_EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+)\s*$")
_METRIC_RES = {
    'train_loss':    re.compile(r"^\s*Train loss\s*:\s*([\d.]+)"),
    'val_loss':      re.compile(r"^\s*Val loss\s*:\s*([\d.]+)"),
    'val_f1':        re.compile(r"^\s*F1\s*:\s*([\d.]+)%"),
    'val_precision': re.compile(r"^\s*Precision\s*:\s*([\d.]+)%"),
    'val_recall':    re.compile(r"^\s*Recall\s*:\s*([\d.]+)%"),
    'val_auroc':     re.compile(r"^\s*AUROC\s*:\s*([\d.]+)%"),
    'lr':            re.compile(r"^\s*LR\s*:\s*([\d.e+-]+)"),
}
_TEST_RES = {
    'f1':        re.compile(r"^\s*Test F1\s*:\s*([\d.]+)%"),
    'precision': re.compile(r"^\s*Test Precision\s*:\s*([\d.]+)%"),
    'recall':    re.compile(r"^\s*Test Recall\s*:\s*([\d.]+)%"),
    'auroc':     re.compile(r"^\s*Test AUROC\s*:\s*([\d.]+)%"),
    'loss':      re.compile(r"^\s*Test loss\s*:\s*([\d.]+)"),
}
_ARGS_RE       = re.compile(r"^Train args:\s*(.+)$")
_NAMESPACE_RE  = re.compile(r"^Args:\s*Namespace\((.+)\)\s*$")
_BEST_VAL_RE   = re.compile(r"Training complete! Best val F1:\s*([\d.]+)%")
_EPOCHS_RUN_RE = re.compile(r"Epochs trained:\s*(\d+)")
_EARLYSTOP_RE  = re.compile(r"Early stopping: no val F1 improvement for (\d+) epochs "
                            r"\(best=([\d.]+)% at epoch (\d+)\)")


def _parse_namespace(text):
    """Turn the argparse Namespace repr into a dict (best effort)."""
    cfg = {}
    for m in re.finditer(r"(\w+)=('([^']*)'|[^,]+?)(?=,\s*\w+=|$)", text):
        key, raw = m.group(1), m.group(2).strip()
        if raw.startswith("'") and raw.endswith("'"):
            val = raw[1:-1]
        elif raw == 'True':
            val = True
        elif raw == 'False':
            val = False
        elif raw == 'None':
            val = None
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        cfg[key] = val
    return cfg


def parse_log(path):
    """Parse one training log. Returns a dict, or None if it isn't a training log."""
    lines = Path(path).read_text(errors='replace').splitlines()

    rec = {
        'log_file':   Path(path).name,
        'train_args': None,
        'config':     {},
        'history':    [],
        'test':       {},
        'best_val_f1':      None,
        'epochs_trained':   None,
        'early_stopped':    False,
        'early_stop_epoch': None,
        'completed':        False,
    }

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        m = _ARGS_RE.match(line)
        if m:
            rec['train_args'] = m.group(1).strip()
            i += 1
            continue

        m = _NAMESPACE_RE.match(line)
        if m:
            rec['config'] = _parse_namespace(m.group(1))
            i += 1
            continue

        m = _EPOCH_RE.match(line)
        if m:
            entry = {'epoch': int(m.group(1))}
            # metrics appear in the ~10 lines following the "Epoch N/M" header
            for j in range(i + 1, min(i + 12, n)):
                for key, rx in _METRIC_RES.items():
                    if key in entry:
                        continue
                    mm = rx.match(lines[j])
                    if mm:
                        entry[key] = float(mm.group(1))
                if _EPOCH_RE.match(lines[j]):
                    break
            # only keep blocks that actually carried metrics
            if 'val_f1' in entry:
                rec['history'].append(entry)
            i += 1
            continue

        for key, rx in _TEST_RES.items():
            mm = rx.match(line)
            if mm:
                rec['test'][key] = float(mm.group(1))

        mm = _BEST_VAL_RE.search(line)
        if mm:
            rec['best_val_f1'] = float(mm.group(1))
            rec['completed'] = True

        mm = _EPOCHS_RUN_RE.search(line)
        if mm:
            rec['epochs_trained'] = int(mm.group(1))

        mm = _EARLYSTOP_RE.search(line)
        if mm:
            rec['early_stopped'] = True
            rec['early_stop_epoch'] = int(mm.group(3))

        i += 1

    if not rec['history'] and not rec['test']:
        return None

    if rec['epochs_trained'] is None:
        rec['epochs_trained'] = len(rec['history'])
    # a run is only fully done if it produced test metrics
    rec['completed'] = rec['completed'] and bool(rec['test'])
    return rec


def config_label(rec):
    """Short human-readable description of the run's configuration."""
    c = rec['config']
    if not c:
        return rec['log_file']
    parts = [c.get('model', '?')]
    if c.get('shuffle'):            parts.append('shuffled')
    if c.get('augment'):            parts.append('aug')
    if c.get('channels') == 'core': parts.append('core')
    parts.append(f"loss={c.get('loss', 'focal')}")
    parts.append(f"wd={c.get('weight_decay', 1e-4):g}")
    if c.get('patience'):                     parts.append(f"pat={c['patience']}")
    if c.get('lr_schedule', 'cosine') != 'cosine':
        parts.append(f"lr={c['lr_schedule']}")
    if c.get('seed', 42) != 42:               parts.append(f"seed={c['seed']}")
    return ' | '.join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--log_dir', default='.', help='Directory containing the .log files')
    ap.add_argument('--out', default='results_summary.json')
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    records = []
    for path in sorted(log_dir.glob('*.log')):
        rec = parse_log(path)
        if rec is None:
            print(f"  (skipped, not a training log or empty: {path.name})")
            continue
        rec['label'] = config_label(rec)
        records.append(rec)

    with open(args.out, 'w') as f:
        json.dump(records, f, indent=2)

    print(f"\nParsed {len(records)} runs -> {args.out}\n")
    hdr = f"{'run':<62} {'ep':>4} {'bestVal':>8} {'testF1':>7} {'testAU':>7} {'done':>5}"
    print(hdr)
    print('-' * len(hdr))
    for r in records:
        t = r['test']
        print(f"{r['label']:<62} {r['epochs_trained'] or 0:>4} "
              f"{(r['best_val_f1'] or float('nan')):>8.2f} "
              f"{t.get('f1', float('nan')):>7.2f} {t.get('auroc', float('nan')):>7.2f} "
              f"{'yes' if r['completed'] else 'NO':>5}")

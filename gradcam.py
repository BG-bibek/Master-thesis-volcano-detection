"""
Grad-CAM for the volcanic deformation classifiers, scored against the
ground-truth deformation masks the classification pipeline normally discards.

Why this exists
---------------
The three-seed shuffle ablation showed frame ORDER carries no signal (the
`any(labels)` union label is permutation-invariant by construction), yet
temporal architectures still beat channel-stacking by a wide margin. That
leaves an open question: if not ordering, what is the ConvLSTM actually
doing better? This answers it with evidence rather than inference — it asks
*where* each model looks, and scores that against the expert-annotated
deformation masks in labels.pth.

Two things make it worth doing:
  1. ConvLSTMClassifier keeps spatial hidden states h_t at every timestep, so
     its per-timestep maps show attention AFTER integrating frames 1..t — the
     recurrence is visible in them. CNNLSTMClassifier global-average-pools each
     frame before its LSTM, so its per-frame maps come from the encoder alone
     and carry no temporal coupling. Same number of maps, different meaning:
     a direct contrast between the two designs.
  2. The paper does no spatial saliency analysis at all (their Suppl. C only
     inspects input-channel fusion weights), so this is unexplored ground.

Metrics (both standard for coarse CAMs; IoU is deliberately NOT used — a
16x16 feature map upsampled to 512x512 gives 32x32-pixel cells, far too coarse
for meaningful overlap against thin deformation fringes):
  * Pointing game : does the CAM's peak fall inside the ground-truth mask?
  * Energy ratio  : what fraction of total CAM mass lands inside the mask?

A random-baseline column is reported alongside, since a mask covering X% of
the frame would score X% by chance.

Usage
-----
    python gradcam.py --model convlstm \\
        --checkpoint outputs/best_convlstm_aug_ce_wd0.01_es20_nonelr.pth \\
        --n_samples 40 --figures 6

    # contrast against the channel-stacking baseline
    python gradcam.py --model baseline \\
        --checkpoint outputs/best_baseline_aug_ce_wd0.01_es20.pth --n_samples 40
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_loader_fixed import create_loaders
from train import (BaselineResNet50, CNNLSTMClassifier, ConvLSTMClassifier,
                   _load_into_model)

DEFAULT_DATA_ROOT  = "/SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3"
DEFAULT_STATS_PATH = "/SCE_Data/gautbib/thalia_temporal3/statistics.json"


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ============================================================================
# GRAD-CAM
# ============================================================================

class GradCAM:
    """Grad-CAM via hooks — no changes to the model classes in train.py.

    A forward hook grabs the target layer's activations and registers a tensor
    hook on them to capture the gradient flowing back. Because ConvLSTMCell is
    invoked once per timestep inside a loop, the hook fires T times and we keep
    every call — giving one saliency map per timestep for free.
    """

    def __init__(self, target_layer):
        self.activations = []   # one entry per forward call of the target layer
        self.gradients   = []   # gradients stored at the SAME index as their activation
        self._handles = [target_layer.register_forward_hook(self._forward_hook)]

    def _forward_hook(self, module, inputs, output):
        # ConvLSTMCell returns (h, c); conv layers return a bare tensor.
        act = output[0] if isinstance(output, tuple) else output
        if not act.requires_grad:
            return
        idx = len(self.activations)
        self.activations.append(act)
        self.gradients.append(None)
        # Bind the index into the closure so each gradient lands next to its own
        # activation. Relying on backward firing in exact reverse order would
        # also work here, but this is order-independent and cannot silently
        # mis-pair a timestep's gradient with another timestep's activation.
        act.register_hook(lambda g, i=idx: self._store_grad(i, g))

    def _store_grad(self, idx, grad):
        self.gradients[idx] = grad

    def reset(self):
        self.activations, self.gradients = [], []

    def close(self):
        for h in self._handles:
            h.remove()

    def compute(self, out_size):
        """Return a list of [H, W] CAMs in timestep order.

        Two ways a target layer yields multiple timesteps:
          * called once per timestep (ConvLSTMCell in its loop)   -> one entry each
          * called once on a [B*T, ...] batch (the CNN encoders)  -> T batch rows
        Both are flattened out here, so every model returns one CAM per frame.
        Assumes B=1, which is what this script runs with.
        """
        cams = []
        for act, grad in zip(self.activations, self.gradients):
            if grad is None:      # activation never received a gradient
                continue
            # alpha_k = GAP of the gradient over space; CAM = ReLU(sum_k alpha_k * A_k)
            weights = grad.mean(dim=(2, 3), keepdim=True)           # [N, C, 1, 1]
            cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # [N, 1, h, w]
            cam = F.interpolate(cam, size=out_size, mode='bilinear', align_corners=False)
            cam = cam.detach().float().cpu().numpy()                # [N, 1, H, W]
            for n in range(cam.shape[0]):                           # N == T when B=1
                c = cam[n, 0]
                m = c.max()
                cams.append(c / m if m > 0 else c)
        return cams


def prepare_for_gradcam(model, model_name):
    """Put the model in eval mode, with one necessary exception.

    cuDNN's fused RNN kernel refuses to compute gradients in eval mode:
        RuntimeError: cudnn RNN backward can only be called in training mode
    CNNLSTMClassifier's nn.LSTM hits this. Putting *only* that submodule in
    train mode unlocks the backward path and is behaviourally a no-op here,
    because the LSTM is num_layers=1 with dropout=0.0 — train() and eval()
    compute exactly the same function for it. The encoder and head stay in
    eval mode, so BatchNorm statistics and Dropout are untouched and the CAMs
    reflect the deployed model.

    ConvLSTMClassifier is unaffected (ConvLSTMCell is plain nn.Conv2d), and
    BaselineResNet50 has no RNN.
    """
    model.eval()
    if model_name == 'cnn_lstm':
        lstm = model.lstm
        assert lstm.num_layers == 1 and float(lstm.dropout) == 0.0, (
            "prepare_for_gradcam assumes the LSTM has no inter-layer dropout, so "
            "train() is a behavioural no-op. That is not true for this model "
            f"(num_layers={lstm.num_layers}, dropout={lstm.dropout}) — enabling "
            "train mode here would change the CAMs. Use "
            "torch.backends.cudnn.flags(enabled=False) around the forward/backward "
            "instead."
        )
        lstm.train()
    return model


def resolve_target_layer(model, model_name):
    """Pick the layer whose activations the CAM is computed from.

    convlstm : the ConvLSTM cell — fires once per timestep, so we get the
               temporal evolution of spatial attention (the interesting case).
    others   : the encoder's last conv stage, the standard Grad-CAM target.
    """
    if model_name == 'latefusion':
        # No recurrent cell exists; the per-timestep spatial projection is the
        # structural analogue and fires once per frame.
        return model.temporal_proj, 'temporal_proj (per-timestep projection)'
    if model_name in ('convlstm', 'convgru', 'convlstm_max'):
        kind = {'convgru': 'ConvGRU'}.get(model_name, 'ConvLSTM')
        return model.conv_lstm, f'{kind} cell (per-timestep hidden state)'
    if model_name == 'cnn_lstm':
        return model.cnn.layer4, 'cnn.layer4 (per-frame encoder features)'
    if model_name == 'baseline':
        return model.model.layer4, 'model.layer4 (final conv stage)'
    raise ValueError(f"Unknown model: {model_name}")


# ============================================================================
# METRICS
# ============================================================================

def pointing_game(cam, mask):
    """True if the CAM's peak pixel lies inside the ground-truth mask."""
    idx = np.unravel_index(np.argmax(cam), cam.shape)
    return bool(mask[idx])


def energy_ratio(cam, mask):
    """Fraction of total CAM mass falling inside the mask.

    Scale-invariant, so the per-CAM max-normalisation in GradCAM.compute()
    does not affect this value.
    """
    total = cam.sum()
    return float(cam[mask].sum() / total) if total > 0 else 0.0


# ── Concentration metrics: how spread out is the attention, mask aside? ──
#
# energy_ratio asks "is the attention ON the deformation". These ask "is the
# attention FOCUSED anywhere at all" — mask-independent. Motivation: better
# classifiers here place LESS mass inside the annotated deformation while
# classifying more accurately, and the proposed explanation is that they read
# context outside the fringe (coherence, topography, atmospheric channels) to
# separate real deformation from atmospheric artefact. If that is right, their
# maps should be measurably more DIFFUSE. These turn that inference into a
# measurement. Both are scale-invariant, so the per-CAM max-normalisation in
# GradCAM.compute() does not affect them.

def concentration(cam, frac=0.5):
    """Fraction of pixels holding `frac` of the total CAM mass.

    Lower = more peaked (a few pixels carry the mass).
    Higher = more diffuse (mass spread over many pixels).
    """
    flat = np.sort(cam.ravel())[::-1]
    total = flat.sum()
    if total <= 0:
        return 1.0
    k = int(np.searchsorted(np.cumsum(flat), frac * total) + 1)
    return float(k / flat.size)


def dilate_mask(mask, radius):
    """Expand a boolean mask by `radius` pixels (square structuring element).

    Why: the CAM is computed at 16x16 and upsampled to 512x512, so one CAM
    cell covers ~32x32 input pixels. Attention landing just OUTSIDE the
    annotated deformation may be pure boundary blur from that upsampling
    rather than the model looking somewhere else. Re-scoring against a mask
    dilated by one or two cell widths separates the two explanations.

    A square element is used deliberately — it matches the shape of a CAM cell.
    """
    if radius <= 0:
        return mask
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    out = F.max_pool2d(t, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return out[0, 0].numpy() > 0.5


def spatial_entropy(cam):
    """Shannon entropy of the CAM read as a spatial distribution, normalised
    to [0, 1] by log(n_pixels). Higher = more diffuse."""
    p = cam.ravel().astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(p.size))


# Which per-frame map to headline, per architecture. This matters: averaging
# every timestep understates ConvLSTM badly, because its classification head
# reads ONLY the final hidden state h_T — the earlier states reach the output
# indirectly through the recurrence, so their gradients are attenuated and
# their maps are correspondingly noisy. The other two models feed every frame's
# features to the classifier, so the mean is the honest aggregate there.
# The max-logit arms route every timestep to the shared head, so - like
# cnn_lstm and baseline - all frames reach the output directly and 'mean' is
# the honest aggregate. Only the final-state arms need 'last'.
PRIMARY_AGG = {'convlstm': 'last', 'convgru': 'last',
               'cnn_lstm': 'mean', 'baseline': 'mean',
               'convlstm_max': 'mean', 'latefusion': 'mean'}


def aggregate_cams(cams, how):
    """Combine per-frame CAMs into the single map used for the headline score."""
    if how == 'last':
        agg = cams[-1].copy()
    else:
        agg = np.mean(np.stack(cams), axis=0)
    m = agg.max()
    return agg / m if m > 0 else agg


# ============================================================================
# MAIN
# ============================================================================

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


def save_figure(image, cams, mask, meta, prob, path, n_ch_per_frame):
    """Overlay each timestep's CAM on that timestep's InSAR phase channel."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T = len(cams)
    fig, axes = plt.subplots(2, T, figsize=(4.2 * T, 8.2), squeeze=False)
    frame_id = meta.get('frame_id', '?')
    labels   = meta.get('label', '?')

    for t in range(T):
        # insar_difference is channel 0 within each timestep's block
        phase = image[t * n_ch_per_frame].cpu().numpy()
        axes[0][t].imshow(phase, cmap='twilight')
        axes[0][t].set_title(f't={t}  InSAR phase', fontsize=10)
        axes[1][t].imshow(phase, cmap='gray', alpha=0.85)
        axes[1][t].imshow(cams[t], cmap='inferno', alpha=0.5)
        if mask is not None:
            axes[1][t].contour(mask.astype(float), levels=[0.5],
                               colors='#1baf7a', linewidths=1.6)
        axes[1][t].set_title(f't={t}  Grad-CAM', fontsize=10)
        for r in (0, 1):
            axes[r][t].set_xticks([]); axes[r][t].set_yticks([])

    fig.suptitle(f"{frame_id}   label={labels}   p(deformation)={prob:.3f}"
                 f"{'   (green contour = ground truth)' if mask is not None else ''}",
                 fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True,
                    choices=['baseline', 'cnn_lstm', 'convlstm', 'convgru',
                             'convlstm_max', 'latefusion'])
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--channels', default='all', choices=['all', 'core'],
                    help='Must match what the checkpoint was trained with')
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    ap.add_argument('--n_samples', type=int, default=40,
                    help='Number of POSITIVE samples to score (masks are empty for negatives)')
    ap.add_argument('--figures', type=int, default=6,
                    help='How many qualitative overlay figures to save')
    ap.add_argument('--dilate', type=int, nargs='+', default=[0, 32, 64],
                    help="Mask dilation radii (pixels) for the sensitivity scan. "
                         "The CAM is 16x16 upsampled to 512x512, so one CAM cell "
                         "is ~32 px: 0 = exact mask, 32 = one cell of slack, "
                         "64 = two. Reported with the DILATED chance level and an "
                         "enrichment ratio, because raw energy rises with mask "
                         "size for any map — enrichment is what stays comparable.")
    ap.add_argument('--aggregate', default='auto', choices=['auto', 'last', 'mean'],
                    help="How to combine per-frame CAMs for the headline score. "
                         "'auto' (default) uses 'last' for convlstm — its head reads "
                         "only the final hidden state, so averaging in the noisy "
                         "early-timestep maps understates it — and 'mean' otherwise. "
                         "Both aggregates are always reported.")
    ap.add_argument('--out_dir', default='outputs/gradcam')
    ap.add_argument('--data_root',  default=DEFAULT_DATA_ROOT)
    ap.add_argument('--stats_path', default=DEFAULT_STATS_PATH)
    ap.add_argument('--num_workers', type=int, default=0,
                    help="Default 0 — and it should stay 0. The eval loader uses "
                         "shardshuffle=False, so with a single process the sample "
                         "order is deterministic and every model scores the SAME "
                         "first --n_samples positives, which is what makes the "
                         "cross-model comparison paired. With num_workers>0, "
                         "webdataset splits shards across workers and the "
                         "DataLoader interleaves whichever worker yields first, so "
                         "each run scores a different subset and the comparison "
                         "silently becomes unpaired.")
    args = ap.parse_args()

    device = get_device()
    print(f"Device: {device}")

    n_ch_per_frame, T = (3 if args.channels == 'core' else 9), 3
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.split} split (channels={args.channels}, masks on)...")
    _, val_loader, test_loader = create_loaders(
        data_root=args.data_root, stats_path=args.stats_path,
        timeseries_length=T, batch_size=1, num_workers=args.num_workers,
        shuffle_frames=False, augment=False, channels=args.channels,
        load_masks=True,
    )
    loader = test_loader if args.split == 'test' else val_loader

    print(f"Building {args.model}, loading {args.checkpoint}")
    model = build_model(args.model, n_ch_per_frame, T)
    _load_into_model(model, torch.load(args.checkpoint, map_location=device,
                                       weights_only=True))
    model = prepare_for_gradcam(model.to(device), args.model)

    target_layer, target_desc = resolve_target_layer(model, args.model)
    agg_mode = PRIMARY_AGG[args.model] if args.aggregate == 'auto' else args.aggregate
    print(f"Grad-CAM target: {target_desc}")
    print(f"Headline aggregate: {agg_mode}"
          + ("  (final hidden state — the only one the head reads)"
             if agg_mode == 'last' else "  (mean over frames)"))
    cam_engine = GradCAM(target_layer)

    records, n_figs, skipped_no_mask = [], 0, 0
    for batch in loader:
        if batch is None:
            continue
        images, labels, metas = batch
        if int(labels[0]) != 1:      # masks only meaningful for positives
            continue
        meta = metas[0]
        mask = meta.get('deformation_mask_union')
        if mask is None:
            skipped_no_mask += 1
            continue
        mask = mask.cpu().numpy().astype(bool)
        if not mask.any():
            skipped_no_mask += 1
            continue

        images = images.to(device)
        cam_engine.reset()

        # Grad-CAM needs gradients even though the model is in eval mode.
        with torch.enable_grad():
            images.requires_grad_(False)
            logits = model(images)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
            model.zero_grad(set_to_none=True)
            logits[0, 1].backward()          # attribute the POSITIVE class

        cams = cam_engine.compute(out_size=images.shape[-2:])
        if not cams:
            continue

        primary = aggregate_cams(cams, agg_mode)
        mean_map = aggregate_cams(cams, 'mean')

        records.append({
            'frame_id':   meta.get('frame_id', '?'),
            'prob':       prob,
            'correct':    bool(prob >= 0.5),   # sample is positive by construction
            'n_cams':     len(cams),
            'mask_frac':  float(mask.mean()),
            'pointing':   pointing_game(primary, mask),
            'energy':     energy_ratio(primary, mask),
            'concentration50': concentration(primary, 0.5),
            'entropy':         spatial_entropy(primary),
            'pointing_mean': pointing_game(mean_map, mask),
            'energy_mean':   energy_ratio(mean_map, mask),
            'pointing_per_t': [pointing_game(c, mask) for c in cams],
            'energy_per_t':   [energy_ratio(c, mask) for c in cams],
            # Sensitivity to how strictly "on the deformation" is defined.
            'dilation_scan': [
                {'radius':    r,
                 'mask_frac': float(dm.mean()),
                 'pointing':  pointing_game(primary, dm),
                 'energy':    energy_ratio(primary, dm)}
                for r, dm in ((r, dilate_mask(mask, r)) for r in args.dilate)
            ],
        })

        if n_figs < args.figures:
            save_figure(images[0].detach().cpu(), cams, mask, meta, prob,
                        out_dir / f"cam_{args.model}_{n_figs:02d}_"
                                  f"{meta.get('frame_id','x')}.png",
                        n_ch_per_frame)
            n_figs += 1

        if len(records) >= args.n_samples:
            break

    cam_engine.close()

    if not records:
        print("\nNo positive samples with a usable mask were found — "
              "check that labels.pth exists in these shards.")
        raise SystemExit(1)

    pct = lambda key: float(np.mean([r[key] for r in records])) * 100
    pointing, energy   = pct('pointing'), pct('energy')
    pointing_m, energy_m = pct('pointing_mean'), pct('energy_mean')
    chance = pct('mask_frac')
    n_t    = max(r['n_cams'] for r in records)

    print("\n" + "=" * 66)
    print(f"Grad-CAM localisation — {args.model} on {args.split} "
          f"({len(records)} positive samples)")
    print("=" * 66)
    print(f"  HEADLINE (aggregate = {agg_mode})")
    print(f"    Pointing game (peak in mask) : {pointing:6.2f}%")
    print(f"    Energy inside mask           : {energy:6.2f}%")
    print(f"    Lift over chance             : {energy - chance:+6.2f}pp (energy)")
    if agg_mode != 'mean':
        print(f"  For reference, mean-over-frames aggregate:")
        print(f"    Pointing {pointing_m:6.2f}%   Energy {energy_m:6.2f}%")
    print(f"  Random chance (mask area)      : {chance:6.2f}%")
    conc50  = pct('concentration50')
    entropy = float(np.mean([r['entropy'] for r in records]))
    print(f"  CONCENTRATION (mask-independent — how focused is the attention)")
    print(f"    Pixels holding 50% of mass   : {conc50:6.2f}%  (lower = more peaked)")
    print(f"    Spatial entropy (0-1)        : {entropy:6.4f}  (higher = more diffuse)")
    if skipped_no_mask:
        print(f"  (skipped {skipped_no_mask} positives lacking a usable mask)")

    if n_t > 1:
        print(f"\n  Per-timestep (target fired {n_t}x per forward pass):")
        for t in range(n_t):
            pt = [r['pointing_per_t'][t] for r in records if len(r['pointing_per_t']) > t]
            et = [r['energy_per_t'][t]   for r in records if len(r['energy_per_t'])   > t]
            print(f"    t={t}:  pointing {np.mean(pt)*100:6.2f}%   "
                  f"energy {np.mean(et)*100:6.2f}%")

    # How much of the "outside the mask" energy is just upsampling blur?
    dil_summary = []
    if records and records[0].get('dilation_scan'):
        print(f"\n  DILATION SCAN (is attention just outside the mask boundary?)")
        print(f"    {'radius':>7} {'mask area':>10} {'pointing':>9} {'energy':>8} "
              f"{'enrichment':>11}")
        for k, r in enumerate(args.dilate):
            mf = float(np.mean([rec['dilation_scan'][k]['mask_frac'] for rec in records]))
            pg = float(np.mean([rec['dilation_scan'][k]['pointing'] for rec in records]))
            en = float(np.mean([rec['dilation_scan'][k]['energy'] for rec in records]))
            enr = en / mf if mf > 0 else float('nan')
            dil_summary.append({'radius': r, 'mask_frac_pct': mf * 100,
                                'pointing_pct': pg * 100, 'energy_pct': en * 100,
                                'enrichment': enr})
            print(f"    {r:>6}px {mf*100:>9.2f}% {pg*100:>8.2f}% {en*100:>7.2f}% "
                  f"{enr:>10.2f}x")
        print("    Raw energy MUST rise with radius (the mask is bigger) — compare")
        print("    the enrichment column, which divides that out.")

    # Does the model localise better on the positives it actually gets right?
    hit  = [r for r in records if r['correct']]
    miss = [r for r in records if not r['correct']]
    if hit and miss:
        print(f"\n  Split by prediction (headline aggregate):")
        for name, grp in (('correctly detected', hit), ('missed', miss)):
            p = np.mean([r['pointing'] for r in grp]) * 100
            e = np.mean([r['energy']   for r in grp]) * 100
            print(f"    {name:<20} n={len(grp):3d}  pointing {p:6.2f}%   energy {e:6.2f}%")
    print("=" * 66)

    summary_path = out_dir / f"gradcam_{args.model}_{args.split}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'model': args.model, 'checkpoint': args.checkpoint,
            'channels': args.channels, 'split': args.split,
            'target_layer': target_desc, 'n_samples': len(records),
            'aggregate': agg_mode,
            'pointing_game_pct': pointing, 'energy_in_mask_pct': energy,
            'pointing_game_pct_mean_agg': pointing_m,
            'energy_in_mask_pct_mean_agg': energy_m,
            'chance_pct': chance,
            'concentration50_pct': conc50,
            'spatial_entropy': entropy,
            'dilation_scan': dil_summary,
            'per_timestep': [
                {'t': t,
                 'pointing_pct': float(np.mean([r['pointing_per_t'][t] for r in records
                                                if len(r['pointing_per_t']) > t])) * 100,
                 'energy_pct':   float(np.mean([r['energy_per_t'][t] for r in records
                                                if len(r['energy_per_t']) > t])) * 100}
                for t in range(n_t)
            ],
            'records': records,
        }, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print(f"Figures: {n_figs} saved to {out_dir}/")

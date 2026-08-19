"""
Training script for Thalia CNN-LSTM thesis.
Tests both baseline (ResNet50) and CNN-LSTM models.

SERVER VERSION (par-sce-ai-02):
  - Auto-detects CUDA / MPS / CPU
  - Default paths point to /SCE_Data/gautbib/thalia_temporal3
  - Checkpoint/resume support (--resume flag)
  - CosineAnnealingLR scheduler
  - Logs metrics to JSON per run (train + val per epoch, test at end)
  - Final test set evaluation using best checkpoint
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import timm
from focal_loss import FocalLoss
from glob import glob

# ============================================================================
# SETUP
# ============================================================================

def get_device():
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

DEVICE = get_device()
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Available GPUs: {torch.cuda.device_count()}")

# Default paths — server layout. Override via --data_root / --stats_path if needed.
DEFAULT_DATA_ROOT  = "/SCE_Data/gautbib/thalia_temporal3/webdatasets/temporal/3"
DEFAULT_STATS_PATH = "/SCE_Data/gautbib/thalia_temporal3/statistics.json"

CFG = {
    'device': DEVICE,
    'task': 'classification',
    'num_classes': 2,
    'epochs': 3,           # 3 for sanity check, 90 for full training (overridden by --epochs)
    'batch_size': 8,
    'lr': 1e-5,
    'weight_decay': 1e-4,   # our established recipe; override with --weight_decay 1e-2 to
                             # match Thalia's Suppl. B "Training Setup" text (--loss ce too)
    'gradient_clip': 1.0,
    'timeseries_length': 3,
    'n_channels_per_timestep': 9,  # 3 geo + 6 atm; set to 3 when --channels=core
    'model_name': 'baseline',      # overridden by --model
}

print(f"Config: epochs={CFG['epochs']}, lr={CFG['lr']}, batch_size={CFG['batch_size']}")


# ============================================================================
# MODELS
# ============================================================================

class BaselineResNet50(nn.Module):
    """Paper's approach: sees all timesteps as channels (no temporal modelling)."""
    def __init__(self, in_channels=27, num_classes=2):
        super().__init__()
        self.model = timm.create_model(
            'resnet50',
            pretrained=True,
            num_classes=num_classes,
            in_chans=in_channels
        )

    def forward(self, x):
        # x: (B, T*C, H, W) = (B, 27, 512, 512) for 'all', (B, 9, 512, 512) for 'core'
        return self.model(x)


class CNNLSTMClassifier(nn.Module):
    """Thesis contribution: CNN on each frame, LSTM over time."""
    def __init__(
        self,
        backbone='resnet50',
        in_channels_per_frame=9,
        timeseries_len=3,
        lstm_hidden=256,
        num_classes=2,
        dropout=0.3,
        pretrained=True
    ):
        super().__init__()
        self.T = timeseries_len
        self.C = in_channels_per_frame

        self.cnn = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg',
            in_chans=in_channels_per_frame
        )
        cnn_out = self.cnn.num_features  # 2048 for ResNet50

        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0.0
        )

        self.head = nn.Sequential(
            nn.LayerNorm(lstm_hidden),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T*C, H, W)
        B, TC, H, W = x.shape
        T, C = self.T, self.C
        assert TC == T * C

        x = x.view(B, T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)       # (B*T, cnn_out)
        feats = feats.view(B, T, -1)   # (B, T, cnn_out)

        lstm_out, _ = self.lstm(feats) # (B, T, lstm_hidden)
        last = lstm_out[:, -1, :]

        return self.head(last)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell (Shi et al., 2015).

    Same gating as a standard LSTM, but every gate is a convolution over a
    spatial feature map instead of a matmul over a flat vector — so hidden/cell
    state stay [C, H, W] tensors and 'where' information survives the
    recurrence. All 4 gates are produced by one conv (input and previous
    hidden state concatenated on the channel dim), the standard efficient
    formulation.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        combined = torch.cat([x, h_prev], dim=1)
        i, f, o, g = self.gates(combined).chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch_size, height, width, device):
        shape = (batch_size, self.hidden_channels, height, width)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


class ConvLSTMClassifier(nn.Module):
    """Thesis contribution #2: a *true* ConvLSTM (Shi et al., 2015) with a
    ResNet-50 encoder — Thalia's paper reports benchmark numbers for exactly
    this kind of model (best time-series classification result in their
    Table 3: F1=78.89%, AUROC=96.19%) but never published its architecture.

    Differs from CNNLSTMClassifier: the encoder is NOT global-average-pooled
    per frame before the recurrence. Instead it keeps spatial feature maps
    alive through the ConvLSTM, so the model can track *where* deformation
    is and how that location evolves across timesteps, not just *how much*
    of each feature fired.
    """
    def __init__(
        self,
        backbone='resnet50',
        in_channels_per_frame=9,
        timeseries_len=3,
        bottleneck_channels=256,
        hidden_channels=128,
        kernel_size=3,
        num_classes=2,
        dropout=0.3,
        pretrained=True,
    ):
        super().__init__()
        self.T = timeseries_len
        self.C = in_channels_per_frame

        # features_only=True: keep the last conv stage's spatial feature map
        # instead of pooling it away (e.g. [2048, 16, 16] for 512x512 input).
        self.cnn = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels_per_frame,
            features_only=True,
            out_indices=(4,),
        )
        cnn_out_channels = self.cnn.feature_info.channels()[-1]  # 2048 for ResNet50

        # 1x1 conv to shrink channel count before the ConvLSTM gates — keeps
        # gate-conv params in the same ballpark as CNNLSTMClassifier's LSTM.
        # Dropout2d here regularizes the encoder->recurrence interface: all 3
        # ConvLSTM runs so far show severe overfitting (train loss -> 0.0000
        # while val loss keeps climbing), so this needed real regularization,
        # not just a different loss/weight_decay recipe.
        self.bottleneck = nn.Sequential(
            nn.Conv2d(cnn_out_channels, bottleneck_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.conv_lstm = ConvLSTMCell(bottleneck_channels, hidden_channels, kernel_size)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        # x: (B, T*C, H, W)
        B, TC, H, W = x.shape
        T, C = self.T, self.C
        assert TC == T * C

        x = x.view(B, T, C, H, W)
        x_flat = x.view(B * T, C, H, W)
        feats = self.cnn(x_flat)[-1]            # (B*T, cnn_out, H', W')
        feats = self.bottleneck(feats)          # (B*T, bottleneck, H', W')
        _, Cb, Hf, Wf = feats.shape
        feats = feats.view(B, T, Cb, Hf, Wf)

        h, c = self.conv_lstm.init_state(B, Hf, Wf, feats.device)
        for t in range(T):
            h, c = self.conv_lstm(feats[:, t], (h, c))

        pooled = h.mean(dim=(2, 3))  # global average pool final hidden state -> (B, hidden_channels)
        return self.head(pooled)


# ============================================================================
# DATA LOADING
# ============================================================================

def create_loaders(data_root, stats_path, shuffle_frames=False, augment=False, channels='all'):
    from data_loader_fixed import create_loaders as _create_loaders
    return _create_loaders(
        data_root=data_root,
        stats_path=stats_path,
        timeseries_length=3,
        batch_size=CFG['batch_size'],
        num_workers=4,  # start at 4; increase if stable, drop to 0 if hangs
        seed=42,
        shuffle_frames=shuffle_frames,
        augment=augment,
        channels=channels,
    )


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(all_labels, all_probs):
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    if len(np.unique(all_labels)) < 2:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'auroc': 50.0}

    return {
        'precision': precision_score(all_labels, all_preds, zero_division=0) * 100,
        'recall':    recall_score(all_labels, all_preds, zero_division=0) * 100,
        'f1':        f1_score(all_labels, all_preds, zero_division=0) * 100,
        'auroc':     roc_auc_score(all_labels, all_probs) * 100,
    }


def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    n_batches  = 0

    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images = images.to(CFG['device'])
        labels = labels.to(CFG['device'])

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CFG['gradient_clip'])
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if n_batches % 10 == 0:
            print(f"  Epoch {epoch} | Batch {n_batches:3d} | loss: {loss.item():.4f}")

    avg_loss = total_loss / max(n_batches, 1)
    print(f"  Epoch {epoch} | Avg loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    n_batches  = 0
    all_labels, all_probs = [], []

    for batch in loader:
        if batch is None:
            continue
        images, labels, _ = batch
        images = images.to(CFG['device'])
        labels = labels.to(CFG['device'])

        logits = model(images)
        loss   = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        total_loss += loss.item()
        n_batches  += 1

    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / max(n_batches, 1)
    return metrics


# ============================================================================
# CHECKPOINT / RESUME
# ============================================================================

def _model_state(model):
    """Return state dict without DataParallel's 'module.' prefix."""
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()


def _load_into_model(model, state_dict):
    """Load a state dict saved via _model_state() (no 'module.' prefix) into
    model, transparently handling DataParallel wrapping. Without this, loading
    on a multi-GPU run raises a key-mismatch error (state dict has no
    'module.' prefix, but a DataParallel-wrapped model expects one)."""
    target = model.module if isinstance(model, nn.DataParallel) else model
    target.load_state_dict(state_dict)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1, history, channels):
    """Save full training state for exact resume."""
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     _model_state(model),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_f1':              best_f1,
        'history':              history,
        'channels':             channels,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, channels):
    """Load training state. Returns (start_epoch, best_f1, history)."""
    ckpt = torch.load(path, map_location=CFG['device'], weights_only=False)
    ckpt_channels = ckpt.get('channels', 'all')
    if ckpt_channels != channels:
        raise ValueError(
            f"Checkpoint at {path} was trained with channels='{ckpt_channels}', "
            f"but this run is using channels='{channels}'. Refusing to load a "
            "checkpoint with a mismatched channel count / input shape."
        )
    _load_into_model(model, ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_f1     = ckpt['best_f1']
    history     = ckpt.get('history', [])
    print(f"Resumed from checkpoint: epoch {ckpt['epoch']} complete, "
          f"best_f1={best_f1:.2f}%, continuing from epoch {start_epoch}")
    return start_epoch, best_f1, history


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    default='cnn_lstm', choices=['baseline', 'cnn_lstm', 'convlstm'])
    parser.add_argument('--epochs',   type=int, default=CFG['epochs'])
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping: stop if val F1 has not improved for this many '
                             'epochs (0 = disabled, always train the full --epochs). The best '
                             'checkpoint by val F1 is saved throughout regardless — this only '
                             'controls when training stops, not what gets saved.')
    parser.add_argument('--shuffle',  action='store_true', help='Shuffle frames (ablation)')
    parser.add_argument('--augment',  action='store_true',
                        help='Apply Thalia-identical spatial augmentation to all training samples')
    parser.add_argument('--channels', default='all', choices=['all', 'core'],
                        help="'all' = 9 channels/timestep (27 total); "
                             "'core' = insar_difference/insar_coherence/dem only "
                             "(9 total) — ablation dropping the 6 atmospheric channels")
    parser.add_argument('--loss', default='focal', choices=['focal', 'ce'],
                        help="'focal' (default) = FocalLoss(gamma=2), our established recipe "
                             "(all prior results use this). 'ce' = CrossEntropyLoss, matching "
                             "Thalia paper's Suppl. B Training Setup text — the recipe that "
                             "actually produced their published benchmark numbers (the checked-in "
                             "configs.json default of FocalLoss/wd=1e-4 does not match that text).")
    parser.add_argument('--weight_decay', type=float, default=CFG['weight_decay'],
                        help=f"AdamW weight decay (default {CFG['weight_decay']:g}, our recipe). "
                             "Thalia's Suppl. B reports 1e-2 for their published numbers — pass "
                             "--weight_decay 1e-2 together with --loss ce to match their recipe "
                             "exactly for a head-to-head comparison.")
    parser.add_argument('--data_root',  default=DEFAULT_DATA_ROOT)
    parser.add_argument('--stats_path', default=DEFAULT_STATS_PATH)
    parser.add_argument('--resume',   action='store_true',
                        help='Resume from last checkpoint for this run, if it exists')
    parser.add_argument('--checkpoint_every', type=int, default=1,
                        help='Save a resumable checkpoint every N epochs (default: every epoch)')
    args = parser.parse_args()
    print("Args:", args)

    CFG['model_name'] = args.model
    CFG['epochs']     = args.epochs
    CFG['n_channels_per_timestep'] = 3 if args.channels == 'core' else 9

    Path("outputs").mkdir(exist_ok=True)

    shuffle_tag  = "_shuffled" if args.shuffle else ""
    aug_tag      = "_aug"      if args.augment else ""
    channels_tag = "_core"     if args.channels == 'core' else ""
    loss_tag     = "_ce"       if args.loss == 'ce' else ""
    # Compare against the original default (before CFG['weight_decay'] is overwritten below)
    wd_tag       = "" if args.weight_decay == CFG['weight_decay'] else f"_wd{args.weight_decay:g}"
    patience_tag = f"_es{args.patience}" if args.patience > 0 else ""
    run_name         = f"{CFG['model_name']}{shuffle_tag}{aug_tag}{channels_tag}{loss_tag}{wd_tag}{patience_tag}"
    best_ckpt_path   = f"outputs/best_{run_name}.pth"
    resume_ckpt_path = f"outputs/resume_{run_name}.pth"
    metrics_log_path = f"outputs/metrics_{run_name}.json"

    CFG['weight_decay'] = args.weight_decay

    print("\n" + "=" * 70)
    print(f"Training: {run_name.upper()}  |  epochs={CFG['epochs']}")
    if args.shuffle:
        print("ABLATION MODE: frames are shuffled (temporal order destroyed)")
    if args.augment:
        print("AUGMENT MODE: Thalia-identical spatial augmentation on all training samples")
    if args.channels == 'core':
        print("CHANNEL ABLATION: core channels only (insar_difference, insar_coherence, dem)")
    print(f"Recipe: loss={args.loss}  weight_decay={args.weight_decay:g}"
          + ("  (matches Thalia Suppl. B)" if (args.loss == 'ce' and args.weight_decay == 1e-2) else ""))
    if args.patience > 0:
        print(f"EARLY STOPPING: patience={args.patience} epochs on val F1")
    print("=" * 70)

    # Verify paths before anything expensive
    if not Path(args.data_root).exists():
        raise FileNotFoundError(f"data_root not found: {args.data_root}")
    if not Path(args.stats_path).exists():
        raise FileNotFoundError(
            f"stats_path not found: {args.stats_path}\n"
            "Copy statistics.json to the server alongside train.py."
        )

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_loaders(
        data_root=args.data_root,
        stats_path=args.stats_path,
        shuffle_frames=args.shuffle,
        augment=args.augment,
        channels=args.channels,
    )

    # Print shard/sample counts so we know what data is being loaded
    data_root_path = Path(args.data_root)
    n_pos   = len(glob(str(data_root_path / "train_pos" / "*.tar")))
    n_neg   = len(glob(str(data_root_path / "train_neg" / "*.tar")))
    n_val   = len(glob(str(data_root_path / "val"       / "*.tar")))
    n_test  = len(glob(str(data_root_path / "test"      / "*.tar")))
    max_shard = 128  # max_samples_per_shard from Thalia configs.json
    print(f"\nDataset shards:")
    print(f"  train_pos : {n_pos:3d} shards  (~{n_pos  * max_shard:5d} samples)")
    print(f"  train_neg : {n_neg:3d} shards  (~{n_neg  * max_shard:5d} samples)")
    print(f"  val       : {n_val:3d} shards  (~{n_val  * max_shard:5d} samples)")
    print(f"  test      : {n_test:3d} shards  (~{n_test * max_shard:5d} samples)")
    print(f"  train total: ~{(n_pos + n_neg) * max_shard} samples "
          f"(RandomMix balances to ~{min(n_pos, n_neg) * max_shard * 2} per epoch)\n")

    # Create model — input channels depend on --channels flag
    print(f"\nCreating {CFG['model_name']} model...")
    n_ch_per_frame = CFG['n_channels_per_timestep']  # 9 ('all') or 3 ('core')
    if CFG['model_name'] == 'baseline':
        model = BaselineResNet50(
            in_channels=n_ch_per_frame * CFG['timeseries_length'],
            num_classes=CFG['num_classes'],
        )
    elif CFG['model_name'] == 'cnn_lstm':
        model = CNNLSTMClassifier(
            backbone='resnet50',
            in_channels_per_frame=n_ch_per_frame,
            timeseries_len=CFG['timeseries_length'],
            lstm_hidden=256,
            num_classes=CFG['num_classes'],
        )
    elif CFG['model_name'] == 'convlstm':
        model = ConvLSTMClassifier(
            backbone='resnet50',
            in_channels_per_frame=n_ch_per_frame,
            timeseries_len=CFG['timeseries_length'],
            bottleneck_channels=256,
            hidden_channels=128,
            num_classes=CFG['num_classes'],
        )
    else:
        raise ValueError(f"Unknown model: {CFG['model_name']}")

    model = model.to(CFG['device'])
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model created ({n_params:.1f}M parameters)")

    if CFG['device'].type == 'cuda' and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # Optimizer, scheduler, loss
    # --loss selects the recipe:
    #   'focal' (default) — FocalLoss(gamma=2). CrossEntropyLoss with imbalanced
    #                        data and no undersampling caused the model to predict
    #                        all-negative -> F1=0%; this is our established recipe.
    #   'ce'               — CrossEntropyLoss, matching Thalia's Suppl. B "Training
    #                        Setup" text, which is what actually produced their
    #                        published numbers (undersampling via RandomMix still
    #                        applies, so all-negative collapse isn't a risk here).
    criterion = nn.CrossEntropyLoss() if args.loss == 'ce' else FocalLoss(gamma=2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CFG['epochs'],
        eta_min=1e-7,
    )

    # Resume logic
    start_epoch = 1
    best_f1     = -1.0  # -1 guarantees a checkpoint is saved after epoch 1 even if F1=0%
    history     = []
    epochs_without_improvement = 0  # for --patience; resets on resume (not persisted in checkpoint)

    if args.resume:
        if Path(resume_ckpt_path).exists():
            start_epoch, best_f1, history = load_checkpoint(
                resume_ckpt_path, model, optimizer, scheduler, args.channels
            )
        else:
            print(f"--resume passed but no checkpoint found at {resume_ckpt_path}. "
                  "Starting fresh.")

    if start_epoch > CFG['epochs']:
        print(f"Already at epoch {start_epoch - 1}, target is {CFG['epochs']}. Nothing to do.")
        exit(0)

    # Training loop
    print("\n" + "=" * 70)
    print(f"Training (epochs {start_epoch} -> {CFG['epochs']})")
    print("=" * 70 + "\n")

    train_start = time.time()
    for epoch in range(start_epoch, CFG['epochs'] + 1):
        t0 = time.time()

        train_loss  = train_epoch(model, train_loader, optimizer, criterion, epoch)
        val_metrics = evaluate(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{CFG['epochs']}")
        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_metrics['loss']:.4f}")
        print(f"  F1         : {val_metrics['f1']:6.2f}%")
        print(f"  Precision  : {val_metrics['precision']:6.2f}%")
        print(f"  Recall     : {val_metrics['recall']:6.2f}%")
        print(f"  AUROC      : {val_metrics['auroc']:6.2f}%")
        print(f"  LR         : {current_lr:.2e}")
        print(f"  Time       : {elapsed:.0f}s\n")

        history.append({
            'epoch':          epoch,
            'train_loss':     train_loss,
            'val_loss':       val_metrics['loss'],
            'val_f1':         val_metrics['f1'],
            'val_precision':  val_metrics['precision'],
            'val_recall':     val_metrics['recall'],
            'val_auroc':      val_metrics['auroc'],
            'lr':             current_lr,
            'epoch_time_sec': elapsed,
        })

        # Save metrics log every epoch for live monitoring
        with open(metrics_log_path, 'w') as f:
            json.dump({
                'run_name':       run_name,
                'model':          args.model,
                'channels':       args.channels,
                'loss':           args.loss,
                'weight_decay':   args.weight_decay,
                'shuffle_frames': args.shuffle,
                'augment':        args.augment,
                'patience':       args.patience,
                'history':        history,
            }, f, indent=2)

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            epochs_without_improvement = 0
            torch.save(_model_state(model), best_ckpt_path)
            print(f"  Best checkpoint saved (F1={best_f1:.1f}%)\n")
        else:
            epochs_without_improvement += 1

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                resume_ckpt_path, model, optimizer, scheduler, epoch, best_f1, history,
                args.channels,
            )

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no val F1 improvement for {args.patience} epochs "
                  f"(best={best_f1:.1f}% at epoch {epoch - epochs_without_improvement}).")
            break

    total_elapsed = time.time() - train_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)
    print("=" * 70)
    print(f"Training complete! Best val F1: {best_f1:.1f}%")
    print(f"Epochs trained: {len(history)}" + (f" (early-stopped, target was {CFG['epochs']})" if len(history) < CFG['epochs'] else ""))
    print(f"Total training time: {total_mins}m {total_secs}s")

    # ── Final test set evaluation using best checkpoint ──────────────────
    print("\nRunning final test set evaluation (best checkpoint)...")
    best_state = torch.load(best_ckpt_path, map_location=CFG['device'], weights_only=True)
    _load_into_model(model, best_state)
    test_metrics = evaluate(model, test_loader, criterion)

    print(f"  Test F1        : {test_metrics['f1']:6.2f}%")
    print(f"  Test Precision : {test_metrics['precision']:6.2f}%")
    print(f"  Test Recall    : {test_metrics['recall']:6.2f}%")
    print(f"  Test AUROC     : {test_metrics['auroc']:6.2f}%")
    print(f"  Test loss      : {test_metrics['loss']:.4f}")

    # Append test metrics to the JSON log
    with open(metrics_log_path, 'r') as f:
        log = json.load(f)
    log['test_metrics'] = test_metrics
    with open(metrics_log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"\nMetrics log: {metrics_log_path}")
    print("=" * 70)

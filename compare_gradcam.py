"""
Cross-model comparison of the Grad-CAM localisation results.

Why this exists
---------------
Comparing "localisation on samples the model detected" ACROSS models is
confounded: each model detects a different subset of positives, and a model
with low recall only fires on the easy, obvious cases — where localisation is
trivially good. Its detected-group score is therefore flattered by selection,
not by better perception.

This script does three things the per-model runs cannot:
  1. Puts the three models in one table.
  2. Tests the selection-effect hypothesis directly, by asking whether the
     samples a model detects have LARGER ground-truth masks (bigger, more
     obvious deformation) than the ones it misses.
  3. Recomputes localisation on the COMMON subset of frames that every model
     detected — a like-for-like comparison with the selection effect removed.

Usage:
    python compare_gradcam.py                       # reads outputs/gradcam/
    python compare_gradcam.py --dir outputs/gradcam
"""
import argparse
import json
import statistics as st
from pathlib import Path

MODELS = ['baseline', 'cnn_lstm', 'convlstm']

# ── PRE-REGISTERED DECISION RULE ─────────────────────────────────────────
# Fixed BEFORE running the dilation scan, from the already-observed r=0
# enrichments on the commonly-detected samples:
#     baseline 40.60/9.42 = 4.31x    convlstm 32.80/9.42 = 3.48x
#     gap(r=0) = 0.83x  ->  threshold = half of that = 0.415x
#
#     gap(r=max) <  0.415x  ->  CONVERGED
#     gap(r=max) >= 0.415x  ->  NOT CONVERGED
#
# "Converged" means the localisation gap shrank by more than half once one
# or two CAM cells of boundary slack are allowed — i.e. much of the apparent
# difference was blur from the coarse 16x16 CAM rather than the models
# attending to different places. It does NOT establish that the models are
# equivalent; it only bounds how much of the gap survives.
#
# DO NOT change these constants after seeing the scan output — that would
# make the analysis post-hoc.
PREREG_R0_GAP    = 0.83
PREREG_THRESHOLD = 0.415
PREREG_PAIR      = ('baseline', 'convlstm')


def load(dirpath, split='test'):
    out = {}
    for m in MODELS:
        p = Path(dirpath) / f"gradcam_{m}_{split}.json"
        if p.exists():
            out[m] = json.load(open(p))
        else:
            print(f"  (missing: {p})")
    return out


def mean(vals):
    return sum(vals) / len(vals) if vals else float('nan')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='outputs/gradcam')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    data = load(args.dir, args.split)
    if len(data) < 2:
        raise SystemExit("Need at least two model result files to compare.")

    # ── 0. Are the models scored on the same samples, in the same order? ─
    #
    # NOTE: frame_id is a LiCSAR *geographic frame* identifier, NOT a unique
    # sample id — many time-series come from the same frame, so 60 samples may
    # span only ~16 distinct frame_ids. Keying records by frame_id would
    # silently collapse them. With --num_workers 0 the eval loader is
    # deterministic, so records[i] is the SAME sample in every run: pair by
    # position, then verify that pairing using (frame_id, mask_frac).
    n_common = min(len(d['records']) for d in data.values())
    def sig(rec):
        return (rec['frame_id'], round(rec['mask_frac'], 9))
    aligned = all(
        sig(d['records'][i]) == sig(next(iter(data.values()))['records'][i])
        for d in data.values() for i in range(n_common)
    )
    counts = ', '.join(f"{m}={len(d['records'])}" for m, d in data.items())
    if not aligned:
        print("\n" + "!" * 78)
        print("WARNING: the models were NOT scored on the same samples in the same order.")
        print(f"  per-model sample counts: {counts}")
        print("  The cross-model table below is therefore UNPAIRED — each model was")
        print("  scored on a different (though same-distribution) subset, so small")
        print("  differences between models are not meaningful.")
        print("  Cause: gradcam.py run with --num_workers > 0 makes the webdataset")
        print("  sample order non-deterministic. Re-run all models with")
        print("  --num_workers 0 (the default) for a paired comparison.")
        print("  NOTE: the DETECTED vs MISSED section below is computed WITHIN each")
        print("  model, so it stays valid regardless.")
        print("!" * 78)
    identical = aligned

    # ── 1. Headline table ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    hdr_note = "same samples per model" if identical else "DIFFERENT samples per model — unpaired"
    print(f"GRAD-CAM LOCALISATION — all positives scored ({hdr_note})")
    print("=" * 78)
    print(f"{'model':<10} {'agg':<6} {'n':>4} {'pointing':>9} {'energy':>8} "
          f"{'chance':>7} {'lift':>7}")
    print("-" * 78)
    for m, d in data.items():
        print(f"{m:<10} {d['aggregate']:<6} {d['n_samples']:>4} "
              f"{d['pointing_game_pct']:>8.2f}% {d['energy_in_mask_pct']:>7.2f}% "
              f"{d['chance_pct']:>6.2f}% {d['energy_in_mask_pct']-d['chance_pct']:>+6.2f}")

    # ── 2. Detected vs missed, and the selection-effect test ─────────────
    print("\n" + "=" * 78)
    print("DETECTED vs MISSED — and whether detected samples are simply EASIER")
    print("=" * 78)
    print(f"{'model':<10} {'group':<10} {'n':>4} {'pointing':>9} {'energy':>8} "
          f"{'mask area':>10}")
    print("-" * 78)
    for m, d in data.items():
        recs = d['records']
        for name, grp in (('detected', [r for r in recs if r['correct']]),
                          ('missed',   [r for r in recs if not r['correct']])):
            if not grp:
                continue
            print(f"{m:<10} {name:<10} {len(grp):>4} "
                  f"{mean([r['pointing'] for r in grp])*100:>8.2f}% "
                  f"{mean([r['energy'] for r in grp])*100:>7.2f}% "
                  f"{mean([r['mask_frac'] for r in grp])*100:>9.2f}%")
        print("-" * 78)
    print("If 'mask area' is consistently larger for detected than missed, the")
    print("model is firing on bigger/more obvious deformation — so its detected-")
    print("group localisation score is inflated by selection, not perception.")

    # ── 3. Common-subset comparison (selection effect removed) ───────────
    # Paired by position (see note above), not by frame_id.
    common_detected = [i for i in range(n_common)
                       if all(d['records'][i]['correct'] for d in data.values())]

    print("\n" + "=" * 78)
    print("LIKE-FOR-LIKE: only samples that EVERY model detected")
    print("=" * 78)
    print(f"samples scored by all models : {n_common}"
          + ("" if aligned else "  (NOT aligned — see warning above)"))
    print(f"detected by all models       : {len(common_detected)}")
    MIN_COMMON = 15
    if len(common_detected) < MIN_COMMON:
        print(f"\n  Too few ({len(common_detected)}) commonly-detected frames for a "
              f"stable comparison (want >= {MIN_COMMON}).")
        if not identical:
            print("  This is a direct consequence of the unpaired sampling flagged")
            print("  above — re-run all models with --num_workers 0 first.")
    if len(common_detected) >= MIN_COMMON:
        print()
        has_conc = all('concentration50' in d['records'][0] for d in data.values())
        hdr = f"{'model':<10} {'pointing':>9} {'energy':>8} {'mask area':>10}"
        if has_conc:
            hdr += f" {'conc@50%':>9} {'entropy':>8}"
        print(hdr)
        print("-" * len(hdr))
        for m, d in data.items():
            grp = [d['records'][i] for i in common_detected]
            row = (f"{m:<10} {mean([r['pointing'] for r in grp])*100:>8.2f}% "
                   f"{mean([r['energy'] for r in grp])*100:>7.2f}% "
                   f"{mean([r['mask_frac'] for r in grp])*100:>9.2f}%")
            if has_conc:
                row += (f" {mean([r['concentration50'] for r in grp])*100:>8.2f}%"
                        f" {mean([r['entropy'] for r in grp]):>8.4f}")
            print(row)
        print()
        print("This is the confound-free comparison: identical samples, all of")
        print("which every model got right. Differences here reflect the models,")
        print("not which subset each one happened to detect.")
        if has_conc:
            print()
            print("conc@50% = fraction of pixels holding half the CAM mass, and")
            print("entropy = spread of the map — both MASK-INDEPENDENT. If the")
            print("better classifiers show HIGHER values here, their lower energy-")
            print("in-mask is genuinely diffuse attention (reading context beyond")
            print("the annotated fringe), not merely mis-aimed attention.")

    # ── 3b. Dilation sensitivity, on the commonly-detected samples ───────
    have_dil = all(d['records'] and d['records'][0].get('dilation_scan')
                   for d in data.values())
    if have_dil and len(common_detected) >= MIN_COMMON:
        radii = [s['radius'] for s in next(iter(data.values()))['records'][0]['dilation_scan']]
        print("\n" + "=" * 78)
        print("DILATION SENSITIVITY — is the gap real, or just CAM-resolution blur?")
        print("=" * 78)
        print("Scored on the same commonly-detected samples. One CAM cell is ~32 px,")
        print("so radius 32/64 allows one/two cells of boundary slack.")
        print()
        print(f"{'model':<10}" + ''.join(f"{'r=' + str(r) + 'px':>22}" for r in radii))
        print(f"{'':<10}" + ''.join(f"{'energy':>11}{'enrich':>11}" for _ in radii))
        print("-" * (10 + 22 * len(radii)))
        for m, d in data.items():
            grp = [d['records'][i] for i in common_detected]
            row = f"{m:<10}"
            for k in range(len(radii)):
                en = mean([r['dilation_scan'][k]['energy'] for r in grp])
                mf = mean([r['dilation_scan'][k]['mask_frac'] for r in grp])
                row += f"{en*100:>10.2f}%{(en/mf if mf else float('nan')):>10.2f}x"
            print(row)
        print()
        print("Read the ENRICHMENT columns, not raw energy — energy rises with")
        print("radius for any map because the mask grows. If enrichment CONVERGES")
        print("across models as radius grows, the apparent localisation gap was")
        print("boundary blur from the coarse CAM. If the models stay separated,")
        print("the difference in where they attend is real.")

        # ── Apply the pre-registered rule mechanically ───────────────────
        a, b = PREREG_PAIR
        if a in data and b in data:
            def enrich(model, k):
                grp = [data[model]['records'][i] for i in common_detected]
                en = mean([r['dilation_scan'][k]['energy'] for r in grp])
                mf = mean([r['dilation_scan'][k]['mask_frac'] for r in grp])
                return en / mf if mf else float('nan')

            gap0   = enrich(a, 0) - enrich(b, 0)
            kmax   = len(radii) - 1
            gapmax = enrich(a, kmax) - enrich(b, kmax)
            converged = gapmax < PREREG_THRESHOLD

            print("\n" + "-" * 78)
            print(f"PRE-REGISTERED RULE  ({a} vs {b}), fixed before this scan:")
            print(f"  threshold: gap at r={radii[kmax]}px < {PREREG_THRESHOLD}x  "
                  f"(half the {PREREG_R0_GAP}x gap observed at r=0)")
            print(f"  observed gap at r={radii[0]:>3}px : {gap0:.3f}x"
                  + (f"   [reproduces the {PREREG_R0_GAP}x the rule was set from]"
                     if abs(gap0 - PREREG_R0_GAP) < 0.05 else
                     f"   [WARNING: differs from the {PREREG_R0_GAP}x assumed — "
                     f"note this when reporting]"))
            print(f"  observed gap at r={radii[kmax]:>3}px : {gapmax:.3f}x")
            print()
            print(f"  VERDICT: {'CONVERGED' if converged else 'NOT CONVERGED'} "
                  f"({gapmax:.3f}x {'<' if converged else '>='} {PREREG_THRESHOLD}x)")
            print()
            if converged:
                print("  More than half the localisation gap disappears once boundary")
                print("  slack is allowed, so much of it was CAM-resolution blur.")
                print("  This does NOT show the models attend identically — it bounds")
                print("  how much of the difference survives.")
            else:
                print("  The gap persists beyond boundary slack: the models genuinely")
                print("  concentrate attention differently, not merely at different")
                print("  distances from the mask edge.")
            print("-" * 78)

    # ── 4. Per-timestep progression (models that expose one) ─────────────
    multi = {m: d for m, d in data.items() if len(d.get('per_timestep', [])) > 1}
    if multi:
        print("\n" + "=" * 78)
        print("PER-TIMESTEP PROGRESSION (energy inside mask)")
        print("=" * 78)
        for m, d in multi.items():
            ts = d['per_timestep']
            vals = [f"{t['energy_pct']:.2f}%" for t in ts]
            ratio = ts[-1]['energy_pct'] / ts[0]['energy_pct'] if ts[0]['energy_pct'] else float('nan')
            print(f"  {m:<10} {' -> '.join(vals):<32} rise {ratio:.2f}x")
        print()
        print("cnn_lstm's CAM target (the encoder) has NO recurrent coupling, so")
        print("its rise measures the gradient-reach confound alone. Any steeper")
        print("rise in convlstm is what the recurrence adds on top of that.")
    print("=" * 78)

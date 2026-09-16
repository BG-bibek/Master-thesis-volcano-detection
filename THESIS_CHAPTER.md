# Temporal architectures for InSAR volcanic deformation detection: a reproduction and mechanism study

*Results chapter — draft. Thalia benchmark, `temporal/3` split. Three seeds per
configuration; all figures are test-set results at the 0.5 decision threshold.*

> **Provenance.** Every number traceable to a training log was re-derived
> programmatically from `results_summary.json` (produced by `parse_logs.py`),
> not transcribed. Saliency figures come from `gradcam.py` / `compare_gradcam.py`
> outputs. Web version: https://claude.ai/code/artifact/dc8ab516-b4ba-4411-aa6d-2ae12866ea95

---

## Abstract

The Thalia benchmark reports a ConvLSTM as its strongest time-series model for
binary volcanic deformation classification in InSAR imagery, attributing its
advantage to the explicit modelling of sequential structure. The architecture was
never published. We reimplement it from the single sentence of prose describing it
and reach a test F1 of 78.14 ± 1.36 against the reported 78.89 ± 1.09, a difference
smaller than either standard deviation.

We then test the stated mechanism and find it unsupported. Destroying temporal
frame order does not significantly change performance (t = 1.39); replacing the
ConvLSTM cell with a ConvGRU, removing 25% of recurrent parameters, does not
either (t = 1.42). We show the first result is forced by the benchmark's own task
definition: the union label is permutation-invariant by construction, so frame
order cannot carry learnable signal.

Grad-CAM saliency scored against expert deformation masks shows that localisation
quality also fails to separate architectures whose F1 spans 26 points, and that an
apparent difference is an artefact of saliency-map resolution, confirmed by a
pre-registered dilation test. What does distinguish the temporal models is
detection sensitivity: the channel-stacking baseline fails on deformation events
averaging 6.45% of frame area, the ConvLSTM only on events averaging 2.22%. We
argue this follows from where fusion occurs — early channel concatenation destroys
the cross-frame comparison that separates persistent deformation from transient
atmospheric artefact — and note that such consistency-checking is itself
permutation-invariant, reconciling the sensitivity gain with the null order result.

---

## 1. Motivation and scope

Interferometric Synthetic Aperture Radar (InSAR) enables systematic, global
monitoring of volcanic ground deformation, but its interpretation is difficult.
Genuine deformation fringes are visually similar to atmospheric phase delays, and
the signals of operational interest — the subtle precursors that precede eruption —
are precisely those closest to the noise floor. The Thalia benchmark supplies
machine-learning-ready InSAR products with expert annotations and establishes
reference results for both single-image and time-series classification.

Within its time-series setting, the benchmark's channel-stacking models — ResNet-50,
MobileNetV3, EfficientNetV2, ConvNeXt and ViT — concatenate the three interferograms
of a sequence along the input channel axis and process them with a standard 2D
network. Such models have no architectural mechanism for representing time; the
three acquisitions enter a single convolution as interchangeable channels. The
benchmark additionally reports a ConvLSTM with a ResNet-50 encoder, described as
modelling "the sequential nature of the input through recurrent processing, in
contrast to the channel-stacking approach adopted by the other architectures." It is
the strongest time-series classifier they report.

That model is absent from the released repository. Its published description
consists of one sentence; no hidden dimensions, layer counts, encoder truncation
point, or model-specific training details are given. This chapter has two aims. The
first is reproduction: can the reported result be recovered from the description
alone? The second, and the more substantial contribution, is mechanistic: if it can,
is the advantage attributable to the sequence modelling the benchmark credits?

---

## 2. Data and experimental setup

### 2.1 Dataset and split

All experiments use the Thalia `temporal/3` variant, in which each sample is a
sequence of three interferograms sharing a primary SAR acquisition date and
differing in their secondary dates, ordered chronologically. A sample is a tensor of
shape [27, 512, 512], comprising three timesteps of nine channels each: the wrapped
interferometric phase difference, coherence, and a digital elevation model, plus six
atmospheric variables (total column water vapour, surface pressure, and vertical
integral of temperature, each at the primary and secondary acquisition dates).

We adopt the benchmark's temporal split unchanged: training from January 2014 to
May 2019 (701 positive, 2,626 negative sequences), validation from June to December
2019 (75 / 728), and test from January 2020 to December 2021 (225 / 1,776).
Following the benchmark, a sequence is labelled positive if any of its three
constituent interferograms is annotated as showing deformation. This labelling
choice becomes analytically important in Section 4.1.

Channels are z-score normalised using dataset-level statistics before any other
processing. Class imbalance is handled by undersampling: each epoch draws positive
and negative sequences in equal proportion, matching the benchmark's stated
procedure.

### 2.2 Architectures

Four models are compared.

**Baseline (ResNet-50)** reproduces the benchmark's channel-stacking approach: an
ImageNet-pretrained ResNet-50 with its first convolution adapted to 27 input
channels (23.6M parameters). All three timesteps are fused in that first layer.

**CNN-LSTM** encodes each of the three frames independently with a shared ResNet-50
(9 input channels), global-average-pools each to a 2048-dimensional vector, and
passes the resulting sequence through a single-layer LSTM with 256 hidden units,
classifying from the final hidden state (25.9M parameters). Fusion is deferred until
after per-frame encoding, but spatial structure is discarded before the recurrence.

**ConvLSTM** is our reconstruction of the benchmark's unpublished model. The
ResNet-50 encoder is truncated at its final convolutional stage and its spatial
feature map retained rather than pooled, yielding a [2048, 16, 16] tensor per frame.
A 1×1 convolution with ReLU and spatial dropout reduces this to 256 channels, which
is processed by a ConvLSTM cell (128 hidden channels, 3×3 kernels) in the standard
formulation of Shi et al., where every LSTM gate is a convolution and the hidden and
cell states remain [C, H, W] tensors. The final hidden state is global-average-pooled
and classified (25.8M parameters). Parameter count was deliberately matched to the
CNN-LSTM so that any difference between them cannot be attributed to capacity.

**ConvGRU** is identical to the ConvLSTM in every respect except the recurrent cell,
which uses GRU gating: reset and update gates and no separate cell state, giving
1,327,488 recurrent parameters against the ConvLSTM's 1,769,984 — exactly 75%
(25.4M total). Encoder, bottleneck, dropout and classification head are
byte-identical between the two, so the comparison isolates the gating mechanism.

All four share the same classification head: layer normalisation, dropout, a
128-unit projection with GELU activation, further dropout, and a linear layer to two
classes.

### 2.3 Training protocol

The final protocol uses AdamW with a fixed learning rate of 10⁻⁵ and weight decay
10⁻², cross-entropy loss, batch size 8, gradient clipping at 1.0, and a maximum of
90 epochs with early stopping after 20 epochs without validation F1 improvement.
Spatial augmentation — horizontal and vertical flips, rotation, Gaussian blur and
random resized crop — is applied identically across all three timesteps of a
sequence, preserving inter-frame spatial alignment. The checkpoint with the best
validation F1 is used for test evaluation. Section 3.2 documents how this protocol
was reached; each component was adopted in response to a diagnosed failure rather
than by search.

Every configuration is run at three seeds (42, 7, 1234) and reported as mean ±
standard deviation, matching the benchmark's own protocol.

---

## 3. Reproduction

### 3.1 Headline result

Our ConvLSTM reaches a test F1 of 78.14 ± 1.36 against the benchmark's reported
78.89 ± 1.09. The difference of 0.75 points is smaller than either standard
deviation and the intervals overlap substantially; on the evidence available the two
are indistinguishable. Both of our temporal models exceed the benchmark's own
channel-stacking ResNet-50 by more than thirteen points of F1.

**Table 1.** Test-set performance, mean ± standard deviation over three seeds. Paper
figures are as reported for the time-series setting with atmospheric channels.

| Model | F1 | AUROC | Precision | Recall |
|---|---|---|---|---|
| *This work* | | | | |
| ConvLSTM | **78.14 ± 1.36** | 94.25 ± 1.45 | **80.10 ± 0.74** | 76.30 ± 2.72 |
| ConvGRU | 76.96 ± 0.45 | 90.45 ± 2.19 | 78.36 ± 2.46 | 75.70 ± 2.45 |
| *Thalia benchmark, as reported* | | | | |
| ConvLSTM | **78.89 ± 1.09** | **96.19 ± 0.85** | 77.01 ± 0.38 | **80.89 ± 2.21** |
| ResNet-50 | 63.66 ± 2.08 | 88.00 ± 2.33 | 68.65 ± 0.85 | 59.41 ± 3.27 |

Two differences deserve note rather than concealment. Our AUROC remains
approximately 1.9 points below the reported figure, with only marginal interval
overlap; ranking quality is therefore plausibly, though not conclusively, worse. And
the same F1 is reached through a different operating point: our model is more
precise (80.10 vs 77.01) and less sensitive (76.30 vs 80.89) than theirs. Since F1
is the harmonic mean of the two, equal F1 by different trade-offs is unsurprising,
but it indicates the two implementations are not identical even where their headline
numbers agree.

### 3.2 How the protocol was reached

Initial ConvLSTM runs performed worse than the plain baseline, reaching only 70.64
F1. The diagnosis was severe overfitting: training loss collapsed to 0.0000 on the
majority of batches while validation loss rose, on a training set of roughly 2,560
undersampled sequences per epoch against a 25.8M-parameter model. Four changes
followed, each addressing an identified failure.

**Table 2.** Diagnostic sequence for the ConvLSTM, seed 42. Each row adds one change
to the row above.

| Configuration | F1 | AUROC |
|---|---|---|
| FocalLoss, wd 10⁻⁴, no augmentation | 70.64 | 80.09 |
| + augmentation, cross-entropy, wd 10⁻² | 73.17 | 88.66 |
| + spatial dropout at the encoder–recurrence interface | 72.68 | 91.90 |
| + early stopping, patience 20 | 75.70 | 93.12 |
| + fixed learning rate (cosine schedule removed) | **78.82** | **93.23** |

The final change produced the largest single gain and is worth stating as a general
observation. Cosine annealing was configured to decay to its floor over the full 90
epochs, but early stopping consistently halted training around epoch 44. The
schedule therefore never approached the learning rate it was designed to end at,
leaving the rate elevated relative to the curve's intent at the point of convergence.
Removing the schedule entirely — which also matches the benchmark's stated use of a
fixed learning rate — added 3.1 points of F1. Practitioners combining cosine
annealing with early stopping should be aware that the two interact in this way.

### 3.3 Architecture comparison under a matched protocol

Under an identical training protocol, the three architectures separate sharply:
52.30 F1 for the channel-stacking baseline, 71.27 for the CNN-LSTM, and 75.70 for
the ConvLSTM. The baseline's failure is concentrated in recall (40.44%), indicating
it detects fewer than half of positive sequences while retaining reasonable precision
(73.98%).

An observation from this comparison is worth recording. The baseline's best all-channel result
anywhere in this project is 70.64 F1 (augmentation, focal loss, wd 10⁻⁴, 90 epochs, no
early stopping; recorded in `outputs/metrics_baseline_aug.json`), obtained under
an *earlier*, weaker protocol. A core-channel baseline variant reaches 72.81, which is
the architecture's true maximum here but is not comparable to the 9-channel models.
The protocol that maximises the temporal models drops the all-channel baseline
to 52.30 — an 18-point fall. The regularisation the higher-capacity recurrent models
require is actively harmful to the channel-stacking model, which is indirect evidence
that the architectures use the data differently rather than merely responding to
hyperparameters along a common axis.

---

## 4. Testing the stated mechanism

The benchmark attributes the ConvLSTM's advantage to explicit sequence modelling. We
tested candidate explanations with falsification experiments specified before their
results were known.

### 4.1 Temporal order carries no learnable signal

Frames were randomly permuted during training while validation and test evaluation
retained correct ordering, testing whether the model can learn order-dependent
features. Over three ordered runs and four shuffled runs (the latter spanning three
distinct seeds, with seed 42 run twice — see Section 7.2), ordered training gives
78.14 ± 1.36 and shuffled 75.31 ± 3.77 (Welch t = 1.39). The ranges overlap almost
entirely — [76.57, 79.02] against [71.15, 80.27] — and the single best result across
the entire study, 80.27 F1, comes from a shuffled run. AUROC is likewise unaffected
(94.25 ± 1.45 against 92.48 ± 1.80).

This should not be read as a failure of the architecture, because the outcome is
forced by the task definition. The sequence label is the disjunction of the three
frame labels: a sequence is positive if any frame is positive. This function is
invariant to permutation of its arguments, so no reordering of the three frames
changes the correct answer for any sample in the dataset. The benchmark's own
formulation therefore admits no reward for learning temporal order in the
classification task, and the same argument applies to its segmentation task, where
the target is the union of the frame masks.

> Frame ordering cannot carry learnable signal in this benchmark, by construction.
> Any claim that a model "explicitly models the sequential nature of the input" is,
> for this task, both untestable and unrewarded.

### 4.2 The recurrent cell design is not responsible

Replacing the ConvLSTM cell with a ConvGRU removes one of four gates and the separate
cell state, reducing recurrent parameters by 442,496 (25%) while leaving encoder,
bottleneck, dropout and head byte-identical. Test F1 falls from 78.14 ± 1.36 to
76.96 ± 0.45 (t = 1.42, not significant at n = 3 per arm). The ConvGRU's full range,
[76.57, 77.45], lies inside the ConvLSTM's [76.57, 79.02]; on seed 7 the two produce
identical F1 to two decimal places.

AUROC shows a larger difference (94.25 ± 1.45 against 90.45 ± 2.19, t = 2.51) which
is consistent in direction across all three seeds. We do not claim it as significant
at this sample size, but note it as the one place the two cells may genuinely differ:
matched decisions at the operating threshold, with poorer confidence ranking behind
them. This warrants further investigation and is not resolved here.

---

## 5. Where the models look

Having found that neither temporal ordering nor gate design explains the performance
ordering, we turned to spatial evidence. The dataset supplies pixel-level deformation
masks that the classification pipeline discards; we used them to ask directly where
each model draws its evidence from, and whether better classifiers look in better
places.

### 5.1 Method

Grad-CAM saliency maps were computed for 60 positive test sequences, identical
samples across all models, guaranteed by a deterministic single-process loader. For
the ConvLSTM and ConvGRU the target is the recurrent cell, which fires once per
timestep and therefore yields one saliency map per frame; for the CNN-LSTM it is the
encoder's final convolutional stage, which sees all three frames as a batch and
likewise yields three maps; for the baseline, whose three timesteps are already fused
in its first layer, only one map exists. The ConvLSTM's headline map is taken at the
final timestep, since its classification head reads only the final hidden state; the
others are averaged across frames.

Two scoring measures standard for coarse saliency were used: the *pointing game*,
whether the map's peak falls inside the annotated mask, and *energy ratio*, the
fraction of saliency mass inside it. Intersection-over-union was deliberately
avoided. The map is computed at 16×16 and upsampled to 512×512, so one saliency cell
spans 32 input pixels — far too coarse for meaningful overlap against thin fringes.
Because a mask covering *x*% of the frame scores *x*% by chance, we report enrichment
(energy divided by mask area) wherever mask size varies.

> **Figure 1 — to insert.** Per-timestep saliency overlaid on the wrapped
> interferometric phase, with the expert deformation mask outlined, for the same
> sequence under each architecture. Generated by `gradcam.py` as
> `outputs/gradcam/cam_<model>_NN_<frame_id>.png`; match panels across models by the
> index prefix, not the frame identifier, since Thalia frame IDs denote a geographic
> LiCSAR frame rather than a unique sequence. Recommended: one sequence all three
> models detect, and one the baseline misses but the ConvLSTM detects — the second
> makes the sensitivity result in Section 6 visible rather than tabular.

### 5.2 Localisation does not separate the architectures

Across all 60 samples, energy inside the mask is 28.67% for the baseline, 28.40% for
the CNN-LSTM and 29.58% for the ConvLSTM, against a chance level of 7.64%. All three
are strongly enriched — roughly 3.7 to 3.9 times chance — and the spread between them
is 1.2 points, for models whose F1 spans 26 points. On this evidence, the ability to
find deformation is not what separates them.

Restricting to the 24 sequences every model classified correctly, so that no model is
scored on an easier subset than another, a difference does emerge: the baseline
places more saliency inside the mask than the ConvLSTM (40.60% against 32.80%), with
the CNN-LSTM between them (35.64%). Counter-intuitively, the ordering is inverse to
classification accuracy.

### 5.3 That difference is a resolution artefact

We tested whether the inverse ordering survives the saliency map's resolution limit.
Since one cell covers 32 pixels, saliency falling just outside the annotated boundary
may be upsampling blur rather than misdirection. Masks were dilated by one and two
cell widths and the maps re-scored, comparing enrichment so that the mechanical rise
in raw energy with mask size is divided out. A decision rule was fixed before the test
was run: the enrichment gap between baseline and ConvLSTM at 64 pixels of slack must
fall below 0.415×, half the 0.828× gap observed at zero slack.

**Table 3.** Saliency enrichment (energy ÷ mask area) as the mask is dilated, on the
24 commonly detected sequences. Enrichment is used rather than raw energy because a
larger mask captures more saliency mechanically.

| Model | r = 0 px | r = 32 px | r = 64 px |
|---|---|---|---|
| Baseline (ResNet-50) | 4.31× | 4.42× | 3.38× |
| CNN-LSTM | 3.78× | 3.65× | 2.61× |
| ConvLSTM | 3.48× | 3.96× | 3.07× |
| **Gap, baseline − ConvLSTM** | 0.828× | 0.462× | **0.311×** |

The gap fell to 0.311×, satisfying the pre-registered criterion. Most of the apparent
localisation difference is attributable to saliency-map resolution rather than to the
models attending to genuinely different locations. The criterion bounds how much of
the difference survives; it does not establish that the models attend identically.

The pointing game makes the same point more sharply. At zero slack the ConvLSTM
appears worst of the three (46.67% against the baseline's 60.00%), but with one cell
of slack the ordering inverts and it becomes the best (80.00% against 66.67%). Its
saliency peak is therefore typically *near but just outside* the annotated boundary,
within a single saliency cell. Its enrichment also rises from r = 0 to r = 32 (3.48×
to 3.96×), the signature of attention concentrated at a boundary rather than
scattered.

One speculative reading, offered as interpretation rather than result: wrapped InSAR
phase represents deformation as concentric fringes, and the information about
deformation gradient is densest where fringes are closest together, typically at the
flanks of a deforming lobe rather than its centre. An annotation marks the deformed
*area*, interior included. A model keyed on phase gradient would therefore peak
slightly off-centre relative to the mask. We did not test this, and it should not be
presented as established.

### 5.4 A rejected explanation

Before the resolution test, we hypothesised that the better classifiers place less
saliency inside the mask because they read a wider spatial context — coherence,
topography and atmospheric structure surrounding the fringe — to separate real
deformation from atmospheric artefact. This predicts measurably more diffuse maps for
the stronger models.

We tested it with two mask-independent concentration measures: the fraction of pixels
holding half the saliency mass, and normalised spatial entropy. On the commonly
detected sequences these give 4.42% / 0.849 for the baseline, 5.25% / 0.883 for the
CNN-LSTM and 5.14% / 0.852 for the ConvLSTM. The values are close and not monotonic
in classification performance; the prediction fails. We report the hypothesis and its
rejection because the alternative — presenting only the surviving explanation — would
misrepresent how the analysis proceeded. A caveat applies: the CNN-LSTM's map is an
average of three, which mechanically blurs it and probably inflates its diffuseness
relative to the two single-map models.

### 5.5 Saliency sharpens as frames are integrated

Because the ConvLSTM exposes a spatial hidden state at every timestep, saliency can be
computed at each step of the recurrence. Energy inside the mask rises from 8.91% at
the first frame to 14.44% at the second and 29.58% at the third — a 3.32-fold
increase. The equivalent progression for the CNN-LSTM, whose saliency target is the
encoder and therefore carries no recurrent coupling, rises only 1.21-fold (26.84% to
32.37%).

Both models attenuate gradients toward earlier timesteps, since those reach the output
through more recurrent steps; the CNN-LSTM's shallow rise provides an estimate of that
confound alone, and the ConvLSTM's is roughly 2.7 times steeper than it. Note that
both scoring measures are scale-invariant and each map is normalised, so raw gradient
magnitude does not mechanically produce this. The observation is consistent with
attention converging on the deformation as evidence accumulates across frames, though
the two effects are not fully separable from this experiment.

---

## 6. What distinguishes the architectures

### 6.1 Detection sensitivity

Having excluded ordering, gate design and localisation quality, one explanation
remains supported. All three architectures succeed on deformation events of
comparable extent — mean annotated areas of 9.42%, 9.50% and 9.14% of the frame for
the events each detects. They differ substantially in what they fail to detect.

**Table 4.** Detection behaviour on 60 positive test sequences. Mask area is the mean
fraction of the frame annotated as deformed; a smaller missed-event size indicates
greater sensitivity.

| Model | Detected | Pointing, detected | Mask area, detected | Mask area, missed |
|---|---|---|---|---|
| Baseline (ResNet-50) | 24 / 60 | **91.67%** | 9.42% | 6.45% |
| CNN-LSTM | 43 / 60 | 69.77% | 9.50% | 2.93% |
| ConvLSTM | 47 / 60 | 51.06% | 9.14% | **2.22%** |

The channel-stacking baseline fails on deformation events averaging 6.45% of frame
area — substantial, visually evident signals. The ConvLSTM fails only on events
averaging 2.22%, approximately three times smaller. The baseline is best understood as
a high-threshold detector: on the 24 sequences it does detect, its saliency peak falls
inside the annotated deformation 91.67% of the time, the highest of any model. It
localises what it finds extremely well, and finds comparatively little.

### 6.2 Why fusion depth produces this

The three architectures differ in one structural respect that maps directly onto the
observed ordering: where the three timesteps are combined.

The baseline concatenates them along the channel axis and fuses them in its first
convolution. From that layer onward no representation of "frame 1 versus frame 2"
exists; the network sees a single 27-channel image. It therefore cannot ask whether a
fringe pattern *persists* across acquisitions, and must decide from within-frame
appearance alone — how strongly a given region resembles deformation in a single
composite view. This is exactly the behaviour observed: a detector that fires only on
strong, unambiguous fringe patterns, localises them superbly, and misses anything
weaker.

The CNN-LSTM and ConvLSTM encode each frame independently before combining them, so a
comparison across acquisitions remains available to the classifier. This matters
because of the physics of the measurement. Atmospheric phase delay varies between
acquisition dates while genuine ground deformation persists; the same distinction
motivates the benchmark's inclusion of atmospheric variables in the first place. A
model that can compare frames can treat weak-but-consistent signal as evidence,
whereas a model that cannot must rely on signal strength within a single fused view.
Lowering the effective detection threshold in this way is precisely what the
missed-event sizes in Table 4 show.

The ConvLSTM's additional advantage over the CNN-LSTM follows the same logic one level
down. The CNN-LSTM pools each frame to a single vector before comparing, so its
cross-frame comparison is between global descriptors — it can ask whether the frames
agree overall, but not whether they agree *in the same place*. The ConvLSTM retains
spatial structure through the recurrence and can therefore compare frames position by
position. The per-timestep sharpening in Section 5.5 is consistent with that reading.

> The ConvLSTM's advantage is not sequence modelling, and not spatial precision. It is
> sensitivity — detection of deformation too subtle for a channel-stacking model to
> register, at roughly one third the event size — and it follows from deferring fusion
> until after per-frame encoding, which preserves the cross-frame comparison that
> distinguishes persistent deformation from transient atmospheric artefact.

This account also resolves an apparent tension. Section 4.1 shows that frame *order*
carries no signal, yet Section 6.1 shows that multi-frame processing carries a great
deal. These are consistent, because checking whether a signal persists across three
observations does not require knowing which came first. Consistency is itself a
permutation-invariant property. The benchmark's union labelling rewards evidence
aggregation and is indifferent to sequence, and the architectures that win are
precisely those that aggregate — not those that order.

### 6.3 Atmospheric channels

The account above predicts that the temporal models should depend on the atmospheric
variables, since those are what make a fringe attributable to weather rather than
ground motion. Removing the six atmospheric channels and retaining only phase,
coherence and elevation reduces F1 from 78.82 to 72.06 and AUROC from 93.23 to 87.28
(single seed). The direction matches the benchmark's own finding, but the magnitude is
roughly double: 6.76 points against their 3.56.

We note this as consistent with the mechanism rather than as a test of it, and flag two
caveats: our protocol was tuned on the full-channel configuration, so the
reduced-channel run is not separately optimised, and this ablation was run at a single
seed.

---

## 7. Methodological findings

### 7.1 The stated recipe contradicts the released configuration

The benchmark's supplementary material specifies cross-entropy loss, weight decay
10⁻², and a fixed learning rate. Its released `configs.json` ships focal loss and
weight decay 10⁻⁴. These are different training recipes, and the difference is worth
several points of F1 in our experiments (Table 2, rows 1–2). The configuration also
sets gradient clipping to null, whereas clipping is frequently assumed. Anyone
attempting to reproduce the benchmark from the repository will train a different model
from the one described in the paper. We report this without attributing intent; it is
the kind of drift that occurs when a configuration file evolves after a manuscript is
written.

### 7.2 Same-seed variance exceeds reported error bars

Two runs launched with identical arguments and the same random seed produced test F1 of
74.51 and 71.15, and AUROC of 89.87 and 93.60. Seeding `torch`, `numpy` and `random`
does not determine the sampling order of a multi-worker WebDataset pipeline, shard
shuffling, the augmentation RNG inside worker processes, or CUDA kernel
non-determinism.

The resulting 3.36-point same-seed spread in F1 exceeds the 1.36-point between-seed
standard deviation we report as error bars, and exceeds the 1.09-point standard
deviation the benchmark reports for the same model. If comparable non-determinism
exists in other pipelines — and nothing about ours is unusual — then error bars
computed across seeds may systematically understate true run-to-run variance in this
literature. We report seed-based figures for comparability with the benchmark, but flag
that they are a lower bound on the uncertainty.

---

## 8. Limitations

- **Sample size.** Three seeds per configuration follows the benchmark's convention but
  gives limited power. The null results in Section 4 are failures to reject, not
  demonstrations of equivalence; a genuine effect smaller than roughly 2 points of F1
  would not be reliably detected here.
- **Single-seed ablations.** The atmospheric-channel ablation, the matched-protocol
  architecture comparison and all saliency analyses were run at one seed and should be
  treated as indicative.
- **Saliency comparability.** Grad-CAM was computed at structurally different layers
  across models — the encoder's final stage for the baseline and CNN-LSTM, the
  recurrent hidden state for the ConvLSTM. The baseline-to-CNN-LSTM comparison shares a
  target layer and is the more controlled of the two; conclusions involving the
  ConvLSTM's absolute saliency values carry more uncertainty.
- **The fusion-depth account is an interpretation.** Section 6.2 is consistent with
  every measurement reported here, but no experiment isolates fusion depth as a causal
  variable. A direct test — a late-fusion model with no recurrence at all, aggregating
  per-frame encodings by mean pooling — would separate "deferred fusion" from
  "recurrence" and is the obvious next experiment.
- **Interrupted runs.** One ConvGRU run was interrupted by a CUDA fault and resumed.
  Because the early-stopping counter is not persisted across resumption, that run
  trained longer than an uninterrupted one would have. Its result lies within the spread
  of the others, but the stopping rule differed.
- **Reproduction is not identity.** Matching a reported F1 does not establish that our
  architecture matches the unpublished one. The divergence in precision/recall balance
  and in AUROC suggests it does not.

---

## 9. Conclusion

An unpublished benchmark model was reconstructed from a one-sentence description and
reproduced its reported F1 to within the noise of either measurement. The mechanism the
benchmark credits for that performance — explicit modelling of sequential structure — is
not supported. Frame order carries no learnable signal, and cannot, because the task's
union labelling is permutation-invariant by construction. The specific recurrent gating
is not responsible either: a ConvGRU with a quarter fewer recurrent parameters performs
equivalently. Spatial localisation quality does not separate architectures whose accuracy
differs by 26 points of F1, and an apparent difference proves to be an artefact of
saliency-map resolution.

What the temporal architectures provide is sensitivity to weak signal. They detect
deformation roughly three times smaller than the channel-stacking baseline registers,
while localising what they detect no more precisely. We attribute this to fusion depth:
combining acquisitions only after per-frame encoding preserves the cross-frame comparison
that separates persistent ground deformation from transient atmospheric delay, and that
comparison is order-free, which is why the sensitivity gain and the null ordering result
are consistent rather than contradictory.

For operational monitoring, where the events of interest are the subtle precursors a
high-threshold detector misses, sensitivity is the more consequential property — and it
is not the one the benchmark's framing emphasises. Three independent falsification
attempts returned null, and the surviving explanation is supported by direct measurement.
That structure, closing off competing explanations rather than accumulating
confirmations, is what gives the remaining claim its weight.

---

## References

Papadopoulos, N., Bountos, N. I., Sdraka, M., Karavias, A., Camps-Valls, G., and
Papoutsis, I. *Thalia: A global, multi-modal dataset for volcanic activity monitoring.*
arXiv:2505.17782.

Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-K., and Woo, W.-c. *Convolutional
LSTM network: a machine learning approach for precipitation nowcasting.* NeurIPS, 2015.

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D.
*Grad-CAM: visual explanations from deep networks via gradient-based localization.* ICCV,
2017.

Bountos, N. I., et al. *Hephaestus: A large scale multitask dataset towards InSAR
understanding.* CVPR Workshops, 2022.

---

### Draft notes

- Section numbering assumes this sits as a standalone results chapter; renumber to fit
  the surrounding document.
- Citation formatting follows no particular style guide. Apply your department's.
- The four references are those load-bearing for methods. Related-work citations belong
  in Section 1.
- Figure 1 (Grad-CAM panels) still to be inserted — see Section 5.1 for what to pick.
- Reproduce every table with: `python parse_logs.py` then the queries in
  `results_summary.json`.

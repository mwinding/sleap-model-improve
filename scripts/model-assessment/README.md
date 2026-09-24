# Centroid model assessment

Reproducible version of the previous ChatGPT comparison: per-model recall,
precision, F1, localisation error, hard/other frame subsets, score sweeps,
comparison plots and paired image overlays.

Run from the repository root, using the existing `sleap165` environment:

```bash
conda activate sleap165
python scripts/model-assessment/assess_centroids.py
python scripts/model-assessment/plot_assessment.py
```

On this Mac, the interpreter is also available directly at
`/Users/windinm/miniconda3/envs/sleap165/bin/python`.
Dependencies: `sleap-io`, `numpy`, `scipy`, `opencv-python`, `h5py`, `matplotlib`.
No training or prediction jobs are launched and the mounted source files are read only.

## Inputs and frame identity

Defaults:

- GT: `data/combined_ground_truth_cleaned_visibility-fixed.pkg.slp`
- Manifest: `scripts/model-assessment/comparison_manifest.csv` (copied from the mounted benchmark)
- Predictions: `/Volumes/lab-windingm/home/shared/sleap/model-tests/predictions/*.slp`
- Video: `/Volumes/lab-windingm/home/shared/sleap/model-tests/comparison_frames_plain.mp4`
- Output: `outputs/model_assessment/`

The manifest identifies 14 unique source images and 368 annotated animals.
Each image occupies ten consecutive video frames: order 1 maps to 0–9, order 2
to 10–19, and so on. The evaluator verifies **every** encoded video frame against
all 14 embedded source images using thumbnail pixel differences. It also checks
image dimensions and expected GT counts. It fails rather than silently assuming
a different frame order or coordinate scale. Embedded source images are resolved
by exact `videoX/frame_numbers` lookup, never by their label-list position.

Only centroid-only prediction SLPs are accepted. Prediction metadata must describe
140 frames. An omitted labeled-frame entry is counted as zero detections and is
listed in `assessment.json`; this is the same scoring outcome as a failed/missing
prediction, not a reason to omit that image.

To run on CAMP, override the mounted paths:

```bash
python scripts/model-assessment/assess_centroids.py \
  --predictions-dir /camp/lab/windingm/home/shared/sleap/model-tests/predictions \
  --comparison-video /camp/lab/windingm/home/shared/sleap/model-tests/comparison_frames_plain.mp4
```

`--ground-truth`, `--manifest`, `--output`, and `--models MODEL_NAME ...` are also
available. Model names omit the `.slp` extension. Without `--models`, all `.slp`
files in the predictions directory are assessed.

## Matching and the historical target discrepancy

The default `--target body` uses the same anatomical body point for every model.
Matching maximizes the number of one-to-one assignments within **15 px inclusive**,
then minimizes their total distance. Unmatched GT are false negatives; unmatched
predictions are false positives. This avoids greedy matching errors in dense piles.

The stored single-node prediction confidence must be at least **0.2**. Sweeps at
0.2, 0.3, 0.4 and 0.5 only filter saved predictions. They cannot recover detections
suppressed when `sleap predict` originally ran. Use `--threshold`, `--thresholds`,
and `--tolerance` to change these settings.

The previous ChatGPT table did not consistently use body-point targets, despite
its description. Mean-of-visible-keypoint targets reproduce the old baseline
numbers (69.3% recall), whereas fixed body points produce 64.4%. This discrepancy
also affects other models without a body anchor. An explicit historical mapping
is provided to reproduce those comparisons:

- Mean of visible keypoints: `centroid_baseline`, `centroid_fullres_sigma5`,
  `centroid_halfres_sigma2p5`, `centroid_fullres_sigma2p5`.
- Body: the eight named body-anchored models in `assess_centroids.py`.
- Unknown models in historical mode fail with an explanatory error.

```bash
python scripts/model-assessment/assess_centroids.py \
  --target historical --output outputs/model_assessment/historical
python scripts/model-assessment/plot_assessment.py \
  --assessment outputs/model_assessment/historical
```

`--target mean` applies the same mean target to *all* models. Historical plots are
explicitly labeled as historical; they measure models against different targets.
Use fixed body targets for a comparison of body detection performance. Neither
mode changes the SLP annotations or marks transparent overlapping larvae invisible.

Each repeat is scored separately. TP/FP/FN are averaged over the ten repeats and
recall/precision are computed from pooled counts (equivalent with equal repeats).
`per_image.csv` reports one row per unique image/model and min/max TP across repeats.
The 140 encoded frames are **not 140 independent test images**. Localisation error
is the mean distance of matched pairs only; it excludes missed animals.

The historical hard subset is manifest category `hard_early`: video 9 frames
28639, 23376, 7717 and 18156 (77 animals). It is a selected frame-level subset,
not an objective per-animal crossing or pile classification. Do not interpret it
as a formally measured crossing-specific recall.

## Outputs

- `report.md`: readable model table and evaluation notes.
- `summary.csv`: all/hard/other metrics per model.
- `per_image.csv`: repeat-averaged metrics for each unique image.
- `per_frame.csv`: raw TP/FP/FN for every encoded repeat.
- `matches.csv`: every TP/FN/FP, coordinates, scores and match distances at the primary threshold.
- `threshold_sweep.csv`: metrics at each saved-score threshold.
- `frame_mapping.csv`: verified video-to-source mapping and thumbnail errors.
- `assessment.json`: arguments, target conventions, prediction/GT/manifest hashes, omitted-frame records.
- `plots/`: initial development, cfull tweaks, synthetic comparison, all models,
  paired per-image recall, and threshold sweep; PNG. Each model-comparison
  figure has separate overall/hard-frame recall subplots; precision is saved in a
  separate `*_precision` figure.
- `examples/`: all 14 baseline/candidate image pairs and zoomed comparisons.

Overlays use repeat 0 by default; choose another with `--repeat`.
The aggregate plots always average all repeats. Cyan circles are predictions;
green crosses are detected GT; red crosses are missed GT; yellow rings identify
GT missed by the baseline and recovered by the candidate.

```bash
python scripts/model-assessment/plot_assessment.py \
  --baseline centroid_fullres_body \
  --candidate centroid_body_synth_fullframe_darkoverlap_v1 \
  --zoom-orders 2 3 5 --zoom-size 480
```

Zoom regions are chosen from the densest GT neighbourhood, independently of which
model improves there. Exact crop bounds are saved in `examples/regions.json`.
Historical grouping plots include the original 12 models; the all-model plot also
includes newly added model files.

## Training overlap sensitivity

Training/test independence is **not certified** by these scripts. Source frame
41972 (manifest order 11) was proposed as a synthetic training background.
It and every other benchmark source must be checked against actual real training
frames and synthetic donors/backgrounds before claiming an independent benchmark.
To reproduce the earlier sensitivity analysis without that image:

```bash
python scripts/model-assessment/assess_centroids.py \
  --models centroid_fullres_body centroid_body_synth_fullframe_darkoverlap_v1 \
  --exclude-orders 11 --output outputs/model_assessment/excluding_41972
python scripts/model-assessment/plot_assessment.py \
  --assessment outputs/model_assessment/excluding_41972
```

Exclusion uses original manifest order; it never shifts the video-frame mapping.
Use a separate output directory for each evaluation protocol or exclusion set.
Existing named output files are overwritten on rerun.

## Checks

```bash
python -m unittest discover -s scripts/model-assessment -v
```

Tests cover one-to-one matching in ambiguous piles, invalid edges, duplicate
predictions, tolerance boundaries, empty detections, repeat accounting and omitted
prediction frames. End-to-end validation uses the actual mounted prediction files.

## Held-out retrains

The `_holdout` models (see `scripts/sleap-training-exps/README.md`) are assessed into a
separate directory so the original results are kept:

```bash
python scripts/model-assessment/assess_centroids.py --models centroid_*_holdout --output outputs/model_assessment/holdout
python scripts/model-assessment/plot_assessment.py --assessment outputs/model_assessment/holdout --suffix _holdout
python scripts/model-assessment/plot_holdout_comparison.py
```

`--suffix` maps the plotted groups onto the retrained model names. `plot_holdout_comparison.py`
draws each model's original score (benchmark frames in training) next to its held-out retrain
and writes the differences to `holdout_vs_original.csv`.

## Miss analysis

```bash
python scripts/model-assessment/analyse_misses.py --models centroid_fullres_body centroid_body_synth_fullframe_darkoverlap_v1
```

Writes `<assessment>/misses/<model>_misses.csv` and a montage of crops around every missed
larva, categorised by distance to the nearest other larva's skeleton (overlapping / close /
frame edge / isolated), with the nearest unmatched prediction, local contrast and the number of
repeats in which it was missed.

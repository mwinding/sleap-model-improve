# Centroid training experiments

## Held-out benchmark frames

The 14 benchmark images in `scripts/model-assessment/comparison_manifest.csv` come from
the ground-truth SLP, so earlier models were trained on the frames they were tested on.
`make_holdout_split.py` removes those frames, plus any labeled frame within 100 frames of
one in the same video, from the real training labels:

```bash
python scripts/sleap-training-exps/make_holdout_split.py
```

This writes `data/combined_ground_truth_train.pkg.slp` (203 of 222 frames) and
`data/holdout_frames.json` (14 benchmark + 5 buffer frames). Video indices are unchanged,
so donor records and the manifest still refer to the same videos.

The synthetic-data scripts read `data/holdout_frames.json` by default and skip held-out
frames as backgrounds and as donor sources (`--no-holdout` disables this).
`configs/centroid_holdout.yaml` trains on the held-out split, and `make_centroid_config.py`
(used by `train_sleap.sh --synthetic`) now uses it as its base config.

Each label file is split into train/validation separately with `trainer_config.seed` (42),
so the real-label validation split is identical with or without synthetic data.

## Held-out reruns

`configs/holdout/` has one config per model trained before the holdout. Each is that
model's saved `initial_config.yaml` with only two changes: the real labels point at
`combined_ground_truth_train.pkg.slp` (and synthetic labels at their regenerated
`synthetic_data_holdout/` versions), and `run_name` gains a `_holdout` suffix, so the
original models and predictions are kept for comparison.

The regenerated synthetic sets use the original settings (seed 165):

```bash
O=outputs/synthetic_data_holdout
python scripts/synthetic-data/generate_full_frame_crossings.py --source data/combined_ground_truth_train.pkg.slp --output $O
python scripts/synthetic-data/generate_crossings.py --count 1000 --mode darken_overlap --output $O/crops
python scripts/synthetic-data/filter_extra_animals.py --crossings $O/crops
python scripts/synthetic-data/generate_hard_crossings.py --output $O
```

These, `data/combined_ground_truth_train.pkg.slp` and `data/holdout_frames.json` are copied
to `/camp/lab/windingm/home/shared/sleap/groundtruth/combined/` (synthetic sets under
`synthetic_data_holdout/`, crop crossings under `synthetic_data_holdout/crops/`).

Train on CAMP, from the repository root, one config per job:

```bash
sbatch scripts/sleap-training-exps/train_sleap.sh scripts/sleap-training-exps/configs/holdout/centroid_fullres_body_holdout.yaml
```

Predict, then assess (assessment runs on the Mac against the mounted predictions):

```bash
sbatch scripts/sleap-training-exps/predict_centroids.sh 'centroid_*_holdout'
python scripts/model-assessment/assess_centroids.py --output outputs/model_assessment/holdout
```

Once the training recipe is chosen, retrain it on all 222 frames (`configs/centroid.yaml`,
synthetic data with `--no-holdout`). That final model has seen the benchmark frames, so
report the held-out results, not its benchmark score.

## Peak-separation experiments

The miss analysis (`scripts/model-assessment/analyse_misses.py`) showed that essentially every
missed larva overlaps another one, and that the model produces a single confidence-map peak
for two body points closer than ~10 px (σ 2.5 × output stride 2 = 5 px in the image). Two
single-change configs off the full-frame-synthetic holdout recipe target this:

- `configs/holdout/centroid_synth_fullframe_stride1_sigma2_holdout.yaml`: output stride 1, σ 2
  (2 px peaks instead of 5 px).
- `configs/holdout/centroid_synth_fullframe_mouthhooks_holdout.yaml`: mouthhooks anchor, the
  single node least often fused with a neighbour in the benchmark frames. Assess it with
  `assess_centroids.py --target-node mouthhooks`.

## Parallel crop crossings

`generate_crossings.py --parallel-probability P` places the second larva side by side with the
first (heading within `--parallel-jitter` degrees, body points `--parallel-offset` px apart,
both larvae at least `--parallel-min-straightness` straight) instead of centred on its midline.
This targets the parallel stacks along the wall that the centroid models miss. Default P = 0
reproduces the earlier crop sets exactly.

```bash
O=outputs/synthetic_data_holdout
python scripts/synthetic-data/generate_crossings.py --count 1400 --mode darken_overlap --parallel-probability 0.7 --output $O/crops_parallel_1k
python scripts/synthetic-data/filter_extra_animals.py --crossings $O/crops_parallel_1k
python scripts/synthetic-data/generate_crossings.py --count 4100 --mode darken_overlap --parallel-probability 0.7 --output $O/crops_parallel_3k
python scripts/synthetic-data/filter_extra_animals.py --crossings $O/crops_parallel_3k
```

1,103 and 3,245 crops are kept. Same seed, so the 3k set contains the 1k set. Configs:
`configs/holdout/centroid_synth_crops_parallel1k_holdout.yaml` and `..._parallel3k_holdout.yaml`.

## Anchor × synthetic-data grid

A 2×2 design, everything else fixed at the full res + body settings, each cell trained with
seed 42 and seed 7 (the seed also sets the train/validation split):

| | Real data only | Real + full-frame synthetic |
|---|---|---|
| Body anchor | `centroid_fullres_body_holdout` | `centroid_body_synth_fullframe_darkoverlap_v1_holdout` |
| Mouthhooks anchor | `centroid_fullres_mouthhooks_holdout` | `centroid_synth_fullframe_mouthhooks_holdout` |

Seed-7 configs have the same names with `_seed7`. Mouthhooks models are assessed with
`assess_centroids.py --target-node mouthhooks`.

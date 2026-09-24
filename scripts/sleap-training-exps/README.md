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

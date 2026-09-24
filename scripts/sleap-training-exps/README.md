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

## Held-out comparison run

Copy the inputs to CAMP (from the Mac, with the share mounted):

```bash
COMBINED=/Volumes/lab-windingm/home/shared/sleap/groundtruth/combined
cp data/combined_ground_truth_train.pkg.slp data/holdout_frames.json "$COMBINED/"
mkdir -p "$COMBINED/synthetic_data_holdout"
cp outputs/synthetic_data_holdout/crossings_full-frame.slp "$COMBINED/synthetic_data_holdout/"
```

Train on CAMP, from the repository root:

```bash
sbatch scripts/sleap-training-exps/train_sleap.sh scripts/sleap-training-exps/configs/centroid_holdout.yaml
sbatch scripts/sleap-training-exps/train_sleap.sh --synthetic \
    --run-name centroid_body_synth_fullframe_darkoverlap_holdout \
    /camp/lab/windingm/home/shared/sleap/groundtruth/combined/synthetic_data_holdout/crossings_full-frame.slp
```

Predict, then assess (assessment runs on the Mac against the mounted predictions):

```bash
sbatch scripts/sleap-training-exps/predict_centroids.sh 'centroid_*_holdout'
python scripts/model-assessment/assess_centroids.py \
    --models centroid_fullres_body_holdout centroid_body_synth_fullframe_darkoverlap_holdout \
    --output outputs/model_assessment/holdout
python scripts/model-assessment/plot_assessment.py --assessment outputs/model_assessment/holdout \
    --baseline centroid_fullres_body_holdout --candidate centroid_body_synth_fullframe_darkoverlap_holdout
```

Once the training recipe is chosen, retrain it on all 222 frames (`configs/centroid.yaml`,
synthetic data with `--no-holdout`). That final model has seen the benchmark frames, so
report the held-out results, not its benchmark score.

#!/bin/bash

#SBATCH --job-name=centroid_pred
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --partition=ga100
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out

set -e

ml purge
ml Anaconda3/2024.10
source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap165

VIDEO=/camp/lab/windingm/home/shared/sleap/model-tests/comparison_frames_plain.mp4
MODELS=/camp/lab/windingm/home/shared/models/sideview/experiments
OUT=/camp/lab/windingm/home/shared/sleap/model-tests/predictions

mkdir -p "$OUT"

for MODEL in \
    centroid_baseline \
    centroid_fullres_sigma5 \
    centroid_halfres_sigma2p5 \
    centroid_halfres_body \
    centroid_fullres_body
do
    echo "======================================"
    echo "Running $MODEL"
    echo "======================================"

    sleap predict \
        -i "$VIDEO" \
        -m "$MODELS/$MODEL" \
        -o "$OUT/$MODEL.slp" \
        --peak_threshold 0.2 \
        --batch_size 8
done

echo "All predictions complete."
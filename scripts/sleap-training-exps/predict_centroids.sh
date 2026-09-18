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

# No arguments = run all centroid models
if [ "$#" -eq 0 ]; then
    PATTERNS=("centroid_*")
else
    PATTERNS=("$@")
fi

for PATTERN in "${PATTERNS[@]}"; do
    for MODEL_PATH in "$MODELS"/$PATTERN; do
        [ -d "$MODEL_PATH" ] || continue

        MODEL=$(basename "$MODEL_PATH")
        OUTPUT="$OUT/$MODEL.slp"

        if [ -f "$OUTPUT" ]; then
            echo "Skipping $MODEL - prediction already exists"
            continue
        fi

        echo "======================================"
        echo "Running $MODEL"
        echo "======================================"

        sleap predict \
            -i "$VIDEO" \
            -m "$MODEL_PATH" \
            -o "$OUTPUT" \
            --peak_threshold 0.2 \
            --batch_size 8
    done
done

echo "All predictions complete."
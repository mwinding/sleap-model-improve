#!/bin/bash

#SBATCH --job-name=SLEAP_train
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --partition=ga100
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-user=$(whoami)@crick.ac.uk
#SBATCH --mail-type=FAIL

set -e

ml purge
ml Anaconda3/2024.10

source /camp/apps/eb/software/Anaconda/conda.env.sh
conda activate /camp/lab/windingm/home/shared/conda-envs/sleap165

cd "$SLURM_SUBMIT_DIR"

CONFIG=$1
shift

sleap train "$CONFIG" "$@"
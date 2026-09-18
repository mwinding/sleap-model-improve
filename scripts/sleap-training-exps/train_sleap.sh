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

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=""
for CANDIDATE in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/../.." "$PWD" "$PWD/../.." "$SCRIPT_DIR/../.."; do
	[[ -n "$CANDIDATE" ]] || continue
	CANDIDATE=$(cd -- "$CANDIDATE" 2>/dev/null && pwd) || continue
	if [[ -f "$CANDIDATE/scripts/sleap-training-exps/make_centroid_config.py" ]]; then
		REPO_ROOT="$CANDIDATE"
		break
	fi
done
if [[ -z "$REPO_ROOT" ]]; then
	echo "Could not locate repository containing make_centroid_config.py" >&2
	exit 2
fi
SCRIPT_DIR="$REPO_ROOT/scripts/sleap-training-exps"
cd "$REPO_ROOT"

if [[ "${1:-}" == "--synthetic" ]]; then
	shift
	if [[ "${1:-}" != "--run-name" || -z "${2:-}" ]]; then
		echo "Usage: sbatch train_sleap.sh --synthetic --run-name NAME PATH [PATH ...]" >&2
		exit 2
	fi
	RUN_NAME=$2
	shift 2
	if [[ "$#" -eq 0 ]]; then
		echo "At least one synthetic SLP is required" >&2
		exit 2
	fi
	CONFIG=$(mktemp "${TMPDIR:-/tmp}/centroid_synthetic_XXXXXX.yaml")
	trap 'rm -f "$CONFIG"' EXIT
	python "$REPO_ROOT/scripts/sleap-training-exps/make_centroid_config.py" \
		--output "$CONFIG" \
		--run-name "$RUN_NAME" \
		--synthetic "$@"
	set --
else
	CONFIG=${1:?"Usage: sbatch train_sleap.sh path/to/config.yaml [overrides...]"}
	shift
	if [[ ! -f "$CONFIG" ]]; then
		CONFIG="$REPO_ROOT/$CONFIG"
	fi
	if [[ ! -f "$CONFIG" ]]; then
		echo "Config file not found: $CONFIG" >&2
		exit 2
	fi
fi

echo "Training with config: $CONFIG"
sleap train --config "$CONFIG" "$@"
#!/bin/bash
# Run the benchmark predictions on a benchmark video (e.g. one written by make_benchmark_video.py), locally on the Mac.
#
#   scripts/model-assessment/predict_benchmark.sh VIDEO OUTDIR [--centroids MODEL ...] [--pipelines DETECTOR__POSE ...]
#
# OUTDIR/centroids/<model>.slp            detector-only predictions (default: every *_holdout model with a prediction
#                                          in the shared predictions folder)
# OUTDIR/pipeline/<detector>__<pose>.slp  top-down pipelines (default: the backbone comparison)
# OUTDIR/merge_seed42, merge_seed7        the two-step mouthhooks + body + tail merge: topdown_*, body_points_*,
#                                          redrawn_* and final_twostep_mhbt.slp
# Existing outputs are kept, so a rerun only fills in what is missing.
set -euo pipefail

VIDEO=${1:?usage: predict_benchmark.sh VIDEO OUTDIR [--centroids MODEL ...] [--pipelines DETECTOR__POSE ...]}
OUT=${2:?usage: predict_benchmark.sh VIDEO OUTDIR}
shift 2
ENV=/Users/windinm/miniconda3/envs/sleap165/bin
MODELS=/Volumes/lab-windingm/home/shared/models/sideview/experiments
OLD_PREDICTIONS=/Volumes/lab-windingm/home/shared/sleap/model-tests/predictions
SCRIPTS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CENTROIDS=()
PIPELINES=()
LIST=
for ARG in "$@"; do
    case $ARG in
        --centroids) LIST=centroids ;;
        --pipelines) LIST=pipelines ;;
        *) if [[ $LIST == centroids ]]; then CENTROIDS+=("$ARG"); elif [[ $LIST == pipelines ]]; then PIPELINES+=("$ARG");
           else echo "Unexpected argument: $ARG" >&2; exit 2; fi ;;
    esac
done
DEFAULTS=$(( $# == 0 ))

if (( DEFAULTS )); then
    for F in "$OLD_PREDICTIONS"/*_holdout*.slp; do CENTROIDS+=("$(basename "$F" .slp)"); done
    DET=centroid_body_synth_fullframe_darkoverlap_v1
    for S in "" _seed7; do
        for B in "" _convnext_pretrained _convnext_scratch _swint_pretrained; do
            PIPELINES+=("${DET}_holdout__centered_instance_body${B}_holdout${S}")
            PIPELINES+=("${DET}${B}_holdout${S}__centered_instance_body_holdout")
        done
        PIPELINES+=("${DET}_convnext_pretrained_holdout${S}__centered_instance_body_convnext_pretrained_holdout")
    done
fi

predict() {  # output, input, model...
    local OUTPUT=$1 INPUT=$2
    shift 2
    [[ -f $OUTPUT ]] && return
    local ARGS=()
    for MODEL in "$@"; do ARGS+=(-m "$MODELS/$MODEL"); done
    echo "== $(basename "$OUTPUT")"
    "$ENV/sleap" predict -i "$INPUT" "${ARGS[@]}" -o "$OUTPUT" --peak_threshold 0.2 --batch_size 4 \
        > "${OUTPUT%.slp}.log" 2>&1
}

mkdir -p "$OUT/centroids" "$OUT/pipeline"
for MODEL in ${CENTROIDS[@]+"${CENTROIDS[@]}"}; do
    predict "$OUT/centroids/$MODEL.slp" "$VIDEO" "$MODEL"
done
for PAIR in ${PIPELINES[@]+"${PIPELINES[@]}"}; do
    predict "$OUT/pipeline/$PAIR.slp" "$VIDEO" "${PAIR%%__*}" "${PAIR##*__}"
done

(( DEFAULTS )) || exit 0
for S in "" _seed7; do
    D="$OUT/merge_seed$([[ -z $S ]] && echo 42 || echo 7)"
    mkdir -p "$D"
    for NODE in mouthhooks body tail; do
        predict "$D/topdown_$NODE.slp" "$VIDEO" "centroid_fullres_${NODE}_holdout$S" "centered_instance_${NODE}_holdout"
    done
    for NODE in mouthhooks tail; do
        [[ -f $D/body_points_$NODE.slp ]] || "$ENV/python" "$SCRIPTS/body_points_for_redraw.py" \
            "$D/topdown_$NODE.slp" "$D/body_points_$NODE.slp" 2>/dev/null
        predict "$D/redrawn_$NODE.slp" "$D/body_points_$NODE.slp" centered_instance_body_holdout
    done
    [[ -f $D/final_twostep_mhbt.slp ]] || "$ENV/python" "$SCRIPTS/merge_poses.py" \
        --input mouthhooks "$D/redrawn_mouthhooks.slp" --input body "$D/topdown_body.slp" \
        --input tail "$D/redrawn_tail.slp" --match-px 15 --min-support 2 --single-min-score 0.7 \
        --output "$D/final_twostep_mhbt.slp" 2>/dev/null
done
echo "All predictions complete."

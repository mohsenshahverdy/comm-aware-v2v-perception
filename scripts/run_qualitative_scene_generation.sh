#!/usr/bin/env bash
# Generate qualitative scene candidates and figures for CARLA and Culver.
#
# Usage:
#   bash scripts/run_qualitative_scene_generation.sh --dry-run
#   bash scripts/run_qualitative_scene_generation.sh --generate
#
# Edit the run-directory variables below, or override them from the shell:
#   CARLA_FULL=/path/to/full ... bash scripts/run_qualitative_scene_generation.sh --dry-run

set -euo pipefail

MODE="${1:---dry-run}"
case "$MODE" in
  --dry-run|--dry_run)
    DRY_RUN=1
    ;;
  --generate|--full)
    DRY_RUN=0
    ;;
  -h|--help)
    sed -n '1,45p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Use --dry-run or --generate" >&2
    exit 2
    ;;
esac

# Resolve repository root from this script location; no username-specific paths.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/Classical_Format_Thesis/figures/qualitative}"
MAX_SCENES="${MAX_SCENES:-5}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.7}"
RENDER_STYLE="${RENDER_STYLE:-classic}"

# -----------------------------------------------------------------------------
# CARLA run directories. Edit these defaults or override them as env variables.
# -----------------------------------------------------------------------------
CARLA_FULL="${CARLA_FULL:-/path/to/carla/full_communication_run}"
CARLA_TOPK="${CARLA_TOPK:-/path/to/carla/selective_topk_energy_10_run}"
CARLA_RECEIVER="${CARLA_RECEIVER:-/path/to/carla/receiver_request_energy_topk_10_run}"
CARLA_TEMPORAL="${CARLA_TEMPORAL:-/path/to/carla/temporal_receiver_request_energy_topk_10_run}"
CARLA_LEARNED="${CARLA_LEARNED:-/path/to/carla/learned_temporal_receiver_request_10_run}"

# -----------------------------------------------------------------------------
# Culver run directories. Edit these defaults or override them as env variables.
# -----------------------------------------------------------------------------
CULVER_FULL="${CULVER_FULL:-/path/to/culver/full_communication_run}"
CULVER_TOPK="${CULVER_TOPK:-/path/to/culver/selective_topk_energy_10_run}"
CULVER_RECEIVER="${CULVER_RECEIVER:-/path/to/culver/receiver_request_energy_topk_10_run}"
CULVER_TEMPORAL="${CULVER_TEMPORAL:-/path/to/culver/temporal_receiver_request_energy_topk_10_run}"
CULVER_LEARNED="${CULVER_LEARNED:-/path/to/culver/learned_temporal_receiver_request_10_run}"

run_dataset() {
  local dataset_name="$1"
  shift
  local run_dirs=("$@")

  local cmd=(
    "$PYTHON_BIN" tools/qualitative_scene_analysis.py
    --dataset_name "$dataset_name"
    --run_dirs "${run_dirs[@]}"
    --method_names Full Top-K Receiver Temporal Learned
    --output_dir "$OUTPUT_DIR"
    --candidate_mode auto
    --max_scenes "$MAX_SCENES"
    --iou_threshold "$IOU_THRESHOLD"
    --render_style "$RENDER_STYLE"
  )

  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry_run)
  fi

  echo
  echo "===== ${dataset_name^^} qualitative scene generation ====="
  echo "Mode: $([[ "$DRY_RUN" -eq 1 ]] && echo dry-run || echo generate)"
  echo "Output: $OUTPUT_DIR"
  echo "Render style: $RENDER_STYLE"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

mkdir -p "$OUTPUT_DIR"

# CARLA block
run_dataset carla \
  "$CARLA_FULL" \
  "$CARLA_TOPK" \
  "$CARLA_RECEIVER" \
  "$CARLA_TEMPORAL" \
  "$CARLA_LEARNED"

# Culver block
run_dataset culver \
  "$CULVER_FULL" \
  "$CULVER_TOPK" \
  "$CULVER_RECEIVER" \
  "$CULVER_TEMPORAL" \
  "$CULVER_LEARNED"

echo
echo "Done. Outputs are under: $OUTPUT_DIR"

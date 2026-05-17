#!/bin/bash
#SBATCH --job-name=ihd_hyperspectral
#SBATCH --output=logs/out/%j_ihd_hyperspectral.out
#SBATCH --error=logs/err/%j_ihd_hyperspectral.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=25G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

DEFAULT_REPO_ROOT='/home/malurool/Polarizaci-n/Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging'
DEFAULT_SCENE_HDR='/disk/IHTest_202104_DistStA/Path5_DistStA/Path5_Step1_DistStA/IHTest_202104_Path5_Step1_LWHSI1_collect0_DistStA.hdr'

# ── Selected pixels for per-pixel analysis (1-indexed row,col; semicolon-separated)
# Change here to update all figure scripts at once.
DEFAULT_SELECTED_PIXELS="143,62;160,62;130,62;80,200;40,62;5,62"

HDR_PATH="${1:-$DEFAULT_SCENE_HDR}"
SELECTED_PIXELS="${2:-$DEFAULT_SELECTED_PIXELS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "$DEFAULT_REPO_ROOT" ]]; then
    REPO_DIR="$DEFAULT_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_DIR="$SLURM_SUBMIT_DIR"
else
    REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err" "$REPO_DIR/results" \
         "$REPO_DIR/Figures/Ablation/pixel_analysis" \
         "$REPO_DIR/Figures/Hyperspectral_correction/Temperature" \
         "$REPO_DIR/Figures/Hyperspectral_correction/Emissivity_clusters" \
         "$REPO_DIR/Figures/Hyperspectral_correction/Range"

export HDR_PATH
export DATA_DIR="$REPO_DIR/data"
export RESULTS_DIR="$REPO_DIR/results"
export SCENE_NAME="$(basename "${HDR_PATH%.*}")"
export SELECTED_PIXELS
export T_AIR="${T_AIR:-289.7}"

# ── Step 1: Hyperspectral estimation without downwelling correction (fast) ──
echo "=== [1/4] Hyperspectral estimation — WITHOUT downwelling ==="
DOWNWELLING_FLAG=False python "$REPO_DIR/hyperspectral_estimation.py"

# ── Step 2: Hyperspectral estimation with downwelling correction ─────────────
echo "=== [2/4] Hyperspectral estimation — WITH downwelling ==="
DOWNWELLING_FLAG=True python "$REPO_DIR/hyperspectral_estimation.py"

# ── Step 3: Hyperspectral figures (T, emissivity, range, pixel overlay) ──────
echo "=== [3/4] Generating hyperspectral figures ==="
python "$REPO_DIR/python/Figure_hyperspectral_results.py"

# ── Step 4: Ablation study (bispectral vs quadspectral vs hyperspectral) ──────
echo "=== [4/4] Generating ablation figures ==="
python "$REPO_DIR/python/Figure_ablation.py"

echo ""
echo "=== Pipeline complete ==="
echo "  Results : $REPO_DIR/results/"
echo "  Figures : $REPO_DIR/Figures/"

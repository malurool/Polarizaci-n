#!/bin/bash
#SBATCH --job-name=ihd_hyperspectral
#SBATCH --output=logs/out/%j_ihd_hyperspectral.out
#SBATCH --error=logs/err/%j_ihd_hyperspectral.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

# Optional: hardcode the repo root so you can run `sbatch` from any directory.
# If empty, the script uses SLURM_SUBMIT_DIR (preferred) or falls back to a relative path.
DEFAULT_REPO_ROOT='/home/malurool/Polarizaci-n/Ozone-Cues-Mitigate-Reflected-Downwelling-Radiance-in-LWIR-Absorption-Based-Ranging'

DEFAULT_SCENE_HDR='/disk/IHTest_202009_DistStA/Path3_DistStA/Path3_Step1_DistStA/IHTest_202009_Path3_Step1_LWHSI1__DistStA.hdr'

HDR_PATH="${1:-$DEFAULT_SCENE_HDR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "$DEFAULT_REPO_ROOT" ]]; then
	REPO_DIR="$DEFAULT_REPO_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	REPO_DIR="$SLURM_SUBMIT_DIR"
else
	REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

mkdir -p "$REPO_DIR/logs/out" "$REPO_DIR/logs/err"

export HDR_PATH
export HSI_DIR="$(dirname "$HDR_PATH")"
export HSI_FILENAME="$(basename "${HDR_PATH%.*}.mat")"
export DATA_DIR="$REPO_DIR/data"

python "$REPO_DIR/hyperspectral_estimation.py"

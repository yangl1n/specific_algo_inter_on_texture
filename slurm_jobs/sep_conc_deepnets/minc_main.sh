#!/bin/bash -l
# ScatNet block-analysis on MINC-2500 over train sizes and seeds.
#SBATCH --job-name=scd_minc_main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --partition PARTITION_NAME
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --array=0-9 # 2 train sizes * 5 seeds

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1

REPO_DIR="$HOME/sep_conc_deepnets"
DATA_DIR="$HOME/dense/data"
cd "${REPO_DIR}"

WORKERS=10
BATCH_SIZE=32
EPOCHS=200
LR=0.01
LR_FREQ=60

TRAIN_SIZES=(0.005 0.0025)
SEEDS=(42 43 44 45 46)
RUN_PREFIX="clean_zarka/minc2500/scatnetblockanalysis"

BASE_COMMAND="python main.py \
  --dataset minc2500 \
  --data ${DATA_DIR} \
  --arch scatnetblockanalysis \
  --workers ${WORKERS} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --learning-rate-adjust-frequency ${LR_FREQ} \
  --n-blocks 4 \
  --P-proj-size 64 128 256 512 \
  --non-linearity relu \
  --frame-width 512 1024 1024 2048 \
  --data-split-mode deep_scatter \
  --val-frac 0.2"

run_one() {
  local train_size="$1"
  local size_tag
  size_tag="${train_size//./p}"
  local seed="$2"
  local seed_tag="seed${seed}"
  local run_dir="${RUN_PREFIX}_s${size_tag}_${seed_tag}"
  local cmd="${BASE_COMMAND} --train-size ${train_size} --seed ${seed} --dir ${run_dir}"
  echo "[$(date)] Running train_size=${train_size} seed=${seed} -> dir=${run_dir}"
  echo "Command: ${cmd}"
  bash -lc "${cmd}"
}

NUM_JOBS=$(( ${#TRAIN_SIZES[@]} * ${#SEEDS[@]}))

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${NUM_JOBS} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for TRAIN_SIZES" >&2
    exit 2
  fi
  TRAIN_SIZE_IDX=$((SLURM_ARRAY_TASK_ID / (${#SEEDS[@]})))
  SEED_IDX=$((SLURM_ARRAY_TASK_ID % (${#SEEDS[@]})))
  run_one "${TRAIN_SIZES[TRAIN_SIZE_IDX]}" "${SEEDS[SEED_IDX]}"
else
  for train_size in "${TRAIN_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "${train_size}" "${seed}"
    done
  done
fi

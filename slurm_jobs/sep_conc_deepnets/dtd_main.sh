#!/bin/bash -l
# ScatNet block-analysis on DTD over partitions, train sizes and seeds.
#SBATCH --job-name=scd_dtd_main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:30:00
#SBATCH --partition PARTITION_NAME
#SBATCH --requeue
#SBATCH --output=slurm-%x-%j.out
#SBATCH --array=0-74

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME

REPO_DIR="$HOME/sep_conc_deepnets"
cd "${REPO_DIR}"

WORKERS=16
BATCH_SIZE=128
EPOCHS=200
LR=0.01
LR_FREQ=60

TRAIN_SIZES=(1.0 0.5 0.3 0.2 0.1)
SEEDS=(42 43 44)
PARTITIONS=(1 2 3 4 5)
RUN_PREFIX="clean_zarka/dtd/scatnetblockanalysis_6blocks"

BASE_COMMAND="python main.py \
  --dataset dtd \
  --arch scatnetblockanalysis \
  --workers ${WORKERS} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --learning-rate-adjust-frequency ${LR_FREQ} \
  --n-blocks 6 \
  --P-proj-size 64 128 256 512 512 512 \
  --non-linearity relu \
  --frame-width 512 1024 1024 2048 2048 2048 \
  --data-split-mode deep_scatter \
  --val-frac 0.2"

run_one() {
  local train_size="$1"
  local size_tag
  size_tag="${train_size//./p}"
  local seed="$2"
  local seed_tag="seed${seed}"
  local partition="$3"
  local partition_tag="part${partition}"
  local run_dir="${RUN_PREFIX}_s${size_tag}_${seed_tag}_${partition_tag}"
  local cmd="${BASE_COMMAND} --train-size ${train_size} --seed ${seed} --dir ${run_dir} --dtd-partition ${partition}"
  echo "[$(date)] Running train_size=${train_size} -> dir=${run_dir}"
  echo "Command: ${cmd}"
  bash -lc "${cmd}"
}

NUM_JOBS=$(( ${#TRAIN_SIZES[@]} * ${#SEEDS[@]} * ${#PARTITIONS[@]} ))

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${NUM_JOBS} )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range for TRAIN_SIZES" >&2
    exit 2
  fi
  TRAIN_SIZE_IDX=$((SLURM_ARRAY_TASK_ID / (${#SEEDS[@]} * ${#PARTITIONS[@]})))
  SEED_IDX=$((SLURM_ARRAY_TASK_ID % (${#SEEDS[@]} * ${#PARTITIONS[@]}) / ${#PARTITIONS[@]}))
  PARTITION_IDX=$((SLURM_ARRAY_TASK_ID % ${#PARTITIONS[@]}))
  run_one "${TRAIN_SIZES[TRAIN_SIZE_IDX]}" "${SEEDS[SEED_IDX]}" "${PARTITIONS[PARTITION_IDX]}"
else
  for train_size in "${TRAIN_SIZES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for partition in "${PARTITIONS[@]}"; do
        run_one "${train_size}" "${seed}" "${partition}"
      done
    done
  done
fi

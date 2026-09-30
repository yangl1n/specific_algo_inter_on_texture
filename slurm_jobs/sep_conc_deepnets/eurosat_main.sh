#!/bin/bash -l
# ScatNet block-analysis on EuroSAT over train sizes and seeds.
#SBATCH --job-name=scd_eurosat_main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition PARTITION_NAME
#SBATCH --requeue
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-8   # 9 train sizes; the 3 seeds are looped inside

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1

REPO_DIR="$HOME/sep_conc_deepnets"
cd "${REPO_DIR}" || exit 1

DATASET=eurosat
TRAIN_SIZES=(1.0 0.5 0.3 0.2 0.15 0.1 0.05 0.01 0.005)
SEEDS=(42 43 44)

ARCH="scatnetblockanalysis"
WORKERS=16
BATCH_SIZE=128
EPOCHS=300
LR=0.01
LR_FREQ=70

TOTAL_TASKS=${#TRAIN_SIZES[@]}
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL_TASKS - 1))]" >&2
    exit 2
fi
TRAIN_SIZE=${TRAIN_SIZES[${SLURM_ARRAY_TASK_ID}]}
SIZE_TAG=${TRAIN_SIZE//./p}

DATA_ROOT="$HOME/data"
N_BLOCKS=5
P_PROJ_SIZE="64 128 256 512 512"
FRAME_WIDTH="1024 2048 4096 8192 8192"
NL="softshrink"

RUN_PREFIX="clean_zarka/${DATASET}_v2/${ARCH}"

BASE_COMMAND="python main.py \
  --data ${DATA_ROOT} \
  --dataset ${DATASET} \
  --arch ${ARCH} \
  --workers ${WORKERS} \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH_SIZE} \
  --lr ${LR} \
  --learning-rate-adjust-frequency ${LR_FREQ} \
  --n-blocks ${N_BLOCKS} \
  --P-proj-size ${P_PROJ_SIZE} \
  --non-linearity ${NL} \
  --frame-width ${FRAME_WIDTH} \
  --data-split-mode deep_scatter \
  --val-frac 0.2"

RESERVE_SEC=$((15 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((12 * 3600))
fi

echo "[$(date)] dataset=${DATASET} train_size=${TRAIN_SIZE} n_blocks=${N_BLOCKS} nl=${NL}"

OK=0; FAILED=0; SKIPPED=0
for SEED in "${SEEDS[@]}"; do
    RUN_DIR="${RUN_PREFIX}_s${SIZE_TAG}_seed${SEED}"
    LOGDIR="./training_logs/${RUN_DIR}"
    CKPT="./checkpoints/${RUN_DIR}/${ARCH}_batchsize_${BATCH_SIZE}_lrfreq_${LR_FREQ}.pth.tar"

    if compgen -G "${LOGDIR}/results_*.json" > /dev/null; then
        echo "[$(date)] SKIP seed=${SEED} -- already complete (${RUN_DIR})"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    REMAINING=$((TIME_LIMIT_SEC - SECONDS - RESERVE_SEC))
    if (( REMAINING <= 0 )); then
        echo "[$(date)] SKIP seed=${SEED} -- out of wall clock (resubmit the array)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    RESUME_ARG=""
    if [[ -f "${CKPT}" ]]; then
        RESUME_ARG="--resume ${CKPT}"
        echo "[$(date)] resuming seed=${SEED} from ${CKPT}"
    fi

    CKPT_FREQ=5
    case "${TRAIN_SIZE}" in
      1.0|0.5|0.3) CKPT_FREQ=1 ;;
    esac

    echo "[$(date)] RUN seed=${SEED} -> ${RUN_DIR}"
    srun bash -lc "${BASE_COMMAND} --train-size ${TRAIN_SIZE} --seed ${SEED} \
        --checkpoint-save-frequency ${CKPT_FREQ} ${RESUME_ARG} --dir ${RUN_DIR}"
    RC=$?
    if (( RC == 0 )); then OK=$((OK + 1)); else FAILED=$((FAILED + 1)); echo "  seed ${SEED} FAILED rc=${RC}"; fi
done

echo "[$(date)] ${DATASET} s${TRAIN_SIZE}: ok=${OK} failed=${FAILED} skipped=${SKIPPED}"
if (( OK == 0 && FAILED > 0 )); then echo "nothing succeeded" >&2; exit 1; fi

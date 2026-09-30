#!/bin/bash -l
# Deep scattering on MINC-2500: L2 penalty toward zero at the smallest train sizes.
#SBATCH --nodes=1
#SBATCH --job-name=ds_minc_l2_zero_lowdata
#SBATCH --time=18:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-3 # 2 train sizes * 2 l2 arms, 5 seeds batched per task = 20 runs

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
cd $HOME/deep_scatter

DATA_DIR="$HOME/dense/data"

N_BLOCKS=6
L=4
KERNEL_SIZE=5
MODULUS_TYPE="complex_modulus"
MIXING_HORIZON=3

LR_HEAD=1.0e-3
LR_EXTRACTOR=1.0e-4
WEIGHT_DECAY=5.0e-3
PHASE_A_EPOCHS=50     # train.py --lr-epochs
PHASE_B_EPOCHS=300    # train.py --epochs
BATCH_SIZE=128

TRAIN_SIZES=(0.0025 0.005)
SEEDS=(42 43 44 45 46)
L2_PENALTIES=(0.1 1.0)

TASKS=()
for TS in "${TRAIN_SIZES[@]}"; do
    for L2 in "${L2_PENALTIES[@]}"; do
        TASKS+=("${TS}|${L2}|$(IFS=,; echo "${SEEDS[*]}")")
    done
done
TOTAL_TASKS=${#TASKS[@]}

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL_TASKS - 1))]" >&2
    echo "  (TOTAL_TASKS=${TOTAL_TASKS}; if you changed TRAIN_SIZES / L2_PENALTIES, update --array)" >&2
    exit 2
fi

IFS='|' read -r TRAIN_SIZE L2_PENALTY TASK_SEED_CSV <<< "${TASKS[${SLURM_ARRAY_TASK_ID}]}"
IFS=',' read -r -a TASK_SEEDS <<< "${TASK_SEED_CSV}"

case "${MODULUS_TYPE}" in
    complex_modulus) MOD_TAG="cmod" ;;
    phase_relu)      MOD_TAG="prelu" ;;
    *)               MOD_TAG="${MODULUS_TYPE}" ;;
esac
TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}
L2_PENALTY_TAG=${L2_PENALTY//./p}
STEM="${N_BLOCKS}b_${MOD_TAG}_${L}L_k${KERNEL_SIZE}_mh${MIXING_HORIZON}_bn_s${TRAIN_SIZE_TAG}_l2${L2_PENALTY_TAG}_randomInit_l2zero"

RESERVE_SEC=$((20 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((18 * 3600))
fi

echo "=========================================================="
echo "[$(date)] MinC-2500 l2-disentangle LOW-DATA -- task ${SLURM_ARRAY_TASK_ID} of ${TOTAL_TASKS}"
echo "  train_size : ${TRAIN_SIZE}"
echo "  l2_penalty : ${L2_PENALTY}  (random-init => distance-from-0 / classic L2)"
echo "  seeds      : ${TASK_SEEDS[*]}"
echo "  guard      : ${TIME_LIMIT_SEC}s budget, reserving $((RESERVE_SEC/60)) min"
echo "=========================================================="

OK=0; FAILED=0; SKIPPED=0
FAILED_SEEDS=""; SKIPPED_SEEDS=""

for SEED in "${TASK_SEEDS[@]}"; do
    SAVE_DIR="clean_updateNC/l2_disentangle/minc2500/${STEM}_seed${SEED}"

    if compgen -G "./training_logs/${SAVE_DIR}/results_*.json" > /dev/null; then
        echo "[$(date)] SKIP seed=${SEED} -- already complete (${SAVE_DIR})"
        SKIPPED=$((SKIPPED + 1)); continue
    fi

    REMAINING=$((TIME_LIMIT_SEC - SECONDS - RESERVE_SEC))
    if (( REMAINING <= 0 )); then
        echo "[$(date)] SKIP seed=${SEED} -- out of wall clock (resubmit the array)"
        SKIPPED=$((SKIPPED + 1)); SKIPPED_SEEDS="${SKIPPED_SEEDS} ${SEED}"; continue
    fi

    echo ""
    echo "[$(date)] RUN seed=${SEED} (${REMAINING}s usable) -> ${SAVE_DIR}"
    python train.py \
        --dataset minc2500 \
        --data-dir ${DATA_DIR} \
        --lr-epochs ${PHASE_A_EPOCHS} \
        --epochs ${PHASE_B_EPOCHS} \
        --batch-size-per-device ${BATCH_SIZE} \
        --print-freq 200 \
        --mixing-horizon ${MIXING_HORIZON} \
        --lowpass-last \
        --global-avg-pool \
        --use-batchnorm \
        --workers 10 \
        --kernel-size ${KERNEL_SIZE} \
        --n-blocks ${N_BLOCKS} \
        --L ${L} \
        --modulus-type ${MODULUS_TYPE} \
        --train-size ${TRAIN_SIZE} \
        --l2-penalty ${L2_PENALTY} \
        --random-init \
        --seed ${SEED} \
        --lr-head ${LR_HEAD} \
        --lr-extractor ${LR_EXTRACTOR} \
        --weight-decay ${WEIGHT_DECAY} \
        --save-dir ${SAVE_DIR}

    RC=$?
    if (( RC == 0 )); then
        OK=$((OK + 1)); echo "[$(date)] seed ${SEED} OK"
    else
        FAILED=$((FAILED + 1)); FAILED_SEEDS="${FAILED_SEEDS} ${SEED}"
        echo "[$(date)] seed ${SEED} FAILED rc=${RC} (continuing with remaining seeds)" >&2
    fi
done

echo ""
echo "=========================================================="
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done: s=${TRAIN_SIZE} l2=${L2_PENALTY}"
echo "  ok=${OK}  failed=${FAILED}  skipped=${SKIPPED}"
echo "  elapsed $((SECONDS / 60)) min of $((TIME_LIMIT_SEC / 60)) min"
[[ -n "${FAILED_SEEDS}"  ]] && echo "  FAILED seeds:${FAILED_SEEDS}"
[[ -n "${SKIPPED_SEEDS}" ]] && echo "  SKIPPED for time, resubmit:${SKIPPED_SEEDS}"
echo "=========================================================="

if (( OK == 0 && FAILED > 0 )); then exit 1; fi
exit 0

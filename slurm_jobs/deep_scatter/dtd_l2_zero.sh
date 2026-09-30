#!/bin/bash -l
# Deep scattering on DTD: L2 penalty toward zero, over partitions, train sizes and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=ds_dtd_l2_zero
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-49 # 5 partitions * 5 train sizes * 2 l2 arms; 3 seeds looped inside

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
cd $HOME/deep_scatter

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

TRAIN_SIZES=(0.1 0.2 0.3 0.5 1.0)
SEEDS=(42 43 44)
PARTITIONS=(1 2 3 4 5)

RUN_L2_ZERO=0
if (( RUN_L2_ZERO == 1 )); then
    L2_PENALTIES=(0.0 0.1 1.0)
else
    L2_PENALTIES=(0.1 1.0)
fi

NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}
NUM_L2=${#L2_PENALTIES[@]}
NUM_PARTITIONS=${#PARTITIONS[@]}
TOTAL_TASKS=$((NUM_PARTITIONS * NUM_TRAIN_SIZES * NUM_L2))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL_TASKS - 1))]" >&2
    echo "  (TOTAL_TASKS=${TOTAL_TASKS}; if you flipped RUN_L2_ZERO, update --array)" >&2
    exit 2
fi

PART_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_TRAIN_SIZES * NUM_L2)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_TRAIN_SIZES * NUM_L2)))
TRAIN_SIZE_IDX=$((REMAINDER / NUM_L2))
L2_IDX=$((REMAINDER % NUM_L2))

DTD_PARTITION=${PARTITIONS[${PART_IDX}]}
TRAIN_SIZE=${TRAIN_SIZES[${TRAIN_SIZE_IDX}]}
L2_PENALTY=${L2_PENALTIES[${L2_IDX}]}

case "${MODULUS_TYPE}" in
    complex_modulus) MOD_TAG="cmod" ;;
    phase_relu)      MOD_TAG="prelu" ;;
    *)               MOD_TAG="${MODULUS_TYPE}" ;;
esac
TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}
L2_PENALTY_TAG=${L2_PENALTY//./p}
STEM="${N_BLOCKS}b_${MOD_TAG}_${L}L_k${KERNEL_SIZE}_mh${MIXING_HORIZON}_s${TRAIN_SIZE_TAG}_l2${L2_PENALTY_TAG}_randomInit_l2zero"

RESERVE_SEC=$((20 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((12 * 3600))
fi

echo "=========================================================="
echo "[$(date)] DTD v2 l2-disentangle -- task ${SLURM_ARRAY_TASK_ID} of ${TOTAL_TASKS}"
echo "  partition  : ${DTD_PARTITION}"
echo "  train_size : ${TRAIN_SIZE}"
echo "  l2_penalty : ${L2_PENALTY}  (random-init => distance-from-0 / classic L2)"
echo "  seeds      : ${SEEDS[*]}"
echo "  guard      : ${TIME_LIMIT_SEC}s budget, reserving $((RESERVE_SEC/60)) min"
echo "=========================================================="

OK=0; FAILED=0; SKIPPED=0
FAILED_SEEDS=""; SKIPPED_SEEDS=""

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="clean_updateNC/l2_disentangle/dtd_v2/${STEM}_seed${SEED}/part${DTD_PARTITION}"

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
        --dataset dtd \
        --dtd-partition ${DTD_PARTITION} \
        --lr-epochs ${PHASE_A_EPOCHS} \
        --epochs ${PHASE_B_EPOCHS} \
        --batch-size-per-device ${BATCH_SIZE} \
        --print-freq 200 \
        --mixing-horizon ${MIXING_HORIZON} \
        --lowpass-last \
        --global-avg-pool \
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
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done: part=${DTD_PARTITION} s=${TRAIN_SIZE} l2=${L2_PENALTY}"
echo "  ok=${OK}  failed=${FAILED}  skipped=${SKIPPED}"
echo "  elapsed $((SECONDS / 60)) min of $((TIME_LIMIT_SEC / 60)) min"
[[ -n "${FAILED_SEEDS}"  ]] && echo "  FAILED seeds:${FAILED_SEEDS}"
[[ -n "${SKIPPED_SEEDS}" ]] && echo "  SKIPPED for time, resubmit:${SKIPPED_SEEDS}"
echo "=========================================================="

if (( OK == 0 && FAILED > 0 )); then exit 1; fi
exit 0

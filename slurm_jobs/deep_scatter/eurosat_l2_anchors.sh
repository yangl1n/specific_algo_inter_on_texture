#!/bin/bash -l
# Deep scattering on EuroSAT: L2 penalty toward an anchor (scattering or init filters).
#SBATCH --nodes=1
#SBATCH --job-name=ds_eurosat_l2_anchors
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-35 # 2 anchors x 9 sizes x 2 lambdas = 108 runs; 3 seeds looped inside each task

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
cd $HOME/deep_scatter || exit 2

N_BLOCKS=4
KERNEL_SIZE=5
MIXING_HORIZON=27
L=4
MODULUS_TYPE="complex_modulus"

LR_HEAD=1.0e-3
LR_EXTRACTOR=1.0e-4
WEIGHT_DECAY=1.0e-4
BATCH_SIZE=128

TRAIN_SIZES=(0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.5 1.0)
SEEDS=(42 43 44)
L2_PENALTIES=(0.1 1.0)
ANCHORS=(scattering init)
SPLIT_SIZES=()

TASKS=()
for ANC in "${ANCHORS[@]}"; do
    for TS in "${TRAIN_SIZES[@]}"; do
        IS_SPLIT=0
        for SS in "${SPLIT_SIZES[@]}"; do [[ "${TS}" == "${SS}" ]] && IS_SPLIT=1; done
        for L2 in "${L2_PENALTIES[@]}"; do
            if (( IS_SPLIT == 1 )); then
                for SD in "${SEEDS[@]}"; do TASKS+=("${ANC}|${TS}|${L2}|${SD}"); done
            else
                TASKS+=("${ANC}|${TS}|${L2}|$(IFS=,; echo "${SEEDS[*]}")")
            fi
        done
    done
done
TOTAL_TASKS=${#TASKS[@]}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID}" -lt 0 || "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]]; then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>} out of range [0, $((TOTAL_TASKS - 1))]" >&2
    echo "  (TOTAL_TASKS=${TOTAL_TASKS}; if you edited the grid, update --array)" >&2
    exit 2
fi

IFS='|' read -r L2_ANCHOR TRAIN_SIZE L2_PENALTY TASK_SEED_CSV <<< "${TASKS[${SLURM_ARRAY_TASK_ID}]}"
IFS=',' read -r -a TASK_SEEDS <<< "${TASK_SEED_CSV}"

OUT_DIR="clean_updateNC/l2_disentangle/eurosat_anchors"
case "${L2_ANCHOR}" in
    scattering) ANCHOR_TAG="l2scat"; ANCHOR_DESC="L2 toward the wavelet init of this architecture" ;;
    init)       ANCHOR_TAG="l2init"; ANCHOR_DESC="L2-SP about this run's own random init" ;;
    zero)       ANCHOR_TAG="l2zero"; ANCHOR_DESC="classic L2 toward 0" ;;
    *)          echo "unhandled anchor '${L2_ANCHOR}' -- add a tag for it" >&2; exit 2 ;;
esac

TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}
L2_PENALTY_TAG=${L2_PENALTY//./p}
STEM="${N_BLOCKS}b_cmod_4L_s${TRAIN_SIZE_TAG}_l2${L2_PENALTY_TAG}_randomInit_${ANCHOR_TAG}"

RESERVE_SEC=$((20 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((3 * 3600))
fi

echo "=========================================================="
echo "[$(date)] EuroSAT l2-ANCHOR sweep -- task ${SLURM_ARRAY_TASK_ID} of ${TOTAL_TASKS}"
echo "  train_size : ${TRAIN_SIZE}"
echo "  l2_penalty : ${L2_PENALTY}"
echo "  l2_anchor  : ${L2_ANCHOR}  (${ANCHOR_DESC})"
echo "  seeds      : ${TASK_SEEDS[*]}"
echo "  out        : training_logs/${OUT_DIR}/${STEM}_seed*"
echo "  guard      : ${TIME_LIMIT_SEC}s budget, reserving $((RESERVE_SEC/60)) min"
echo "=========================================================="

OK=0; FAILED=0; SKIPPED=0; DONE=0
FAILED_SEEDS=""; SKIPPED_SEEDS=""

for SEED in "${TASK_SEEDS[@]}"; do
    SAVE_DIR="${OUT_DIR}/${STEM}_seed${SEED}"

    if compgen -G "./training_logs/${SAVE_DIR}/results_*.json" > /dev/null; then
        echo "[$(date)] SKIP seed=${SEED} -- already complete (${SAVE_DIR})"
        DONE=$((DONE + 1)); continue
    fi

    REMAINING=$((TIME_LIMIT_SEC - SECONDS - RESERVE_SEC))
    if (( REMAINING <= 0 )); then
        echo "[$(date)] SKIP seed=${SEED} -- out of wall clock (resubmit the array)"
        SKIPPED=$((SKIPPED + 1)); SKIPPED_SEEDS="${SKIPPED_SEEDS} ${SEED}"; continue
    fi

    echo ""
    echo "[$(date)] RUN seed=${SEED} (${REMAINING}s usable) -> ${SAVE_DIR}"
    python train.py \
        --dataset eurosat \
        --data-dir $HOME/data \
        --lr-epochs 20 \
        --epochs 300 \
        --batch-size-per-device ${BATCH_SIZE} \
        --print-freq 200 \
        --workers 10 \
        --n-blocks ${N_BLOCKS} \
        --kernel-size ${KERNEL_SIZE} \
        --L ${L} \
        --modulus-type ${MODULUS_TYPE} \
        --mixing-horizon ${MIXING_HORIZON} --lowpass-last \
        --train-size ${TRAIN_SIZE} \
        --l2-penalty ${L2_PENALTY} \
        --l2-anchor ${L2_ANCHOR} \
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
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done: s=${TRAIN_SIZE} l2=${L2_PENALTY} anchor=${L2_ANCHOR}"
echo "  ok=${OK}  failed=${FAILED}  already-done=${DONE}  skipped-for-time=${SKIPPED}"
echo "  elapsed $((SECONDS / 60)) min of $((TIME_LIMIT_SEC / 60)) min"
[[ -n "${FAILED_SEEDS}"  ]] && echo "  FAILED seeds:${FAILED_SEEDS}"
[[ -n "${SKIPPED_SEEDS}" ]] && echo "  SKIPPED for time, resubmit:${SKIPPED_SEEDS}"
echo "=========================================================="

if (( OK == 0 && DONE == 0 )); then exit 1; fi
exit 0

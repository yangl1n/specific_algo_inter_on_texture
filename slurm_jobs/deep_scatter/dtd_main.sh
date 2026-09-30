#!/bin/bash -l
# Deep scattering on DTD: two-phase and random-init training over partitions, train sizes, L2 arms and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=ds_dtd_main
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-79 # 4 partitions * 5 train sizes * 4 reg arms; 3 seeds looped inside

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1
cd $HOME/deep_scatter

DATA_DIR="$HOME/data/dtd"

N_BLOCKS=6
L=4
KERNEL_SIZE=5
MODULUS_TYPE="complex_modulus"
MIXING_HORIZON=3

LR_HEAD=1.0e-3
LR_EXTRACTOR=1.0e-4
WEIGHT_DECAY=5.0e-3
PHASE_A_EPOCHS=50    # train.py --lr-epochs
PHASE_B_EPOCHS=300   # train.py --epochs
BATCH_SIZE=128

TRAIN_SIZES=(0.1 0.2 0.3 0.5 1.0)
SEEDS=(42 43 44)

REG_NAMES=(l2_0p0 l2_0p1 l2_1p0 randinit)
REG_L2=(0.0 0.1 1.0 0.0)
REG_RANDINIT=(0 0 0 1)   # 1 => pass --random-init (single-phase joint training)

PARTITIONS=(2 3 4 5)

NUM_PARTITIONS=${#PARTITIONS[@]}
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}
NUM_REG=${#REG_NAMES[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_COMBINATIONS=$((NUM_PARTITIONS * NUM_TRAIN_SIZES * NUM_REG))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL_COMBINATIONS - 1))]"
    exit 1
fi

PART_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_TRAIN_SIZES * NUM_REG)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_TRAIN_SIZES * NUM_REG)))
TRAIN_SIZE_IDX=$((REMAINDER / NUM_REG))
REG_IDX=$((REMAINDER % NUM_REG))

DTD_PARTITION=${PARTITIONS[${PART_IDX}]}
TRAIN_SIZE=${TRAIN_SIZES[${TRAIN_SIZE_IDX}]}
L2_PENALTY=${REG_L2[${REG_IDX}]}
REG_TAG=${REG_NAMES[${REG_IDX}]}

RANDINIT_FLAG=""
if (( ${REG_RANDINIT[${REG_IDX}]} == 1 )); then
    RANDINIT_FLAG="--random-init"
fi

case "${MODULUS_TYPE}" in
    complex_modulus) MOD_TAG="cmod" ;;
    phase_relu)      MOD_TAG="prelu" ;;
    *)               MOD_TAG="${MODULUS_TYPE}" ;;
esac

TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}
STEM="${N_BLOCKS}b_${MOD_TAG}_${L}L_k${KERNEL_SIZE}_mh${MIXING_HORIZON}_s${TRAIN_SIZE_TAG}_${REG_TAG}"

RESERVE_SEC=$((20 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((12 * 3600))
fi

echo "[$(date)] partition=${DTD_PARTITION} train_size=${TRAIN_SIZE} reg=${REG_TAG} (l2=${L2_PENALTY} randinit=${RANDINIT_FLAG:-no})"

OK=0; FAILED=0; SKIPPED=0
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="clean_v2/dtd/${STEM}_seed${SEED}_part${DTD_PARTITION}"
    LEGACY_DIR="clean_v2/dtd/${STEM}_seed${SEED}"

    if compgen -G "./training_logs/${SAVE_DIR}/results_*.json" > /dev/null; then
        echo "[$(date)] SKIP seed=${SEED} -- already complete (${SAVE_DIR})"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    if (( DTD_PARTITION == 1 )) && compgen -G "./training_logs/${LEGACY_DIR}/results_*.json" > /dev/null; then
        echo "[$(date)] SKIP seed=${SEED} -- partition 1 already exists under the LEGACY name"
        echo "           ${LEGACY_DIR}"
        echo "           Run dtd_v2_rename_part1.sh so paper_plots.py can see it as part 1."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    REMAINING=$((TIME_LIMIT_SEC - SECONDS - RESERVE_SEC))
    if (( REMAINING <= 0 )); then
        echo "[$(date)] SKIP seed=${SEED} -- out of wall clock (resubmit the array)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    echo "[$(date)] RUN seed=${SEED} -> ${SAVE_DIR}"
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
        ${RANDINIT_FLAG} \
        --seed ${SEED} \
        --lr-head ${LR_HEAD} \
        --lr-extractor ${LR_EXTRACTOR} \
        --weight-decay ${WEIGHT_DECAY} \
        --save-dir ${SAVE_DIR}
    RC=$?
    if (( RC == 0 )); then OK=$((OK + 1)); else FAILED=$((FAILED + 1)); echo "  seed ${SEED} FAILED rc=${RC}"; fi
done

echo "[$(date)] part${DTD_PARTITION} s${TRAIN_SIZE} ${REG_TAG}: ok=${OK} failed=${FAILED} skipped=${SKIPPED}"
if (( OK == 0 && FAILED > 0 )); then echo "nothing succeeded" >&2; exit 1; fi

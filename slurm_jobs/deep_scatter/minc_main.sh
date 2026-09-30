#!/bin/bash -l
# Deep scattering on MINC-2500: two-phase and random-init training over all train sizes, L2 arms and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=ds_minc_main
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-123 # 7 sizes * 4 reg arms * 3 seeds (0-83), then 2 low-data sizes * 4 reg arms * 5 seeds (84-123)

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1
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
PHASE_A_EPOCHS=50    # train.py --lr-epochs
PHASE_B_EPOCHS=300   # train.py --epochs
BATCH_SIZE=128

REG_NAMES=(l2_0p0 l2_0p1 l2_1p0 randinit)
REG_L2=(0.0 0.1 1.0 0.0)
REG_RANDINIT=(0 0 0 1)   # 1 => pass --random-init (single-phase joint training)

# Train sizes below 0.01 use 5 seeds and are saved separately (paper_plots.py: low_data).
MAIN_SIZES=(0.01 0.05 0.1 0.2 0.3 0.5 1.0)
MAIN_SEEDS=(42 43 44)
LOWDATA_SIZES=(0.0025 0.005)
LOWDATA_SEEDS=(42 43 44 45 46)

TASKS=()   # train_size:reg_idx:seed:save_root
for ts in "${MAIN_SIZES[@]}"; do
    for r in "${!REG_NAMES[@]}"; do
        for s in "${MAIN_SEEDS[@]}"; do TASKS+=("${ts}:${r}:${s}:clean_v2/minc"); done
    done
done
for ts in "${LOWDATA_SIZES[@]}"; do
    for r in "${!REG_NAMES[@]}"; do
        for s in "${LOWDATA_SEEDS[@]}"; do TASKS+=("${ts}:${r}:${s}:clean_v2/minc_lowdata"); done
    done
done
TOTAL_COMBINATIONS=${#TASKS[@]}

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL_COMBINATIONS - 1))]"
    exit 1
fi

IFS=':' read -r TRAIN_SIZE REG_IDX SEED SAVE_ROOT <<< "${TASKS[${SLURM_ARRAY_TASK_ID}]}"
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

SAVE_DIR="${SAVE_ROOT}/${N_BLOCKS}b_${MOD_TAG}_${L}L_k${KERNEL_SIZE}_mh${MIXING_HORIZON}_bn_s${TRAIN_SIZE_TAG}_${REG_TAG}_seed${SEED}"

if compgen -G "./training_logs/${SAVE_DIR}/results_*.json" > /dev/null; then
    echo "[$(date)] SKIP -- already complete (${SAVE_DIR})"
    exit 0
fi

echo "[$(date)] train_size=${TRAIN_SIZE} reg=${REG_TAG} (l2=${L2_PENALTY} randinit=${RANDINIT_FLAG:-no}) seed=${SEED} -> ${SAVE_DIR}"

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
    ${RANDINIT_FLAG} \
    --seed ${SEED} \
    --lr-head ${LR_HEAD} \
    --lr-extractor ${LR_EXTRACTOR} \
    --weight-decay ${WEIGHT_DECAY} \
    --save-dir ${SAVE_DIR}

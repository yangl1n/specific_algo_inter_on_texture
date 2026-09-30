#!/bin/bash -l
# Deep scattering on EuroSAT: random-init baseline over train sizes and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=ds_eurosat_random_init
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-26

eval "$(conda shell.bash hook)"

L=4
N_BLOCKS=4
MODULUS_TYPE="complex_modulus"

TRAIN_SIZES=(1.0 0.5 0.3 0.2 0.15 0.1 0.05 0.01 0.005)
SEEDS=(42 43 44)
L2_PENALTIES=(0.0)

LR_HEAD=1.0e-3
LR_EXTRACTOR=1.0e-4

NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}
NUM_L2_PENALTIES=${#L2_PENALTIES[@]}
NUM_SEEDS=${#SEEDS[@]}

TOTAL_COMBINATIONS=$((NUM_TRAIN_SIZES * NUM_L2_PENALTIES * NUM_SEEDS))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_COMBINATIONS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} is out of range [0, $((TOTAL_COMBINATIONS - 1))]"
    exit 1
fi

TRAIN_SIZE_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_L2_PENALTIES * NUM_SEEDS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_L2_PENALTIES * NUM_SEEDS)))
L2_PENALTY_IDX=$((REMAINDER / NUM_SEEDS))
SEED_IDX=$((REMAINDER % NUM_SEEDS))

TRAIN_SIZE=${TRAIN_SIZES[${TRAIN_SIZE_IDX}]}
L2_PENALTY=${L2_PENALTIES[${L2_PENALTY_IDX}]}
SEED=${SEEDS[${SEED_IDX}]}

case "${MODULUS_TYPE}" in
    complex_modulus) MOD_TAG="cmod" ;;
    phase_relu) MOD_TAG="prelu" ;;
    *) MOD_TAG="${MODULUS_TYPE}" ;;
esac

TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}
L2_PENALTY_TAG=${L2_PENALTY//./p}

SEED_TAG=${SEED}
LR_HEAD_TAG=${LR_HEAD//./p}
LR_HEAD_TAG=${LR_HEAD_TAG//-/m}

SAVE_DIR="./clean_updateNC/eurosat/${N_BLOCKS}b_${MOD_TAG}_${L}L_s${TRAIN_SIZE_TAG}_l2${L2_PENALTY_TAG}_seed${SEED_TAG}_randomInit"

conda activate ENV_NAME
cd $HOME/deep_scatter

python train.py --dataset eurosat \
    --data-dir $HOME/data \
    --lr-epochs 20 --epochs 300 --batch-size-per-device 128 \
    --print-freq 200 --mixing-horizon 27 --lowpass-last \
    --workers 10 --kernel-size 5 --n-blocks ${N_BLOCKS} \
    --modulus-type ${MODULUS_TYPE} --L ${L} --train-size ${TRAIN_SIZE} \
    --l2-penalty ${L2_PENALTY} --seed ${SEED} --save-dir ${SAVE_DIR} \
    --lr-head $LR_HEAD --lr-extractor $LR_EXTRACTOR --retrain --random-init

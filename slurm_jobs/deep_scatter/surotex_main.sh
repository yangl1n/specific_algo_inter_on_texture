#!/bin/bash -l
# Deep scattering on SuroTex: two-phase and random-init training over train sizes, L2 arms and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=ds_surotex_main
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=10
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=8-23 # 2 lambdas (0.1, 1.0) * 8 train sizes; 3 seeds looped inside

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1
cd $HOME/deep_scatter || exit 2

DATA_DIR="$HOME/data"

INPUT_SIZE=224
PHASE_A_EPOCHS=50    # --lr-epochs. Ignored by the --random-init arm.
PHASE_B_EPOCHS=300   # --epochs. This is the joint budget for --random-init.
BATCH_SIZE=128
WEIGHT_DECAY=5.0e-3

LR_SETTLED=1
LR_SETTLED_NOTE="LR probe: head1p0em2_ext1p0em4 won on val at both train sizes (+0.75 @2%, +4.33 @5%), 6/6 seed-pairs over the old default head1p0em3_ext1p0em4"

if (( LR_SETTLED != 1 )); then
    echo "ERROR: this job is GATED on the DS learning-rate probe and has not been ungated." >&2
    echo "" >&2
    echo "  The SCD side is tuned over six learning rates on SuroTex; the DS side is still" >&2
    echo "  at the untuned default (--lr-head 1.0e-3 --lr-extractor 1.0e-4). Running the" >&2
    echo "  head-to-head now would compare a tuned baseline against an untuned method." >&2
    echo "" >&2
    echo "  The probe has been run and (1.0e-2, 1.0e-4) won; if you are" >&2
    echo "  seeing this, someone re-zeroed LR_SETTLED. Confirm" >&2
    echo "  that the FLAGS table still carries the winning pair, and set" >&2
    echo "  LR_SETTLED=1 again." >&2
    exit 2
fi
echo "[lr-gate] ungated: ${LR_SETTLED_NOTE:-<no note recorded -- fill in LR_SETTLED_NOTE>}"

RESERVE_SEC=$((70 * 60))
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    TIME_LIMIT_SEC=$(( SLURM_JOB_END_TIME - $(date +%s) ))
else
    TIME_LIMIT_SEC=$((8 * 3600))   # keep in sync with #SBATCH --time
fi
echo "[guard] budget $((TIME_LIMIT_SEC/60)) min, reserving $((RESERVE_SEC/60)) min"

NAMES=(
  bn_lr1em2        # 0  two-phase
  bn_lr1em2        # 1  same arm, lambda=0.1
  bn_lr1em2        # 2  same arm, lambda=1.0
  bn_rinit_lr1em3  # 3  --random-init, lambda=0
)

L2_PENALTIES=(0.0 0.1 1.0 0.0)

FLAGS=(
  "--n-blocks 4 --kernel-size 5 --mixing-horizon 3 --lowpass-last --global-avg-pool --L 4 --use-batchnorm --lr-head 1.0e-2 --lr-extractor 1.0e-4"
  "--n-blocks 4 --kernel-size 5 --mixing-horizon 3 --lowpass-last --global-avg-pool --L 4 --use-batchnorm --lr-head 1.0e-2 --lr-extractor 1.0e-4"
  "--n-blocks 4 --kernel-size 5 --mixing-horizon 3 --lowpass-last --global-avg-pool --L 4 --use-batchnorm --lr-head 1.0e-2 --lr-extractor 1.0e-4"
  "--n-blocks 4 --kernel-size 5 --mixing-horizon 3 --lowpass-last --global-avg-pool --L 4 --use-batchnorm --lr-head 1.0e-3 --lr-extractor 1.0e-4"
)

RANDINIT=(0 0 0 1)   # 1 => pass --random-init (single-phase joint training)

TRAIN_SIZES=(0.02 0.05 0.10 0.15 0.20 0.30 0.50 1.0)
TRAIN_N=(50 150 300 450 600 950 1600 3200)   # parallel; logged, and checked after the run
SEEDS=(42 43 44)

SAVE_PREFIX="surotex_h2h"
LOG_ROOT="training_logs"

NUM_CONFIGS=${#NAMES[@]}
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}
TOTAL_TASKS=$((NUM_CONFIGS * NUM_TRAIN_SIZES))

if (( ${#FLAGS[@]} != NUM_CONFIGS || ${#RANDINIT[@]} != NUM_CONFIGS \
      || ${#L2_PENALTIES[@]} != NUM_CONFIGS )); then
    echo "ERROR: NAMES (${NUM_CONFIGS}) / FLAGS (${#FLAGS[@]}) / RANDINIT (${#RANDINIT[@]}) /" >&2
    echo "       L2_PENALTIES (${#L2_PENALTIES[@]}) length mismatch" >&2
    exit 2
fi
if (( ${#TRAIN_N[@]} != NUM_TRAIN_SIZES )); then
    echo "ERROR: TRAIN_SIZES (${NUM_TRAIN_SIZES}) vs TRAIN_N (${#TRAIN_N[@]}) length mismatch" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0,$((TOTAL_TASKS-1))]" >&2; exit 2
fi

CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TRAIN_SIZES))
TRAIN_SIZE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TRAIN_SIZES))
NAME=${NAMES[${CONFIG_IDX}]}
EXTRA_FLAGS=${FLAGS[${CONFIG_IDX}]}
L2_PENALTY=${L2_PENALTIES[${CONFIG_IDX}]}
TRAIN_SIZE=${TRAIN_SIZES[${TRAIN_SIZE_IDX}]}
N_TRAIN=${TRAIN_N[${TRAIN_SIZE_IDX}]}
TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}

if [[ "${L2_PENALTY}" == "0.0" || "${L2_PENALTY}" == "0" ]]; then
    L2_TAG=""
else
    L2_TAG="_l2${L2_PENALTY//./p}"
fi

FLAG_LR_HEAD=$(sed -n 's/.*--lr-head[[:space:]]\+\([^[:space:]]\+\).*/\1/p' <<< "${EXTRA_FLAGS}")
if [[ -z "${FLAG_LR_HEAD}" ]]; then
    echo "ERROR: no --lr-head found in FLAGS[${CONFIG_IDX}]: ${EXTRA_FLAGS}" >&2; exit 2
fi
LR_TAG_FROM_FLAG=$(awk -v lr="${FLAG_LR_HEAD}" 'BEGIN{
    e = 0; m = lr + 0
    if (m <= 0) { print "lrINVALID"; exit }
    while (m < 1) { m *= 10; e++ }
    s = sprintf("%g", m); gsub(/\./, "p", s)
    printf "lr%sem%d", s, e }')
if [[ "${NAME}" != *"_${LR_TAG_FROM_FLAG}" ]]; then
    echo "ERROR: arm name / learning-rate mismatch in the ARM TABLE." >&2
    echo "  NAMES[${CONFIG_IDX}] = '${NAME}'" >&2
    echo "  FLAGS[${CONFIG_IDX}] has --lr-head ${FLAG_LR_HEAD}, which tags as '${LR_TAG_FROM_FLAG}'" >&2
    echo "  Rename the arm to end in '_${LR_TAG_FROM_FLAG}', or fix the flag. Refusing to run." >&2
    exit 2
fi
echo "[name-guard] ${NAME} <-> --lr-head ${FLAG_LR_HEAD}  OK"

if (( ${RANDINIT[${CONFIG_IDX}]} == 1 )) && [[ "${L2_PENALTY}" != "0.0" && "${L2_PENALTY}" != "0" ]]; then
    echo "ERROR: config ${CONFIG_IDX} (${NAME}) is random-init AND lambda=${L2_PENALTY}." >&2
    echo "  Under --random-init the penalty anchors at zero (l2_zero), not at the wavelet" >&2
    echo "  init (l2_sp), so it is not the same lambda axis as arm 0. Put it in" >&2
    echo "  training_logs/clean_updateNC/l2_disentangle/surotex/ instead. Refusing to run." >&2
    exit 2
fi

RANDINIT_FLAG=""
PHASE_DESC="two-phase (A ${PHASE_A_EPOCHS} + B ${PHASE_B_EPOCHS})"
if (( ${RANDINIT[${CONFIG_IDX}]} == 1 )); then
    RANDINIT_FLAG="--random-init"
    PHASE_DESC="single-phase joint (${PHASE_B_EPOCHS} epochs), random init"
fi

echo "=========================================================="
echo "[$(date)] SUROTEX DS head-to-head -- task ${SLURM_ARRAY_TASK_ID}/${TOTAL_TASKS}"
echo "  config=${NAME}  l2=${L2_PENALTY} (dir tag '${L2_TAG:-<none, = lambda 0>}')"
echo "  train_size=${TRAIN_SIZE} (expect n_train=${N_TRAIN})  seeds=${SEEDS[*]}"
echo "  schedule: ${PHASE_DESC}  batch=${BATCH_SIZE}  wd=${WEIGHT_DECAY}"
echo "  flags: ${EXTRA_FLAGS} ${RANDINIT_FLAG}"
echo "=========================================================="

N_OK=0; N_FAIL=0; N_SKIP=0; N_DONE=0; FAILED=""; SKIPPED=""; DONE=""
for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="${SAVE_PREFIX}/${NAME}_s${TRAIN_SIZE_TAG}${L2_TAG}_seed${SEED}"
    FULL_DIR="${LOG_ROOT}/${SAVE_DIR}"

    if compgen -G "${FULL_DIR}/results_*.json" > /dev/null; then
        N_DONE=$((N_DONE+1)); DONE="${DONE} ${SEED}"
        echo "---- seed ${SEED}: results json already present in ${FULL_DIR} -- SKIPPING"
        continue
    fi

    if [[ -d "${FULL_DIR}" ]]; then
        STALE="${FULL_DIR}.stale-$(date +%Y%m%d_%H%M%S)"
        echo "---- seed ${SEED}: ${FULL_DIR} exists with no results json (incomplete run)." >&2
        echo "     Moving it to ${STALE} so train.py cannot auto-resume it. Nothing deleted." >&2
        mv "${FULL_DIR}" "${STALE}" || { echo "     mv FAILED -- skipping this seed" >&2; \
            N_FAIL=$((N_FAIL+1)); FAILED="${FAILED} ${SEED}"; continue; }
    fi

    if (( TIME_LIMIT_SEC - SECONDS - RESERVE_SEC <= 0 )); then
        N_SKIP=$((N_SKIP+1)); SKIPPED="${SKIPPED} ${SEED}"; continue
    fi

    echo ""; echo "---- [$(date)] seed ${SEED} -> ${FULL_DIR}"
    python train.py \
        --dataset surotex \
        --data-dir ${DATA_DIR} \
        --input-size ${INPUT_SIZE} \
        --lr-epochs ${PHASE_A_EPOCHS} \
        --epochs ${PHASE_B_EPOCHS} \
        --batch-size-per-device ${BATCH_SIZE} \
        --print-freq 100 --workers 10 \
        --modulus-type complex_modulus \
        --train-size ${TRAIN_SIZE} --l2-penalty ${L2_PENALTY} --seed ${SEED} \
        --weight-decay ${WEIGHT_DECAY} --save-dir ${SAVE_DIR} \
        ${EXTRA_FLAGS} ${RANDINIT_FLAG}
    if (( $? == 0 )); then
        N_OK=$((N_OK+1)); echo "---- seed ${SEED} OK"
        ACTUAL=$(python - "${FULL_DIR}" <<'PY'
import glob, json, sys
js = sorted(glob.glob(sys.argv[1] + "/results_*.json"))
print(json.load(open(js[-1]))["results"]["actual_train_samples"] if js else "?")
PY
)
        if [[ "${ACTUAL}" != "${N_TRAIN}" ]]; then
            echo "---- WARNING seed ${SEED}: actual_train_samples=${ACTUAL}, grid expects ${N_TRAIN}" >&2
        fi
    else
        N_FAIL=$((N_FAIL+1)); FAILED="${FAILED} ${SEED}"; echo "---- seed ${SEED} FAILED (continuing)" >&2
    fi
done

echo ""; echo "=========================================================="
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done: ${NAME} @ ts=${TRAIN_SIZE} l2=${L2_PENALTY}"
echo "  ran=${N_OK} failed=${N_FAIL} already-done=${N_DONE} skipped-for-time=${N_SKIP}  elapsed $((SECONDS/60)) min"
[[ -n "${DONE}" ]]    && echo "  ALREADY DONE (not recomputed):${DONE}"
[[ -n "${FAILED}" ]]  && echo "  FAILED seeds:${FAILED}"
[[ -n "${SKIPPED}" ]] && echo "  SKIPPED (time), rerun:${SKIPPED}"
echo "=========================================================="
if (( N_OK == 0 && N_DONE == 0 )); then exit 1; fi
exit 0

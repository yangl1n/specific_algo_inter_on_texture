#!/bin/bash -l
# ScatNet block-analysis on SuroTex over two LR arms, train sizes and seeds.
#SBATCH --nodes=1
#SBATCH --job-name=scd_surotex_main
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-47 # 2 LR arms * 8 train sizes * 3 seeds, one run per task

set -o pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1

REPO_DIR="$HOME/sep_conc_deepnets"
DATA_DIR="$HOME/data"
cd "${REPO_DIR}" || exit 2

WORKERS=16
BATCH_SIZE=25     # must divide every entry of TRAIN_N
INPUT_SIZE=224
EPOCHS=300
LR_FREQ=70
WD=1e-4
BETA=5e-4

ARM_TAGS=(lr2p5em3  lr1em3)
ARM_LR=(  0.0025    0.001 )

ARCH_NAME="blk6_whigh_nlsoftshrink"
NL=softshrink
N_BLOCKS=6
P_PROJ_SIZE="64 128 256 512 512 512"
FRAME_WIDTH="1024 2048 4096 8192 8192 8192"

REF_LR_DONE=(0.05 0.01 0.005 0.0005)

TRAIN_SIZES=(0.02 0.05 0.10 0.15 0.20 0.30 0.50 1.0)
TRAIN_N=(50 150 300 450 600 950 1600 3200)   # parallel; used by the preflight
SEEDS=(42 43 44)

RUN_PREFIX="clean_zarka/surotex_h2h"
DONOR_PREFIX="clean_zarka/surotex_lrprobe"  # completed cells reused by symlink
LOG_ROOT="training_logs"                    # ./training_logs/<--dir>
CKPT_ROOT="checkpoints"                     # ./checkpoints/<--dir>

NUM_ARMS=${#ARM_TAGS[@]}
NUM_TRAIN_SIZES=${#TRAIN_SIZES[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_TASKS=$((NUM_ARMS * NUM_TRAIN_SIZES * NUM_SEEDS))

if (( ${#ARM_LR[@]} != NUM_ARMS )); then
    echo "ERROR: ARM_TAGS (${NUM_ARMS}) vs ARM_LR (${#ARM_LR[@]}) length mismatch" >&2; exit 2
fi
if (( ${#TRAIN_N[@]} != NUM_TRAIN_SIZES )); then
    echo "ERROR: TRAIN_SIZES (${NUM_TRAIN_SIZES}) vs TRAIN_N (${#TRAIN_N[@]}) length mismatch" >&2; exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_TASKS )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0,$((TOTAL_TASKS-1))]" >&2; exit 2
fi

CELLS_PER_ARM=$((NUM_TRAIN_SIZES * NUM_SEEDS))
ARM_IDX=$((SLURM_ARRAY_TASK_ID / CELLS_PER_ARM))
REM=$((SLURM_ARRAY_TASK_ID % CELLS_PER_ARM))
TRAIN_SIZE_IDX=$((REM / NUM_SEEDS))
SEED_IDX=$((REM % NUM_SEEDS))

ARM_TAG=${ARM_TAGS[${ARM_IDX}]}
LR=${ARM_LR[${ARM_IDX}]}
TRAIN_SIZE=${TRAIN_SIZES[${TRAIN_SIZE_IDX}]}
N_TRAIN=${TRAIN_N[${TRAIN_SIZE_IDX}]}
SEED=${SEEDS[${SEED_IDX}]}
TRAIN_SIZE_TAG=${TRAIN_SIZE//./p}

NAME="${ARCH_NAME}_${ARM_TAG}"

LR_TAG_FROM_LR=$(awk -v lr="${LR}" 'BEGIN{
    e = 0; m = lr + 0
    while (m < 1) { m *= 10; e++ }
    s = sprintf("%g", m); gsub(/\./, "p", s)
    printf "lr%sem%d", s, e
}')
if [[ "${ARM_TAG}" != "${LR_TAG_FROM_LR}" ]]; then
    echo "ERROR: arm tag '${ARM_TAG}' does not match lr=${LR} (expected '${LR_TAG_FROM_LR}')." >&2
    echo "       ARM_TAGS and ARM_LR have drifted out of step -- fix the table." >&2
    exit 2
fi

for REF in "${REF_LR_DONE[@]}"; do
    if awk -v a="${LR}" -v b="${REF}" 'BEGIN{exit !(a==b)}'; then
        echo "ERROR: arm '${ARM_TAG}' uses lr=${LR}, which already has results elsewhere" >&2
        echo "       (clean_zarka/surotex_final, surotex_regprobe or surotex_lrprobe)." >&2
        echo "       Re-running it here would duplicate cells across directories." >&2
        exit 2
    fi
done

if (( N_TRAIN % BATCH_SIZE != 0 )); then
    echo "ERROR: n_train=${N_TRAIN} is not divisible by batch_size=${BATCH_SIZE}" >&2
    echo "       tail batch would be $((N_TRAIN % BATCH_SIZE)) images -- see header." >&2
    echo "       Pick a batch size dividing all of: ${TRAIN_N[*]}  (gcd is 50; 25 also works)" >&2
    exit 2
fi
echo "[preflight] ts=${TRAIN_SIZE} n_train=${N_TRAIN} batch=${BATCH_SIZE}" \
     "-> $((N_TRAIN / BATCH_SIZE)) full steps/epoch, no tail. OK"

RUN_DIR="${RUN_PREFIX}/${NAME}_s${TRAIN_SIZE_TAG}_seed${SEED}"
FULL_LOG_DIR="${LOG_ROOT}/${RUN_DIR}"
FULL_CKPT_DIR="${CKPT_ROOT}/${RUN_DIR}"
DONOR_LOG_DIR="${LOG_ROOT}/${DONOR_PREFIX}/${ARM_TAG}_s${TRAIN_SIZE_TAG}_seed${SEED}"

echo "=========================================================="
echo "[$(date)] SUROTEX SCD head-to-head -- task ${SLURM_ARRAY_TASK_ID}/${TOTAL_TASKS}"
echo "  arm=${ARM_TAG} (lr=${LR})  train_size=${TRAIN_SIZE} (n_train=${N_TRAIN})  seed=${SEED}"
echo "  arch=${ARCH_NAME}  n_blocks=${N_BLOCKS}  P-proj=[${P_PROJ_SIZE}]  frame=[${FRAME_WIDTH}]"
echo "  nl=${NL}  epochs=${EPOCHS}  lr_freq=${LR_FREQ}  batch=${BATCH_SIZE}  wd=${WD}  beta=${BETA}"
echo "  -> ${FULL_LOG_DIR}"
echo "=========================================================="

if compgen -G "${FULL_LOG_DIR}/results_*.json" > /dev/null; then
    echo "[$(date)] results json already present -- SKIPPING (cell is done)."
    echo "  Delete ${FULL_LOG_DIR} by hand if you really want it recomputed."
    exit 0
fi

if [[ ! -e "${FULL_LOG_DIR}" ]] && compgen -G "${DONOR_LOG_DIR}/results_*.json" > /dev/null; then
    echo "[$(date)] donor cell found: ${DONOR_LOG_DIR}"
    echo "  checking config equality before reusing it..."
    MATCH=$(python - "${DONOR_LOG_DIR}" "${LR}" "${EPOCHS}" "${BATCH_SIZE}" "${LR_FREQ}" \
                     "${WD}" "${BETA}" "${N_BLOCKS}" "${NL}" "${TRAIN_SIZE}" "${SEED}" \
                     "${INPUT_SIZE}" "${FRAME_WIDTH}" "${P_PROJ_SIZE}" <<'PY'
import glob, json, sys

(donor, lr, epochs, batch, lr_freq, wd, beta, nb, nl, ts, seed,
 insize, frame, pproj) = sys.argv[1:15]

js = sorted(glob.glob(donor + "/results_*.json"))
if not js:
    print("NO:no results json"); raise SystemExit
blob = json.load(open(js[-1]))
c = blob.get("config", {})
if blob.get("results", {}).get("run_status") != "success":
    print("NO:donor run_status is not success"); raise SystemExit

def num(x):
    return float(x)

checks = [
    ("lr",                             num(c.get("lr", -1)),            num(lr)),
    ("epochs",                         num(c.get("epochs", -1)),        num(epochs)),
    ("batch_size",                     num(c.get("batch_size", -1)),    num(batch)),
    ("learning_rate_adjust_frequency", num(c.get("learning_rate_adjust_frequency", -1)), num(lr_freq)),
    ("weight_decay",                   num(c.get("weight_decay", -1)),  num(wd)),
    ("pars_beta",                      num(c.get("pars_beta", -1)),     num(beta)),
    ("n_blocks",                       num(c.get("n_blocks", -1)),      num(nb)),
    ("train_size",                     num(c.get("train_size", -1)),    num(ts)),
    ("seed",                           num(c.get("seed", -1)),          num(seed)),
    ("input_size",                     num(c.get("input_size", -1)),    num(insize)),
    ("val_frac",                       num(c.get("val_frac", -1)),      0.2),
]
bad = [f"{k}({got}!={want})" for k, got, want in checks
       if abs(got - want) > 1e-12 * max(1.0, abs(want))]

if c.get("non_linearity") != nl:
    bad.append(f"non_linearity({c.get('non_linearity')}!={nl})")
if c.get("data_split_mode") != "deep_scatter":
    bad.append(f"data_split_mode({c.get('data_split_mode')})")
if [int(v) for v in (c.get("frame_width") or [])] != [int(v) for v in frame.split()]:
    bad.append("frame_width")
if [int(v) for v in (c.get("P_proj_size") or [])] != [int(v) for v in pproj.split()]:
    bad.append("P_proj_size")

print("YES" if not bad else "NO:" + ",".join(bad))
PY
)
    if [[ "${MATCH}" == "YES" ]]; then
        mkdir -p "$(dirname "${FULL_LOG_DIR}")" || exit 2
        if ln -s "../${DONOR_PREFIX##*/}/${ARM_TAG}_s${TRAIN_SIZE_TAG}_seed${SEED}" \
                 "${FULL_LOG_DIR}"; then
            echo "[$(date)] config matches -- SYMLINKED donor cell in as ${RUN_DIR}."
            echo "  analyze.py labels it by the LINK name (${NAME}_s${TRAIN_SIZE_TAG}_seed${SEED}),"
            echo "  because SCD result JSONs carry no save_dir key. No training needed."
            exit 0
        fi
        echo "WARNING: symlink failed -- falling through to a normal run." >&2
    else
        echo "WARNING: donor cell REJECTED (${MATCH})." >&2
        echo "         Training this cell normally instead of reusing it." >&2
    fi
fi

if [[ -d "${FULL_LOG_DIR}" || -d "${FULL_CKPT_DIR}" ]]; then
    STAMP=$(date +%Y%m%d_%H%M%S)
    echo "[$(date)] run dir exists with no results json (incomplete run)." >&2
    echo "  Moving it aside; nothing is deleted." >&2
    [[ -d "${FULL_LOG_DIR}"  ]] && { mv "${FULL_LOG_DIR}"  "${FULL_LOG_DIR}.stale-${STAMP}"  || exit 2; }
    [[ -d "${FULL_CKPT_DIR}" ]] && { mv "${FULL_CKPT_DIR}" "${FULL_CKPT_DIR}.stale-${STAMP}" || exit 2; }
fi

python main.py \
    --data ${DATA_DIR} \
    --dataset surotex \
    --arch scatnetblockanalysis \
    --input-size ${INPUT_SIZE} \
    --workers ${WORKERS} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --wd ${WD} \
    --pars-beta ${BETA} \
    --learning-rate-adjust-frequency ${LR_FREQ} \
    --n-blocks ${N_BLOCKS} \
    --P-proj-size ${P_PROJ_SIZE} \
    --frame-width ${FRAME_WIDTH} \
    --non-linearity ${NL} \
    --data-split-mode deep_scatter \
    --val-frac 0.2 \
    --train-size ${TRAIN_SIZE} \
    --seed ${SEED} \
    --dir ${RUN_DIR}
STATUS=$?

if (( STATUS == 0 )); then
    python - "${FULL_LOG_DIR}" "${N_TRAIN}" "${EPOCHS}" <<'PY'
import glob, json, sys
d, want_n, epochs = sys.argv[1], sys.argv[2], int(sys.argv[3])
js = sorted(glob.glob(d + "/results_*.json"))
if not js:
    print("WARNING: exit 0 but no results json written."); raise SystemExit
res = json.load(open(js[-1]))["results"]

actual = res.get("train_samples_optimized", "?")
if str(actual) != str(want_n):
    print(f"WARNING: train_samples_optimized={actual}, grid expects {want_n} --")
    print(f"         the no-tail guarantee was made about {want_n}. Check before using.")

be = res.get("best_epoch_acc1")
if isinstance(be, int) and be >= 0.85 * epochs:
    print(f"---- WARNING: best epoch {be} of {epochs} is in the last 15% of training.")
    print( "     This arm is likely UNDER-CONVERGED: the accuracy was still climbing when")
    print( "     the budget ran out, so it is a floor, not a converged number. Do not put")
    print( "     it in the curve without extending --epochs for this cell.")
PY
fi

echo "=========================================================="
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} done: ${NAME} @ ts=${TRAIN_SIZE} seed=${SEED}"
echo "  exit=${STATUS}  elapsed $((SECONDS/60)) min"
echo "=========================================================="
exit ${STATUS}

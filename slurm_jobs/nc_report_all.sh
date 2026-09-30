#!/bin/bash -l
# Offline neural-collapse recompute (nc_report.py) over each results directory.
#SBATCH --nodes=1
#SBATCH --job-name=nc_report_all
#SBATCH --time=8:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --partition PARTITION_NAME
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=EMAIL@ADDRESS.COM
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --array=0-13 # one task per results directory in DIRS

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate ENV_NAME
export PYTHONNOUSERSITE=1

DS_REPO="$HOME/deep_scatter"
SCD_REPO="$HOME/sep_conc_deepnets"

REPOS=(ds ds ds ds ds ds ds ds ds ds scd scd scd scd)
DIRS=(
  "training_logs/clean_v2/dtd"                            #  0
  "training_logs/clean_updateNC/eurosat"                  #  1
  "training_logs/clean_v2/minc"                           #  2
  "training_logs/clean_v2/minc_lowdata"                   #  3
  "training_logs/surotex_h2h"                             #  4
  "training_logs/phase_a/minc2500"                        #  5
  "training_logs/clean_updateNC/l2_disentangle/dtd_v2"    #  6
  "training_logs/clean_updateNC/l2_disentangle/eurosat"   #  7
  "training_logs/clean_updateNC/l2_disentangle/minc2500"  #  8
  "training_logs/clean_updateNC/l2_disentangle/surotex"   #  9
  "training_logs/clean_zarka/dtd"                         # 10
  "training_logs/clean_zarka/eurosat_v2"                  # 11
  "training_logs/clean_zarka/minc2500"                    # 12
  "training_logs/clean_zarka/surotex_h2h"                 # 13
)

TOTAL=${#DIRS[@]}
if (( ${#REPOS[@]} != TOTAL )); then
    echo "REPOS (${#REPOS[@]}) and DIRS (${TOTAL}) are out of sync" >&2
    exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
    echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range [0, $((TOTAL - 1))]" >&2
    exit 2
fi

SIDE=${REPOS[${SLURM_ARRAY_TASK_ID}]}
RESULTS_DIR=${DIRS[${SLURM_ARRAY_TASK_ID}]}

if [[ "${SIDE}" == "ds" ]]; then
    REPO_DIR="${DS_REPO}"
    BATCH_SIZE=128
else
    REPO_DIR="${SCD_REPO}"
    BATCH_SIZE=64
fi

cd "${REPO_DIR}"
echo "[$(date)] side=${SIDE} repo=${REPO_DIR}"
echo "[$(date)] results-dir=${RESULTS_DIR}"

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "ERROR: ${REPO_DIR}/${RESULTS_DIR} does not exist" >&2
    exit 3
fi

CSV="${RESULTS_DIR}/nc_recomputed.csv"
PRE="${RESULTS_DIR}/nc_recomputed.pre_skew.csv"
if [[ -f "${CSV}" && ! -f "${PRE}" ]]; then
    if ! head -1 "${CSV}" | grep -q "nc1_skew"; then
        cp -p "${CSV}" "${PRE}"
        echo "[$(date)] kept the pre-skew CSV as ${PRE}"
    fi
fi

python -u nc_report.py \
    --results-dir "${RESULTS_DIR}" \
    --recursive \
    --device cuda \
    --batch-size ${BATCH_SIZE} \
    --workers 8 \
    --threads 16 \
    --compare

echo
echo "[$(date)] new CSV vs pre-skew snapshot"
if [[ -f "${PRE}" ]]; then
python -u - "${RESULTS_DIR}" "${SIDE}" <<'PY'
import os, sys
import pandas as pd
d, side = sys.argv[1], sys.argv[2]
new = pd.read_csv(os.path.join(d, "nc_recomputed.csv"))
old = pd.read_csv(os.path.join(d, "nc_recomputed.pre_skew.csv"))
key = ["results_json", "phase", "split"]
m = new.merge(old.drop_duplicates(key), on=key, suffixes=("", "__prev"))  # nc1_old is a real column
print(f"  {len(new)} new rows, {len(old)} pre-skew rows, {len(m)} matched on {key}")
cols = ["nc1", "nc1_trace", "between_class_trace", "nc2"]
if side == "ds":
    cols.append("nc3")
for c in cols:
    if c + "__prev" not in m:
        continue
    sub = m
    if c == "nc3":
        sub = m[~m["nc_source"].astype(str).str.contains("config-rebuilt")]
    both = sub[[c, c + "__prev"]].dropna()
    if both.empty:
        print(f"  {c:20s} (no overlapping values)")
        continue
    rel = (both[c] - both[c + "__prev"]).abs() / both[c + "__prev"].abs()
    print(f"  {c:20s} {int((rel <= 1e-3).sum())}/{len(both)} within rtol 1e-3  "
          f"(median {rel.median():.1e}, max {rel.max():.1e})")
if side == "scd" and "nc3__prev" in m:
    print(f"  nc3 (SCD fix)        was NaN in {int(m['nc3__prev'].isna().sum())} rows; "
          f"now NaN in {int(m['nc3'].isna().sum())}")
print(f"  nc1_skew non-null    {int(new['nc1_skew'].notna().sum())}/{len(new)}")
PY
else
    echo "  (no pre-skew nc_recomputed.csv here -- first recompute for this directory)"
fi

echo "[$(date)] done -> ${REPO_DIR}/${CSV}"

#!/usr/bin/env bash
#$ -cwd
#$ -l node_q=1
#$ -l h_rt=24:00:00
#$ -t 1-5
#$ -j y

set -euo pipefail

TARGET="${TARGET:-${JOB_NAME:-}}"
TASK_ID="${SGE_TASK_ID:-1}"
INIT_SMILES="${INIT_SMILES:-c1ccccc1}"
N_STEP_COARSE="${N_STEP_COARSE:-1000}"
N_STEP_FINE="${N_STEP_FINE:-20}"
GOBLET_ENV="${GOBLET_ENV:-goblet}"
BOLTZ_ENV="${BOLTZ_ENV:-$GOBLET_ENV}"
PROJECT_DIR="${SGE_O_WORKDIR:-$PWD}"
WORKSPACE_REL="/wkdir/${TARGET}_${TASK_ID}/"
WORKSPACE_DIR="${PROJECT_DIR}${WORKSPACE_REL}"
OUTPUT_NAME="${TARGET}_${TASK_ID}.ASBOLTZ.csv"
OUTPUT_FILE="${PROJECT_DIR}/data/result/${OUTPUT_NAME}"
META_FILE="${PROJECT_DIR}/data/result/${TARGET}_${TASK_ID}.meta.txt"
API_LOG="${PROJECT_DIR}/data/result/${TARGET}_${TASK_ID}.api.log"

case "$TARGET" in
    akt1|ampc|cxcr4|cp3a4|hivrt|hivpr|gcr|kif11) ;;
    *)
        echo "ERROR: TARGET must be one of akt1, ampc, cxcr4, cp3a4, hivrt, hivpr, gcr, or kif11: $TARGET" >&2
        exit 2
        ;;
esac
for value_name in N_STEP_COARSE N_STEP_FINE; do
    value="${!value_name}"
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer: $value" >&2
        exit 2
    fi
done

cd "$PROJECT_DIR"
mkdir -p "$WORKSPACE_DIR" data/result

if [[ -e "$OUTPUT_FILE" ]]; then
    echo "ERROR: refusing to overwrite existing result: $OUTPUT_FILE" >&2
    exit 2
fi

INPUT_YAML="data/sample_yamls/boltz_${TARGET}.yaml"
INPUT_MSA="data/sample_yamls/msa_${TARGET}.a3m"
for required_file in \
    "$INPUT_YAML" \
    "$INPUT_MSA" \
    ckpt/pretrain/ver2/feature-extractor-ep20.pth \
    ckpt/pretrain/ver2/node-header-ep20.pth \
    ckpt/pretrain/ver2/edge-header-ep20.pth; do
    if [[ ! -s "$required_file" ]]; then
        echo "ERROR: required input is missing or empty: $required_file" >&2
        exit 2
    fi
done

cp "$INPUT_YAML" "$WORKSPACE_DIR/input.yaml"
cp "$INPUT_MSA" "$WORKSPACE_DIR/"

if [[ -f /etc/profile.d/modules.sh ]]; then
    # Exact software modules used for the reported T4-cluster calculations.
    source /etc/profile.d/modules.sh
    module load cuda/12.1.0 gcc miniconda/24.1.2
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available" >&2
    exit 2
fi
eval "$(conda shell.bash hook)"
conda activate "$GOBLET_ENV"

export HYDRA_FULL_ERROR=1
export T4TMPDIR="${T4TMPDIR:-${TMPDIR:-/tmp}}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${T4TMPDIR}/numba_cache}"
export SOURCE_BOLTZ_CACHE="${SOURCE_BOLTZ_CACHE:-${HOME}/.boltz}"
export BOLTZ_CACHE_DIR="${BOLTZ_CACHE_DIR:-${T4TMPDIR}/boltz_cache}"
mkdir -p "$BOLTZ_CACHE_DIR" "$NUMBA_CACHE_DIR"
if [[ ! -d "$SOURCE_BOLTZ_CACHE" ]]; then
    echo "ERROR: Boltz cache directory is missing: $SOURCE_BOLTZ_CACHE" >&2
    exit 2
fi
if [[ "$(realpath "$SOURCE_BOLTZ_CACHE")" != "$(realpath "$BOLTZ_CACHE_DIR")" ]]; then
    cp -a "${SOURCE_BOLTZ_CACHE}/." "$BOLTZ_CACHE_DIR/"
fi
for cache_file in boltz2_conf.ckpt boltz2_aff.ckpt; do
    if [[ ! -s "${BOLTZ_CACHE_DIR}/${cache_file}" ]]; then
        echo "ERROR: Boltz model is missing: ${BOLTZ_CACHE_DIR}/${cache_file}" >&2
        exit 2
    fi
done

{
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "job_id=${JOB_ID:-manual}"
    echo "task_id=$TASK_ID"
    echo "target=$TARGET"
    echo "init_smiles=$INIT_SMILES"
    echo "reward_name=ASBoltz"
    echo "n_step_coarse=$N_STEP_COARSE"
    echo "n_step_fine=$N_STEP_FINE"
    echo "generator_model_version=20"
    echo "goblet_env=$GOBLET_ENV"
    echo "boltz_env=$BOLTZ_ENV"
    echo "output_file=$OUTPUT_FILE"
    python -c 'import importlib.metadata as m; import platform; print("python=" + platform.python_version()); print("boltz=" + m.version("boltz")); print("torch=" + m.version("torch")); print("rdkit=" + m.version("rdkit")); print("dgl=" + m.version("dgl"))'
    conda run -n "$BOLTZ_ENV" python -c 'import importlib.metadata as m; print("boltz_server=" + m.version("boltz"))'
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    sha256sum \
        ckpt/pretrain/ver2/feature-extractor-ep20.pth \
        ckpt/pretrain/ver2/node-header-ep20.pth \
        ckpt/pretrain/ver2/edge-header-ep20.pth \
        "$INPUT_YAML" "$INPUT_MSA" \
        "${BOLTZ_CACHE_DIR}/boltz2_conf.ckpt" \
        "${BOLTZ_CACHE_DIR}/boltz2_aff.ckpt"
} > "$META_FILE"

API_PID=""
cleanup() {
    status=$?
    trap - EXIT
    if [[ -n "$API_PID" ]] && kill -0 "$API_PID" 2>/dev/null; then
        kill "$API_PID" 2>/dev/null || true
        wait "$API_PID" 2>/dev/null || true
    fi
    {
        echo "ended_at=$(date --iso-8601=seconds)"
        echo "exit_status=$status"
    } >> "$META_FILE"
    exit "$status"
}
trap cleanup EXIT INT TERM

FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export API_PORT="$FREE_PORT"
echo "api_port=$API_PORT" >> "$META_FILE"
conda run --no-capture-output -n "$BOLTZ_ENV" \
    python -u Utils/boltz_api_constraints.py > "$API_LOG" 2>&1 &
API_PID=$!

api_ready=false
for _ in $(seq 1 180); do
    if ! kill -0 "$API_PID" 2>/dev/null; then
        echo "ERROR: Boltz API exited during startup; see $API_LOG" >&2
        exit 1
    fi
    if curl --fail --silent "http://127.0.0.1:${API_PORT}/docs" >/dev/null; then
        api_ready=true
        break
    fi
    sleep 2
done
if [[ "$api_ready" != true ]]; then
    echo "ERROR: Boltz API did not become ready within 6 minutes; see $API_LOG" >&2
    exit 1
fi

python -u Generator/mcts.py \
    "mcts.init_smiles='${INIT_SMILES}'" \
    mcts.reward_name=ASBoltz \
    "mcts.output_fname=${OUTPUT_NAME}" \
    "mcts.work_space=${WORKSPACE_REL}" \
    "mcts.n_step=${N_STEP_COARSE}" \
    "mcts.n_step_fine=${N_STEP_FINE}"

if [[ ! -s "$OUTPUT_FILE" ]]; then
    echo "ERROR: MCTS completed without producing $OUTPUT_FILE" >&2
    exit 1
fi
echo "result_rows=$(($(wc -l < "$OUTPUT_FILE") - 1))" >> "$META_FILE"
echo "Result: $OUTPUT_FILE"
echo "Metadata: $META_FILE"
echo "Boltz API log: $API_LOG"

#!/usr/bin/env bash
set -euo pipefail

# 数学任务熵分析一键脚本
#
# 用法：
#   1. 先修改下面“配置区”的变量
#   2. 在仓库根目录执行：bash scripts/run_math_entropy_analysis.sh
#
# 说明：
#   - generate 阶段支持 transformers / vllm
#   - rescore 阶段当前固定走 transformers，并且要使用同一个模型
#   - 如果你只想用某一张卡，优先设置 CUDA_VISIBLE_DEVICES
#   - 设置了 CUDA_VISIBLE_DEVICES 之后，脚本内 DEVICE 通常写成 cuda 即可

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# =========================
# 配置区：基础路径
# =========================

# Python 可执行文件
PYTHON_BIN="${PYTHON_BIN:-python}"

# 模型名称或本地路径
MODEL_PATH="/home/cxy/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B"
PROBLEMS_PATH="/home/cxy/verl_async/DeepscalerDataset/converted/aime2025_converted_verl_fixed.parquet"

# 输出目录；实际产物会写到 ${OUTPUT_DIR}/${RUN_NAME}/ 下
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/artifacts}"
RUN_NAME="${RUN_NAME:-math_entropy_run}"
ARTIFACT_DIR="${OUTPUT_DIR}/${RUN_NAME}"

# =========================
# 配置区：设备与后端
# =========================

# 选卡方式 1：直接限制进程可见卡
# 示例：export CUDA_VISIBLE_DEVICES=0
# 示例：export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 选卡方式 2：传给 transformers 的 device
# 常见写法：cuda / cuda:0 / cpu
# 注意：如果已经设置 CUDA_VISIBLE_DEVICES=1，那么这里的 cuda 或 cuda:0
# 实际上对应物理上的第 1 张卡。
DEVICE="${DEVICE:-cuda}"

# generate 阶段后端：transformers 或 vllm
GENERATE_BACKEND="${GENERATE_BACKEND:-transformers}"

# =========================
# 配置区：生成参数
# =========================

# greedy 或 sampling
DECODE_MODE="${DECODE_MODE:-greedy}"

# 最大生成 token 数
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"

# 只有 sampling 模式时有意义
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-1.0}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-1}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"

# generate 阶段 batch size
# 显存不够时优先调小这个值
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"

# ??? N ???????? 3 ????????????????
MAX_SAMPLES="${MAX_SAMPLES:-3}"

# =========================
# 配置区：可视化
# =========================

# 是否额外导出单条样本的熵热力图 HTML
# 取值：1 / 0
EXPORT_SINGLE_HEATMAP="${EXPORT_SINGLE_HEATMAP:-1}"

# 如果不指定 sample id，visualize.py 会默认选择第一条样本
VIS_SAMPLE_ID="${VIS_SAMPLE_ID:-}"
VIS_DECODE_ID="${VIS_DECODE_ID:-0}"
VIS_TITLE="${VIS_TITLE:-Math Entropy Heatmap}"

# =========================
# 基础检查
# =========================

if [[ "${MODEL_PATH}" == "/path/to/your/model" ]]; then
  echo "[ERROR] 请先设置 MODEL_PATH"
  exit 1
fi

if [[ "${PROBLEMS_PATH}" == "/path/to/your/problems.parquet" ]]; then
  echo "[ERROR] 请先设置 PROBLEMS_PATH"
  exit 1
fi

if [[ ! -f "${PROBLEMS_PATH}" ]]; then
  echo "[ERROR] 测试集文件不存在: ${PROBLEMS_PATH}"
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}"
mkdir -p "${ARTIFACT_DIR}/plots"

# =========================
# 打印配置，方便回溯
# =========================

echo "========== Math Entropy Analysis =========="
echo "REPO_ROOT             = ${REPO_ROOT}"
echo "PYTHON_BIN            = ${PYTHON_BIN}"
echo "MODEL_PATH            = ${MODEL_PATH}"
echo "PROBLEMS_PATH         = ${PROBLEMS_PATH}"
echo "ARTIFACT_DIR          = ${ARTIFACT_DIR}"
echo "CUDA_VISIBLE_DEVICES  = ${CUDA_VISIBLE_DEVICES}"
echo "DEVICE                = ${DEVICE}"
echo "GENERATE_BACKEND      = ${GENERATE_BACKEND}"
echo "DECODE_MODE           = ${DECODE_MODE}"
echo "MAX_NEW_TOKENS        = ${MAX_NEW_TOKENS}"
echo "TEMPERATURE           = ${TEMPERATURE}"
echo "TOP_P                 = ${TOP_P}"
echo "NUM_RETURN_SEQUENCES  = ${NUM_RETURN_SEQUENCES}"
echo "REPETITION_PENALTY    = ${REPETITION_PENALTY}"
echo "GEN_BATCH_SIZE        = ${GEN_BATCH_SIZE}"
echo "MAX_SAMPLES           = ${MAX_SAMPLES:-ALL}"
echo "EXPORT_SINGLE_HEATMAP = ${EXPORT_SINGLE_HEATMAP}"
echo "==========================================="

# =========================
# 第 1 步：生成答案
# =========================

echo "[1/5] Running generation ..."
GEN_ARGS=(
  --model "${MODEL_PATH}"
  --problems "${PROBLEMS_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --run-name "${RUN_NAME}"
  --backend "${GENERATE_BACKEND}"
  --decode-mode "${DECODE_MODE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --num-return-sequences "${NUM_RETURN_SEQUENCES}"
  --repetition-penalty "${REPETITION_PENALTY}"
  --batch-size "${GEN_BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  GEN_ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

"${PYTHON_BIN}" -m math_entropy_analysis.generate "${GEN_ARGS[@]}"

# =========================
# 第 2 步：teacher-forcing 重打分
# =========================
# 这一步会计算完整 logits 上的 entropy / logprob / top1_prob / margin 等。
# 如果显存不足，优先：
#   1. 减小 MAX_NEW_TOKENS
#   2. 换更小模型
#   3. 调低输入测试集规模

echo "[2/5] Running rescoring ..."
"${PYTHON_BIN}" -m math_entropy_analysis.rescore \
  --model "${MODEL_PATH}" \
  --problems "${ARTIFACT_DIR}/problems.parquet" \
  --generations "${ARTIFACT_DIR}/generations.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --device "${DEVICE}"

# =========================
# 第 3 步：聚合序列级特征
# =========================

echo "[3/5] Aggregating sequence features ..."
"${PYTHON_BIN}" -m math_entropy_analysis.features \
  --generations "${ARTIFACT_DIR}/generations.parquet" \
  --token-metrics "${ARTIFACT_DIR}/token_metrics.parquet" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}"

# =========================
# 第 4 步：生成全局统计图
# =========================

echo "[4/5] Rendering summary plots ..."
"${PYTHON_BIN}" -m math_entropy_analysis.plots \
  --sequence-features "${ARTIFACT_DIR}/sequence_features.parquet" \
  --token-metrics "${ARTIFACT_DIR}/token_metrics.parquet" \
  --output-dir "${ARTIFACT_DIR}/plots"

# =========================
# 第 5 步：导出单条样本热力图（可选）
# =========================

if [[ "${EXPORT_SINGLE_HEATMAP}" == "1" ]]; then
  echo "[5/5] Rendering single-sample heatmap ..."

  VIS_ARGS=(
    --token-metrics "${ARTIFACT_DIR}/token_metrics.parquet"
    --generations "${ARTIFACT_DIR}/generations.parquet"
    --output "${ARTIFACT_DIR}/plots/sample_entropy_heatmap.html"
    --decode-id "${VIS_DECODE_ID}"
    --title "${VIS_TITLE}"
  )

  if [[ -n "${VIS_SAMPLE_ID}" ]]; then
    VIS_ARGS+=(--sample-id "${VIS_SAMPLE_ID}")
  fi

  "${PYTHON_BIN}" -m math_entropy_analysis.visualize "${VIS_ARGS[@]}"
fi

# =========================
# 完成提示
# =========================

echo "Done. Artifacts are under: ${ARTIFACT_DIR}"
echo "- problems.parquet"
echo "- generations.parquet"
echo "- token_metrics.parquet"
echo "- sequence_features.parquet"
echo "- plots/"

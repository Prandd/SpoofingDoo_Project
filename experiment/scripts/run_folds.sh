#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_folds.sh — Run the 5-fold cross-validation for SpoofingDoo Stage 2/3.
#
# Usage (from SpoofingDoo_Project root):
#   bash experiment/scripts/run_folds.sh [OPTIONS]
#
# Optional overrides (env vars):
#   CHECKPOINT   path to AASIST-L .pth  (default: data/pretrained/AASIST-L.pth)
#   MANIFEST     path to prosody JSON    (default: data/features/prosody_manifest.json)
#   OUT_PREFIX   run directory prefix    (default: experiment/runs/stage2)
#   EPOCHS       max epochs per fold     (default: 100)
#   LR           learning rate           (default: 1e-4)
#   PATIENCE     early-stop patience     (default: 10)
#   BATCH        batch size              (default: 16)
#
# Example with custom checkpoint:
#   CHECKPOINT=data/pretrained/AASIST-L.pth bash experiment/scripts/run_folds.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-data/pretrained/AASIST-L.pth}"
MANIFEST="${MANIFEST:-data/features/prosody_manifest.json}"
OUT_PREFIX="${OUT_PREFIX:-experiment/runs/stage2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-10}"
BATCH="${BATCH:-16}"

LOG_DIR="${OUT_PREFIX}_cv_logs"
mkdir -p "${LOG_DIR}"

SUMMARY_CSV="${LOG_DIR}/cv_summary.csv"
echo "fold,best_eer_pct,auc,best_epoch,total_epochs,out_dir" > "${SUMMARY_CSV}"

echo "========================================================"
echo "  SpoofingDoo 5-Fold CV Run"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Manifest   : ${MANIFEST}"
echo "  Out prefix : ${OUT_PREFIX}"
echo "  Epochs     : ${EPOCHS}  LR: ${LR}  Patience: ${PATIENCE}"
echo "========================================================"

TOTAL_EER=0
TOTAL_AUC=0

for FOLD_ID in 00 01 02 03 04; do
    FOLD_JSON="experiment/protocols/folds/fold_${FOLD_ID}.json"
    OUT_DIR="${OUT_PREFIX}_fold${FOLD_ID}"
    FIG_DIR="${OUT_DIR}/figures"
    LOG_FILE="${LOG_DIR}/fold_${FOLD_ID}.log"

    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  Fold ${FOLD_ID}  →  ${OUT_DIR}"
    echo "──────────────────────────────────────────────────────"

    python3 experiment/scripts/baseline_train.py \
        --fold       "${FOLD_JSON}" \
        --manifest   "${MANIFEST}" \
        --checkpoint "${CHECKPOINT}" \
        --out_dir    "${OUT_DIR}" \
        --fig_dir    "${FIG_DIR}" \
        --epochs     "${EPOCHS}" \
        --lr         "${LR}" \
        --batch_size "${BATCH}" \
        --patience   "${PATIENCE}" \
        2>&1 | tee "${LOG_FILE}"

    # Parse final EER and AUC from the log
    FINAL_LINE=$(grep -E "^\[Final\]" "${LOG_FILE}" | tail -1 || true)
    BEST_LINE=$(grep -E "^\[Done\]" "${LOG_FILE}" | tail -1 || true)

    EER=$(echo "${FINAL_LINE}" | grep -oE "EER=[0-9.]+%" | grep -oE "[0-9.]+" || echo "N/A")
    AUC=$(echo "${FINAL_LINE}" | grep -oE "AUC=[0-9.]+" | grep -oE "[0-9.]+" || echo "N/A")
    BEST_EP=$(grep -E "✓ Best EER" "${LOG_FILE}" | tail -1 | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+" || echo "N/A")
    TOTAL_EP=$(grep -E "^Epoch " "${LOG_FILE}" | tail -1 | grep -oE "^Epoch [0-9]+" | grep -oE "[0-9]+" || echo "N/A")

    echo "fold_${FOLD_ID}: EER=${EER}%  AUC=${AUC}" | tee -a "${LOG_DIR}/summary.txt"
    echo "${FOLD_ID},${EER},${AUC},${BEST_EP},${TOTAL_EP},${OUT_DIR}" >> "${SUMMARY_CSV}"

done

echo ""
echo "========================================================"
echo "  All folds complete."
echo "  Summary CSV  : ${SUMMARY_CSV}"
echo "  Full logs    : ${LOG_DIR}/"
echo ""
echo "  Run collect_results.py to generate a combined report:"
echo "    python3 experiment/scripts/collect_results.py \\"
echo "        --runs_root experiment/runs \\"
echo "        --prefix stage2"
echo "========================================================"

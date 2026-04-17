#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Pretrain Pipeline — 3-Phase Training Script
# Chains Phase 1 → Phase 2 → Phase 3 sequentially.
# Edit the variables below to configure your run.
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────

MODEL_NAME="${MODEL_NAME:-pretrain_model}"
VOCODER="${VOCODER:-ChouwaGAN}"
ARCHITECTURE="${ARCHITECTURE:-Fork}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OPTIMIZER="${OPTIMIZER:-AdamW}"
FP16="${FP16:-true}"

# Phase 1: Decoder-only pretraining
PHASE1_EPOCHS="${PHASE1_EPOCHS:-200}"
PHASE1_LR_G="${PHASE1_LR_G:-2e-4}"
PHASE1_LR_D="${PHASE1_LR_D:-2e-4}"
PHASE1_SR="${PHASE1_SR:-48000}"
PHASE1_SAVE_EVERY="${PHASE1_SAVE_EVERY:-20}"

# Phase 2: Full VITS pretrain
PHASE2_EPOCHS="${PHASE2_EPOCHS:-500}"
PHASE2_LR_G="${PHASE2_LR_G:-1e-4}"
PHASE2_LR_D="${PHASE2_LR_D:-1e-4}"
PHASE2_SR="${PHASE2_SR:-48000}"
PHASE2_SAVE_EVERY="${PHASE2_SAVE_EVERY:-20}"
KL_ANNEAL_STEPS="${KL_ANNEAL_STEPS:-50000}"
KL_FREE_BITS="${KL_FREE_BITS:-0.25}"
DECODER_FREEZE_STEPS="${DECODER_FREEZE_STEPS:-10000}"

# Phase 3: SR adaptation
PHASE3_EPOCHS="${PHASE3_EPOCHS:-50}"
PHASE3_LR_G="${PHASE3_LR_G:-5e-5}"
PHASE3_LR_D="${PHASE3_LR_D:-5e-5}"
PHASE3_TARGET_SRS="${PHASE3_TARGET_SRS:-40000 32000}"
PHASE3_SAVE_EVERY="${PHASE3_SAVE_EVERY:-10}"

# Loss config
SPECTRAL_LOSS="${SPECTRAL_LOSS:-L1 Mel Loss}"
ADVERSARIAL_LOSS="${ADVERSARIAL_LOSS:-lsgan}"

# ─── Precision flags ─────────────────────────────────────────────

FP16_FLAG=""
if [ "$FP16" = "true" ]; then
    FP16_FLAG="--fp16"
else
    FP16_FLAG="--no_fp16"
fi

# ─── Script dir ──────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_SCRIPT="${SCRIPT_DIR}/rvc/train/pretrain.py"
PYTHON="${PYTHON:-python}"
EXPERIMENT_DIR="${SCRIPT_DIR}/logs/${MODEL_NAME}"

echo "═══════════════════════════════════════════════════════════"
echo "  RVC Pretrain Pipeline"
echo "  Model:    ${MODEL_NAME}"
echo "  Vocoder:  ${VOCODER}"
echo "  GPU:      ${GPU}"
echo "═══════════════════════════════════════════════════════════"

# ─── Phase 1 ─────────────────────────────────────────────────────

echo ""
echo "▶ PHASE 1: Decoder-only pretraining (${VOCODER} @ ${PHASE1_SR}Hz)"
echo "  Epochs: ${PHASE1_EPOCHS}, LR: ${PHASE1_LR_G}/${PHASE1_LR_D}"
echo ""

$PYTHON "$PRETRAIN_SCRIPT" \
    --phase 1 \
    --model_name "$MODEL_NAME" \
    --vocoder "$VOCODER" \
    --architecture "$ARCHITECTURE" \
    --sample_rate "$PHASE1_SR" \
    --batch_size "$BATCH_SIZE" \
    --total_epochs "$PHASE1_EPOCHS" \
    --save_every "$PHASE1_SAVE_EVERY" \
    --gpu "$GPU" \
    --lr_g "$PHASE1_LR_G" \
    --lr_d "$PHASE1_LR_D" \
    --optimizer "$OPTIMIZER" \
    --spectral_loss "$SPECTRAL_LOSS" \
    --adversarial_loss "$ADVERSARIAL_LOSS" \
    $FP16_FLAG \
    --use_tf32

echo ""
echo "✓ Phase 1 complete!"

# ─── Phase 2 ─────────────────────────────────────────────────────

# Find Phase 1 checkpoints
PHASE1_G=$(ls -t "${EXPERIMENT_DIR}"/G_phase1_*.pth 2>/dev/null | head -1 || true)
PHASE1_D=$(ls -t "${EXPERIMENT_DIR}"/D_phase1_*.pth 2>/dev/null | head -1 || true)

echo ""
echo "▶ PHASE 2: Full VITS pretrain (@ ${PHASE2_SR}Hz)"
echo "  Epochs: ${PHASE2_EPOCHS}, LR: ${PHASE2_LR_G}/${PHASE2_LR_D}"
echo "  KL anneal: ${KL_ANNEAL_STEPS} steps, free bits: ${KL_FREE_BITS}"
echo "  Decoder freeze: ${DECODER_FREEZE_STEPS} steps"
if [ -n "$PHASE1_G" ]; then
    echo "  Phase 1 G checkpoint: $PHASE1_G"
fi
echo ""

PHASE2_CMD="$PYTHON $PRETRAIN_SCRIPT \
    --phase 2 \
    --model_name $MODEL_NAME \
    --vocoder $VOCODER \
    --architecture $ARCHITECTURE \
    --sample_rate $PHASE2_SR \
    --batch_size $BATCH_SIZE \
    --total_epochs $PHASE2_EPOCHS \
    --save_every $PHASE2_SAVE_EVERY \
    --gpu $GPU \
    --lr_g $PHASE2_LR_G \
    --lr_d $PHASE2_LR_D \
    --optimizer $OPTIMIZER \
    --spectral_loss \"$SPECTRAL_LOSS\" \
    --adversarial_loss $ADVERSARIAL_LOSS \
    --kl_anneal_steps $KL_ANNEAL_STEPS \
    --kl_free_bits $KL_FREE_BITS \
    --decoder_freeze_steps $DECODER_FREEZE_STEPS \
    $FP16_FLAG \
    --use_tf32"

if [ -n "$PHASE1_G" ]; then
    PHASE2_CMD="$PHASE2_CMD --phase1_ckpt_g $PHASE1_G"
fi
if [ -n "$PHASE1_D" ]; then
    PHASE2_CMD="$PHASE2_CMD --phase1_ckpt_d $PHASE1_D"
fi

eval "$PHASE2_CMD"

echo ""
echo "✓ Phase 2 complete!"

# ─── Phase 3 ─────────────────────────────────────────────────────

PHASE2_G=$(ls -t "${EXPERIMENT_DIR}"/G_phase2_*.pth 2>/dev/null | head -1 || true)
PHASE2_D=$(ls -t "${EXPERIMENT_DIR}"/D_phase2_*.pth 2>/dev/null | head -1 || true)

for TARGET_SR in $PHASE3_TARGET_SRS; do
    echo ""
    echo "▶ PHASE 3: SR adaptation → ${TARGET_SR}Hz"
    echo "  Epochs: ${PHASE3_EPOCHS}, LR: ${PHASE3_LR_G}/${PHASE3_LR_D}"
    if [ -n "$PHASE2_G" ]; then
        echo "  Phase 2 G checkpoint: $PHASE2_G"
    fi
    echo ""

    PHASE3_MODEL="${MODEL_NAME}_${TARGET_SR}"

    PHASE3_CMD="$PYTHON $PRETRAIN_SCRIPT \
        --phase 3 \
        --model_name $PHASE3_MODEL \
        --vocoder $VOCODER \
        --architecture $ARCHITECTURE \
        --sample_rate $TARGET_SR \
        --batch_size $BATCH_SIZE \
        --total_epochs $PHASE3_EPOCHS \
        --save_every $PHASE3_SAVE_EVERY \
        --gpu $GPU \
        --lr_g $PHASE3_LR_G \
        --lr_d $PHASE3_LR_D \
        --optimizer $OPTIMIZER \
        --spectral_loss \"$SPECTRAL_LOSS\" \
        --adversarial_loss $ADVERSARIAL_LOSS \
        $FP16_FLAG \
        --use_tf32"

    if [ -n "$PHASE2_G" ]; then
        PHASE3_CMD="$PHASE3_CMD --phase2_ckpt_g $PHASE2_G"
    fi
    if [ -n "$PHASE2_D" ]; then
        PHASE3_CMD="$PHASE3_CMD --phase2_ckpt_d $PHASE2_D"
    fi

    eval "$PHASE3_CMD"

    echo ""
    echo "✓ Phase 3 (${TARGET_SR}Hz) complete!"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All phases completed successfully!"
echo "  Checkpoints saved in: ${EXPERIMENT_DIR}"
echo "═══════════════════════════════════════════════════════════"

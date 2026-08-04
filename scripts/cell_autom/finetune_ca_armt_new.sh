#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=1
NP=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export NCCL_ASYNC_ERROR_HANDLING=0
export WANDB_PROJECT=${WANDB_PROJECT:-cellular_automata}
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export CUDA_LAUNCH_BLOCKING=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_ROOT"

TASK_NAME=CA
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:GPTNeoXForCausalLM
DATASET_PATH=${DATASET_PATH:-irodkin/1dCA_r2s20T20}

ITERS=40000
TBS=256
MAX_N_SEGMENTS=10
MAX_VAL_SEGMENTS=10
SHIFT=2
LR=3e-4
BS=256
MEMORY_SIZE=16
INPUT_TOKENS=20
D_MEM=32
DIM=128
NUM_LAYERS=4
RUN_ID=51
SCHEDULER=linear
SEGMENT_ORDERING=regular
K2=-1

MODEL_NAME=armt_new_neox_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}
MODEL_CFG="$REPO_ROOT/base_models/gptconfigs/neox_tiny_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}.json"
if [[ ! -f "$MODEL_CFG" ]]; then
    (cd base_models/gptconfigs && python create_config.py \
        --hidden_size "$DIM" \
        --num_hidden_layers "$NUM_LAYERS" \
        --num_attention_heads "$NUM_LAYERS")
fi

INPUT_SIZE=$INPUT_TOKENS
INPUT_SEQ_LEN=$((INPUT_SIZE * MAX_N_SEGMENTS))
MODEL_CPT=None

echo "Running new ARMT: shift=$SHIFT gpu=$CUDA_VISIBLE_DEVICES run=$RUN_ID"

export WANDB_NAME=armt_new_s${SHIFT}_run${RUN_ID}
accelerate launch \
    --num_processes "$NP" \
    --config_file ./accelerate.yaml \
    --main_process_port "$((29500 + RUN_ID))" \
    run_finetuning_cell_autom.py \
    --task_name "$TASK_NAME" \
    --model_path "../runs/lm_long/armt/${TASK_NAME}/${MODEL_NAME}/lr${LR}_${SCHEDULER}_dmem${D_MEM}_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}_shift${SHIFT}/run_${RUN_ID}" \
    --model_cfg "$MODEL_CFG" \
    --dataset_path "$DATASET_PATH" \
    --model_type "$MODEL_TYPE" \
    --memory_cell_cls "$MEMORY_CELL" \
    --recurrent_wrapper_cls "$RECURRENT_WRAPPER" \
    --model_cls "$BACKBONE_CLS" \
    --model_cpt "$MODEL_CPT" \
    --segment_size "$INPUT_TOKENS" \
    --input_size "$INPUT_SIZE" \
    --max_n_segments "$MAX_N_SEGMENTS" \
    --num_mem_tokens "$MEMORY_SIZE" \
    --num_timesteps "$MAX_N_SEGMENTS" \
    --num_test_timesteps "$MAX_VAL_SEGMENTS" \
    --prediction_shift "$SHIFT" \
    --optimize_metric exact_match --optimize_mode max \
    --batch_size "$BS" \
    --gradient_accumulation_steps "$((TBS / BS / NP))" \
    --iters "$ITERS" \
    --num_training_steps "$((ITERS * 2))" \
    --optimizer AdamW --weight_decay 0.01 \
    --lr "$LR" --lr_scheduler "$SCHEDULER" --num_warmup_steps 1000 \
    --data_n_workers 2 \
    --log_interval 50 --valid_interval 250 \
    --show_valid_examples 5 \
    --early_stopping_patience 30 \
    --seed "$((RUN_ID + 42 * (SHIFT - 1)))" \
    --clip_grad_value 0.1 \
    --save_best \
    --d_mem "$D_MEM" \
    --layers_attr gpt_neox.layers \
    --repeat_state \
    --armt_impl new

echo "done"

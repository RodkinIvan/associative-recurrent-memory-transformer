#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
export NCCL_ASYNC_ERROR_HANDLING=0
# set -e
cd ../..
export WANDB_PROJECT=groupmul
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
TASK_NAME=CA
MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:GPTNeoXForCausalLM

DATASET_PATH=XXXX/groupmul_A5_split

ITERS=40000
TBS=512

MAX_N_SEGMENTSS=(8 10)
LENGTHS=(15 20)
LR=3e-4
BSS=(64 64)

MEMORY_SIZE=4
INPUT_TOKENS=2
D_MEM=32

ACT_TYPE=layer
MAX_HOP=4

DIM=512
NUM_LAYERS=2
N_ATTN_HEADS=8

cd base_models/gptconfigs
python create_config.py --hidden_size $DIM --num_hidden_layers $NUM_LAYERS --num_attention_heads $N_ATTN_HEADS
cd ../..
MODEL_CFG=~/rmt/wip/base_models/gptconfigs/neox_tiny_${NUM_LAYERS}l${N_ATTN_HEADS}hd${DIM}.json


START_CPT=../runs/lm_long/gpt_neox/CA//lr3e-4_linear_dmem32_-5x_mem4_bs512_iters40000_regular_bptt--1_actlayer_length10/run_25


for N in 25
do

NEW_CPT=$START_CPT

for (( j=0; j<${#LENGTHS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}

# LR_=${LRS[j]}
LR_=${LR}
LENGTH=${LENGTHS[j]}

BS=${BSS[j]}
K2=-1
for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in $LR_
do


MODEL_CPT=$NEW_CPT

NEW_CPT=../runs/lm_long/gpt_neox/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_dmem${D_MEM}_-${MAX_N_SEGMENTS}x_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}_act${ACT_TYPE}_length${LENGTH}/run_$N
echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $BACKBONE_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port $((29500 + $N)) run_finetuning_groupmul.py \
        --model_path $NEW_CPT \
        --model_cfg $MODEL_CFG \
        --dataset_path $DATASET_PATH \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt $MODEL_CPT \
        --segment_size $INPUT_TOKENS \
        --max_n_segments $MAX_N_SEGMENTS \
        --num_mem_tokens $MEMORY_SIZE \
        --optimize_metric bit_accuracy --optimize_mode max \
        --batch_size $BS \
        --gradient_accumulation_steps $(($TBS/$BS/$NP)) \
        --iters $ITERS \
        --num_training_steps $(($ITERS*2))\
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --show_valid_examples 5 \
        --early_stopping_patience 30 \
        --seed $(($N+42)) \
        --clip_grad_value 0.5 \
        --save_best \
        --d_mem $D_MEM \
        --layers_attr gpt_neox.layers \
        --length $LENGTH
        # --act_on \
        # --max_hop $MAX_HOP \
        # --time_penalty 3e-4 \
        # --act_type $ACT_TYPE \
done
done
done
done
done
echo "done"


#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
# export WANDB_PROJECT=t5-experiments
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}') # ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=wikitext-103-v1

ITERS=36000
TBS=32

MAX_N_SEGMENTSS=(8)
MAX_VAL_SEGMENTSS=(16)
MEMORY_SIZES=(16)
INPUT_TOKENS=128
LRS=(1e-5)
MODEL=meta-llama/Llama-3.2-1B
BSS=(1)

D_MEM=64
N_HEADS=1



for N in 4
do

for MODEL_NAME in $MODEL
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
MAX_VAL_SEGMENTS=${MAX_VAL_SEGMENTSS[j]}

INPUT_SIZE=$(($INPUT_TOKENS))
INPUT_SEQ_LEN=$(((INPUT_SIZE)*MAX_N_SEGMENTS))
TGT_LEN=$INPUT_SEQ_LEN
LR_=${LRS[j]}
VAL_SEQ_LEN=$(((INPUT_SIZE)*MAX_VAL_SEGMENTS))
ALPHA=${ALPHAS[j]}

BS=${BSS[j]}

GRAD_ACC_STEPS=$(($TBS/$BS/$NP))

K2=8
for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in $LR_
do

if [[ j -gt 0 ]]
then
    PREV_SEQ_LEN=$(((INPUT_SIZE)*${MAX_N_SEGMENTSS[j-1]}))
    MODEL_CPT=../runs/lm_long/amt/${TASK_NAME}/$MODEL_NAME/lr${LRS[j-1]}_${SCHEDULER}_alpha${ALPHAS[j-1]}_dmem${D_MEM}_${PREV_SEQ_LEN}-${MAX_N_SEGMENTSS[j-1]}x${INPUT_SIZE}_mem${MEMORY_SIZES[j-1]}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N 
else
    MODEL_CPT=None
fi
# cd accel_configs/
# python create_config.py \
#         --bf16 \
#         --train_batch_size $TBS\
#         --train_micro_batch_size_per_gpu $BS\
#         --gradient_accumulation_steps $GRAD_ACC_STEPS\
#         --np $NP\
#         --gradient_clipping 1.0
# cd ..
# ACCEL_CONFIG=~/rmt/wip/accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
ACCEL_CONFIG=./accel_configs/accelerate_bf16.yaml

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file  $ACCEL_CONFIG --mixed_precision bf16 --main_process_port $(($N + 29500)) run_finetuning_lm_rmt_distil.py \
        --task_name $TASK_NAME \
        --model_path ../runs/lm_long/amt/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_alpha${ALPHA}_dmem${D_MEM}_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt $MODEL_CPT \
        --input_seq_len $INPUT_SEQ_LEN \
        --block_size $INPUT_TOKENS \
        --val_seq_len $VAL_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --max_val_segments $MAX_VAL_SEGMENTS\
        --batch_size $BS \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --iters $ITERS \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 10 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42*$j)) \
        --clip_grad_norm 1.0 \
        --save_best \
        --d_mem $D_MEM \
        --n_heads $N_HEADS \
        --layers_attr model.layers \
        --prev_seg_kv \
        --use_sink
done
done
done
done
done
done
echo "done"


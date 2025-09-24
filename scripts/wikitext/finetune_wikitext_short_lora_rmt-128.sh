#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
NP=4 # ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=wikitext-2-v1

ITERS=6000
TBS=32


MAX_N_SEGMENTSS=(2)
MAX_VAL_SEGMENTSS=(15 15 15 15 15)
MEMORY_SIZES=(4 4 4 4 4)
INPUT_TOKENS=512
LRS=(1e-4 5e-5 3e-5 2e-5 1e-5)

BSS=(2 2 1 1 1)

for N in 1
do

for MODEL_NAME in XXXX/gpt2-wiki2
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
MAX_VAL_SEGMENTS=${MAX_VAL_SEGMENTSS[j]}

INPUT_SIZE=$(($INPUT_TOKENS+2*$MEMORY_SIZE))
INPUT_SEQ_LEN=$(((INPUT_SIZE-2*MEMORY_SIZE)*MAX_N_SEGMENTS))
TGT_LEN=$INPUT_SEQ_LEN
LR_=${LRS[j]}
VAL_SEQ_LEN=$(((INPUT_SIZE-2*MEMORY_SIZE)*MAX_VAL_SEGMENTS))

BS=${BSS[j]}
K2=8
for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in $LR_
do

if [[ j -gt 0 ]]
then
    PREV_SEQ_LEN=$(((INPUT_SIZE-2*MEMORY_SIZE)*${MAX_N_SEGMENTSS[j-1]}))
    MODEL_CPT=../runs/lm_long/${TASK_NAME}/$MODEL_NAME/lr${LRS[j-1]}_${SCHEDULER}_adamw_wd1e-03_${PREV_SEQ_LEN}-${MAX_N_SEGMENTSS[j-1]}x${INPUT_SIZE}_mem${MEMORY_SIZES[j-1]}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}_lora_freeze/run_$N 
else
    MODEL_CPT=None
fi

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file ./accelerate.yaml --main_process_port 29502 run_finetuning_lm_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/lm_long/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}_lora_freeze/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt $MODEL_CPT \
        --input_seq_len $INPUT_SEQ_LEN \
        --val_seq_len $VAL_SEQ_LEN \
        --block_size $INPUT_TOKENS \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --max_val_segments $MAX_VAL_SEGMENTS\
        --batch_size $BS \
        --gradient_accumulation_steps $(($TBS/$BS/$NP)) \
        --iters $ITERS \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 500 \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42*$j)) \
        --clip_grad_value 5.0 \
        --save_best \
        --tokenizer 'gpt2'    
done
done
done
done
done
done
echo "done"

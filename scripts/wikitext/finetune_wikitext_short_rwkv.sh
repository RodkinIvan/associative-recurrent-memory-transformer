#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1
export RWKV_NO_CUDA=1
export RWKV_JIT_ON=0
export CUDA_LAUNCH_BLOCKING=1
export CUBLAS_WORKSPACE_CONFIG=:4096:2
export TORCH_USE_CUDA_DSA=1
export WANDB_PROJECT=t5-experiments
NP=2
set -e
cd ../..

MODEL_TYPE=decoder
MEMORY_CELL=baselines.rwkv.language_modeling:MemoryCell
RECURRENT_WRAPPER=baselines.rwkv.language_modeling:RecurrentWrapper
BACKBONE_CLS=baselines.rwkv.language_modeling:RWKV_v6
TASK_NAME=wikitext-103-v1

ITERS=10000
TBS=64

MAX_N_SEGMENTSS=(1)
MAX_VAL_SEGMENTSS=(2)
INPUT_TOKENS=1024
# MAX_N_SEGMENTSS=(1)
# MAX_VAL_SEGMENTSS=(1)
# INPUT_TOKENS=1024
LRS=(1e-4)
MODEL=trash
TOKENIZER=EleutherAI/pythia-160m
BSS=(8)



for N in 2
do

for MODEL_NAME in $MODEL
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
MAX_VAL_SEGMENTS=${MAX_VAL_SEGMENTSS[j]}

INPUT_SIZE=$(($INPUT_TOKENS))
INPUT_SEQ_LEN=$(((INPUT_SIZE)*MAX_N_SEGMENTS))
TGT_LEN=$INPUT_SEQ_LEN
LR_=${LRS[j]}
VAL_SEQ_LEN=$(((INPUT_SIZE)*MAX_VAL_SEGMENTS))

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
    PREV_SEQ_LEN=$(((INPUT_SIZE)*${MAX_N_SEGMENTSS[j-1]}))
    MODEL_CPT=../runs/lm_long/rwkv/${TASK_NAME}/$MODEL_NAME/lr${LRS[j-1]}_${SCHEDULER}_alpha${ALPHAS[j-1]}_${PREV_SEQ_LEN}-${MAX_N_SEGMENTSS[j-1]}x${INPUT_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N 
else
    MODEL_CPT=None
fi

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port 29501 run_finetuning_lm_rmt_distil.py \
        --task_name $TASK_NAME \
        --model_path ../runs/lm_long/rwkv/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_alpha${ALPHA}_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N \
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
        --max_n_segments $MAX_N_SEGMENTS\
        --max_val_segments $MAX_VAL_SEGMENTS\
        --batch_size $BS \
        --gradient_accumulation_steps $(($TBS/$BS/$NP)) \
        --iters $ITERS \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 1000 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42*$j)) \
        --clip_grad_value 1.0 \
        --save_best \
        --tokenizer $TOKENIZER
        # --tokenized_dataset irodkin/wikitext-103-raw-v1-rwkv-v5-tokenized
done
done
done
done
done
done
echo "done"


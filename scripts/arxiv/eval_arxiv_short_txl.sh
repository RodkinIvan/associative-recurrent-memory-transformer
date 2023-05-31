#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MODEL_CLS=modeling_rmt.language_modeling:RMTDecoderXLCache
BACKBONE_CLS=transformers:AutoModelForCausalLM
TASK_NAME=arxiv

ITERS=250000
TBS=128

TGT_LEN=128
INPUT_SIZE=128

MAX_N_SEGMENTSS=(1 2 3 4 5 6 7 8 9 10 16 32 64)
BSS=(128 64 64 32 32 32 16 16 16 16 8 4)

MAX_N_SEGMENTSS=(4 5 6 7 8 9 10 16 32 64)
BSS=(32 32 32 16 16 16 16 8 4)

for N in 1
do

MODEL_NAME=gpt2

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MEMORY_SIZE=0
XL_CACHE_SIZE=64
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
INPUT_SEQ_LEN=$(((INPUT_SIZE-2*MEMORY_SIZE-XL_CACHE_SIZE)*MAX_N_SEGMENTS))
BS=${BSS[j]}

SOURCE_N_SEGMENTS=5
K2=${SOURCE_N_SEGMENTS}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in 1e-05
do

echo RUNNING: TASK_NAME MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR
horovodrun --gloo -np $NP python run_finetuning_arxiv_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_xl${XL_CACHE_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_from_cpt_${SOURCE_N_SEGMENTS}-${MAX_N_SEGMENTS}_eval/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_cpt ../runs/${TASK_NAME}/gpt2/linear_adamw_wd1e-03_320-128-5x128_mem0_xl64_bs32_regular_bptt-5_from_cpt_4-5/run_1 \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --xl_cache_size $XL_CACHE_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --validate_only \
        --iters $ITERS \
        --k1 -1 --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
done
done
echo "done"
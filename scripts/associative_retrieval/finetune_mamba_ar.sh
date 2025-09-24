#!/usr/bin/env bash
export WANDB_PROJECT=associative_retrieval
export CUDA_VISIBLE_DEVICES=4,5,6,7
NP=4
cd ../..
set -e
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=baselines.mamba.language_modeling:MemoryCell
RECURRENT_WRAPPER=baselines.mamba.language_modeling:RecurrentWrapper
BACKBONE_CLS=baselines.mamba.language_modeling:Mamba_tiny
TASK_NAME=associative_retrieval_v3
METRIC=exact_match


MODEL_NAME=gpt-neox

TBS=512
INPUT_SIZE=2048

NUMS_PAIRS=(1 2 3 5 10 20 40 50 50 200)
# NUMS_PAIRS=(200)

KEY_SIZES=(1 1 1 1 1 2 2 2 3 3)
# KEY_SIZES=(3)

VALUE_SIZES=(1 1 1 1 1 1 1 1 1 1)

BSS=(128 128 128 128 128 128 64 64 64 16)
# BSS=(16)

ITERSS=(2000 10000 10000 10000 10000 10000 10000 10000 10000 30000)
# ITERSS=(30000)

# ITERSS=(1 1 1 1 1 1 1 1)

DIM=128
NUM_LAYERS=4


for N in 4
do

for (( j=0; j<${#NUMS_PAIRS[@]}; j++ ))
do
NUM_PAIRS=${NUMS_PAIRS[j]}
KEY_SIZE=${KEY_SIZES[j]}
VALUE_SIZE=${VALUE_SIZES[j]}
MAX_N_SEGMENTS=$((NUM_PAIRS + 1))
BS=${BSS[j]}
ITERS=${ITERSS[j]}


BLOCK_SIZE=$((KEY_SIZE + VALUE_SIZE + 2))
for LR in 3e-04
do

K2=${MAX_N_SEGMENTS}

for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

if [[ j -gt 0 ]]
then
    PREV_NUM_PAIRS=${NUMS_PAIRS[j-1]}
    PREV_MAX_N_SEGMENTS=$((PREV_NUM_PAIRS + 1))
    MODEL_CPT=../runs/${TASK_NAME}/mamba/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZES[j-1]}-v${VALUE_SIZES[j-1]}-p${PREV_NUM_PAIRS}-${PREV_MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${PREV_MAX_N_SEGMENTS}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N 
else
    MODEL_CPT=None
fi

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
ACCEL_CONFIG=./accel_configs/accelerate.yaml

echo gradient accumulation steps $GRAD_ACC_STEPS

echo RUNNING: TASK_NAME MEMORY_SIZE KEY_SIZE VALUE_SIZE N_SEG  MODEL_NAME MODEL_CLS LR N
echo RUNNING: $TASK_NAME $MEMORY_SIZE $KEY_SIZE $VALUE_SIZE $MAX_N_SEGMENTS $MODEL_NAME $MODEL_CLS  $LR $N
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29571 run_finetuning_associative_retrieval.py \
        --task_name $TASK_NAME \
        --model_path ../runs/${TASK_NAME}/mamba/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_k${KEY_SIZE}-v${VALUE_SIZE}-p${NUM_PAIRS}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_${SEGMENT_ORDERING}_bptt-${K2}_${NUM_LAYERS}l${NUM_LAYERS}hd${DIM}/run_$N \
        --model_cls $BACKBONE_CLS \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --segment_size $BLOCK_SIZE \
        --key_size $KEY_SIZE \
        --value_size $VALUE_SIZE \
        --num_pairs $NUM_PAIRS \
        --max_n_segments $MAX_N_SEGMENTS\
        --use_generate_on_valid \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --num_training_steps $((ITERS*2)) \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 30 \
        --seed $(($N+42)) \
        --clip_grad_value 1.0 \
        --dataset_path /home/XXXX/armt/datasets/associative_retrieval \
        --train_size 1000000 \
        --valid_size 1000 \
        --test_size 10000 \
        --model_cpt $MODEL_CPT \
        --save_best \
        --vary_n_segments \
        --from_pretrained trash
done
done
done
done
done
echo "done"
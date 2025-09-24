#!/usr/bin/env bash

export WANDB_PROJECT=babilong
export CUDA_VISIBLE_DEVICES=4,5,6,7
NP=4
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=baselines.mamba.language_modeling:MemoryCell
RECURRENT_WRAPPER=baselines.mamba.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:MambaForCausalLM
# TASK_DATASET=qa1_single-supporting-fact
# TASK_DATASET=qa2_two-supporting-facts
TASK_DATASET=qa3_three-supporting-facts
NOISE_DATASET=pg19
METRIC=exact_match


MODEL_NAME=state-spaces/mamba-130m-hf  # backbone model
SEGMENT_SIZE=512 # size of one segment in tokens
TBS=128

MAX_N_SEGMENTSS=(2 3 5 8 16 32)
TEST_N_SEGMENTSS=(2 3 5 8 16 128)
ITERSS=(2000 10000 10000 10000 10000 30000)
# ITERSS=(1 1 1 1 1)
BSS=(8 4 2 2 1 1)


for (( j=4; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do

BS=${BSS[j]}
ITERS=${ITERSS[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
TEST_N_SEGMENTS=${TEST_N_SEGMENTSS[j]}

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
LR=3e-4



for N in 2
do

K2=-1   # BPTT unroll length

SAMPLE_SIZE=$(($SEGMENT_SIZE*$MAX_N_SEGMENTS)) # length of task sample in tokens
MAX_N_FACTS=$(($SAMPLE_SIZE/10))
TEST_SAMPLE_SIZE=$(($SEGMENT_SIZE*$TEST_N_SEGMENTS))
ACCEL_CONFIG=./accel_configs/accelerate.yaml


if [[ j -gt 0 ]]
then
    MODEL_CPT=/home/XXXX/armt/runs/babilong/${TASK_DATASET}/mamba/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTSS[j-1]}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}/run_$N 
else
    MODEL_CPT=None
fi

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29712 run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/XXXX/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
        --model_path  /home/XXXX/armt/runs/babilong/${TASK_DATASET}/mamba/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 2 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $((ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0 \
        --model_cpt $MODEL_CPT \
        --use_generate_on_valid \
	    --test_sample_size $TEST_SAMPLE_SIZE \
        --vary_n_segments \
        --max_n_facts $MAX_N_FACTS
done
done
echo "done"

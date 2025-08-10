#!/usr/bin/env bash

export WANDB_PROJECT=babilong
export CUDA_VISIBLE_DEVICES=0
export RWKV_JIT_ON=0
export RWKV_NO_CUDA=1
export CHUNK_LEN=1
NP=1
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=baselines.rwkv.language_modeling:RWKV_v6

TASK_DATASET=qa2_two-supporting-facts
NOISE_DATASET=pg19
METRIC=exact_match


MODEL_NAME=~/lab/rwkv-x060-173m-pile-20240515-ctx4k.pth
TOKENIZER=EleutherAI/pythia-160m  # backbone model
SEGMENT_SIZE=512 # size of one segment in tokens
TBS=64

MEMORY_SIZE=10
D_MEM=64

MAX_N_SEGMENTSS=(2 3 5 8 16 32)
MAX_VAL_SEGMENTSS=(2 3 5 8 16 128)
ITERSS=(5000 10000 10000 10000 10000 10000)
# ITERSS=(1)
BSS=(32 32 32 32 32 32)


for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do

BS=${BSS[j]}
ITERS=${ITERSS[j]}
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
MAX_VAL_SEGMENTS=${MAX_VAL_SEGMENTSS[j]}

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear
LR=1e-04



for N in 2
do

K2=-1   # BPTT unroll length

SAMPLE_SIZE=$(($SEGMENT_SIZE*$MAX_N_SEGMENTS)) # length of task sample in tokens
TEST_SAMPLE_SIZE=$(($SEGMENT_SIZE*$MAX_VAL_SEGMENTS))
# ACCEL_CONFIG=./accel_configs/deepspeed.yaml

cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0
cd ..
ACCEL_CONFIG=~/rmt/wip/accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
# ACCEL_CONFIG=./accel_configs/deepspeed.yaml

if [[ j -gt 0 ]]
then
    MODEL_CPT=~/runs/babilong/${TASK_DATASET}/rwkv_armt/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-02_${MAX_N_SEGMENTSS[j-1]}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}/run_$N 
else
    MODEL_CPT=None
fi

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE 
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29702 --mixed_precision bf16 --num_processes $NP run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path ~/lab/associative-recurrent-memory-transformer/data/tasks_1-20_v1-2/en-10k \
        --model_path ~/runs/babilong/${TASK_DATASET}/rwkv_armt/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-02_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --test_sample_size $TEST_SAMPLE_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --batch_size $BS --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --save_best \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0 \
        --model_cpt $MODEL_CPT \
        --tokenizer $TOKENIZER \
        --d_mem $D_MEM \
        --layers_attr model.blocks \
        --num_mem_tokens $MEMORY_SIZE \
        --no_denom \
        --grad_cp
done
done
echo "done"

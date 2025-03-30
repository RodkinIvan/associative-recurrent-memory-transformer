export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_BLOCKING_WAIT=0
export WANDB_PROJECT=llm_pretrain
NP=4
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_amt.language_modeling:AssociativeMemoryCell
RECURRENT_WRAPPER=modeling_amt.language_modeling:AssociativeRecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
DATASET_NAME=pg19

MODEL_NAME=meta-llama/Llama-3.2-1B
MODEL_PATH=$MODEL_NAME

TOKENIZED_DATASET=~/rmt/datasets/pg19/pg19_tokenized

ITERS=50000
TBS=256
# TBS=32
BS=2

LR=1e-05
SEGMENT_SIZE=512
MAX_N_SEGMENTS=4
MEMORY_SIZE=32
D_MEM=64
LAYERS_ATTR=model.layers

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear

for N in 1
do

K2=-1   # BPTT unroll length

MODEL_CPT=../runs/pg19/meta-llama/Llama-3.2-1B/linear_adamw_wd1e-03_4x512_mem32_bs256_bptt--1_nfs_dmem64/run_1/checkpoint-7000/pytorch_model.bin

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
# ACCEL_CONFIG=~/rmt/dev/accel_configs/accelerate_ds_bf16.yaml
# DEEPSPEED_CONFIG=~/rmt/dev/accel_configs/deepspeed_bf16.json

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

# python run_finetuning_lm_rmt.py \
accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29002 --num_processes $NP run_finetuning_lm_rmt_hf.py \
        --tokenized_dataset $TOKENIZED_DATASET \
        --output_dir ../runs/${DATASET_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_nfs_dmem${D_MEM}/run_$N \
        --from_pretrained $MODEL_PATH \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --val_sample_size $SAMPLE_SIZE \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --min_sample_len 16000 \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better false \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 25 --eval_steps 100 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --d_mem $D_MEM \
        --layers_attr $LAYERS_ATTR \
        --no_loss_from_first_segment \
        --valid_tokens tokens \
        --train_tokens tokens \
        --attend_to_previous_input \
        --model_cpt $MODEL_CPT
        
done
echo "done"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,COLL
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# Flight recorder for stack traces PyTorch mentions
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576   # 1MB per rank is fine to start

# Write per-rank logs (optional but helpful)
export NCCL_DEBUG_FILE=/tmp/nccl_rank_%r.log

export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_BLOCKING_WAIT=0
export WANDB_PROJECT=llm_pretrain
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
BACKBONE_CLS=transformers:AutoModelForCausalLM


# DATASET_NAME=BramVanroy/CommonCrawl-CreativeCommons
# DATASET_NAME=deepmind/pg19
DATASET_NAME=karpathy/fineweb-edu-100b-shuffle
# VALID_DATASET_NAME=deepmind/pg19

MODEL_NAME=google/gemma-3-1b-it
# MODEL_NAME=meta-llama/Llama-3.2-1B
MODEL_PATH=$MODEL_NAME


ITERS=50000
TBS=64
# TBS=32
BS=1

LR=1e-5
SEGMENT_SIZE=1024
MAX_N_SEGMENTS=8
MEMORY_SIZE=32
D_MEM=64
LAYERS_ATTR=model.layers

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
SCHEDULER=linear

for N in 34
do

cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0\
        --stage 3
cd ..
ACCEL_CONFIG=$(pwd)/accel_configs/exp/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml # DEEPSPEED
DEEPSPEED_CONFIG=$(pwd)/accel_configs/exp/deepspeed/0s3_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0.json # DEEPSPEED


# ACCEL_CONFIG=~/rmt/dev/accel_configs/accelerate_ds_bf16.yaml
# DEEPSPEED_CONFIG=~/rmt/dev/accel_configs/deepspeed_bf16.json

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

export WANDB_NAME=armt_${DATASET_NAME}
# export NOT_INVERT_ATTN_MASK=1
# python run_finetuning_lm_rmt.py \
# export ARMT_DEBUG_NAN=1
accelerate launch --config_file $ACCEL_CONFIG --main_process_port $((29000+$N)) --num_processes $NP --mixed_precision bf16 run_finetuning_lm_rmt_hf_armt.py \
        --task_name $DATASET_NAME \
        --output_dir ../runs/${DATASET_NAME}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_hf_armt_dmem${D_MEM}/run_$N \
        --from_pretrained $MODEL_PATH \
        --model_type $MODEL_TYPE \
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
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --logging_steps 25 --eval_steps 100 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --d_mem $D_MEM \
        --layers_attr $LAYERS_ATTR \
        --valid_tokens tokens \
        --train_tokens tokens \
        --attn_implementation flash_attention_2 \
        --armt_impl inner  \
        --deepspeed $DEEPSPEED_CONFIG \
        --max_grad_norm 1.0 \
        --streaming --stream_chunk_docs 10000 \
        --model_dtype bfloat16 \
        --memory_dtype bfloat16 \
        --attn_implementation flash_attention_2
        # --tokenized_dataset /mnt/data/users/ivan.rodkin/lab/datasets/pg19_tokenized
done
echo "done"

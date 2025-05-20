#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
NP=1 # ./test_bert_sparse_pretrain_train_valid.sh
export NCCL_ASYNC_ERROR_HANDLING=0
set -e
cd ../..
CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
MODEL_TYPE=decoder
MEMORY_CELL=baselines.dummy.language_modeling:MemoryCell
RECURRENT_WRAPPER=baselines.dummy.language_modeling:RecurrentWrapper
BACKBONE_CLS=modeling_lstm.language_modeling:DoubleLSTMModel


# DATASET_NAME=ca
DATASET_NAME=addition_binary
# DATASET_NAME=reverse_binary
# DATASET_NAME=copy_binary

export WANDB_PROJECT=$DATASET_NAME
TASK_NAME=$DATASET_NAME

ITERS=30000
TBS=256

MAX_HOP=4
ACT_TYPE=layer
TIME_PENALTY=3e-4

MAX_N_SEGMENTSS=(2)
MAX_VAL_SEGMENTSS=(2)
SHIFTS=(1)
LRS=(1e-3)
BSS=(256)

INPUT_TOKENS=40

DIM=128
EMBED_DIM=128
NUM_LAYERS=4

cd base_models/configs/lstmconfigs
python create_config.py --hidden_size $DIM --num_layers $NUM_LAYERS --embedding_dim $EMBED_DIM
cd ../../..
pwd=$(pwd)
echo $pwd
MODEL_CFG=$pwd/base_models/configs/lstmconfigs/lstm_${NUM_LAYERS}ed${EMBED_DIM}hd${DIM}.json



for N in 7
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
SHIFT=${SHIFTS[j]}

BS=${BSS[j]}
K2=-1
for SEGMENT_ORDERING in regular
do

for SCHEDULER in linear
do

for LR in $LR_
do

# if [[ j -gt 0 ]]
# then
#     PREV_SEQ_LEN=$(((INPUT_SIZE)*${MAX_N_SEGMENTSS[j-1]}))
#     MODEL_CPT=../runs/lm_long/lstm/${TASK_NAME}/$MODEL_NAME/lr${LRS[j-1]}_${SCHEDULER}_dmem${D_MEM}_${PREV_SEQ_LEN}-${MAX_N_SEGMENTSS[j-1]}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N 
# else
#     MODEL_CPT=None
# fi
MODEL_CPT=None

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $BACKBONE_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port 29508 run_finetuning_gpt_neox.py \
        --task_name $TASK_NAME \
        --model_path ../runs/lm_long/lstm/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_dmem${D_MEM}_${INPUT_SEQ_LEN}-${MAX_N_SEGMENTS}x${INPUT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_bptt-${K2}/run_$N \
        --model_cfg $MODEL_CFG \
        --dataset_name $DATASET_NAME \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --model_cpt $MODEL_CPT \
        --segment_size $INPUT_TOKENS \
        --input_size $INPUT_SIZE \
        --num_timesteps $MAX_N_SEGMENTS \
        --num_test_timesteps $MAX_VAL_SEGMENTS \
        --prediction_shift $SHIFT \
        --optimize_metric exact_match --optimize_mode max \
        --batch_size $BS \
        --gradient_accumulation_steps $(($TBS/$BS/$NP)) \
        --iters $ITERS \
        --num_training_steps $(($ITERS*2))\
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 0 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 250 \
        --show_valid_examples 5 \
        --early_stopping_patience 15892245 \
        --seed $(($N+42*$j)) \
        --clip_grad_value 1.0 \
        --save_best \
        --act_on \
        --act_type $ACT_TYPE \
        --max_hop $MAX_HOP  \
        --time_penalty $TIME_PENALTY 

done
done
done
done
done
echo "done"


export CUDA_VISIBLE_DEVICES=1
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_CFG=./base_models/gptconfigs/neox_tiny

ITERS=10000
# MAX_LENGTH=512
LR=3e-4
BS=1024
N_GENS=256

BETA_KL=0

cd ../..

accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port 29501 train_grpo_gpt_neox.py \
    --model_cfg $MODEL_CFG \
    --iters $ITERS \
    --lr $LR \
    --batch_size $BS \
    --num_generations $N_GENS \
    --beta_kl $BETA_KL

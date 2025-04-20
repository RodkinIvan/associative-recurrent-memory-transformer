export CUDA_VISIBLE_DEVICES=1
export WANDB_PROJECT=grpo
export WANDB_NAME=gptneox

NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_CFG=./base_models/gptconfigs/neox_tiny
MODEL_CPT=./check_points/neox_supervised
MODEL_SAVE=./check_points/neox_grpo_run1

ITERS=10000
LR=3e-4
TBS=2048
N_GENS=8
BETA_KL=0
SHIFT=1
BS=1024
GRAD_ACC_STEPS=$(($TBS/$BS/$NP))
N=7

cd ../..

accelerate launch --num_processes $NP --config_file ./accelerate.yaml --main_process_port $((29500 + 24)) train_grpo_gpt_neox.py \
    --model_cfg $MODEL_CFG \
    --model_cls transformers:AutoModelForCausalLM \
    --from_pretrained $MODEL_CPT \
    --output_dir $MODEL_SAVE \
    --iters $ITERS \
    --lr $LR \
    --batch_size $BS \
    --num_generations $N_GENS \
    --beta_kl $BETA_KL \
    --prediction_shift $SHIFT \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --seed $(($N + 42))

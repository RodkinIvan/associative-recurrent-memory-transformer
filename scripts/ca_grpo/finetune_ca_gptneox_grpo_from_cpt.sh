export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT=grpo
export WANDB_NAME=gptneox
NP=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

MODEL_CFG=./base_models/gptconfigs/neox_tiny


# REASONER_CLS=modeling_reasoning.reasoning_wrapper:Reasoner
MODEL_CLS=transformers:AutoModelForCausalLM

ITERS=10000
# MAX_LENGTH=512
LR=3e-4
TBS=2048
N_GENS=8

BETA_KL=0.01
SHIFT=1

BS=1024

GRAD_ACC_STEPS=$(($TBS/$BS/$NP))

MODEL_CPT=../runs/lm_long/gpt_neox/CA//lr3e-4_linear_dmem1_10000-10x1000_mem1_bs256_iters40000_regular_bptt--1_act1/run_10
N=7
cd ../..

accelerate launch --num_processes $NP --config_file  ./accelerate.yaml --main_process_port $((29500 + $N)) train_grpo_gpt_neox.py \
    --model_cfg $MODEL_CFG \
    --model_cls $MODEL_CLS \
    --iters $ITERS \
    --lr $LR \
    --batch_size $BS \
    --num_generations $N_GENS \
    --beta_kl $BETA_KL \
    --prediction_shift $SHIFT \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --seed $(($N + 42)) \
    --model_cpt $MODEL_CPT
    # --reasoning

export CUDA_VISIBLE_DEVICES=0
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
SHIFTS=(2 3 4)

BS=1024

GRAD_ACC_STEPS=$(($TBS/$BS/$NP))


N=10

MODEL_CPT=../checkpoints/gptneox_s1/
cd ../..

for SHIFT in ${SHIFTS[@]}
do

N=$(($N + 1))

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
    --model_cpt $MODEL_CPT \
    --reasoning \
    --early_stopping_patience 5

done

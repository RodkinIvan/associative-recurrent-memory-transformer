import os
import torch
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
import argparse
from lm_experiments_tools.utils import get_cls_by_name
from transformers.integrations import WandbCallback
from transformers import TrainerCallback
import pandas as pd
import wandb
import accelerate
# Set environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--model_cfg', type=str, help='path to model configuration file')
parser.add_argument('--lr', type=float, help='learning_rate', default=1e-4)
parser.add_argument('--beta_kl', type=float, help='KL coefficient', default=0.0)
parser.add_argument('--iters', type=int, help='number of iterations', default=40000)
parser.add_argument('--batch_size', type=int, help='batch size', default=128)
parser.add_argument('--num_generations', type=int, help='num grpo generations', default=32)
parser.add_argument('--prediction_shift', type=int, help='t, so that we predict 10+t\'s state, in 1-indexing', default=1)
parser.add_argument('--model_cls', type=str, help='path to model class implementation')
# parser.add_argument('--reasoner_cls', type=str, help='path to reasoner class implementation')
parser.add_argument('--gradient_accumulation_steps', type=int, help='', default=1)
parser.add_argument('--max_length', type=int, help='maximum completion length', default=256)
parser.add_argument('--reasoning', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()


accelerator = accelerate.Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
from accelerate.logging import get_logger
logger = get_logger('')

# Load tokenizer and add special tokens
tokenizer = AutoTokenizer.from_pretrained(args.model_cfg, use_fast=True)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["<sep>", "<gen>"],
    "pad_token": "[PAD]",
    "unk_token": "[UNK]",
})

SEP_TOKEN = "<sep>"
GEN_TOKEN = "<gen>"
EOS_TOKEN = "[EOS]"

# Load datasets
train_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="train")
val_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="validation")
test_dataset = load_dataset("irodkin/1dCA_r2s20T20", split="test")

# Process: Insert <sep> between input states, <gen> before target
def process_sample(sample):
    input_chunks = [sample[f"input_ids_{i}"] for i in range(10)]
    prompt_ids = []
    for chunk in input_chunks:
        prompt_ids.append(SEP_TOKEN)
        prompt_ids.extend(chunk)

    prompt_str = " ".join(map(str, prompt_ids))
    t = 9 + args.prediction_shift
    target_ids = sample[f"input_ids_{t}"]
    target_str = " ".join(map(str, target_ids))
    is_eos = [False] * len(target_ids) + [True] if target_ids else []
    
    
    new_sample = {
        "prompt": f"{prompt_str} {GEN_TOKEN}",
        "target": f"{target_str} {EOS_TOKEN}",
        "is_eos": is_eos,
    }
    # print(new_sample)
    return new_sample

# Apply processing
print("*** Start processing dataset ***")
train_dataset = train_dataset.map(process_sample, batched=False, desc='Processing train')
val_dataset = val_dataset.map(process_sample, batched=False, desc='Processing eval')
test_dataset = test_dataset.map(process_sample, batched=False, desc='Processing test')

print("*** Done processing dataset ***")

# Reward: token-level accuracy
def reward_token_accuracy(completions, **kwargs):
    targets = kwargs.get("target")
    if targets is None:
        raise ValueError("Targets are required for reward calculation.")
    rewards = []
    for comp_str, tgt_str in zip(completions, targets):
        if args.reasoning:
            answer = comp_str.split("<sep>")[-1]
        else:
            answer = comp_str

        pred_tokens = answer.strip().split()
        tgt_tokens = tgt_str.strip().split()
        correct = sum(p == t for p, t in zip(pred_tokens, tgt_tokens))
        acc = correct / len(tgt_tokens) if tgt_tokens else 0.0
        rewards.append(acc)
    return rewards

# Load model config and model
print(f"*** Loading model of class {args.model_cls} ***")
config = AutoConfig.from_pretrained(args.model_cfg)

model_cls = get_cls_by_name(args.model_cls)
model = model_cls.from_config(config)
model.resize_token_embeddings(len(tokenizer))
print("*** Done loaing Model ***")

def predict(model, tokenizer, sample):
    """Predicts the model outputs for a given dataset.

    Args:
        model (nn.Module): The model to use for prediction.
        tokenizer (PreTrainedTokenizerFast): The tokenizer to use.
        dataset (Dataset): The dataset to predict on.

    Returns:
        list: The predicted outputs.
    """
    inputs = tokenizer(sample["prompt"], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_length, do_sample=False)
    return tokenizer.batch_decode(outputs)


_, train_dataset, val_dataset, test_dataset = accelerator.prepare(model, train_dataset, val_dataset, test_dataset)
# GRPO Training config
training_args = GRPOConfig(
    output_dir="optimized-GRPO",
    learning_rate=args.lr,
    logging_steps=10,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_generations=args.num_generations,
    max_steps=args.iters,
    fp16=True,
    beta=args.beta_kl,
    eval_strategy='steps',
    eval_steps=100,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    report_to='wandb',
    max_completion_length=args.max_length,
    skip_memory_metrics=True
)

# Trainer
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    reward_funcs=reward_token_accuracy,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

def get_trainer_wandb_run(trainer):
    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, WandbCallback):
            return callback._wandb
    return None

class WandbPredictionProgressCallback(TrainerCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, val_dataset, num_samples=10):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 10.
        """
        super().__init__()
        self.sample_dataset = val_dataset.select(range(num_samples))
    
    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        predictions = predict(
            trainer.model,
            tokenizer,
            self.sample_dataset,
        )
        # print(predictions)
        # decode predictions and labels
        # add predictions to a wandb.Table
        predictions_df = pd.DataFrame(predictions)
        predictions_df.columns = [str(c) for c in predictions_df.columns]
        predictions_df["epoch"] = state.epoch
        records_table = wandb.Table(dataframe=predictions_df)
        # log the table to wandb
        thinking_len = sum(
            len(pred[len(sample['prompt']):].split()) - len(pred[len(sample['prompt']):].split("<sep>")[-1].split()) for sample, pred in zip(self.sample_dataset, predictions)
        ) / len(self.sample_dataset)
        if accelerator.is_main_process:
            get_trainer_wandb_run(trainer).log({"sample_predictions": records_table, "thinking_len": thinking_len})

progress_callback = WandbPredictionProgressCallback(
    val_dataset=val_dataset,
    num_samples=10
)
class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


trainer.add_callback(progress_callback)
trainer.add_callback(EvaluateFirstStepCallback())

# Train
# trainer.evaluate(val_dataset)
trainer.train()

trainer.evaluate(test_dataset)

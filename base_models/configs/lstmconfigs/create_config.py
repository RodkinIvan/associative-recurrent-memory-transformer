import os
import json
import argparse

save_folder = "."

print(os.getcwd())
default_config = {
  "architectures": [
    "DoubleLSTMModel"
  ],
  "model_type": "gpt_neox",
  "embedding_dim": 8,
  "hidden_size": 128,
  "num_layers": 4,
  "lr": 0.001,
  "vocab_size": 128,
  "intermediate_size": 128, 
  "max_position_embeddings": 2048,
  "bos_token_id": 101,
  "eos_token_id": 102,
  "hidden_act": "gelu",
  "rotary_pct": 0.25,
  "rotary_emb_base": 10000,
  "attention_dropout": 0.0,
  "hidden_dropout": 0.0,
  "classifier_dropout": 0.1,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-5,
  "use_cache": True,
  "tie_word_embeddings": False,
  "use_parallel_residual": True,
  "act_on": False,
  "act_type": "layer",
  "max_hop": 4,
  "time_penalty": 3e-4,
  "constant_depth": False
}

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", default=128)
parser.add_argument("--num_layers", default=1)
parser.add_argument("--embedding_dim", default=8)


args = parser.parse_args()

config = dict(**default_config)
config['hidden_size'] = int(args.hidden_size)
config['num_layers'] = int(args.num_layers)
config['embedding_dim'] = int(args.embedding_dim)

config_name = f"lstm_{args.num_layers}ed{args.embedding_dim}hd{args.hidden_size}"

save_path = os.path.join(save_folder, f'{config_name}.json')
print(f'Saving config {config_name} \n {save_path}')
with open(save_path, 'w') as f:
    json.dump(config, f)
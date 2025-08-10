from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import os
import json

# Build vocabulary: "0" to "127"
vocab = {str(i): i for i in range(100)} | {str(i): i for i in range(105, 126)}

# Add special tokens
vocab["<sep>"] = 100
vocab["<gen>"] = 101
vocab["<eos>"] = 102
vocab["<rule>"] = 103
vocab["<mask>"] = 104
vocab["<pad>"] = 126
vocab["<unk>"] = 127
# Reverse vocab for saving
id_to_token = {v: k for k, v in vocab.items()}

# Create WordLevel tokenizer
tokenizer_model = models.WordLevel(vocab=vocab, unk_token="<unk>")
tokenizer = Tokenizer(tokenizer_model)

# No pre-tokenizer needed — you already have token IDs
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Wrap in HF tokenizer
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    sep_token="<sep>",
    pad_token="<pad>",
    eos_token="<eos>",
    rule_token="<rule>",
)

# Add <gen> as a regular vocabulary token (not special)
wrapped_tokenizer.add_tokens(["<gen>"])

# Save tokenizer properly
save_path = "../neox_tiny"
os.makedirs(save_path, exist_ok=True)
wrapped_tokenizer.save_pretrained(save_path)

# Save vocab.json manually for compatibility
with open(os.path.join(save_path, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

# Save tokenizer config
tokenizer_config = {
    "unk_token": "<unk>",
    "sep_token": "<sep>",
    "pad_token": "<pad>",
    "eos_token": "<eos>",
    "gen_token": "<gen>",
    "rule_token": "<rule>",
}
with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)

print("✅ Tokenizer saved to ../neox_tiny using stringified integer tokens")

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import os
import json

# Build vocabulary: "0" to "121"
vocab = {str(i): i for i in range(122)}

# Add special tokens
vocab["<gen>"] = 123
vocab["<sep>"] = 124
vocab["[UNK]"] = 125
vocab["[SEP]"] = 126
vocab["[PAD]"] = 127

# Reverse vocab for saving
id_to_token = {v: k for k, v in vocab.items()}

# Create WordLevel tokenizer
tokenizer_model = models.WordLevel(vocab=vocab, unk_token="[UNK]")
tokenizer = Tokenizer(tokenizer_model)

# No pre-tokenizer needed — you already have token IDs
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Wrap in HF tokenizer
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    sep_token="[SEP]",
    pad_token="[PAD]",
    additional_special_tokens=["<sep>", "<gen>"]
)

# Save tokenizer properly
save_path = "./neox_tiny"
os.makedirs(save_path, exist_ok=True)
wrapped_tokenizer.save_pretrained(save_path)

# Save vocab.json manually for compatibility
with open(os.path.join(save_path, "vocab.json"), "w") as f:
    json.dump(vocab, f, indent=2)

# Save tokenizer config
tokenizer_config = {
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "additional_special_tokens": ["<sep>", "<gen>"]
}
with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
    json.dump(tokenizer_config, f, indent=2)

print("✅ Tokenizer saved to ./neox_tiny using stringified integer tokens")

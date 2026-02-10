#!/usr/bin/env python3
"""
Simple test to verify alignment between generate and forward methods.
"""
import os

os.environ["ARMT_DISABLE_LIGER_KERNEL"] = "1"
import torch
from modeling_amt.inner_loop import InnerLoopARMTForCausalLM
from modeling_amt.inner_loop import ARMTConfig

device = torch.device("cuda:0")

def main():
    print("Loading inner-loop ARMT model...")
    
    # Create a simple config
    config = ARMTConfig()
    # config.base_model_name = 'meta-llama/Llama-3.2-1B'
    config.base_model_name = 'google/gemma-3-1b-it'
    config.num_mem_tokens = 4
    config.d_mem = 4
    config.segment_size = 5
    config.sliding_window = False
    config.use_sink = False
    
    # Create model
    model = InnerLoopARMTForCausalLM(config)
    for layer in model.get_layers():
        torch.nn.init.normal_(layer.W_mv.weight, mean=0.0, std=0.02)
    model.to(device)
    model.eval()
    
    # Simple test input
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0]}")
    print()
    
    n_generated_tokens = 15
    # Test 1: Generate first token
    print("=== Test 1: First Generated Token ===")
    generated_ids, gen_logits = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=n_generated_tokens, 
        return_logits=True
    )
    
    print(f"Generated token: {generated_ids[0]}")
    print(f"Generate method logits shape: {gen_logits.shape}")
    print(f"Generate method logits (first 5): {gen_logits[0, 0, :5]}")
    
    # Test 2: Forward pass on concatenated input (should match generate method)
    concat_input_ids = torch.cat([input_ids, generated_ids], dim=-1)
    concat_attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids)], dim=-1)
    
    print(f"\nConcatenated input: {concat_input_ids[0]}")
    
    with torch.no_grad():
        forward_output = model(
            input_ids=concat_input_ids,
            attention_mask=concat_attention_mask
        )
        forward_logits = forward_output.logits[:, -n_generated_tokens-1:-1, :]  # Last tokens
    
    print(f"Forward method logits (first 5): {forward_logits[0, 0, :5]}")
    
    # Check alignment
    print(f"\n=== Alignment Check ===")
    gen_tokens = generated_ids[0]
    forward_tokens = torch.argmax(forward_logits[0], dim=-1)
    
    print(f"Generated tokens: {gen_tokens}")
    print(f"Forward predicted tokens: {forward_tokens}")

    if torch.all(gen_tokens == forward_tokens):
        print("✅ SUCCESS: Forward and generate methods are perfectly aligned!")
    else:
        print("❌ FAILURE: Forward and generate methods are misaligned!")
    
    # Check logits match
    print(f"\n=== Logits Comparison ===")
    logits_match = torch.allclose(gen_logits, forward_logits, atol=1e-4)
    
    if logits_match:
        print("✅ SUCCESS: Generate and forward logits are identical!")
    else:
        print("❌ FAILURE: Generate and forward logits differ!")
        max_diff = torch.max(torch.abs(gen_logits - forward_logits))
        print(f"Maximum difference: {max_diff}")
    
if __name__ == "__main__":
    main()

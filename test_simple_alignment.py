#!/usr/bin/env python3
"""
Simple test to verify alignment between generate and forward methods.
"""

import torch
from modeling_amt.inner_loop import InnerLoopARMTForCausalLM
from transformers import AutoConfig

def main():
    print("Loading inner-loop ARMT model...")
    
    # Create a simple config
    config = AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B')
    config.base_model_name = 'meta-llama/Llama-3.2-1B'
    config.num_mem_tokens = 4
    config.d_mem = 4
    config.segment_size = 5
    config.sliding_window = True
    config.use_sink = True
    
    # Create model
    model = InnerLoopARMTForCausalLM(config)
    model.eval()
    
    # Simple test input
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input IDs: {input_ids[0]}")
    print()
    
    # Test 1: Generate first token
    print("=== Test 1: First Generated Token ===")
    generated_ids, gen_logits = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=1, 
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
        forward_logits = forward_output.logits[:, -1:, :]  # Last token
    
    print(f"Forward method logits (first 5): {forward_logits[0, 0, :5]}")
    
    # Check alignment
    print(f"\n=== Alignment Check ===")
    gen_token = generated_ids[0, 0].item()
    forward_token = torch.argmax(forward_logits[0, 0]).item()
    
    print(f"Generated token: {gen_token}")
    print(f"Forward predicted token: {forward_token}")
    
    if gen_token == forward_token:
        print("✅ SUCCESS: Forward and generate methods are perfectly aligned!")
    else:
        print("❌ FAILURE: Forward and generate methods are misaligned!")
    
    # Check logits match
    print(f"\n=== Logits Comparison ===")
    logits_match = torch.allclose(gen_logits, forward_logits, atol=1e-6)
    
    if logits_match:
        print("✅ SUCCESS: Generate and forward logits are identical!")
    else:
        print("❌ FAILURE: Generate and forward logits differ!")
        max_diff = torch.max(torch.abs(gen_logits - forward_logits))
        print(f"Maximum difference: {max_diff}")
    
    # Test 3: Generate multiple tokens
    print(f"\n=== Test 2: Multiple Generated Tokens ===")
    generated_ids_multi, gen_logits_multi = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        max_new_tokens=3, 
        return_logits=True
    )
    
    print(f"Generated tokens: {generated_ids_multi[0]}")
    print(f"Generate method logits shape: {gen_logits_multi.shape}")
    
    # Forward pass on multi-token concatenated input
    concat_multi_input_ids = torch.cat([input_ids, generated_ids_multi], dim=-1)
    concat_multi_attention_mask = torch.cat([attention_mask, torch.ones_like(generated_ids_multi)], dim=-1)
    
    with torch.no_grad():
        forward_multi_output = model(
            input_ids=concat_multi_input_ids,
            attention_mask=concat_multi_attention_mask
        )
        forward_multi_logits = forward_multi_output.logits[:, -3:, :]  # Last 3 tokens
    
    # Check alignment for multiple tokens
    print(f"\n=== Multiple Token Alignment Check ===")
    gen_tokens = generated_ids_multi[0]
    forward_tokens = torch.argmax(forward_multi_logits, dim=-1)
    
    print(f"Generated tokens: {gen_tokens}")
    print(f"Forward predicted tokens: {forward_tokens[0]}")
    
    tokens_match = torch.allclose(gen_tokens, forward_tokens[0])
    if tokens_match:
        print("✅ SUCCESS: Multiple token alignment is perfect!")
    else:
        print("❌ FAILURE: Multiple token alignment failed!")
    
    # Check logits match for multiple tokens
    logits_multi_match = torch.allclose(gen_logits_multi, forward_multi_logits, atol=1e-6)
    
    if logits_multi_match:
        print("✅ SUCCESS: Multiple token logits are identical!")
    else:
        print("❌ FAILURE: Multiple token logits differ!")
        max_diff = torch.max(torch.abs(gen_logits_multi - forward_multi_logits))
        print(f"Maximum difference: {max_diff}")

if __name__ == "__main__":
    main()

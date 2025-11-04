#!/usr/bin/env python3
"""
Benchmark script for evaluating HuggingFace models on SQuAD and GSM8K benchmarks.
"""

import argparse
import torch
import json
import csv
import os
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark a HuggingFace model on SQuAD and GSM8K")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate per dataset (default: all)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate (default: 256)"
    )
    return parser.parse_args()


def create_output_dir(model_name: str, benchmark_name: str) -> Path:
    """Create output directory for benchmark results."""
    # Clean model name for use in path
    clean_model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Create directory structure: benchmark_results/model_name/benchmark_name/
    output_dir = Path("benchmark_results") / clean_model_name / benchmark_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def save_predictions_to_csv(predictions: List[Dict], output_path: Path):
    """Save predictions to CSV file."""
    if not predictions:
        return
    
    # Get all keys from the first prediction
    fieldnames = predictions[0].keys()
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(
            csvfile, 
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,
            escapechar='\\'
        )
        writer.writeheader()
        writer.writerows(predictions)
    
    print(f"Predictions saved to: {output_path}")


def load_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer from HuggingFace."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded successfully on {device}")
    return model, tokenizer


def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """Generate answer using the model."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part (excluding the prompt)
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def evaluate_squad(model, tokenizer, device: str, model_name: str, max_samples: int = None, max_new_tokens: int = 256) -> Dict:
    """Evaluate model on SQuAD dataset."""
    print("\n" + "="*50)
    print("Evaluating on SQuAD")
    print("="*50)
    
    # Create output directory
    output_dir = create_output_dir(model_name, "squad")
    
    # Load SQuAD validation set
    dataset = load_dataset("squad", split="validation")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    predictions = []
    
    for example in tqdm(dataset, desc="SQuAD evaluation"):
        context = example["context"]
        question = example["question"]
        answers = example["answers"]["text"]
        
        # Create prompt
        prompt = f"""Answer the following question based on the context.

Context: {context}

Question: {question}

Answer:"""
        
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        
        # Extract first line/sentence as the answer
        generated_answer_clean = generated_answer.split("\n")[0].strip()
        
        # Check if any of the ground truth answers appear in the generated answer
        is_correct = any(
            answer.lower() in generated_answer_clean.lower() or 
            generated_answer_clean.lower() in answer.lower()
            for answer in answers
        )
        
        if is_correct:
            correct += 1
        total += 1
        
        # Store prediction (replace newlines for cleaner CSV)
        predictions.append({
            "context": context.replace('\n', ' '),
            "question": question.replace('\n', ' '),
            "target": " | ".join(answers),  # Multiple possible answers separated by |
            "prediction": generated_answer_clean.replace('\n', ' '),
            "is_correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    # Save predictions to CSV
    csv_path = output_dir / "predictions.csv"
    save_predictions_to_csv(predictions, csv_path)
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "output_dir": str(output_dir)
    }
    
    print(f"\nSQuAD Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return results


def extract_number_from_text(text: str) -> str:
    """Extract the final numerical answer from generated text."""
    # Look for patterns like "#### 123" (common in GSM8K)
    match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for patterns like "The answer is 123"
    match = re.search(r'(?:answer is|answer:|=)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', text, re.IGNORECASE)
    if match:
        return match.group(1).replace(',', '')
    
    # Look for the last number in the text
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')
    
    return ""


def evaluate_gsm8k(model, tokenizer, device: str, model_name: str, max_samples: int = None, max_new_tokens: int = 256) -> Dict:
    """Evaluate model on GSM8K dataset."""
    print("\n" + "="*50)
    print("Evaluating on GSM8K")
    print("="*50)
    
    # Create output directory
    output_dir = create_output_dir(model_name, "gsm8k")
    
    # Load GSM8K test set
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    predictions = []
    
    for example in tqdm(dataset, desc="GSM8K evaluation"):
        question = example["question"]
        answer = example["answer"]
        
        # Extract the numerical answer from the ground truth
        ground_truth = extract_number_from_text(answer)
        
        # Create prompt
        prompt = f"""Solve the following math problem step by step. Provide your final answer as a number.

Question: {question}

Solution:"""
        
        # Generate answer
        generated_answer = generate_answer(model, tokenizer, prompt, max_new_tokens, device)
        
        # Extract numerical answer from generation
        predicted = extract_number_from_text(generated_answer)
        
        # Compare answers (handle floating point comparison)
        is_correct = False
        try:
            if predicted and ground_truth:
                pred_val = float(predicted)
                gt_val = float(ground_truth)
                if abs(pred_val - gt_val) < 1e-3:
                    correct += 1
                    is_correct = True
        except ValueError:
            pass
        
        total += 1
        
        # Store prediction (GSM8K doesn't have context, so we use empty string)
        # Replace newlines in answer and prediction with space for cleaner CSV
        predictions.append({
            "context": "",  # GSM8K doesn't have context
            "question": question,
            "target": answer.replace('\n', ' '),  # Full solution with answer (newlines removed)
            "prediction": generated_answer.replace('\n', ' '),
            "target_number": ground_truth,
            "predicted_number": predicted,
            "is_correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    
    # Save predictions to CSV
    csv_path = output_dir / "predictions.csv"
    save_predictions_to_csv(predictions, csv_path)
    
    results = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "output_dir": str(output_dir)
    }
    
    print(f"\nGSM8K Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return results


def main():
    """Main function."""
    args = parse_args()
    
    print("="*50)
    print("Model Benchmarking Script")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*50)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)
    
    # Run benchmarks
    results = {}
    
    try:
        results["squad"] = evaluate_squad(
            model, tokenizer, args.device, args.model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens
        )
    except Exception as e:
        print(f"\nError evaluating SQuAD: {e}")
        results["squad"] = {"error": str(e)}
    
    try:
        results["gsm8k"] = evaluate_gsm8k(
            model, tokenizer, args.device, args.model,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens
        )
    except Exception as e:
        print(f"\nError evaluating GSM8K: {e}")
        results["gsm8k"] = {"error": str(e)}
    
    # Print final summary
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    
    for benchmark, result in results.items():
        print(f"\n{benchmark.upper()}:")
        if "error" in result:
            print(f"  Error: {result['error']}")
        else:
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Correct: {result['correct']}/{result['total']}")
            if "output_dir" in result:
                print(f"  CSV saved to: {result['output_dir']}/predictions.csv")
    
    # Save results to JSON
    output_file = f"benchmark_results_{args.model.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "model": args.model,
            "device": args.device,
            "max_samples": args.max_samples,
            "results": results
        }, f, indent=2)
    
    print(f"\nJSON summary saved to: {output_file}")


if __name__ == "__main__":
    main()


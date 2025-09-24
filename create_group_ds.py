#!/usr/bin/env python3
"""
Script to create group datasets and push them to Hugging Face Hub.
Based on the test_groups.ipynb notebook.
"""

import numpy as np
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from abstract_algebra.finite_algebras import (
    FiniteAlgebra,
    generate_cyclic_group,
    generate_symmetric_group,
)


def generate_group(g: str) -> FiniteAlgebra:
    """Generate a group from a string identifier."""
    if g[0] == "S":
        return generate_symmetric_group(int(g[1:]))
    elif g[0] == "Z":
        return generate_cyclic_group(int(g[1:]))
    elif g[0] == "A":
        s_n = generate_symmetric_group(int(g[1:]))
        a_n = s_n.commutator_subalgebra()
        a_n.name = f"A{g[1:]}"
        return a_n
    else:
        raise ValueError("Group must be one of S, Z, or A")


def create_dataset(group, num_samples: int, length: int) -> list:
    """Create a dataset for group multiplication tasks."""
    num_elements = len(group.elements)
    
    seqs = np.random.randint(0, num_elements, size=(num_samples, length))
    dataset = []
    
    for i in tqdm(range(num_samples), desc=f"Creating {num_samples} samples"):
        # Generate random sequence of group elements
        tgts = []
        for j in range(length):
            seq_str = [group.elements[k] for k in seqs[i, :j+1]]
            # Convert to group element
            tgt_str = group.op(*seq_str)
            tgt = group.elements.index(tgt_str)
            tgts.append(tgt)
        
        sample = {
            'input_ids': seqs[i, :].tolist(),
            'labels': tgts
        }
        dataset.append(sample)
    
    return dataset


def create_and_push_dataset(group_name: str, length: int = 40, num_samples: int = 1000000):
    """Create a dataset for a specific group."""
    print(f"Creating dataset for group {group_name} with length {length}")
    
    # Generate the group
    if group_name == "A4xZ5":
        # Special case for A4xZ5
        a_4 = generate_symmetric_group(4).commutator_subalgebra()
        a_4.name = "A4"
        
        Z5 = generate_cyclic_group(5)
        Z5.name = "Z5"
        
        group = a_4 * Z5
        group.name = "A4xZ5"
    else:
        group = generate_group(group_name)
    
    print(f"Group {group_name} has {len(group.elements)} elements")
    
    # Create dataset
    dataset = create_dataset(group, num_samples, length)
    
    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({f'length_{length}': hf_dataset})
    
    return dataset_dict


def create_split_datasets(group_name: str, dataset_dict, length: int = 40):
    """Create train/validation/test splits and push to Hub."""
    dataset_name = f"groupmul_{group_name}"
    
    # Create splits
    ds = dataset_dict
    for k in ds.keys():
        ds[k] = ds[k].train_test_split(0.1)  # 10% test
        train_val = ds[k]['train'].train_test_split(0.05 / 0.9)  # 5% validation from remaining 90%
        ds[k]['validation'] = train_val['test']
        ds[k]['train'] = train_val['train']
    
    # Push split datasets to Hub
    for l in ds.keys():
        print(f"Pushing split dataset: {dataset_name}_split with config {l}")
        ds[l].push_to_hub(
            f"XXXX/{dataset_name}_split",
            config_name=f"{l}",
        )


def main():
    """Main function to create datasets for all specified groups."""
    groups = ["A5", "A4xZ5", "Z60"]
    length = 40
    num_samples = 1000000  # 1M samples per group
    
    print("Creating group datasets for:", groups)
    print(f"Length: {length}, Samples per group: {num_samples}")
    
    for group_name in groups:
        print(f"\n{'='*50}")
        print(f"Processing group: {group_name}")
        print(f"{'='*50}")
        
        try:
            # Create the dataset (without pushing to Hub)
            dataset_dict = create_and_push_dataset(group_name, length, num_samples)
            
            # Create and push only split datasets
            create_split_datasets(group_name, dataset_dict, length)
            
            print(f"Successfully completed processing for {group_name}")
            
        except Exception as e:
            print(f"Error processing {group_name}: {e}")
            continue
    
    print("\nAll split datasets have been created and pushed to Hub!")


if __name__ == "__main__":
    main()

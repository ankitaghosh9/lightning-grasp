# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json 
import pickle 
from pathlib import Path 
import json
import torch

def pathify(p):
    if not isinstance(p, Path):
        return Path(p)
    return p

def save_json(obj, filepath, indent=4):
    """Save a Python object as a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=indent)


def load_json(filepath):
    """Load a JSON file into a Python object."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_pickle(obj, filepath):
    """Save a Python object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """Load a pickle file into a Python object."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def print_dict_structure(d, indent=0):
    """
    Recursively parses a nested dictionary and prints the key structure.
    """
    for key, value in d.items():
        # Print the current key with indentation
        print('  ' * indent + str(key))
        
        # If the value is another dictionary, recurse into it
        if isinstance(value, dict):
            print_dict_structure(value, indent + 1)
        # Optional: Handle lists of dictionaries
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    print_dict_structure(item, indent + 1)
        else:
            print(value.shape)

import json
import torch

def save_results_to_json(results_dict: dict, file_path: str):
    """
    Saves a dictionary containing PyTorch tensors (and nested dicts) to a JSON file.
    
    Args:
        results_dict (dict): Dictionary containing the results.
        file_path (str): The path where the JSON file will be saved.
    """

    def make_serializable(data):
        """
        Recursively converts PyTorch tensors in a data structure to Python lists.
        Handles dictionaries, lists, and direct tensor values.
        """
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().tolist()
        elif isinstance(data, dict):
            return {k: make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [make_serializable(v) for v in data]
        else:
            return data    
    
    # Convert entire dictionary structure to be JSON compatible
    serializable_data = make_serializable(results_dict)

    try:
        with open(file_path, 'w') as f:
            json.dump(serializable_data, f, indent=4)
        print(f"Successfully saved results to {file_path}")
    except Exception as e:
        print(f"Error saving file: {e}")


def load_results_from_json(file_path: str, device: str = 'cpu'):
    """
    Loads a JSON file and converts lists back to PyTorch tensors.
    
    Args:
        file_path (str): Path to the JSON file.
        device (str): Device to load tensors onto ('cpu' or 'cuda').
    
    Returns:
        dict: The results dictionary with PyTorch tensors.
    """

    def recursive_to_tensor(data, device):
        """
        Recursively converts lists inside a dictionary to PyTorch tensors.
        """
        if isinstance(data, dict):
            return {k: recursive_to_tensor(v, device) for k, v in data.items()}
        elif isinstance(data, list):
            # Convert list to tensor and move to specified device
            return torch.tensor(data).to(device)
        else:
            # Return strings, ints, or floats as is
            return data

    try:
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # Convert the raw JSON data (lists) back to tensors
        tensor_dict = recursive_to_tensor(raw_data, device)
        
        print(f"Successfully loaded results from {file_path}")
        return tensor_dict
        
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
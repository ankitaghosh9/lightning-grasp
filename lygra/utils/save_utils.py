# Copyright (c) Zhao-Heng Yin
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json 
import pickle 
from pathlib import Path 


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
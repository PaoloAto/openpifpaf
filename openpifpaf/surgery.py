
import torch
import numpy as np
import os
import os.path as p

state = {}

state["current"] = None

state["cachedir"] = "cache_pifpaf_results"


def set_current_path(path):
    print("SURGERY: Setting current path to", path)
    state["current"] = path

def set_var(name, value):
    print(f"SURGERY: Setting value {name}={value}")
    state[name] = value


def save_tensor(tensor, subdir="cif"):
    path = state["current"]
    if path is None:
        print("SURGERY: No current path")
        return

    subdir = subdir.format_map(state)

    dirpath = p.join(state["cachedir"], subdir)
    os.makedirs(dirpath, exist_ok=True)

    from os.path import basename, splitext

    basepath = basename(path)
    filename, ext = splitext(basepath)

    if isinstance(tensor, torch.Tensor):
        filepath = p.join(dirpath, filename + ".pt")
        print("SURGERY: Saving tensor to", tensor.size(), "to path", filepath)
        torch.save(tensor.cpu(), filepath)
    elif isinstance(tensor, np.ndarray):
        filepath = p.join(dirpath, filename + ".npy")
        print("SURGERY: Saving tensor of size", tensor.shape, "to path", filepath)
        np.save(filepath, tensor)
    else:
        raise Exception("Value not recognized", tensor)
    

def resolve(template: str):
    return template.format_map(state)
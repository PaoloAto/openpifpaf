
import torch
import numpy as np

state = {}

state["current"] = None

state["cachedir"] = "cache_pifpaf_results"


def set_current_path(path):
    print("SURGERY: Setting current path to ", path)
    state["current"] = path

def set_var(name, value):
    print(f"SURGERY: Setting value {name}={value}")
    state[name] = value


def save_tensor(tensor, subdir="cif"):
    path = state["current"]
    if path is None:
        print("SURGERY: Ooops! Path is not yet set :(")
        return
    import os
    import os.path as p

    subdir = subdir.format_map(state)

    dirpath = p.join(state["cachedir"], subdir)
    os.makedirs(dirpath, exist_ok=True)

    if isinstance(tensor, torch.Tensor):
        filepath = p.join(dirpath, path.replace("/", "_") + ".pt")
        print("SURGERY: Saving tensor to", tensor.size(), "to path", filepath)
        torch.save(tensor.cpu(), filepath)
    elif isinstance(tensor, np.ndarray):
        filepath = p.join(dirpath, path.replace("/", "_") + ".npy")
        print("SURGERY: Saving tensor to", tensor.shape, "to path", filepath)
        np.save(filepath, tensor)
    else:
        raise Exception("Value not recognized", tensor)
    

def resolve(template: str):
    return template.format_map(state)

import torch

state = {}

state["current"] = None

cachedir = "cache_wat"


def set_current_path(path):
    print("SURGERY: Setting current path to ", path)
    state["current"] = path


def save_tensor(tensor, t="cif"):
    path = state["current"]
    if path is None:
        print("SURGERY: Ooops! Path is not yet set :(")
        return
    import os
    import os.path as p

    dirpath = p.join(cachedir, t)
    os.makedirs(dirpath, exist_ok=True)

    filepath = p.join(dirpath, path.replace("/", "_") + ".pt")
    print("SURGERY: Saving tensor to", tensor.size(), "to path", filepath)
    
    torch.save(tensor.cpu(), filepath)
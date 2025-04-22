import pathlib
import json
import numpy as np

from .serialisable import NumpyEncoder

def check_create(file_path: pathlib.Path):

    if not file_path.exists():
        paths_to_create = [file_path]
        while not paths_to_create[0].parent.exists():
            paths_to_create.insert(0, paths_to_create[0].parent)
        [f.mkdir() for f in paths_to_create if not f.suffix]
        return False
    return True

def save_npy(path, array):
    with open(path, 'wb+') as f:
        np.save(f, array, allow_pickle=False)


def open_npy(path):
    with open(path, 'rb') as f:
        return np.load(f)


def save_json(filename, save_dict):
    with open(filename, 'w') as f:
        json.dump(save_dict, f, cls=NumpyEncoder, indent=4,
                  separators=(',', ': '))
        
def open_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

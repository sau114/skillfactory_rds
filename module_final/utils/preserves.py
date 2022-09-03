import os
import os.path
import pickle
from typing import Any


PICKLE_SUBDIR = 'storage'


def save_object(obj: Any, obj_name: str) -> None:
    if not os.path.isdir(PICKLE_SUBDIR):
        os.mkdir(PICKLE_SUBDIR)
    filename = os.path.join(PICKLE_SUBDIR, f'{obj_name}.pickle')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.DEFAULT_PROTOCOL)
    return


def load_object(obj_name: str) -> Any:
    if not os.path.isdir(PICKLE_SUBDIR):
        return None
    filename = os.path.join(PICKLE_SUBDIR, f'{obj_name}.pickle')
    if not os.path.isfile(filename):
        return None
    with open(filename, 'rb') as f:
        result = pickle.load(f)
    return result

import time
import importlib
import collections
# helper functions

def exists(val):
    return val is not None

# time helpers

class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.last_time = time.time()

    def elapsed(self):
        return time.time() - self.last_time

# print helpers

def print_ribbon(s, symbol = '=', repeat = 40):
    flank = symbol * repeat
    return f'{flank} {s} {flank}'

# import helpers

def import_or_print_error(pkg_name, err_str = None):
    try:
        return importlib.import_module(pkg_name)
    except ModuleNotFoundError as e:
        if exists(err_str):
            print(err_str)
        exit()

def flatten(d, parent_key="", sep="."):
    """flatten a nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
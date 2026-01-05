from typing import Any, Dict, Optional, Union
import hashlib
import re
import numpy as np
from functools import lru_cache
from scipy.optimize import root_scalar

import torch
torch.set_default_dtype(torch.float64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print("Current device:", current_device)
    print("Current device name:", torch.cuda.get_device_name(current_device))


def str_to_hash(s: str) -> str:
    return hashlib.sha256(s.encode('ascii')).hexdigest()


def _get_dict_item_sort_key(item):
    """
    Generate a sort key for dictionary items that sorts by:
    1. Parameter name (alphabetically)
    2. Numeric value if the value is a number

    This ensures that parameters with numeric values are sorted numerically
    rather than lexicographically (e.g., 0.01, 0.0003, 0.00173 instead of
    0.0003, 0.00173, 0.01).
    """
    key, value = item

    # Try to extract numeric value for proper sorting
    numeric_value = None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
    elif isinstance(value, str):
        try:
            # Try to parse as float (handles scientific notation like 5.2e-05)
            numeric_value = float(value)
        except (ValueError, TypeError):
            # Not a number, will sort by string representation
            pass

    # Return a sort key: (param_name, numeric_value_or_string_repr)
    # If numeric_value is None, use string representation for sorting
    if numeric_value is not None:
        return (key, 0, numeric_value)  # 0 to prioritize numeric sorting
    else:
        return (key, 1, str(value))  # 1 to sort non-numeric values after numeric


def _to_str(x, include_space=False) -> str:
    sep = ', ' if include_space else ','
    if type(x) is dict:
        # Sort items using the custom sort key
        sorted_items = sorted(x.items(), key=_get_dict_item_sort_key)
        return '(' + sep.join(
            key + '=' + _to_str(value)
            for key, value in sorted_items
        ) + ')'
    if type(x) is list:
        return '[' + sep.join(map(_to_str, x)) + ']'
    if type(x) is str:
        return x
    return repr(x)


def dict_to_str(d: Dict[str, Any], include_space=False) -> str:
    if type(d) is not dict:
        raise ValueError(f"Expected a dictionary, got a {type(d).__name__} "
                            f"with value {d!r}")
    return _to_str(d, include_space=include_space)[1:-1]


def dict_to_hash(d: Dict[str, Any]) -> str:
    return str_to_hash(dict_to_str(d))


def sanitize_file_name(file_name: str) -> str:
    # Define a dictionary of characters to replace
    replacements = {
        '/': '_',
        '\\': '_',
        ':': '_',
        '*': '_',
        '?': '_',
        '"': '_',
        '<': '_',
        '>': '_',
        '|': '_',
    }

    # Replace the characters based on the replacements dictionary
    sanitized_name = ''.join(replacements.get(c, c) for c in file_name)

    # Remove characters that are non-printable or not allowed
    sanitized_name = re.sub(r'[^\x20-\x7E]', '', sanitized_name)

    # Remove all whitespace characters
    sanitized_name = re.sub(r'\s+', '', sanitized_name)

    return sanitized_name


def dict_to_fname_str(d: Dict[str, Any]) -> str:
    return sanitize_file_name(dict_to_str(d))


def _int_linspace_naive(start, stop, num):
    return np.unique(np.round(np.linspace(start, stop, num)).astype(int))


@lru_cache(maxsize=128) # Not necessary but why not.
def int_linspace(start, stop, num):
    if not (isinstance(start, int) and isinstance(stop, int)):
        raise ValueError("start and stop should be integers")

    if num > stop - start + 1:
        raise ValueError('num must be less than or equal to stop - start + 1')
    ret = _int_linspace_naive(start, stop, num)
    length = len(ret)
    
    if length < num:
        sol = root_scalar(
            lambda x: len(_int_linspace_naive(start, stop, int(x))) - num,
            method='secant', x0=num, x1=2*num, xtol=1e-12, rtol=1e-12)
        
        k = int(sol.root)
        ret = _int_linspace_naive(start, stop, k)

        if len(ret) != num:
            if len(ret) > num:
                while len(ret) > num:
                    k -= 1
                    ret = _int_linspace_naive(start, stop, k)
            else:
                while len(ret) < num:
                    k += 1
                    ret = _int_linspace_naive(start, stop, k)

    return ret


def to_device(tensor, device):
    if tensor is None or device is None:
        return tensor
    return tensor.to(device)


def dict_to_cmd_args(params: Dict, equals=False) -> list[str]:
    """Convert keyword arguments to command-line argument list."""
    parts = []
    for key, value in sorted(params.items()):
        # If the value is a boolean, only include it if True.
        if isinstance(value, bool):
            if value:
                parts.append(f"--{key}")
        elif value is not None:
            if equals:
                parts.append(f"--{key}={value}")
            else:
                parts.append(f"--{key}")
                parts.append(str(value))
    return parts


def _json_serializable_to_numpy(data: Any, array_keys: Optional[set]=None):
    if isinstance(data, dict):
        return {
            k: np.array(v) if isinstance(v, list) and
            (array_keys is None or k in array_keys)
            else _json_serializable_to_numpy(v, array_keys)
            for k, v in data.items()
        }
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, (list, tuple)):
        return [_json_serializable_to_numpy(x, array_keys) for x in data]
    return data


def json_serializable_to_numpy(data: Any,
                               array_keys: Optional[Union[list,tuple,set]]=None):
    if array_keys is not None:
        array_keys = set(array_keys)
    return _json_serializable_to_numpy(data, array_keys)


def convert_to_json_serializable(data, float_precision=None):
    if isinstance(data, dict):
        return {k: convert_to_json_serializable(v, float_precision)
                for k, v in data.items()}
    if isinstance(data, np.ndarray):
        return convert_to_json_serializable(data.tolist(), float_precision)
    if torch.is_tensor(data):
        return convert_to_json_serializable(
            data.cpu().numpy().tolist(), float_precision)
    if isinstance(data, (list, tuple)):
        return [convert_to_json_serializable(x, float_precision) for x in data]
    if isinstance(data, (int, float, str, bool, type(None))):
        if isinstance(data, float) and float_precision is not None:
            # Round to specified number of significant digits in scientific notation
            return float(f'{data:.{float_precision}e}')
        return data
    if isinstance(data, type):
        return data.__name__
    return str(data)


def safe_issubclass(obj, parent):
    """Returns whether `obj` is a class that is a subclass of `parent`.
    In contrast to `issubclass`, doesn't raise TypeError when `obj` is not a class."""
    return isinstance(obj, type) and issubclass(obj, parent)


def get_arg_names(p) -> list[str]:
    """Get argument names from an argparse parser or group."""
    return [action.dest for action in p._group_actions if action.dest != "help"]


def group_by(items, group_function=lambda x: x):
    """
    Groups items by a grouping function
    parameters:
        items: iterable containing things
        group_function: function that gives the same result when called on two
            items in the same group
    returns: a dict where the keys are results of the function and values are
        lists of items that when passed to group_function give that key
    """
    group_dict = {}
    for item in items:
        value = group_function(item)
        if value in group_dict:
            group_dict[value].append(item)
        else:
            group_dict[value] = [item]
    return group_dict


def _assert_all_have_type(values: list, name: str, t: type):
    for i, v in enumerate(values):
        if not isinstance(v, t):
            raise ValueError(
                f"Expected all elements of {name} to have type {t.__name__}, "
                f"but {name}[{i}] is of type {v.__class__.__name__}")


def _assert_has_type(x: object, name: str, t: type):
    if not isinstance(x, t):
        raise ValueError(f"Expected {name} to have type {t.__name__}, "
                         f"but it is of type {x.__class__.__name__}")


def aggregate_stats_list(stats_list: Union[list[dict[str]], list[np.ndarray],
                                           list[float], list[int], list[str],
                                             list[bool], list[None]]):
    _assert_has_type(stats_list, "stats_list", list)
    stats0 = stats_list[0]
    if isinstance(stats0, dict):
        _assert_all_have_type(stats_list, "stats_list", dict)
        return {
            key: aggregate_stats_list([
                stats[key] for stats in stats_list
            ]) for key in stats0.keys()
        }
    if isinstance(stats0, (np.ndarray, float, int, str, bool, type(None))):
        _assert_all_have_type(stats_list, "stats_list", type(stats0))
        return np.array(stats_list)
    raise ValueError(f"stats_list must be a list of dicts or list of ndarrays, "
                     f"but got stats_list[0]={stats0}")

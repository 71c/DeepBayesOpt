from time import time as _time

_start_times = {}


def tic(name=None, say_name=False):
    if say_name:
        print(name)
    _start_times[name] = _time()
    return _start_times[name]


def toc(name=None):
    t = _time() # calling time() asap makes it as accurate as possible in theory
    if name not in _start_times:
        argument = repr(name) if name is not None else ''
        raise RuntimeError(f"Error using toc: You must call tic({argument}) before calling toc({argument})")
    dt = t - _start_times.pop(name)
    if name is None:
        print(f"Elapsed time: {dt:.6f} seconds")
    else:
        print(f"{name} took {dt:.6f} seconds")

def tocl():
    t = _time()
    if len(_start_times) == 0:
        raise RuntimeError("You didn't set any timers to tocl")
    name, t0 = _start_times.popitem() # python version needs to be >= 3.7
    dt = t - t0
    if name is None:
        print(f"Elapsed time: {dt:.6f} seconds")
    else:
        print(f"{name} took {dt:.6f} seconds")

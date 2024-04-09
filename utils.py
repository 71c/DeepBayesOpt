import torch
from functools import partial


def uniform_randint(min_val, max_val):
    return torch.randint(min_val, max_val+1, (1,), dtype=torch.int32).item()


def get_uniform_randint_generator(min_val, max_val):
    return partial(uniform_randint, min_val, max_val)


def loguniform_randint(min_val, max_val, size=1, pre_offset=0.0, offset=0):
    if not (isinstance(min_val, int) and isinstance(max_val, int) and isinstance(offset, int)):
        raise ValueError("min_val, max_val, and offset must be integers")
    if not (1 <= min_val <= max_val):
        raise ValueError("min_val must be between 1 and max_val")

    min_log = torch.log(torch.tensor(min_val + pre_offset))
    max_log = torch.log(torch.tensor(max_val+1 + pre_offset))
    random_log = torch.rand(size) * (max_log - min_log) + min_log
    ret = (torch.exp(random_log) - pre_offset).to(dtype=torch.int32) + offset
    if torch.numel(ret) == 1:
        return ret.item()
    return ret


def get_loguniform_randint_generator(min_val, max_val, pre_offset=0.0, offset=0):
    return partial(loguniform_randint, min_val, max_val, pre_offset=pre_offset, offset=offset)


# Based on
# https://docs.gpytorch.ai/en/stable/_modules/gpytorch/module.html#Module.initialize
def get_param_value(module, name):
    if "." in name:
        submodule, name = module._get_module_and_name(name)
        if isinstance(submodule, torch.nn.ModuleList):
            idx, name = name.split(".", 1)
            return get_param_value(submodule[int(idx)], name)
        else:
            return get_param_value(submodule, name)
    elif not hasattr(module, name):
        raise AttributeError("Unknown parameter {p} for {c}".format(p=name, c=module.__class__.__name__))
    elif name not in module._parameters and name not in module._buffers:
        return getattr(module, name)
    else:
        return module.__getattr__(name)


# Print out all parameters of a random model:
# random_model = model.pyro_sample_from_prior()
# for name, param in model.named_parameters(): 
#     print(name)
#     print(get_param_value(random_model, name))
#     print()
## OR,
# random_model_params_dict = {
#     name: get_param_value(random_model, name)
#     for name, param in model.named_parameters()
# }


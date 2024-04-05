import torch


def get_uniform_randint_generator(min_val, max_val):
    return lambda: torch.randint(min_val, max_val+1, (1,))[0]


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


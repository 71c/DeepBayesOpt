import torch
from torch.masked import masked_tensor
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


def pad_tensor(vec, length, dim, add_mask=False):
    """Pads a tensor 'vec' to a size 'length' in dimension 'dim' with zeros.
    args:
        vec - tensor to pad
        length - the size to pad to in dimension 'dim'
        dim - dimension to pad
        add_mask - whether to return a MaskedTensor that includes the mask

    return:
        a new tensor padded to 'length' in dimension 'dim'
    """
    pad_size = length - vec.size(dim)
    if pad_size < 0:
        raise ValueError("Tensor cannot be padded to length less than it already is")
    
    pad_shape = list(vec.shape)
    pad_shape[dim] = pad_size
    if pad_size == 0: # Could pad with nothing but that's unnecessary
        padded = vec
    else:
        padding = torch.zeros(*pad_shape, dtype=vec.dtype, device=vec.device)
        padded = torch.cat([vec, padding], dim=dim)

    if add_mask:
        mask_true = torch.ones(vec.shape, dtype=torch.bool, device=vec.device)
        mask_false = torch.zeros(*pad_shape, dtype=torch.bool, device=vec.device)
        mask = torch.cat([mask_true, mask_false], dim=dim)
        padded = masked_tensor(padded, mask)

    return padded


def max_pad_tensors_batch(tensors, dim=0, add_mask=False):
    """Pads a batch of tensors along a dimension to match the maximum length.

    Args:
        tensors (List[torch.Tensor]): A list of tensors to be padded.
        dim (int, default: 0): The dimension along which to pad the tensors.
        add_mask (bool, optional, default: False):
            If add_mask=True AND tensors are of different lengths
            (padding is necessary), add a mask and return a MaskedTensor.
            Otherwise, returns a regular tensor padded with zeros.

    Returns:
        Tensor or MaskedTensor: The padded batch of tensors.
    """
    lengths = [x.shape[dim] for x in tensors]
    max_length = max(lengths)
    if all(length == max_length for length in lengths):
        stacked = torch.stack(tensors) # Don't pad if we don't need to
    else:
        # MaskedTensor doesn't support torch.stack but does support torch.vstack
        # so need to add a dimension to all of them and then vstack which is
        # equivalent.
        padded_tensors = [
            pad_tensor(x.unsqueeze(0), max_length, dim=1+dim, add_mask=add_mask)
            for x in tensors]
        stacked = torch.vstack(padded_tensors)
    
    return stacked


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


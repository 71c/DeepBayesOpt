import math
import numpy as np
import scipy
from scipy.optimize import newton
import torch
from torch import Tensor
from botorch.acquisition.analytic import _log_ei_helper
from botorch.utils.probability.utils import get_constants_like, ndtr as Phi

torch.set_default_dtype(torch.double)


_neg_inv_sqrt_2 = -(1 / math.sqrt(2))
_inv_sqrt_2pi = 1 / math.sqrt(math.tau)
# _constant_1 = math.log(0.5 / math.sqrt(math.tau))
_constant_1 = -0.5 * math.log(8.0 * math.pi)
_constant_2 = 0.5755

def _approx_ei_helper_inverse(v: Tensor) -> Tensor:
    const1, const2 = get_constants_like((_constant_1, _constant_2), v)
    tmp_log = const1 - torch.log(v)
    val_low = -torch.sqrt(2 * (tmp_log - torch.log(tmp_log)))
    val_med = const2 * torch.log(-1 + torch.exp(v / const2))
    return torch.where(
        v <= 0.05, val_low,
        torch.where(v <= 10.0, val_med, v)
    )


def _phi_numpy(x: Tensor) -> Tensor:
    r"""Standard normal PDF."""
    return _inv_sqrt_2pi * np.exp(-0.5 * x**2)


def _Phi_numpy(x):
    r"""Standard normal CDF."""
    return 0.5 * scipy.special.erfc(_neg_inv_sqrt_2 * x)


def _ei_helper_numpy(u):
    return _phi_numpy(u) + u * _Phi_numpy(u)


def ei_helper_inverse(v: Tensor) -> Tensor:
    v = v.detach().cpu()
    if not torch.is_tensor(v):
        raise ValueError("v should be a torch tensor")
    log_v = torch.log(v).numpy()

    def f(x):
        x = torch.from_numpy(np.asarray(x))
        return _log_ei_helper(x).numpy() - log_v
    def fprime(x):
        return _Phi_numpy(x) / _ei_helper_numpy(x)
    def fprime2(x):
        return (
            _phi_numpy(x) * _ei_helper_numpy(x) - _Phi_numpy(x)**2) / _ei_helper_numpy(x)**2

    x0 = _approx_ei_helper_inverse(v).numpy()
    result = newton(f, x0, fprime=fprime, fprime2=fprime2, tol=1e-10, maxiter=50)
    return torch.tensor(result, dtype=v.dtype, device=v.device)


def probability_y_greater_than_gi_normal(cbar: Tensor, sigma: Tensor) -> Tensor:
    u = ei_helper_inverse(cbar / sigma)
    return Phi(u)

import sys
from math import log10, floor

import torch
from numpy.linalg import cond
from scipy.linalg import eigvals
from scipy.stats import wishart


def generate_pd_matrix(n: int, as_torch: bool = True):
    if n < 2:
        raise ValueError(f"Must generate at least a 2x2 matrix, not {n}x{n}")
    A = wishart.rvs(n, tuple(1 for _ in range(n)))
    while not _is_positive_definite(A) or _is_ill_conditioned(A):
        A = wishart.rvs(n, tuple(1 for _ in range(n)))
    if as_torch:
        A_torch = torch.as_tensor(A, dtype=torch.get_default_dtype())
        return A_torch
    return A


def _is_ill_conditioned(A):
    rcond = 1 / cond(A)
    if floor(log10(rcond)) <= -8:
        return True
    return False


def _is_positive_definite(A):
    return all(eig > 0 for eig in eigvals(A))

import numpy as np
import torch
from numpy.linalg import cond
from scipy.linalg import cholesky
from scipy.linalg.misc import LinAlgError
from scipy.stats import wishart

from hessianfree.types import Matrix


def generate_pd_matrix(n: int, as_torch: bool = True):
    """Generate an (n x n) positive-definite matrix"""
    if n < 2:
        raise ValueError(f"Must generate at least a 2x2 matrix, not {n}x{n}")
    A = wishart.rvs(n, tuple(1 for _ in range(n)))
    while not _is_positive_definite(A):
        A = wishart.rvs(n, tuple(1 for _ in range(n)))
        A += A.mean() * np.eye(n)
    if as_torch:
        A_torch = torch.as_tensor(A, dtype=torch.get_default_dtype())
        return A_torch
    return A


def _is_ill_conditioned(A: Matrix):
    """Check if a matrix is ill-conditioned"""
    if cond(A) >= 1 / np.finfo(np.float32).eps:
        return True
    return False


def _is_positive_definite(A: Matrix):
    """Check that a matrix is positive-definite"""
    try:
        cholesky(A)
        return True
    except LinAlgError:
        return False

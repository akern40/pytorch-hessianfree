from typing import Callable, Union

import torch
import numpy as np


def _diagonal(A):
    return torch.diagflat(torch.diagonal(A))


_preconditioners = {"diag": _diagonal}


def pcg(
    A: torch.Tensor,
    b: torch.Tensor,
    x0: Union[torch.Tensor, None] = None,
    preconditioner: str = "diag",  # A pre-defined preconditioner function
    max_iter: int = None,
    err_tol: float = 1e-6,
    callback: Callable = None,
):
    """Compute the solution to Ax=b using the preconditioned conjugate gradient method"""
    if x0 is None:
        x0 = torch.zeros_like(b)
    if max_iter is None:
        max_iter = A.shape[0]

    x = x0
    count = 0
    residual = b - A @ x

    P = _preconditioners[preconditioner](A)
    P_inv = torch.inverse(P)
    direction = P_inv @ residual

    delta0 = residual.t() @ direction
    delta_new = delta0

    while count < max_iter or delta_new / delta0 > (err_tol ** 2):
        q = A @ direction
        alpha = delta_new / (direction.t() @ q)

        if callable(callback):
            callback(x, direction, alpha)

        x += alpha * direction

        if count % 50 == 0:
            residual = b - A @ x
        else:
            residual -= alpha * q

        s = P_inv @ residual
        delta_old = delta_new
        delta_new = residual.t() @ s

        beta = delta_new / delta_old
        direction = s + beta * direction

        count += 1

    return x

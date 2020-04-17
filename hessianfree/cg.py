from typing import Callable, Optional, Any

import torch

from hessianfree.types import LinearOperator

CGCallback = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Any]


def _identity(x):
    return x


def pcg(
    A: LinearOperator,
    b: torch.Tensor,
    max_iter: int,
    x0: Optional[torch.Tensor] = None,
    preconditioner: Optional[LinearOperator] = None,
    shape: Optional[int] = None,
    err_tol: float = 1e-3,
    callback: Optional[CGCallback] = None,
):
    """Compute the solution to Ax=b using the preconditioned conjugate gradient method"""
    if x0 is None:
        x0 = torch.zeros_like(b)
    if max_iter is None:
        max_iter = A.shape[0]

    # Initialize x, residual
    x = x0
    count = 0
    residual = b - A(x)

    # Set preconditioner, if there is none
    if preconditioner is None:
        P = _identity
    else:
        P = preconditioner
    direction = P(residual)

    delta0 = residual.t() @ direction
    delta_new = delta0

    while count < max_iter and delta_new / delta0 > (err_tol ** 2):
        q = A(direction)
        alpha = delta_new / (direction.t() @ q)

        if callable(callback):
            callback(x, direction, alpha)

        x += alpha * direction

        if count % 50 == 0:
            residual = b - A(x)
        else:
            residual -= alpha * q

        s = P(residual)
        delta_old = delta_new
        delta_new = residual.t() @ s

        beta = delta_new / delta_old
        direction = s + beta * direction

        count += 1

    return x, count

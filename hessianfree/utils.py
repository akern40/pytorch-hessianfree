import sys
from math import ceil, log10

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.linalg import cond, solve
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
    if cond(A) >= 1 / np.finfo(np.float32).eps:
        return True
    return False


def _is_positive_definite(A):
    return all(eig > 0 for eig in eigvals(A))


def draw_surface(A: np.array, b: np.array, pcg_solution, points, directions, alphas):
    solution = solve(A, b)
    solution_x = solution[0]
    solution_y = solution[1]

    x = np.linspace(min(solution_x, 0) - 5, max(solution_x, 0) + 5, 100)
    y = np.linspace(min(solution_y, 0) - 5, max(solution_y, 0) + 5, 100)

    X, Y = np.meshgrid(x, y)
    Z = get_z(X, Y, A, b)

    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, Z, 50)
    fig.colorbar(cs)

    pcg_solution = pcg_solution.squeeze()

    ax.plot(pcg_solution[0], pcg_solution[1], "ro")
    ax.plot(solution_x, solution_y, "go")

    for p, d, a in zip(points, directions, alphas):
        ax.plot(p[0], p[1], "wo")
        ax.arrow(p[0], p[1], a * d[0], a * d[1], color="w")

    plt.show()


def draw_surface_3d(A: np.array, b: np.array):
    solution = solve(A, b)
    solution_x = solution[0]
    solution_y = solution[1]

    x = np.linspace(min(solution_x, 0) - 5, max(solution_x, 0) + 5, 100)
    y = np.linspace(min(solution_y, 0) - 5, max(solution_y, 0) + 5, 100)

    X, Y = np.meshgrid(x, y)
    Z = get_z(X, Y, A, b)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis")
    plt.show()
    # ax.scatter(solution_x, solution_y, c="w")


def get_z(X: np.array, Y: np.array, A: np.array, b: np.array):
    inputs = np.stack((X, Y), axis=1)
    square_term = np.einsum("bij,ki,bkj->bj", inputs, A, inputs)
    linear_term = np.einsum("ki,bkj->bj", b, inputs)
    return 0.5 * square_term - linear_term

import unittest
from math import floor, log10

import numpy as np
import torch

from hessianfree.cg import pcg
from hessianfree.utils import generate_pd_matrix


class TestPCG(unittest.TestCase):
    def test_basic(self):
        A = torch.Tensor([[3, 2], [2, 6]])
        b = torch.Tensor([[2], [1]])

        solution = torch.Tensor([[0.7142857909202576], [-0.071428582072258]])

        cg_solution, info = pcg(A, b)

        assert torch.allclose(solution, cg_solution)

    def test_2d(self):
        self._test_nd(2)

    def _test_nd(self, n, ntests=10000, backward_eps=-5):
        num_failed = 0
        n_early_termination = 0
        back_errors = np.zeros(ntests)

        for ii in range(ntests):
            A = generate_pd_matrix(n)
            b = torch.rand(n, 1)

            cond_A = np.linalg.cond(A.numpy())

            torch_solution, _ = torch.solve(b, A)
            cg_solution, info = pcg(A, b)

            if info["n_iter"] < n:
                n_early_termination += 1

            backward_err = torch.dist(b, A @ cg_solution) / (
                torch.norm(A) * torch.norm(cg_solution) + torch.norm(b)
            )
            back_errors[ii] = backward_err.item()

        print(f"{n_early_termination}/{ntests} terminated early")
        print(
            f"Backward errors:\n\tMin: {np.min(back_errors)}\n\tMax: {np.max(back_errors)}\n\tMean: {np.mean(back_errors)}\n\tStd Dev: {np.std(back_errors)}"
        )
        assert floor(log10(np.mean(back_errors) + np.std(back_errors))) <= backward_eps

    def test_multi_dim(self):
        for dim in range(5, 1010, 50):
            print(dim, end=" ")
            if dim == 55:
                self._test_nd(dim, ntests=100, backward_eps=-3)
            elif dim > 500:
                self._test_nd(dim, ntests=5)
            else:
                self._test_nd(dim, ntests=10)


if __name__ == "__main__":
    unittest.main()

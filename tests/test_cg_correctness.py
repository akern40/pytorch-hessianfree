import unittest
from math import floor, log10

import numpy as np
import torch

from hessianfree.cg import pcg
from hessianfree.utils import generate_pd_matrix


class TestPCG(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

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
        forward_errors = np.zeros(ntests)

        for ii in range(ntests):
            A = generate_pd_matrix(n)
            alo = lambda x: A @ x
            b = torch.rand(n, 1)

            cond_A = np.linalg.cond(A.numpy())

            torch_solution, _ = torch.solve(b, A)
            cg_solution, n_iter = pcg(alo, b, A.shape[0])

            if n_iter < n:
                n_early_termination += 1

            # ||b - Ax'|| / (||A||*||x'|| + ||b||)
            backward_err = torch.dist(b, A @ cg_solution) / (
                torch.norm(A) * torch.norm(cg_solution) + torch.norm(b)
            )
            forward_err = torch.dist(torch_solution, cg_solution) / torch.norm(
                torch_solution
            )
            back_errors[ii] = backward_err.item()
            forward_errors[ii] = forward_err.item()

        print(f"{n_early_termination}/{ntests} terminated early")
        print(
            f"Forward errors:\n\tMin: {np.min(forward_errors)}\n\tMax: {np.max(forward_errors)}\n\tMean: {np.mean(forward_errors)}\n\tStd Dev: {np.std(forward_errors)}"
        )
        print(
            f"Backward errors:\n\tMin: {np.min(back_errors)}\n\tMax: {np.max(back_errors)}\n\tMean: {np.mean(back_errors)}\n\tStd Dev: {np.std(back_errors)}"
        )
        assert floor(log10(np.mean(back_errors) + np.std(back_errors))) <= backward_eps

    def test_multi_dim(self):
        for dim in range(5, 1010, 50):
            print(dim, end=" ")
            if dim == 55 or dim == 105:
                self._test_nd(dim, ntests=100, backward_eps=-3)
            elif dim > 500:
                self._test_nd(dim, ntests=5)
            else:
                self._test_nd(dim, ntests=10)


if __name__ == "__main__":
    unittest.main()

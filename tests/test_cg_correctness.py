import unittest
from math import ceil, log10

import numpy as np
import torch
from scipy.linalg import solve

from hessianfree.cg import pcg
from hessianfree.utils import generate_pd_matrix, draw_surface


class TestPCG(unittest.TestCase):
    def test_basic(self):
        A = torch.Tensor([[3, 2], [2, 6]])
        b = torch.Tensor([[2], [1]])

        solution = torch.Tensor([[0.7142857909202576], [-0.071428582072258]])

        cg_solution = pcg(A, b)

        self.assertTrue(torch.allclose(solution, cg_solution))

    def test_2d(self):
        self._test_nd(2)

    def _test_nd(self, n, ok_failure_rate=0.05, ntests=1000):
        num_failed = 0

        for _ in range(ntests):
            A = generate_pd_matrix(n)

            b = torch.rand(n, 1)

            solution = torch.as_tensor(solve(A.numpy(), b.numpy()))
            cg_solution = pcg(A, b)

            try:
                assert torch.allclose(solution, cg_solution)
            except AssertionError:
                num_failed += 1

        print(num_failed)
        assert num_failed <= ntests * ok_failure_rate

    # def test_nd(self):
    #     for dim in range(2, 100):
    #         self._test_nd(dim)


if __name__ == "__main__":
    unittest.main()

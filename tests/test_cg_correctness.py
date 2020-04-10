import unittest
from math import ceil, log10

import numpy as np
import torch
from scipy.linalg import solve
from scipy.stats import wishart

from hessianfree.cg import pcg
from hessianfree.utils import generate_pd_matrix


class TestPCG(unittest.TestCase):
    def test_basic(self):
        A = torch.Tensor([[3, 2], [2, 6]])
        b = torch.Tensor([[2], [1]])

        solution = torch.Tensor([[0.7142857909202576], [-0.071428582072258]])

        cg_solution = pcg(A, b)

        self.assertTrue(torch.allclose(solution, cg_solution))

    def test_2d(self):
        num_passed = 0
        for _ in range(10000):
            A = generate_pd_matrix(2)

            b = torch.rand(2, 1)

            solution = torch.from_numpy(solve(A.numpy(), b.numpy()))
            cg_solution = pcg(A, b)

            largest_diff = max(torch.abs(solution - cg_solution)).item()
            if largest_diff < 1e-5:
                num_passed += 1
        print(num_passed)
        self.assertGreaterEqual(num_passed, 999, msg=f"Only passed {num_passed}/10000")


if __name__ == "__main__":
    unittest.main()

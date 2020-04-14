import unittest

import numpy as np
import torch

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
        self._test_nd(2)

    def _test_nd(self, n, ntests=10000, ok_failure_rate=0.0005):
        num_failed = 0

        for _ in range(ntests):
            A = generate_pd_matrix(n)
            b = torch.rand(n, 1)

            torch_solution, _ = torch.solve(b, A)
            cg_solution = pcg(A, b)

            rel_error = torch.dist(torch_solution, cg_solution) / torch.norm(
                torch_solution
            )
            try:
                assert rel_error < np.linalg.cond(A.numpy()) * np.finfo(np.float32).eps
            except AssertionError as e:
                if ok_failure_rate == 0:
                    raise e
                num_failed += 1

        assert num_failed <= ntests * ok_failure_rate

    def test_multi_dim(self):
        for dim in range(3, 10):
            self._test_nd(dim, ntests=100)


if __name__ == "__main__":
    unittest.main()

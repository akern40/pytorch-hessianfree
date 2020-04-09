import unittest

import numpy as np
import torch

from hessianfree.cg import pcg


class TestPCG(unittest.TestCase):
    def test_basic(self):
        A = torch.rand(2, 2)
        b = torch.rand((2, 1))
        pd_matrix = 0.5 * (A + A.transpose(0, 1))

        A_inv = torch.inverse(A)
        solution = A_inv @ b
        print(solution)

        cg_solution = pcg(A, b)
        print(cg_solution)


if __name__ == "__main__":
    unittest.main()

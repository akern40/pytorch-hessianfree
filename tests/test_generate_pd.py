import sys
import unittest

from numpy.linalg import cond
from scipy.linalg import eigvals

from hessianfree.utils import generate_pd_matrix


class TestGeneratePDMatrix(unittest.TestCase):
    def test_positive_definite(self):
        for ii in range(2, 101):
            A = generate_pd_matrix(ii, as_torch=False)
            assert all(eig > 0 for eig in eigvals(A))

    def test_well_conditioned(self):
        for ii in range(2, 101):
            A = generate_pd_matrix(ii, as_torch=False)
            assert cond(A) < 1 / sys.float_info.epsilon


if __name__ == "__main__":
    unittest.main()

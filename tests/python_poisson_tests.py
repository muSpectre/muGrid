import unittest
import numpy as np

import muGrid


def test_poisson_solver(comm):
    print(comm.rank, comm.size)
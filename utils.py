from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import linalg
from scipy.sparse import csr_matrix 
from scipy.sparse import identity
import numpy as np
import time


def relerr(A, x, b):
    z = A.createVecLeft()
    z.set(0.0)
    z.assemble()
    A.mult(x, z)
    z.aypx(-1.0, b)
    n1 = z.norm(norm_type=PETSc.NormType.NORM_2)
    n2 = b.norm(norm_type=PETSc.NormType.NORM_2)
    return n1/n2
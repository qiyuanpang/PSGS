from mpi4py import MPI
from petsc4py import PETSc
from scipy.sparse import linalg
from scipy.sparse import csr_matrix 
from scipy.sparse import identity
from scipy.sparse import find
import numpy as np
import time

def psgs(A, b, itmax=200, restart=True, K0=2, x0=None):
    # set up 
    a0, a1 = 0, 1
    l = 0
    Kre, K1 = 0, K0

    N, M = A.getSize()
    assert N == M

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    
    remainder = N % comm_size
    dvd = [N//comm_size+1 if i < remainder else N//comm_size for i in range(comm_size)]
    rstart, rend = A.getOwnershipRange()
    assert rend-rstart == dvd[comm_rank]

    cputime = {"Jii": 0, "initialize": 0, "Ax": 0, "getLocalVector": 0, "kspsolve": 0, "setArray": 0, "add": 0, "dot": 0, "update": 0}

    # create Jii
    start = time.time()
    nQij = 0.0
    I, J, V = A.getValuesCSR()
    Qi = csr_matrix((V, J, I), shape = (rend-rstart, N))
    Jii = Qi[:, rstart:rend]
    for j in range(comm_size):
        if j != comm_rank:
            # I, J, V = A.getValuesCSR([i for i in range(rstart, rend)], sum(dvd[:j]):sum(dvd[:j+1]))
            Qij = Qi[:, sum(dvd[:j]):sum(dvd[:j+1])]
            nQij += linalg.norm(Qij, ord='fro')
    Jii += nQij*identity(rend-rstart, format="csr")
    Ii, Ji, Vi = Jii.indptr, Jii.indices, Jii.data

    Jii = PETSc.Mat().create(comm=PETSc.COMM_SELF)
    Jii.setSizes([rend-rstart, rend-rstart])
    Jii.setFromOptions()
    Jii.setUp()
    Jii.setValuesCSR(Ii, Ji, Vi)
    Jii.assemble()
    end = time.time()
    cputime["Jii"] = end-start

    # create x0, y1, x1
    start = time.time()
    if x0 == None:
        x0 = A.createVecRight()
        x0.setRandom()
        x0.assemble()
    y1 = x0.copy()
    y1.assemble()
    x1 = x0.copy()
    x1.assemble()

    # create auxiliary vecs
    z = A.createVecLeft()
    z.set(0.0)
    z.assemble()
    w = A.createVecRight()
    w.set(0.0)
    w.assemble()
    w1 = A.createVecRight()
    w1.set(0.0)
    w1.assemble()

    ksp = PETSc.KSP().create(comm=PETSc.COMM_SELF)
    ksp.setFromOptions()
    ksp.setOperators(Jii)
    end = time.time()
    cputime["initialize"] = end-start

    it = 1
    while it <= itmax:
        start = time.time()
        A.mult(y1, z)
        z.aypx(-1.0, b)
        end = time.time()
        cputime["Ax"] += end-start

        start = time.time()
        localv = z.createLocalVector()
        z.getLocalVector(localv, readonly=False)
        localw = w.createLocalVector()
        w.getLocalVector(localw, readonly=False)
        end = time.time()
        cputime["getLocalVector"] += end-start

        start = time.time()
        ksp.solve(localv, localw) 
        end = time.time()
        cputime["kspsolve"] += end-start
        
        start = time.time()
        w.restoreLocalVector(localw, readonly=False)
        end = time.time()
        cputime["setArray"] += end-start

        start = time.time()
        w.axpy(1.0, y1)
        w.copy(result=x1)
        w.copy(result=w1)
        w.axpy(-1.0, x0)
        end = time.time()
        cputime["add"] += end-start
        
        start = time.time()
        dp = w.dot(z)
        end = time.time()
        cputime["dot"] += end-start

        start = time.time()
        if it > Kre + K1 and restart and dp >= 0.0:
            Kre = it
            K1 = 2*K0
            l += 1
            a1, a0 = 1, 0

            x0.copy(result=y1)
            x0.copy(result=x1)
        else:
            a2 = (1.0 + np.sqrt(1+4.0*a1**2)) / 2.0
            w1.axpby(-(a1-1)/a2, 1+(a1-1)/a2, x0)
            w1.copy(result=y1)

            a1, a0 = a2, a1
            x1.copy(result=x0)
        end = time.time()
        cputime["update"] += end-start

        it += 1

    return x1, it, cputime









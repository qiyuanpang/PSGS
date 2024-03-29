from mpi4py import MPI
import time

import sys, petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc
import numpy

import scipy.io as sio
from scipy.sparse import identity
from scipy import sparse

import time

from psgs import psgs
from utils import relerr

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()
# comm = PETSc.Comm.COMM_WORLD
# comm_size = comm.getSize()
# comm_rank = comm.getRank()

repeats = 10
a0 = 2.0

opts = PETSc.Options()
n = 715176

allopts = opts.getAll()

A = PETSc.Mat().create()
A.setSizes([n, n])
A.setFromOptions()
A.setUp()

filepath = "../oracle/graph/apache2/static/"+str(n)+"/apache2.mat"
data = sio.loadmat(filepath)
L = data["Problem"][0][0][2]

rstart, rend = A.getOwnershipRange()

# rows
for i in range(rstart, rend):
    l = L[i, :]
    I, J, V = sparse.find(l)
    A[i, J.tolist()] = V.tolist()

A.assemble()

b = A.createVecLeft()
b.setRandom()
b.assemble()

x0 = A.createVecRight()
x0.setRandom()
x0.assemble()

rerr0 = relerr(A, x0, b)

itmax = 10
cputime_avg = {}
for i in range(repeats):
    x, it, cputime = psgs(A, b, itmax=itmax, x0=x0)
    for key in cputime:
        if key not in cputime_avg:
            cputime_avg[key] = cputime[key]/repeats
        else:
            cputime_avg[key] += cputime[key]/repeats
time_avg = sum([cputime_avg[key] for key in cputime_avg])

rerr1 = relerr(A, x, b)

Print = PETSc.Sys.Print

Print()
Print("******************************")
Print("******** PSGS Results ********")
Print("******************************")
Print()

Print("Processes: %d" % comm_size)
Print("Problem size: %d" % n)
Print("Matrix: apache2")
Print("File path: %s" % filepath)
Print("Max iteration: %d" % itmax)
Print("Restart: True")
Print("Initial relerr: %.2e" % rerr0)
Print("Computed relerr: %.2e" % rerr1)
Print("Execution time(s): %.2e" % time_avg)
for key in cputime_avg:
    Print("    %s time(s): %.2e" % (key, cputime_avg[key]))





Print()
Print("******************************")
Print("******** KSPCG Results ********")
Print("******************************")
Print()

x = A.createVecRight()
x.setRandom()
x.assemble()

ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
ksp.setFromOptions()
ksp.setTolerances(rtol=0, max_it=100)
ksp.setType(PETSc.KSP.Type.CG)
ksp.setOperators(A)

rerr0 = relerr(A, x, b)
start = time.time()
for i in range(repeats):
    pc = PETSc.PC().create(comm=PETSc.COMM_WORLD)
    pc.setOperators(A)
    ksp.setPC(pc)
    ksp.solve(b, x) 
    pc.destroy()
end = time.time()
time_avg = (end-start)/repeats
rerr1 = relerr(A, x, b)


Print("Processes: %d" % comm_size)
Print("Problem size: %d" % n)
Print("Matrix: apache2")
Print("File path: %s" % filepath)
Print("Preconditioner: BJACOBI")
Print("Max iteration: %d" % ksp.getIterationNumber())
Print("Initial relerr: %.2e" % rerr0)
Print("Computed relerr: %.2e" % rerr1)
Print("Execution time(s): %.2e" % time_avg)


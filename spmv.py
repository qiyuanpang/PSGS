from mpi4py import MPI

import sys, slepc4py
slepc4py.init(sys.argv)

from petsc4py import PETSc
# from slepc4py import SLEPc
import numpy as np

import scipy.io as sio
from scipy.sparse import identity
from scipy import sparse

import time

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()

repeats = 10

opts = PETSc.Options()
n = opts.getInt('n', 30)

Print = PETSc.Sys.Print

allopts = opts.getAll()
# pc = allopts["st_pc_type"]
# bs = allopts["eps_lobpcg_blocksize"]

A = PETSc.Mat().create()
A.setSizes([n, n])
A.setFromOptions()
A.setUp()

Print("Matrix type: %s" % A.getType())

if n == 1000000:
    fname = "graph/LBOLBSV/static/"+str(n)+"/static_lowOverlap_lowBlockSizeVar_"+str(n)+"_nodes_L_"
elif n == 5000000:
    fname = "graph/LBOHBSV/static/"+str(n)+"/static_lowOverlap_highBlockSizeVar_"+str(n)+"_nodes_L_"
elif n == 20000000:
    fname = "graph/LBOHBSV/static/"+str(n)+"/static_lowOverlap_highBlockSizeVar_"+str(n)+"_nodes_L_"
elif n == 18571154:
    fname = "graph/MAWI/static/"+str(n)+"/static_mawi_"+str(n)+"_nodes_L_"
elif n == 16777216:
    fname = "graph/GRAPH500/static/"+str(n)+"/static_graph500-scale24-ef16_"+str(n)+"_nodes_L_"
else:
    fname = "graph/LBOLBSV/static/"+str(n)+"/static_lowOverlap_lowBlockSizeVar_"+str(n)+"_nodes_L_"

data = sio.loadmat(fname+str(comm.rank)+".mat")
L = data["data"]
L = L.tocsr()

rstart, rend = A.getOwnershipRange()

# s = rend - rstart
# mpi_s = 'rank %i has %s'%(comm.rank, s)
# comm.send(mpi_s, tag=11, dest=0)
# mpi_s_list = []
# if comm.rank == 0:
#     mpi_s_list = []
#     for i in range(comm.size):
#         mpi_s_list.append(comm.recv(source=i, tag=11))

#     print(mpi_s_list)

# mpi_s_list = comm.bcast(mpi_s_list, root=0)


# rows
# for i in range(rstart, rend):
I, J, V = L.indptr, L.indices, L.data
assert L.shape[0] == rend - rstart
A.setValuesLocalCSR(I, J, V)
# A[rstart, 1] = 1.0

A.assemble()


mpi_s = 'rank %i has %f, %f'%(comm.rank, np.sum(V), V[0])
comm.send(mpi_s, tag=11, dest=0)
mpi_s_list = []
if comm.rank == 0:
    mpi_s_list = []
    for i in range(comm.size):
        mpi_s_list.append(comm.recv(source=i, tag=11))

    print(mpi_s_list)

mpi_s_list = comm.bcast(mpi_s_list, root=0)


Print("is Symmetry: %s" % A.isSymmetric())

L = None

vl, vr = A.createVecs()

vr.setRandom()

vr.assemble()

vl.set(0.0)

vl.assemble()

cputime = float('inf')
for ii in range(repeats):
    start_time = time.time()
    A.mult(vr, vl)
    end_time = time.time()

    elapsed = end_time - start_time
    cputime = elapsed if ii == 0 else min(cputime, elapsed)


Print()
Print("******************************")
Print("*** SpMV ***")
Print("******************************")
Print()

Print("Nodes: %d" % comm_size)
Print("Problem size: %d" % n)
Print("File path: %s" % fname)
Print("Time: %.2e" % cputime)




import numpy

import scipy.io as sio
from scipy.sparse import identity
from scipy import sparse

import time



def dvd(N, p):
    return [N // p + 1 if i < N % p else N // p for i in range(p)]

if __name__ == "__main__":
    p = 64

    n = 1585478
    mat = "G3_circuit"
    filepath = "../oracle/graph/G3_circuit/static/"+str(n)+"/G3_circuit.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)


    n = 986703
    mat = "bone010"
    filepath = "../oracle/graph/bone010/static/"+str(n)+"/bone010.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)


    n = 715176
    mat = "apache2"
    filepath = "../oracle/graph/apache2/static/"+str(n)+"/apache2.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)


    n = 943695
    mat = "audikw_1"
    filepath = "../oracle/graph/audikw_1/static/"+str(n)+"/audikw_1.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)


    n = 1465137
    mat = "StocF-1465"
    filepath = "../oracle/graph/StocF-1465/static/"+str(n)+"/StocF-1465.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)



    n = 1228045
    mat = "thermal2"
    filepath = "../oracle/graph/thermal2/static/"+str(n)+"/thermal2.mat"
    data = sio.loadmat(filepath)
    L = data["Problem"][0][0][2]
    ttl_nnz = L.count_nonzero()
    d = dvd(n, p)
    max_nnz = 0
    for i in range(p):
        max_nnz = max(max_nnz, L[sum(d[:i]):sum(d[:i+1]), :].count_nonzero())
    print(mat, n, ttl_nnz/n, ttl_nnz, max_nnz*p/ttl_nnz)






import matplotlib.pyplot as plt
import numpy as np


def getdata(data, nps, N, mat, maxit):
    op = {}
    for i in range(len(data)):
        line = data[i]
        if len(line) == 2 and line[0] == "Processes:" and data[i-3][1] == "PSGS":
            if int(data[i][1]) == nps and int(data[i+1][2]) == N and data[i+2][1] == mat: 
                maxit_fac = maxit/int(data[i+4][2])
                op["totals"] = float(data[i+8][2])
                op["setup Jii"] = float(data[i+9][2])
                op["initialize"] = float(data[i+10][2])
                op["totals"] = op["setup Jii"] + op["initialize"] + (op["totals"]-op["setup Jii"]-op["initialize"])*maxit_fac
                op["matvec"] = float(data[i+11][2])*maxit_fac
                op["solve Jii"] = (float(data[i+12][2])+float(data[i+13][2])+float(data[i+14][2]))*maxit_fac
                op["add"] = float(data[i+15][2])*maxit_fac
                op["dot"] = float(data[i+16][2])*maxit_fac
                # op["update"] = float(data[i+17][2])
                break

    return op


def getdata2(data, nps, N, mat, maxit):
    op = {}
    for i in range(len(data)):
        line = data[i]
        if len(line) == 2 and line[0] == "Processes:" and data[i-3][1] == "KSPCG":
            if int(data[i][1]) == nps and int(data[i+1][2]) == N and data[i+2][1] == mat: 
                it = int(data[i+5][2])
                op["totals"] = float(data[i+8][2])*maxit/it
                # op["update"] = float(data[i+17][2])
                break

    return op


if __name__ == "__main__":
    filename1 = "./results/results_test_psgs_scalability_G3_circuit_ksp_p1m.log"
    filename2 = "./results/results_test_psgs_scalability_bone010_ksp_p1m.log"
    filename3 = "./results/results_test_psgs_scalability_apache2_ksp_p715k.log"
    filename4 = "./results/results_test_psgs_scalability_audikw_1_ksp_p943k.log"
    filename5 = "./results/results_test_psgs_scalability_stocf-1465_ksp_p1m.log"
    filename6 = "./results/results_test_psgs_scalability_thermal2_ksp_p1m.log"
    data1 = []
    f = open(filename1, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)
    f = open(filename2, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)
    f = open(filename3, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)
    f = open(filename4, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)
    f = open(filename5, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)
    f = open(filename6, mode='r')
    for line in f:
        line = line.split()
        data1.append(line)

    mats = ["G3_circuit", "bone010", "apache2", "audikw_1", "StocF-1465", "thermal2"]
    
    mat2N = {"G3_circuit": 1585478, "bone010": 986703, "apache2": 715176, "audikw_1": 943695, "StocF-1465": 1465137, "thermal2": 1228045}
    stat1 = {}
    for nps in [4, 16, 64, 121]:
        for mat in mats:
            for maxit in [100]:
                stat1[(nps, mat, maxit)] = getdata(data1, nps, mat2N[mat], mat, maxit)
    
    stat2 = {}
    for nps in [4, 16, 64, 121]:
        for mat in mats:
            for maxit in [100]:
                stat2[(nps, mat, maxit)] = getdata2(data1, nps, mat2N[mat], mat, maxit)


    tosave = "./results/"
    
    for mat in mats:
        npss = [4,16,64,121]
        plt.figure(dpi=200)
        plt.plot(npss, [stat1[(nps, mat, maxit)]["totals"] for nps in npss], 'o-')
        plt.plot(npss, [stat1[(nps, mat, maxit)]["setup Jii"] for nps in npss], 'o-')
        plt.plot(npss, [stat1[(nps, mat, maxit)]["matvec"] for nps in npss], 'o-')
        plt.plot(npss, [stat1[(nps, mat, maxit)]["solve Jii"] for nps in npss], 'o-')
        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xlabel('number of processes')
        plt.ylabel('seconds')
        plt.title(mat)
        plt.legend(['totals', 'setup J', 'matvec', 'solve J'], loc='lower left')
        plt.savefig(tosave+'test_psgs_scalability_'+mat+'.png')
        plt.close()

    
    for mat in mats:
        npss = [4,16,64,121]
        plt.figure(dpi=200)
        plt.plot(npss, [stat1[(nps, mat, maxit)]["totals"] for nps in npss], 'o-')
        plt.plot(npss, [stat2[(nps, mat, maxit)]["totals"] for nps in npss], 'o-')
        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xlabel('number of processes')
        plt.ylabel('seconds')
        plt.title(mat)
        plt.legend(['p acc-jacobi', 'KSPCG'], loc='lower left')
        plt.savefig(tosave+'comp_psgs_scalability_'+mat+'.png')
        plt.close()
    

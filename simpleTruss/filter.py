# THIS FILE NEEDS TO BE FIXED
import numpy as np
from math import floor

def check(nelx, nely, rmin, x, dc):
    dcn = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            sum=0.0
            for k in range(max(i-floor(rmin), 0), min(i+floor(rmin),nelx)):
                for l in range(max(j-floor(rmin), 0), min(j+floor(rmin),nely)):
                    # print("l", l, "k", k)
                    # print("dc[l,k]: ", dc[l,k])
                    fac = rmin-np.sqrt((i-k)**2+(j-l)**2)
                    sum = sum + max(0,fac)
                    dcn[j,i] = dcn[j,i] + max(0,fac)*x[l,k]*dc[l,k]

            dcn[j,i] = dcn[j,i]/(x[j,i]*sum) 
    # print(dcn)
    return dcn
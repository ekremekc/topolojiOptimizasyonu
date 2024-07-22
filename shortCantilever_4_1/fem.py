import numpy as np
from scipy.sparse import coo_matrix, coo_array, csr_matrix, csr_array, lil_array, lil_matrix
from scipy.sparse.linalg import spsolve

def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
	return KE

def FE(nelx, nely, x, penal):
    ndof = 2*(nelx+1)*(nely+1)
    Kn = lil_matrix((ndof,ndof))
    Fn = lil_array((ndof,1))
    Un = lil_array((ndof,1))
    
    KE = lk()

    for ely in range(nely):
        for elx in range(nelx):
            n1 = (nely+1)*(elx)   + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3,2*n1+2, 2*n1+3]
            Kn[np.ix_(edof,edof)] += x[ely,elx]**penal*KE

    Fn[2*(nelx+1)*(nely+1)-1] = -1
    print(Fn)

    dofs  = np.arange(ndof)
    fixed = np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
    free  = np.setdiff1d(dofs,fixed)
    
    Kn = Kn[free,:][:,free]
    Un[free]=spsolve(Kn,Fn[free])
    Un[fixed] = 0

    return Un


if __name__ == "__main__":
    # Default input parameters
    nelx=32
    nely=20
    ndof = 2*(nelx+1)*(nelx+1)
    x = np.full((nelx,nely), 0.4)
    penal = 0.3
    Un = FE(nelx, nely, x, penal)
    print(type(Un))
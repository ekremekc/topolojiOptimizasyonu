import numpy as np
from scipy.sparse import coo_matrix, coo_array, csr_matrix, csr_array, lil_array, lil_matrix
from scipy.sparse.linalg import spsolve
from os import environ
from numpy.linalg import solve as npsolve
from petsc4py import PETSc


# environ['OMP_NUM_THREADS'] = '4'

def lk():
    KE = np.array([ [2/3, -1/6, -1/3, -1/6],
                    [-1/6, 2/3, -1/6, -1/3],
                    [-1/3, -1/6, 2/3, -1/6],
                    [-1/6, -1/3, -1/6, 2/3]])
    return KE

def FE(nelx, nely, x, penal, backend='scipy'):
    
    ndof = (nelx+1)*(nely+1)
    Kn = lil_matrix((ndof,ndof))
    KE = lk()

    for ely in range(nely):
        for elx in range(nelx):
            n1 = (nely+1)*(elx)   + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = [n1-1, n2-1, n2, n1]
            Kn[np.ix_(edof,edof)] += (0.001+0.999*x[ely,elx]**penal)*KE

    

    dofs  = np.arange(ndof,dtype=np.int32)
    fixed = np.arange(int(nely/2+1-(nely/20)),int(nely/2+1+(nely/20)), 1)
    free  = np.setdiff1d(dofs,fixed)
    print(fixed)

    if backend=='scipy':

        # scipy backend
        # We apply heat sink BC over the boundary
        Fn = lil_array((ndof,1))
        Un = lil_array((ndof,1))
        Fn[:,0] = 0.01
        
        Kn = Kn[free,:][:,free]
        Kn_csr = Kn.tocsr()
        Fn_csr = Fn.tocsr()

        Un[free] = spsolve(Kn_csr,Fn_csr[free], permc_spec='MMD_AT_PLUS_A')
        Un[fixed] = 0

    elif backend=='petsc':

        # TODO PETSc backend - in progress (NOT WORKING, there is some problem in the preconditioning and vector F)
        Kn_csr = Kn.tocsr()
        Fn = np.full(ndof, 0.01)

        K_petsc = PETSc.Mat().createAIJ(size=Kn_csr.shape,
                                    csr=(Kn_csr.indptr, Kn_csr.indices,
                                            Kn_csr.data))
        K_petsc.assemble()
        ksp = PETSc.KSP().create()
        ksp.setOperators(K_petsc)
        F_petsc = K_petsc.createVecLeft()
        U_petsc = K_petsc.createVecRight()
        ksp.setType('gmres')
        ksp.setConvergenceHistory()
        # ksp.getPC().setType('lu')
        F_petsc.setArray(Fn)
        ksp.solve(F_petsc[free], U_petsc[free])
        Un = np.asarray(U_petsc.array)
        Un[fixed] = 0


    # print(Un)

    return Un


if __name__ == "__main__":

    import datetime
    start_time = datetime.datetime.now()

    nelx=3
    nely=3
    x = np.full((nelx,nely), 0.4)
    penal = 0.3
    Un_scipy = FE(nelx, nely, x, penal, backend='scipy')
    print(Un_scipy)
    Un_petsc = FE(nelx, nely, x, penal, backend='petsc')
    print(Un_petsc)

    print("Toplam hesaplama suresi: ", datetime.datetime.now()-start_time)
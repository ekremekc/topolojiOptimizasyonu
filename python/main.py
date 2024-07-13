import numpy as np
np.set_printoptions(linewidth=300, precision=3) # Tam matris gösterimleri için

from fem import FE, lk
from filter import check
from sensitivity import oc
import matplotlib.pyplot as plt

def top(nelx, nely, volfrac, penal, rmin):
    x = np.full((nely, nelx), volfrac)
    dc = np.zeros((nely, nelx))
    loop = 0
    change = 1

    while change >0.01:
        print("loop: ", loop, "change: ", change)
        loop += 1
        xold = x
        # Amaç fonksiyonu ve hassasiyet analizi
        U = FE(nelx, nely, x, penal)
        KE = lk()
        # OK TILL HERE
        c = 0
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely+1)*(elx)   + ely
                n2 = (nely+1)*(elx+1) + ely
                edof = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3,2*n1+2, 2*n1+3]
                Ue = U[edof]

                c = c+x[ely, elx]**penal*Ue.T@KE@Ue
                dc[ely, elx] = -penal*x[ely, elx]**(penal-1)*Ue.T@KE@Ue
        

        dc = check(nelx, nely, rmin, x, dc)
        
        x = oc(nelx, nely, x, volfrac, dc)
        
        change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)

    plt.contourf(x)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    nelx = 90
    nely = 30
    volfrac = 0.5
    penal = 3.0
    rmin = 1.5
    top(nelx, nely, volfrac, penal, rmin)
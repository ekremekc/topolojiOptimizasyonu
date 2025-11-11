import numpy as np
np.set_printoptions(linewidth=300, precision=3) # Tam matris gösterimleri için
from fem import FE, lk
from filter import filtre
from sensitivity import oc
import matplotlib.pyplot as plt

def toph(nelx, nely, volfrac, penal, rmin):
    
    x = np.full((nely, nelx), volfrac)
    dc = np.zeros((nely, nelx))
    loop = 0
    change = 1

    # Grafikleri başlatır.
    plt.ion()
    figure, ax = plt.subplots(figsize=(15, 15))
    ax.yaxis.set_inverted(True)

    while change >0.01:
        loop += 1
        print("loop: ", loop, "change: ", change)
        xold = x
        # Amaç fonksiyonu ve hassasiyet analizi
        U = FE(nelx, nely, x, penal)
        # U = FE(nelx, nely, x, penal, backend='petsc')
        KE = lk()
        c = 0
        for ely in range(nely):
            for elx in range(nelx):
                n1 = (nely+1)*(elx)   + ely
                n2 = (nely+1)*(elx+1) + ely
                edof = [n1-1, n2-1, n2, n1]
                Ue = U[edof]

                c = c+(0.001+0.999*x[ely, elx]**penal)*Ue.T@KE@Ue
                dc[ely, elx] = -0.999*penal*x[ely, elx]**(penal-1)*Ue.T@KE@Ue

        dc = filtre(nelx, nely, rmin, x, dc)
        x = oc(nelx, nely, x, volfrac, dc)
        
        change=np.linalg.norm(x.reshape(nelx*nely,1)-xold.reshape(nelx*nely,1),np.inf)

        # guncellenen tasarım alanını tekrar çizer.
        ax.contourf(x, cmap="gray")
        figure.canvas.draw()
    
        # Grafiği sürekli açık tutar.
        figure.canvas.flush_events()
    
    np.save("distribution", x)
    plt.savefig("topoloji.pdf")
    plt.show()


if __name__ == "__main__":

    import datetime

    start_time = datetime.datetime.now()

    nelx = 20   
    nely = 20
    volfrac = 0.4
    penal = 3.0
    rmin = 1.2
        
    toph(nelx, nely, volfrac, penal, rmin)

    print("Toplam hesaplama suresi: ", datetime.datetime.now()-start_time)
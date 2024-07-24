import numpy as np
from math import floor

def filtre(nelx, nely, rmin, x, dc):
    """Bu fonksiyon, bulunan tasarım alanı türevini filtreleyip daha 
    anlamlı tasarım alanına çevirmek için kullanılır.
    Args:
        nelx (_type_): yatay eksendeki eleman sayısı
        nely (_type_): dikey eksendeki eleman sayısı
        rmin (_type_): filtrelemede kullanılan, elemanlar arası olabilecek maksimum yarıçap
        x (_type_): tasarım alanı
        dc (_type_): tasarım alanı türevi

    Returns:
        _type_: _description_
    """
    dcn = np.zeros((nely, nelx))
    for i in range(nelx):
        for j in range(nely):
            sum=0.0
            for k in range(max(i-floor(rmin), 0), min(i+floor(rmin)+1,nelx)):
                for l in range(max(j-floor(rmin), 0), min(j+floor(rmin)+1,nely)):
                    fac = rmin-np.sqrt((i-k)**2+(j-l)**2)
                    sum = sum + max(0,fac)
                    dcn[j,i] = dcn[j,i] + max(0,fac)*x[l,k]*dc[l,k]
            dcn[j,i] = dcn[j,i]/(x[j,i]*sum)
    return dcn
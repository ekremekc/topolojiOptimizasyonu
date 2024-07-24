import numpy as np

# Optimality criterion
def oc(nelx,nely,x,volfrac,dc, passive):
	l1=0
	l2=100000
	move=0.2
	while (l2-l1)>1e-4:
		lmid=0.5*(l2+l1)
		xnew= np.maximum(0.001,np.maximum(x-move,np.minimum(1.0,np.minimum(x+move,x*np.sqrt(-dc/lmid)))))
		xnew[passive == 1.] = 0.001
		if np.sum(np.sum(xnew))-volfrac*nelx*nely>0 :
			l1=lmid
		else:
			l2=lmid
	return xnew
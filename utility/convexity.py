#!/usr/bin/env python

import numpy as np

def posDef(M):

    n = int(np.sqrt(len(M)))
    M.resize((n,n))
    try:
        np.linalg.cholesky(0.5*(M+M.T))
        return 1
    except np.linalg.LinAlgError:
        return 0

def posDefMult(Ms):

    ind = np.zeros(Ms.shape[0],dtype=np.bool)
    for i in range(Ms.shape[0]):
        ind[i] = posDef(Ms[i])

    return ind

def find_wells(idnn,x):

    # Find "wells" (regions of convexity, with low gradient norm)

    # First, rereference the free energy
    pred = idnn.predict([x,x,x])
    mu_test = pred[1]
    eta_test = np.array([[0,0,0,0],
                         [0.25,0.25,0.25,0.25]])
    y = idnn.predict([eta_test,eta_test,eta_test])[0]
    g0 = y[0,0]
    g1 = y[1,0]
    mu_test[:,0] = mu_test[:,0] - 4*(g1 - g0)
    gradNorm = np.sqrt(np.sum(mu_test**2,axis=-1))

    H = pred[2] # get the list of Hessian matrices
    ind2 = posDefMult(H) # indices of points with local convexity
    eta = x[ind2]
    gradNorm = gradNorm[ind2]

    ind3 = np.argsort(gradNorm)
    
    # Return eta values with local convexity, sorted by gradient norm (low to high)

    return eta[ind3]

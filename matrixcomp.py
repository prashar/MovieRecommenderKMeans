#!/usr/bin/python2

from scipy import *
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy.random as rd
import scipy.linalg as la

def load_movielens(fname):
    ret = map(lambda x: map(int, x.split("::")),
              open(fname).read().splitlines())
    ret = array(ret)

    nu = ret[:, 0].max(); nm = ret[:, 1].max()
    ret[:, 0] -= 1; ret[:, 1] -= 1

    mat = sp.lil_matrix((nu, nm))
    mat[ret[:, 0], ret[:, 1]] = ret[:, 2]

    return (ret, mat.tocsc())

#Ignore the author's ignorance of Software engineering.
def initialize_computation(nu, nm, arr, k, lam):
    L = 1e-3 * randn(nu, k)
    R = 1e-3 * randn(nm, k)

    return [arr.copy(), L, R, lam]

def stochastic_gd(ssize, prob):
    (arr, L, R, lam) = prob    
    #Too stochastic ?
    perm = range(arr.shape[0])
    rd.shuffle(perm)    
    mse = 0.0
    for i in perm:
        u = arr[i, 0]
        v = arr[i, 1]
        res = (arr[i, 2] - L[u, :].dot(R[v, :]))
        mse += res**2
        L[u, :] -= ssize * (- res * R[v, :] + lam * L[u, :])
        R[v, :] -= ssize * (- res * L[u, :] + lam * R[v, :])
    #Returns the MSE over the training data set.
    return mse/float(arr.shape[0])

def matrix_factorize(prob, step=0, max_iter=30):
    results = zeros(max_iter)
    idx_results = 0 
    delta = 1e-2
    for i in range(max_iter):
        if(step == 0):
            mse = stochastic_gd(delta/sqrt(i+1), prob)
        elif(step == 1):
            mse = stochastic_gd(delta, prob)
        elif(step == 2):
            mse = stochastic_gd(delta/(i+1), prob)
        elif(step == 3):
            mse = stochastic_gd((delta/((i+1)**0.75)), prob)    
        results[idx_results] = mse    
        idx_results += 1
        print("MSE at the step %s: %s" % (i, mse))
    ''' 
    plt.plot(range(30),results) 
    if(step == 0):
        plt.title("learning_rate == delta/sqrt(t+1)")
    elif (step == 1):
        plt.title("learning_rate == delta")
    elif ( step == 2):
        plt.title("learning_rate = delta/t")
    elif ( step == 3):
        plt.title("learning_rate == delta/i**.75)") 

    plt.ylabel("MSE_Training")
    plt.xlabel("step")
    plt.grid()
    plt.show()
    '''
    return prob

def predict_ratings(tarr, prob):
    (tmp, L, R, lam) = prob        
    carr = zeros(tarr.shape[0])
    for i in range(tarr.shape[0]):
        carr[i] = L[tarr[i, 0], :].dot(R[tarr[i, 1], :])
    return carr

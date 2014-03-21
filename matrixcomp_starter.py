from scipy import *
import numpy.random as rd
import scipy.sparse as sp
import matrixcomp as mf
from matplotlib.pyplot import * 

def user_ratings(ui, mat, movie, n=10):
    vec = mat[ui, :].tocsr()

    ret = []
    idxs = map(lambda x: x[0], movie)
    for vi in vec.indices:
        try:
            mname = movie[idxs.index(vi)][1]
        except ValueError:
            mname = "<Not present>"
        ret.append([mat[ui, vi], mname])
    ret.sort(key=lambda x: -x[0])
    return ret[:n]

def user_predict(ui, prob, movie, n=10):
    (arr, L, R, lam) = prob

    rats = L[ui, :].dot(R.T)
    rank = argsort(-rats)

    ret = []
    idxs = map(lambda x: x[0], movie)
    for vi in rank[:n]:
        try:
            mname = movie[idxs.index(vi)][1]
        except ValueError:
            mname = "<Not present>"

        ret.append([rats[vi], mname])
    ret.sort(key=lambda x: -x[0])
    return ret

#
(oarr, mat) = mf.load_movielens("ml-1m/ratings.dat")
movie = map(lambda x: [int(x[0]) - 1] + x[1:], map(lambda x: x.split("::"), open("ml-1m/movies.dat").read().splitlines()))

rd.shuffle(oarr)
#Number of users
nu = oarr[:, 0].max() + 1
#Number of movies
nm = oarr[:, 1].max() + 1

#Split dataset
tridx = floor(0.6 * oarr.shape[0])
vaidx = floor(0.8 * oarr.shape[0])

trarr = oarr[:tridx, :]
vaarr = oarr[tridx:vaidx, :]
tearr = oarr[vaidx:, :]

lam = 1e-2

'''
range_ks = array([0])
vmse = zeros(4)
tmse = zeros(4)
idx = 0 
for step_size in range_ks:
    prob = mf.initialize_computation(nu, nm, trarr, 100, lam)
    #This passes over the data about 30 times over (see the implementation).
    mf.matrix_factorize(prob,step=step_size)

    #Predict the mse
    t_mse = mean((mf.predict_ratings(trarr, prob) - trarr[:, 2])**2)
    v_mse = mean((mf.predict_ratings(vaarr, prob) - vaarr[:, 2])**2)

    print "TMSE: ", t_mse , "| VMSE: ", v_mse
    vmse[idx] = v_mse
    tmse[idx] = t_mse

    idx += 1

print range_ks
print tmse
print vmse
'''

vmse = zeros(4)
idx = 0 
range_ks = array([100])
for k in range_ks:
    prob = mf.initialize_computation(nu, nm, trarr, k, lam)
    #This passes over the data about 30 times over (see the implementation).
    mf.matrix_factorize(prob)

    #Predict the mse
    mse = mean((mf.predict_ratings(vaarr, prob) - vaarr[:, 2])**2)
    print mse
    vmse[idx] = mse
    idx += 1

print range_ks
print vmse

##########################

#Top predictions
ui = 99
top_pre = user_predict(ui, prob, movie)
top_rks = user_ratings(ui, mat, movie)

print "Top Predictions for ID 100 --------"
print top_pre
print "---------"
print top_rks

ui = 98
top_pre = user_predict(ui, prob, movie)
top_rks = user_ratings(ui, mat, movie)

print "Top Predictions for ID 99 --------"
print top_pre
print "---------"
print top_rks

###################
'''
smallest_mse = argmin(vmse)
k_tilde = range_ks[smallest_mse] 
print "k_star is ", k_tilde, " for ", vmse[smallest_mse] 

# It's time to train again on k_tilde
prob = mf.initialize_computation(nu, nm, trarr, k_tilde, lam)
#This passes over the data about 30 times over (see the implementation).
mf.matrix_factorize(prob)

#Predict the mse
test_mse = mean((mf.predict_ratings(tearr, prob) - tearr[:, 2])**2)
print "Test MSE:" , test_mse
    
#Top predictions
ui = 99
top_pre = user_predict(ui, prob, movie)
top_rks = user_ratings(ui, mat, movie)

print "Top Predictions for ID 100 --------"
print top_pre
print "---------"
print top_rks

ui = 98
top_pre = user_predict(ui, prob, movie)
top_rks = user_ratings(ui, mat, movie)

print "Top Predictions for ID 99 --------"
print top_pre
print "---------"
print top_rks
'''
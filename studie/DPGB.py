import numpy as np
from main import alpha_prediction
import csv
from DPGA import get_output
np.random.seed(42)
#stuksgewijs lineair signaal met sprongen en interacties. Met p = 10 en
#X ∼U[−1,1]p onafhankelijk nemen we
#f_B(x) = (2x2 +1)⊮{x1 ≤ 0}+(−1,5x2 −1)⊮{x1 > 0}+2x3⊮{x4 > 0,5}+x5.
def f(X):
    ind_1 = indicator(X[0],0,1,0) #is x1 x[0] of x[1]
    ind_2 = indicator(X[0],0,0,1)
    ind_3 = indicator(X[3],0.5,0,1)
    return (2*X[1]+1)*ind_1 + (-1.5*X[1] -1)*ind_2 + 2*X[2]*ind_3 + X[4] #x1 = X[0]

def indicator(x,m,a,b):
    if x <= m:
        return a
    else:
        return b

def estimate_sigma(functie):
    SNR = [1, 3, 10]
    sigma_squared = np.zeros(3)
    X = np.random.uniform(-1, 1, size=(10**6,10)) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    for i in range(3):
        sigma_squared[i] = np.var(fun) / SNR[i]
    return sigma_squared

def generate_data(functie,N,sigma_squared):
    X_train, train_y, train_ruis_y = data(functie,N,sigma_squared)
    X_test, test_y, test_ruis_y = data(functie,10000,sigma_squared)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(functie,n,sigma_squared):
    X = np.random.uniform(-1, 1, size=(n,10))
    fun = np.apply_along_axis(functie, 1, X)
    ruis = np.random.normal(loc=0, scale=np.sqrt(sigma_squared), size=n)
    fun_ruis = fun + ruis
    return X,fun, fun_ruis

SIGMA2 = estimate_sigma(f)

get_output("DPGB",f,SIGMA2)
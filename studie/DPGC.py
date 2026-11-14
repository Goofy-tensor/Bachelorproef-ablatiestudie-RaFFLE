from main import alpha_prediction
import numpy as np

np.random.seed(42)
#f(x) = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5

#van paper met friedman:
#zie bron 4 in methodologie
# In the first data set, there are ten independent predictor variables
#zl,..., zlo each of which is uniformly distributed over [0, 1]. The response is given by
#y = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5 + eps (eps is zo de ruis eh)

def f(X):
    return 10*np.sin(np.pi*X[1]*X[2]) + 20*(X[3] -0.5)**2 + 10*X[4] + 5*X[5]

def estimate_sigma(functie,SNR):
    X = np.random.uniform(0, 1, size=(10**6,10)) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    sigma_squared = np.var(fun)/SNR
    return sigma_squared

def generate_data(functie,SNR,N):
    sigma_squared = estimate_sigma(functie,SNR)
    X_train, train_y, train_ruis_y = data(functie,N,sigma_squared)
    X_test, test_y, test_ruis_y = data(functie,10000,sigma_squared)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(functie,n,sigma_squared):
    X = np.random.uniform(0, 1, size=(n,10))
    fun = np.apply_along_axis(functie, 1, X)
    ruis = np.random.normal(loc=0, scale=sigma_squared, size=n)
    fun_ruis = fun + ruis
    return X,fun, fun_ruis

#start met DPG B, SNR 3 en N=1000
X_trainA, train_yA, train_ruis_yA, X_testA, test_yA, test_ruis_yA = generate_data(f,3,100)
out = alpha_prediction(X_trainA,X_testA,train_ruis_yA,train_yA,test_yA)
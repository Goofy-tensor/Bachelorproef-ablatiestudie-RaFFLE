import numpy as np
from DPGA import get_output
#f(x) = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5

#van paper met friedman:
#zie bron 4 in methodologie
# In the first data set, there are ten independent predictor variables
#zl,..., zlo each of which is uniformly distributed over [0, 1]. The response is given by
#y = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5 + eps (eps is zo de ruis eh)

def f(X):
    return 10*np.sin(np.pi*X[0]*X[1]) + 20*(X[2] -0.5)**2 + 10*X[3] + 5*X[4] #x1 = X[0]

def estimate_sigma(functie):
    np.random.seed(42)  # zodat je meerdere keren dezelfde sigma_squared krijgt
    SNR = [1, 3, 10]
    sigma_squared = np.zeros(3)
    X = np.random.uniform(0, 1, size=(10**6,10)) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    for i in range(3):
        sigma_squared[i] = np.var(fun) / SNR[i]
    return sigma_squared

def generate_data(functie,N,sigma_squared,i):
    rng_train = np.random.default_rng([i, 0])
    rng_test  = np.random.default_rng([i, 1])
    X_train, train_y, train_ruis_y = data(functie,N,sigma_squared,rng_train)
    X_test, test_y, test_ruis_y = data(functie,10000,sigma_squared,rng_test)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(functie,n,sigma_squared,rng):
    X = rng.uniform(0, 1, size=(n,10))
    fun = np.apply_along_axis(functie, 1, X)
    ruis = rng.normal(loc=0, scale=np.sqrt(sigma_squared), size=n)
    return X, fun, fun + ruis


if __name__ == "__main__":
    SIGMA2 = estimate_sigma(f)
    get_output("DPGC",f,generate_data,SIGMA2)
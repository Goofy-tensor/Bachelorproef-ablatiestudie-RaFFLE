from main import alpha_prediction
import numpy as np
import csv
np.random.seed(42)
#f(x) = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5

#van paper met friedman:
#zie bron 4 in methodologie
# In the first data set, there are ten independent predictor variables
#zl,..., zlo each of which is uniformly distributed over [0, 1]. The response is given by
#y = 10sin(πx1x2)+20(x3−0,5)^2+10x4+5x5 + eps (eps is zo de ruis eh)

def f(X):
    return 10*np.sin(np.pi*X[0]*X[1]) + 20*(X[2] -0.5)**2 + 10*X[3] + 5*X[4] #x1 = X[0]

def estimate_sigma(functie):
    SNR = [1, 3, 10]
    sigma_squared = np.zeros(3)
    X = np.random.uniform(0, 1, size=(10**6,10)) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    for i in range(3):
        sigma_squared[i] = np.var(fun) / SNR[i]
    return sigma_squared

def generate_data(functie,N,sigma_squared):
    X_train, train_y, train_ruis_y = data(functie,N,sigma_squared)
    X_test, test_y, test_ruis_y = data(functie,10000,sigma_squared)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(functie,n,sigma_squared):
    X = np.random.uniform(0, 1, size=(n,10))
    fun = np.apply_along_axis(functie, 1, X)
    ruis = np.random.normal(loc=0, scale=np.sqrt(sigma_squared), size=n)
    fun_ruis = fun + ruis
    return X,fun, fun_ruis

SIGMA2 = estimate_sigma(f)
for N in [250,1000,4000]:
    for i in range(3):
        X_trainA, train_yA, train_ruis_yA, X_testA, test_yA, test_ruis_yA = generate_data(f, N, SIGMA2[i])
        out = alpha_prediction(X_trainA, X_testA, train_ruis_yA, test_ruis_yA, test_yA)
        with open(f'DPGA{N}-{i}.csv', 'w+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for j in range(9):
                spamwriter.writerow(out[j,])
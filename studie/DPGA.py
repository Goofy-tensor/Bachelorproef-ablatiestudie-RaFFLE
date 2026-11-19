import numpy as np
from main import iterate_predictions
import csv
#Y = f(X) + eps
#eps uit normale verdeling

#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (9 waarden), R = 30.

#Voor elk DGP geldt Y = f(X)+ε met ε ∼ N(0,σ2) onafhankelijk van X. We kiezen normale,
#constante ruis omdat de BIC uit de kansrekening komt voor normaal verdeelde fouten. Ons
#ruismodel past dus bij het criterium dat we onderzoeken.
#De signaal-ruisverhouding is
#SNR = Var(f(X))/σ2
#Het beste dat een model kan halen is een R2 van SNR/(1+SNR). Dat is 0,91 bij SNR = 10, 0,75
#bij SNR = 3 en 0,50 bij SNR = 1. We schatten Var(f(X)) één keer per DGP op 10^6 ruisvrije
#punten en kiezen σ2 = Var(f(X))/SNR. Zo betekent “SNR = 3” voor alle DGP’s hetzelfde. Per
#replicatie genereren we N trainingspunten en een aparte testset van 10000 punten, zowel met
#ruis (voor de test-MSE) als zonder ruis (voor de ruisvrije fout).


#DPG A
#Met p = 10 en X ∼ N(0,Σ), waarbij Σij = 0,3|i−j|, nemen we fA(x) = β⊤x, β =(3, −2, 1,5, −1, 0,5, 0, 0, 0, 0, 0)^⊤
def f(X):
    beta = np.array([3,-2,1.5,-1,0.5,0,0,0,0,0])
    return np.dot(beta,X)

def covariantie(N):
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            if i!=j:
                M[i,j] = 0.3**(abs(i-j))
            else:
                M[i,j] = 1/2
    return M + M.T

def estimate_sigma(m,c,functie):
    np.random.seed(42) #zodat je meerdere keren dezelfde sigma_squared krijgt
    SNR = [1,3,10]
    sigma_squared = np.zeros(3)
    X = np.random.multivariate_normal(m, c, size=10**6) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    for i in range(3):
        sigma_squared[i] = np.var(fun)/SNR[i]
    return sigma_squared

def generate_data(functie,N,sigma_squared,i):
    rng_train = np.random.default_rng([i, 0])   # aparte, reproduceerbare stromen
    rng_test  = np.random.default_rng([i, 1])
    X_train, train_y, train_ruis_y = data(mean,cov,functie,N,sigma_squared,rng_train)
    X_test, test_y, test_ruis_y = data(mean,cov,functie,10000,sigma_squared,rng_test)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(m,c,functie,n,sigma_squared,rng):
    X = rng.multivariate_normal(m, c, size=n)
    fun = np.apply_along_axis(functie, 1, X)
    ruis = rng.normal(loc=0, scale=np.sqrt(sigma_squared), size=n)
    return X, fun, fun + ruis

mean = np.zeros(10)
cov = covariantie(10)

def relatieve_mse(M):#delen door de waarde bij alpha=0.5 om te vergelijken
    #M is de (13,9) matrix, de MSE's zitten in rijen 4 t/m 7 (train/test forest, train/test boom)
    #niet in-place: M blijft intact, dus de absolute waarden blijven bewaard (nodig voor plots en sigma^2-lijn)
    mse = M[4:8,:]
    t_05 = (mse - mse[:,[6]])/mse[:,[6]]#relatieve verandering t.o.v. alpha = 0.5
    t_1 = (mse - mse[:,[8]])/mse[:,[8]]#relatieve verandering t.o.v. alpha = 1 (standaard PILOT)
    #MSE is altijd > 0 (er is ruis), dus geen if nodig
    #(enkel nog relevant bij R2: gebruik daar np.nan in plaats van 0, zodat het zichtbaar blijft)
    return t_05, t_1

def get_output(DPG,fun1,fun2,S):
    SNR = [1,3,10] #zelfde volgorde als in estimate_sigma
    for N in [250,1000,4000]:
        for i in range(3):#aanpassen i naar RNS
            out = iterate_predictions(fun1,fun2,S[i],DPG,N,SNR[i])
            with open(f'data{DPG}{N}-{SNR[i]}.csv', 'w+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                for j in range(13):#alle 13 rijen, absolute waarden (was range(9) en gedeeld door alpha=0.5)
                    spamwriter.writerow(out[0][j,])
            with open(f'relatief{DPG}{N}-{SNR[i]}.csv', 'w+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                t_05, t_1 = relatieve_mse(out[0])
                for rij in np.vstack((t_05, t_1)):#eerst 4 rijen t.o.v. 0.5, dan 4 rijen t.o.v. 1
                    spamwriter.writerow(rij)
            with open(f'hypothese{DPG}{N}-{SNR[i]}.csv', 'w+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                for k in range(1,11):
                    spamwriter.writerow(out[k])
    return

if __name__ == "__main__":
    SIGMA2 = estimate_sigma(mean, cov, f)
    get_output("DPGA",f,generate_data,SIGMA2)
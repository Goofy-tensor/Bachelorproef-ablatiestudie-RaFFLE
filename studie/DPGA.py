import numpy as np
from main import iterate_predictions
import csv
np.random.seed(42)
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

def estimate_sigma(mean,cov,functie):
    SNR = [1,3,10]
    sigma_squared = np.zeros(3)
    X = np.random.multivariate_normal(mean, cov, size=10**6) #Elke rij is 1 prediction (dus 10^6 rijen en len(X) kolommen)
    fun = np.apply_along_axis(functie, 1, X)# axis 0 = volgens kolommen, 1 volgens rijen
    for i in range(3):
        sigma_squared[i] = np.var(fun)/SNR[i]
    return sigma_squared

def generate_data(functie,N,sigma_squared):
    X_train, train_y, train_ruis_y = data(mean,cov,functie,N,sigma_squared)
    X_test, test_y, test_ruis_y = data(mean,cov,functie,10000,sigma_squared)
    return X_train, train_y, train_ruis_y, X_test, test_y, test_ruis_y

def data(mean,cov,functie,n,sigma_squared):
    X = np.random.multivariate_normal(mean, cov, size=n)
    fun = np.apply_along_axis(functie, 1, X)
    ruis = np.random.normal(loc=0, scale=np.sqrt(sigma_squared), size=n)
    fun_ruis = fun + ruis
    return X,fun, fun_ruis

mean = np.zeros(10)
cov = covariantie(10)
SIGMA2 = estimate_sigma(mean,cov,f)

def relative_R2(M):#delen door de waarde bij alpha=0.5 om te vergelijken
    for i in [0,1,2,3]:
        if M[i,6] > 0:
            M[:,i] = M[:,i]/M[6,i]
        else:
            M[:,i] = np.zeros(0)
            #deze rij beschouwen we dan als niet relevant, oi mss best door max delen dan
            #want mss is er dan een waarde die wel positief is dus dan kan je wel nog een relevante conclusie trekken
    return M

def get_output(DPG,fun,S):
    for N in [250,1000,4000]:
        for i in range(3):#aanpassen i naar RNS
            X_trainA, train_yA, train_ruis_yA, X_testA, test_yA, test_ruis_yA = generate_data(fun, N, S[i])
            out = iterate_predictions(X_trainA, X_testA, train_ruis_yA, test_ruis_yA, test_yA)
            with open(f'data{DPG}{N}-{i}.csv', 'w+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                relative_R2(out[0])
                for j in range(9):
                    spamwriter.writerow(out[0][j,])
            with open(f'hypothese{DPG}{N}-{i}.csv', 'w+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile)
                for k in range(1,9):
                    spamwriter.writerow(out[k])

get_output("DPGA",f,SIGMA2)
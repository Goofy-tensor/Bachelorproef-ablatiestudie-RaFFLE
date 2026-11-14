from DPGA import out
import numpy as np
import matplotlib.pyplot as plt

#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (9 waarden) (al in code), R = 30 (al in code)
alpha = [0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1]

def divide_by_max(l): #zo voor MSE dat je het kan vergelijken tov het maximum
    m = max(l) #moet dit min of max zijn?
    return l/m

def relative_RSS(M):
    for i in [0,1,5,6]:
        M[:,i] = divide_by_max(M[:,i])
    return M

def generalisatiegat(M): #test-train
    return M[:,5]-M[:,4], M[:,7]-M[:,6]#dit is fout want moet met MSE niet RSS

#out is voor DPGA SNR=3 N=1000
from tabulate import tabulate
F,G = generalisatiegat(out[0])
relative_RSS(out[0])
# assign data

a =  np.column_stack((np.column_stack((out[0], F)),G))

# create header
headers = ["trainR2", "testR2","train boom R2" , "test boom R2", "trainMSE", "testMSE","train boom MSE" , "test boom MSE", "maxdepth", "maxmodeldepth","ruisvrije error", "foutcorr" , "ambigu","G forest", "G boom"]

print(tabulate(a, headers=headers, tablefmt="grid"))

x = out[0][:,0]
y = out[0][:,1]
z = alpha
plt.figure(1)
plt.plot(z, x, 'r-', label = 'TrainRSS in fucntie van alpha', markersize=5)
plt.plot(z, y, 'b-', label = 'TestRSS in fucntie van alpha', markersize=1)
plt.xlabel('alpha', fontsize=12)
plt.ylabel('relatieve RSS', fontsize=12)
plt.title('relatieve RSS in functie van alpha', fontsize=10)
plt.legend(loc='best', fontsize=10)
plt.show()
#mss leuk om te plotten tegenover de boom MSE om visueel te zien dat optimale alpha forest <= opt alpha boom

#mss best node types weglaten
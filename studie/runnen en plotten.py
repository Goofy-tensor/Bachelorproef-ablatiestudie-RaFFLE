from DPGA import out
import numpy as np
import matplotlib.pyplot as plt

#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (9 waarden) (al in code), R = 30 (al in code)
alpha = [0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1]

def relative_R2(M):#delen door de waarde bij alpha=0.5 om te vergelijken
    for i in [0,1,2,3]:
        M[:,i] = M[:,i]/M[6,i]
    return M

def generalisatiegat(M): #test-train
    return M[:,5]-M[:,4], M[:,7]-M[:,6]

#testplotje
x = out[:,4]
y = out[:,5]
u = out[:,6]
v = out[:,7]
z = [1,2,3,4,5,6,7,8,9]

plt.figure(1)
plt.plot(z, x, 'r-', label = 'TrainMSE in fucntie van alpha', markersize=5)
plt.plot(z, y, 'b-', label = 'TestMSE in fucntie van alpha', markersize=1)
plt.plot(z, u, 'o-', label = 'TrainboomMSE in fucntie van alpha', markersize=5)
plt.plot(z, v, 'g-', label = 'TestboomMSE in fucntie van alpha', markersize=1)
plt.xlabel('alpha', fontsize=12)
plt.ylabel('relatieve MSE', fontsize=12)
plt.title('relatieve RSS in functie van alpha', fontsize=10)
plt.legend(loc='best', fontsize=10)
plt.show()

#algemene plots: (per N en SNR)
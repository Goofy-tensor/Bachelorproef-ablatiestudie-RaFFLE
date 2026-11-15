import numpy as np
import matplotlib.pyplot as plt
#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (9 waarden) (al in code), R = 30 (al in code)
alpha = [0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1]

def relative_R2(M):#delen door de waarde bij alpha=0.5 om te vergelijken
    for i in [0,1,2,3]:
        if M[6,i] > 0:
            M[:,i] = M[:,i]/M[6,i]
    return M

def generalisatiegat(M): #test-train
    return M[:,5]-M[:,4], M[:,7]-M[:,6]

#testplotje
#x = out[:,4]
#y = out[:,5]
#u = out[:,6]
#v = out[:,7]

#plt.figure(1)
#plt.plot(range(1,10), x, 'r-', label = 'TrainMSE in fucntie van alpha', markersize=5)
#plt.plot(range(1,10), y, 'b-', label = 'TestMSE in fucntie van alpha', markersize=1)
#plt.plot(range(1,10), u, 'o-', label = 'TrainboomMSE in fucntie van alpha', markersize=5)
#plt.plot(range(1,10), v, 'g-', label = 'TestboomMSE in fucntie van alpha', markersize=1)
#plt.xlabel('alpha', fontsize=12)
#plt.ylabel('relatieve MSE', fontsize=12)
#plt.title('relatieve RSS in functie van alpha', fontsize=10)
#plt.legend(loc='best', fontsize=10)
#plt.show()

#algemene plots: (per N en SNR)
def plot(DPG):
    N = [250,1000,4000]
    for j in range(len(N)):
        for i in range(len(N)):
            matrix = np.genfromtxt(f'{DPG}{N[j]}-{i}.csv', delimiter=',')
            trainMSE = matrix[:,4]
            testMSE =matrix[:,5]
            trainMSE_boom =matrix[:,6]
            testMSE_boom =matrix[:,7]
            z = [1, 2, 3, 4, 5, 6, 7, 8, 9]

            plt.figure(3*j+i)
            plt.plot(z, trainMSE, 'r-', label='TrainMSE in fucntie van alpha', markersize=5)
            plt.plot(z, testMSE, 'b-', label='TestMSE in fucntie van alpha', markersize=1)
            plt.plot(z, trainMSE_boom, 'o-', label='TrainboomMSE in fucntie van alpha', markersize=5)
            plt.plot(z, testMSE_boom, 'g-', label='TestboomMSE in fucntie van alpha', markersize=1)
            plt.xlabel('alpha', fontsize=12)
            plt.ylabel('relatieve MSE', fontsize=12)
            plt.title('relatieve RSS in functie van alpha', fontsize=10)
            plt.legend(loc='best', fontsize=10)
            plt.show()
    return
plot('DPGA') 
import numpy as np
from scipy import stats
import csv

def bereken_pwaarden(DPG):
    N = [250,1000,4000]
    res_pvalues = np.zeros((8,9))
    betrouwbaarheidsinterval = np.zeros((9,2)) #bootstrapbetrouwbaarheidsintervallen van versch in opt_alpha
    aantal_alpha = np.zeros((9,3)) #aantal pos,neg,en nulwaardes meedelen van het versch in opt_alpha

    for j in range(3):
        for i in range(3): #i aanpassen naar de SNR
            SNR = [1,3,10]
            matrix = np.genfromtxt(f'hypothese{DPG}{N[j]}-{SNR[i]}.csv', delimiter=',') # shape: (8,30)

            res_pvalues[0][3*j + i] = stats.wilcoxon(matrix[0,],alternative="greater").pvalue #rs_trainMSE
            res_pvalues[1][3 * j + i] = stats.wilcoxon(matrix[1,], alternative="less").pvalue #rs_G
            res_pvalues[2][3 * j + i] = stats.wilcoxon(matrix[2,], alternative="two-sided").pvalue #rs_rho
            res_pvalues[3][3 * j + i] = stats.wilcoxon(matrix[3,], alternative="two-sided").pvalue #minMSE_half
            res_pvalues[4][3 * j + i] = stats.wilcoxon(matrix[4,], alternative="two-sided").pvalue #versch_ambigu
            #versch = A(0) - A(1). H3b voorspelt +, H3a voorspelt -
            res_pvalues[5][3 * j + i] = stats.wilcoxon(matrix[5,], alternative="less").pvalue #opt alpha vergelijking

            a = min(matrix[6,])
            b = max(matrix[6,])
            betrouwbaarheidsinterval[3*j + i,] = [a+0.025*(b-a),b-0.025*(b-a)]
            aantal_alpha[3*j + i] = aantal(matrix[6,])

            res_pvalues[6][3 * j + i] = stats.wilcoxon(matrix[6,], alternative="greater").pvalue #c
            res_pvalues[7][3 * j + i] = stats.wilcoxon(matrix[6,], alternative="two-sided").pvalue #min_inwendig
            #p-waarde voor nulhypothse: ligt niet in het inwendige, min_inwendig geeft 0 als niet in inwendege en anders 1
            #dit is dus eig enkel relevant voor waarden waar c pos
    with open(f'p_waarden{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for k in range(8):
            spamwriter.writerow(res_pvalues[k,])
    with open(f'betrouwbaarheidsintervallen{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for l in range(9):
            spamwriter.writerow(betrouwbaarheidsinterval[l,])
    with open(f'aantal_alpha{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for m in range(9):
            spamwriter.writerow(aantal_alpha[m,])
    return res_pvalues

def holmcorrectie(pvalues):
    #pwaarden v.d. verschillende hypothesen sorteren en vergelijken met drempelwaarde p_i <= 0,05/(m-i+), m = tot hypothesen
    sorted_pvalues = np.sort(pvalues) #klein naar groot per rij
    vgl_grens = np.zeros((8,9)) #elke p-waarde vergelijken met de grens uit de Holm methode
    pos = np.zeros(8) #pos van eerste niet verworpen p-waarde
    for i in range(9):
        vgl_grens[:,i] = sorted_pvalues[:,i] - 0.05/(9-i)
    for j in range(8):
        pos[j] = positie(vgl_grens[j,])
    return pos,sorted_pvalues

def positie(l):
    for i in range(len(l)):
        if l[i] > 0:
            return i
    return -1

def aantal(l):
    res = np.zeros(3)
    for i in range(len(l)):
        if l[i] > 0:
            res[0] += 1
        elif l[i] == 0:
            res[1] += 1
        else:
            res[2] += 1
    return res

def verworpen_hypotheses(DPG,pvalues):
    pos,sorted_pvalues = holmcorrectie(pvalues)
    verworpen = np.ones((8,9))
    for i,j in enumerate(pos):
        if j != -1:
           for k in range(int(j),9):
               verworpen[i,np.where(pvalues[i] == sorted_pvalues[i,k])[0][0]] = 0
    with open(f'verworpen_hypotheses{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for m in range(8):
            spamwriter.writerow(verworpen[m,])
    return verworpen #geeft per soort hypothese voor welke SNR en N in het rooster de nulhypothese verworpen werd


if __name__ == "__main__":
    pvalues = bereken_pwaarden("DPGA")
    verworpen = verworpen_hypotheses("DPGA",pvalues)
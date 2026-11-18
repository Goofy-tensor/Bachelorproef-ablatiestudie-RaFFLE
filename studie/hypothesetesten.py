import numpy as np
from scipy import stats
import csv
from statsmodels.stats import multitest

def bereken_pwaarden(DPG):
    N = [250,1000,4000]
    res_pvalues = np.zeros((7,9))
    aantal_alpha = np.zeros((9,3)) #aantal pos,neg,en nulwaardes meedelen van het versch in opt_alpha
    rel_aantal_min_inwendig = np.zeros(9)
    for j in range(3):
        for i in range(3): #i aanpassen naar de SNR
            SNR = [1,3,10]
            matrix = np.genfromtxt(f'hypothese{DPG}{N[j]}-{SNR[i]}.csv', delimiter=',') # shape: (8,30)

            res_pvalues[0][3*j + i] = stats.wilcoxon(matrix[0,],alternative="greater", nan_policy="omit").pvalue #rs_trainMSE
            res_pvalues[1][3 * j + i] = stats.wilcoxon(matrix[1,], alternative="less", nan_policy="omit").pvalue #rs_G
            res_pvalues[2][3 * j + i] = stats.wilcoxon(matrix[2,], alternative="two-sided", nan_policy="omit").pvalue #rs_rho
            res_pvalues[3][3 * j + i] = stats.wilcoxon(matrix[3,], alternative="two-sided", nan_policy="omit").pvalue #minMSE_half
            res_pvalues[4][3 * j + i] = stats.wilcoxon(matrix[4,], alternative="two-sided", nan_policy="omit").pvalue #versch_ambigu
            #versch = A(0) - A(1). H3b voorspelt +, H3a voorspelt -
            res_pvalues[5][3 * j + i] = stats.wilcoxon(matrix[5,], alternative="less", nan_policy="omit").pvalue #opt alpha vergelijking
            aantal_alpha[3*j + i] = aantal(matrix[5,])

            res_pvalues[6][3 * j + i] = stats.wilcoxon(matrix[6,], alternative="greater", nan_policy="omit").pvalue #c
            rel_aantal_min_inwendig = sum(matrix[7,])/len(matrix[7,]) #hoeveel keer min inwendig is (en c > 0)
    with open(f'p_waarden{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for k in range(7):
            spamwriter.writerow(res_pvalues[k,])
    with open(f'relatief_aantal_min_inwendig{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(rel_aantal_min_inwendig)
    with open(f'aantal_alpha{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for m in range(9):
            spamwriter.writerow(aantal_alpha[m,])
    return res_pvalues

def holmcorrectie(DPG,pvalues):
    #pwaarden v.d. verschillende hypothesen sorteren en vergelijken met drempelwaarde p_i <= 0,05/(m-i+), m = tot hypothesen
    rejected = np.full((7, 9), np.nan)
    corrected_pvalues = np.full((7, 9), np.nan)
    for i in range(7):
        rij = pvalues[i,]
        geldig = ~np.isnan(rij)  # True waar er een echte p-waarde staat

        if geldig.sum() != 0:
            rej, corr, _, _ = multitest.multipletests(rij[geldig], method="holm")
            rejected[i, geldig] = rej
            corrected_pvalues[i, geldig] = corr

    with open(f'verworpen_hypotheses{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for m in range(7):
            spamwriter.writerow(rejected[m,])
    with open(f'aangepaste_pwaarden{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for l in range(7):
            spamwriter.writerow(corrected_pvalues[l,])
    return rejected, corrected_pvalues

def positie(l):
    for i in range(len(l)):
        if np.isnan(l[i]):
            return i
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


if __name__ == "__main__":
    pvaluesA = bereken_pwaarden("DPGA")
    verworpen = holmcorrectie("DPGA",pvaluesA)
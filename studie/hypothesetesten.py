import numpy as np
from scipy import stats
import csv
from statsmodels.stats import multitest

def bereken_pwaarden(DPG): #weeer alle indices aanpassen want weer extra waarde bijgekomen
    N = [250,1000,4000]
    res_pvalues = np.zeros((9,9))
    aantal_trainMSE = np.zeros((9, 4))  # aantal nan,pos,neg,en nulwaardes meedelen
    aantal_G = np.zeros((9, 4))
    aantal_rho = np.zeros((9, 4))

    med_trainMSE = np.zeros(9)
    med_G = np.zeros(9)
    med_rho = np.zeros(9)

    aantal_alpha = np.zeros((9,4))
    aantal_min_inwendig_forest = np.zeros((2,9)) #1e rij aantal min in inwendige => Uvorm, 2e rij is aantal monotoom
    aantal_min_inwendig_boom = np.zeros((2, 9))  # 1e rij aantal min in inwendige => Uvorm, 2e rij is aantal monotoom
    omkeringen = np.zeros((2,9))#1e rij gemiddelde en tweede rij max
    for j in range(3):
        for i in range(3): #i aanpassen naar de SNR
            SNR = [1,3,10]
            matrix = np.genfromtxt(f'hypothese{DPG}{N[j]}-{SNR[i]}.csv', delimiter=',') # shape: (8,30)

            res_pvalues[0][3*j + i] = safe_wilcoxon(matrix[0,],"greater") #rs_trainMSE
            omkeringen[0][3 * j + i] = np.average(matrix[1,])
            omkeringen[1][3 * j + i] = max(matrix[1,])
            med_trainMSE[3*j + i], aantal_trainMSE[3*j + i,] = effectgroottes(matrix[0,])


            res_pvalues[1][3 * j + i] = safe_wilcoxon(matrix[2,],"less") #rs_G
            med_G[3 * j + i], aantal_G[3 * j + i,] = effectgroottes(matrix[2,])

            res_pvalues[2][3 * j + i] = safe_wilcoxon(matrix[3,],"less") #rs_rho
            med_rho[3 * j + i], aantal_rho[3 * j + i,] = effectgroottes(matrix[3,])

            res_pvalues[3][3 * j + i] = safe_wilcoxon(matrix[4,],"two-sided") #minMSE_half

            res_pvalues[4][3 * j + i] = safe_wilcoxon(matrix[5,],"less") #opt alpha vergelijking
            aantal_alpha[3*j + i,] = effectgroottes(matrix[5,])[1]

            res_pvalues[5][3 * j + i] = safe_wilcoxon(matrix[6,],"greater") #c_forest
            aantal_min_inwendig_forest[0][3 * j + i] = np.sum(matrix[7,] == 1)  # hoeveel keer min inwendig is (en c > 0)
            aantal_min_inwendig_forest[1][3 * j + i] = np.sum(matrix[7,] == 2)  # hoeveel keer monotoom voor c > 0

            res_pvalues[6][3 * j + i] = safe_wilcoxon(matrix[8,],"greater") #c_boom
            aantal_min_inwendig_boom[0][3 * j + i] = np.sum(matrix[9,] == 1)  # hoeveel keer min inwendig is (en c > 0)
            aantal_min_inwendig_boom[1][3 * j + i] = np.sum(matrix[9,] == 2)  # hoeveel keer monotoom voor c > 0

            #nog aanpassen
            res_pvalues[7][3 * j + i] = safe_wilcoxon(matrix[10,], "greater")  #treeMSE_d0
            res_pvalues[8][3 * j + i] = safe_wilcoxon(matrix[11,], "greater")  #treeMSE_d1

    hulp_write(DPG,"p_waarden",res_pvalues)
    hulp_write(DPG, "aantal_min_inwendig_forest", aantal_min_inwendig_forest)
    hulp_write(DPG, "aantal_min_inwendig_boom", aantal_min_inwendig_boom)
    hulp_write(DPG, "omkeringen", omkeringen)
    hulp_write(DPG, "aantal_alpha", aantal_alpha)
    hulp_write(DPG, "aantal_trainMSE", aantal_trainMSE)
    hulp_write(DPG, "aantal_G", aantal_G)
    hulp_write(DPG, "aantal_rho", aantal_rho)

    hulp_write(DPG, "med_trainMSE", med_trainMSE)
    hulp_write(DPG, "med_G", med_G)
    hulp_write(DPG, "med_rho", med_rho)
    return res_pvalues

def hulp_write(DPG, name, values):
    values = np.atleast_2d(values)      # 1D (9,) wordt (1,9)
    with open(f'{name}{DPG}.csv', 'w+', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        for rij in values:
            spamwriter.writerow(rij)


def safe_wilcoxon(x,alternative): #als allen nul willen we dat het np.nan terug geeft
        goed = ~np.isnan(x)  # drop NaNs
        d = x[goed] #x moet hiervoor een numpy array zijn anders omzetten met np.asarray
        if d.size == 0 or np.all(d == 0):
            return np.nan  # als alle res nul of x alleen nan dan np.nan teruggeven
        else:
            return stats.wilcoxon(d,alternative=alternative).pvalue

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

def effectgroottes(l):
    res = np.zeros(4)
    for i in range(len(l)):
        if np.isnan(l[i]):
            res[0] += 1
        elif l[i] > 0:
            res[1] += 1
        elif l[i] == 0:
            res[2] += 1
        else:
            res[3] += 1
    return np.nanmedian(l),res


if __name__ == "__main__":
    pvaluesA = bereken_pwaarden("DPGA")
    holmcorrectie("DPGA",pvaluesA)
    pvaluesB = bereken_pwaarden("DPGB")
    holmcorrectie("DPGB", pvaluesB)
    pvaluesC = bereken_pwaarden("DPGC")
    holmcorrectie("DPGC", pvaluesC)
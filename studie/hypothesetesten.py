import numpy as np
from scipy import stats
import csv
from statsmodels.stats import multitest
from main import pad, K, NA

#Rijen van hypothese{DPG}{N}-{SNR}.csv (11 rijen, 30 replicaties):
#0 rs_trainMSE, 1 omkeringen, 2 rs_G, 3 rs_rho, 4 delta (forest - boom, rangindex),
#5 c_forest, 6 min_inwendig_forest, 7 c_boom, 8 min_inwendig_boom, 9 treeMSE_d0, 10 treeMSE_d1
#Rijen van res_pvalues (7 x 9 cellen): 0 trainMSE, 1 G, 2 rho, 3 delta (tekentoets), 4 c_forest, 5 d0, 6 d1
#(c_boom wordt niet getoetst: de U-vorm van de boom gebeurt via de kwadratische fit op de gemiddelde curve, zie u_vorm_boom)
#Er worden geen bootstrapintervallen gebruikt: we rapporteren p-waarden, medianen en tellingen.

def bereken_pwaarden(DPG):
    N = [250,1000,4000]
    res_pvalues = np.zeros((7,9))
    aantal_trainMSE = np.zeros((9, 4))  # aantal nan,pos,neg,en nulwaardes meedelen
    aantal_G = np.zeros((9, 4))
    aantal_rho = np.zeros((9, 4))
    aantal_d0 = np.zeros((9, 4))
    aantal_d1 = np.zeros((9, 4))

    med_trainMSE = np.zeros(9)
    med_G = np.zeros(9)
    med_rho = np.zeros(9)
    med_d0 = np.zeros(9)
    med_d1 = np.zeros(9)
    med_a_forest = np.zeros(9)

    aantal_alpha = np.zeros((9,4)) #aantal nan,pos,nul,neg van delta
    aantal_min_inwendig_forest = np.zeros((2,9)) #1e rij aantal min in inwendige => Uvorm, 2e rij is aantal monotoom
    aantal_min_inwendig_boom = np.zeros((2, 9))  # 1e rij aantal min in inwendige => Uvorm, 2e rij is aantal monotoom
    omkeringen = np.zeros((2,9))#1e rij gemiddelde en tweede rij max
    for j in range(3):
        for i in range(3): #i aanpassen naar de SNR
            SNR = [1,3,10]
            matrix = np.genfromtxt(pad(f'hypothese{DPG}{N[j]}-{SNR[i]}.csv'), delimiter=',') # shape: (11,30)
            c = 3*j + i

            res_pvalues[0][c] = safe_wilcoxon(matrix[0,],"greater") #rs_trainMSE
            omkeringen[0][c] = np.average(matrix[1,])
            omkeringen[1][c] = max(matrix[1,])
            med_trainMSE[c], aantal_trainMSE[c,] = effectgroottes(matrix[0,])

            res_pvalues[1][c] = safe_wilcoxon(matrix[2,],"less") #rs_G
            med_G[c], aantal_G[c,] = effectgroottes(matrix[2,])

            res_pvalues[2][c] = safe_wilcoxon(matrix[3,],"less") #rs_rho
            med_rho[c], aantal_rho[c,] = effectgroottes(matrix[3,])

            res_pvalues[3][c] = safe_tekentoets(matrix[4,]) #delta = k*_forest - k*_boom, tekentoets
            aantal_alpha[c,] = effectgroottes(matrix[4,])[1]

            res_pvalues[4][c] = safe_wilcoxon(matrix[5,],"greater") #c_forest
            med_a_forest[c] = np.nanmedian(matrix[5,]) #effectgrootte: mediaan van a (coefficient van k^2)
            aantal_min_inwendig_forest[0][c] = np.sum(matrix[6,] == 1)  # hoeveel keer min inwendig is (en c > 0)
            aantal_min_inwendig_forest[1][c] = np.sum(matrix[6,] == 2)  # hoeveel keer monotoom voor c > 0

            #c_boom wordt niet getoetst, enkel de tellingen bewaren
            aantal_min_inwendig_boom[0][c] = np.sum(matrix[8,] == 1)
            aantal_min_inwendig_boom[1][c] = np.sum(matrix[8,] == 2)

            res_pvalues[5][c] = safe_wilcoxon(matrix[9,], "greater")  #treeMSE_d0 (met leave-one-out optimum)
            med_d0[c], aantal_d0[c,] = effectgroottes(matrix[9,])
            res_pvalues[6][c] = safe_wilcoxon(matrix[10,], "greater")  #treeMSE_d1
            med_d1[c], aantal_d1[c,] = effectgroottes(matrix[10,])

    hulp_write(DPG,"p_waarden",res_pvalues)
    hulp_write(DPG, "aantal_min_inwendig_forest", aantal_min_inwendig_forest)
    hulp_write(DPG, "aantal_min_inwendig_boom", aantal_min_inwendig_boom)
    hulp_write(DPG, "omkeringen", omkeringen)
    hulp_write(DPG, "aantal_alpha", aantal_alpha)
    hulp_write(DPG, "aantal_trainMSE", aantal_trainMSE)
    hulp_write(DPG, "aantal_G", aantal_G)
    hulp_write(DPG, "aantal_rho", aantal_rho)
    hulp_write(DPG, "aantal_d0", aantal_d0)
    hulp_write(DPG, "aantal_d1", aantal_d1)

    hulp_write(DPG, "med_trainMSE", med_trainMSE)
    hulp_write(DPG, "med_G", med_G)
    hulp_write(DPG, "med_rho", med_rho)
    hulp_write(DPG, "med_d0", med_d0)
    hulp_write(DPG, "med_d1", med_d1)
    hulp_write(DPG, "med_a_forest", med_a_forest)
    return res_pvalues

def u_vorm_boom(DPG, R=30):
    #kwadratische fit MSE(k) = c + b k + a k^2 op de gemiddelde test-MSE-curve van de boom (rij 7), rangindex k = 1..9
    #rijen van het resultaat (3 x 9 cellen): 0 a, 1 k*, 2 voldoet aan a > 0 en k* in (1,9) (1 = ja, 0 = nee)
    N = [250,1000,4000]
    SNR = [1,3,10]
    res = np.full((3,9), np.nan)
    for j in range(3):
        for i in range(3):
            alle = np.stack([np.genfromtxt(pad(f'data_per_iteratie{DPG}{N[j]}-{SNR[i]}-{r}.csv'), delimiter=',')
                             for r in range(R)]) # (R,14,9)
            gem_boom = alle[:,7,:].mean(axis=0) #gemiddelde test-MSE-curve van de boom
            a, b, _ = np.polyfit(K, gem_boom, 2)
            kstar = -b/(2*a) if a > 0 else np.nan #minimum enkel zinvol bij a > 0
            res[:, 3*j+i] = [a, kstar, float(a > 0 and 1 < kstar < NA)]
    hulp_write(DPG, "u_vorm_boom", res)
    return res

def hulp_write(DPG, name, values):
    values = np.atleast_2d(values)      # 1D (9,) wordt (1,9)
    with open(pad(f'{name}{DPG}.csv'), 'w+', newline='') as csvfile:
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

def safe_tekentoets(x):
    #eenzijdige tekentoets H0: med(delta) >= 0 tegen Ha: med(delta) < 0
    #nullen worden genegeerd, enkel replicaties met delta != 0 tellen mee
    d = x[~np.isnan(x)]
    n_neg = int(np.sum(d < 0))
    n_pos = int(np.sum(d > 0))
    n = n_neg + n_pos
    if n == 0: #alle verschillen nul => toets niet uitvoerbaar
        return np.nan
    #onder H0 is P(negatief) = 0,5; Ha: meer negatieve dan positieve verschillen
    return stats.binomtest(n_neg, n, p=0.5, alternative="greater").pvalue

def holmcorrectie(P, DPG):
    #P = (7,9)-matrix p-waarden van één DGP; familie = de 9 cellen van die DGP per toets
    rejected = np.full(P.shape, np.nan)
    corrected_pvalues = np.full(P.shape, np.nan)
    families = [[0],[1],[2],[3],[4],[5,6]] #d0 en d1 samen: 18 toetsen per DGP
    for rijen in families:
        flat = P[rijen,:].ravel()
        geldig = ~np.isnan(flat)
        if geldig.sum() == 0:
            continue
        rej, corr, _, _ = multitest.multipletests(flat[geldig], method="holm", alpha=0.05)
        r_flat = np.full(flat.shape, np.nan)
        c_flat = np.full(flat.shape, np.nan)
        r_flat[geldig] = rej
        c_flat[geldig] = corr
        rejected[rijen,:] = r_flat.reshape(len(rijen), -1)
        corrected_pvalues[rijen,:] = c_flat.reshape(len(rijen), -1)

    hulp_write(DPG, "verworpen_hypotheses", rejected)
    hulp_write(DPG, "aangepaste_pwaarden", corrected_pvalues)
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
    for DPG in ["DPGA","DPGB","DPGC"]:
        P = bereken_pwaarden(DPG)
        holmcorrectie(P, DPG)
        u_vorm_boom(DPG)
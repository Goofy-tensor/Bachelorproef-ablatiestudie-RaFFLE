import numpy as np
import scipy
from scipy import stats

def plot(DPG):
    N = [250,1000,4000]
    res_pvalues = np.zeros((9,9))
    for j in range(3):
        for i in range(3): #i aanpassen naar de SNR
            SNR = [1,3,10]
            matrix = np.genfromtxt(f'hypothese{DPG}{N[j]}-{SNR[i]}.csv', delimiter=',')

            res_pvalues[0][3*j + i] = stats.wilcoxon(matrix[:,0],alternative="two-sided") #rs_trainMSE
            #telling omkeringen tussen opeenvolgende roosterpunten, wat betekent dit exact?
            #we verwachten +1 dus kunnen we niet eenzijdig testen
            res_pvalues[1][3 * j + i] = stats.wilcoxon(matrix[:, 1], alternative="two-sided") #rs_G
            #we verwachten negatief dus mss beter eenzijdig?
            res_pvalues[2][3 * j + i] = stats.wilcoxon(matrix[:, 2], alternative="two-sided") #rs_rho
            #tellen in welke cellen wat ondersteund wordt (pos=> H3.., neg=>H3.. welke exact?)
            res_pvalues[3][3 * j + i] = stats.wilcoxon(matrix[:, 3], alternative="two-sided") #minMSE_0
            #ligt min in het inwendige => index meegeven van min alpha best dan?
            #als min bij 0 dan toch niet per se monotoom?
            res_pvalues[4][3 * j + i] = stats.wilcoxon(matrix[:, 4], alternative="two-sided") #minMSE_1
            # ligt min in het inwendige?
            # als min bij 1 dan toch niet per se monotoom?
            res_pvalues[5][3 * j + i] = stats.wilcoxon(matrix[:, 5], alternative="two-sided") #versch_ambigu
            #lagere alpha zou hogere ambiguiteit moeten geven dus verschil moet negatief zijn
            #mss dan ook eenzijdig?
            res_pvalues[6][3 * j + i] = stats.wilcoxon(matrix[:, 6], alternative="less") #opt alpha nergelijking
            #aantal pos,neg,en nulwaardes meedelen
            #bootstrap betrouwbaarheidsinterval???
            res_pvalues[7][3 * j + i] = stats.wilcoxon(matrix[:, 7], alternative="greater") #c
            # min in inwendige,hzo?
            #Met Vergelijking(9) gaan we na of een stijging van ¯ρ bij lage α de daling van ¯σ2 teniet doet
    return res_pvalues

#uiteindelijk holm correctie (zie berichtje Warre)
#moet dit in de code? mss wel handig, of overzicht
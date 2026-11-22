import numpy as np
from statsmodels.stats import multitest

# Hergebruikt hulp_write en safe_wilcoxon uit hypothesetesten.py (geen eigen kopieën, dus
# één definitie voor alles) en importeert NIETS uit main.py: geen pilot nodig, geen cirkelimport.
# Leest enkel bestaande csv's, er hoeft niets opnieuw gesimuleerd te worden.
from hypothesetesten import safe_wilcoxon, hulp_write
from paden import pad

# Vormcodes voor de kwadratische fit op de rangindex k = 1..9:
#   1 = U-vorm (a > 0, minimum strikt in (1,9))   <- enige code die als U telt
#   2 = convex, maar minimum buiten (1,9) (monotoom binnen het bereik)
#   3 = concaaf (a < 0), dus nooit een U
#   4 = (bijna) lineair of vlak (a ~ 0)
# Posities van het argmin (robuuster dan de parabool): 0 = links, 1 = intern, 2 = rechts

N_WAARDEN = [250, 1000, 4000]
SNR_WAARDEN = [1, 3, 10]
K = np.arange(1, 10)


def classificeer(y):
    a, b, _ = np.polyfit(K, y, 2)
    if np.isclose(a, 0):
        return 4
    kstar = -b / (2 * a)
    if a > 0:
        return 1 if 1 < kstar < 9 else 2
    return 3


def positie_min(y):
    k = int(np.argmin(y))
    if k == 0:
        return 0
    if k == 8:
        return 2
    return 1


def rho_tweezijdig(DPG):
    # vervangt de "greater"-toets: één tweezijdige toets, geen twee eenzijdige op dezelfde data
    pvalues = np.zeros(9)
    for j in range(3):
        for i in range(3):
            matrix = np.genfromtxt(
                pad(f'hypothese{DPG}{N_WAARDEN[j]}-{SNR_WAARDEN[i]}.csv'), delimiter=',')  # (11,30)
            pvalues[3 * j + i] = safe_wilcoxon(matrix[3,], "two-sided")  # rs_rho

    rejected = np.full(9, np.nan)
    corrected = np.full(9, np.nan)
    goed = ~np.isnan(pvalues)
    if goed.any():
        rej, corr, _, _ = multitest.multipletests(pvalues[goed], method="holm", alpha=0.05)
        rejected[goed] = rej
        corrected[goed] = corr
    hulp_write(DPG, "verworpen_hypotheses_rho_tweezijdig", rejected)
    hulp_write(DPG, "aangepaste_pwaarden_rho_tweezijdig", corrected)
    return pvalues


def recover_vorm(DPG, R=30):
    res = {}
    for naam, rij in [("forest", 5), ("boom", 7)]:
        vorm = np.zeros((4, 9))             # rijen: code 1,2,3,4
        pos = np.zeros((3, 9))              # rijen: links, intern, rechts
        verschil = np.full((1, 9), np.nan)  # mediaan van MSE(alpha=1) - min(MSE) over replicaties
        for j in range(3):
            for i in range(3):
                c = 3 * j + i
                diffs = np.zeros(R)
                for r in range(R):
                    M = np.genfromtxt(
                        pad(f'data_per_iteratie{DPG}{N_WAARDEN[j]}-{SNR_WAARDEN[i]}-{r}.csv'),
                        delimiter=',')
                    y = M[rij, :]
                    vorm[classificeer(y) - 1, c] += 1
                    pos[positie_min(y), c] += 1
                    diffs[r] = y[-1] - y.min()
                verschil[0, c] = np.median(diffs)
        hulp_write(DPG, f"vorm_{naam}", vorm)
        hulp_write(DPG, f"argmin_pos_{naam}", pos)
        hulp_write(DPG, f"verschil_alpha1_min_{naam}", verschil)
        res[naam] = (vorm, pos, verschil)
    return res


def extra_analyse(DPG):
    # dit is de functie die main.py aanroept
    rho_tweezijdig(DPG)
    return recover_vorm(DPG)


if __name__ == "__main__":
    for DPG in ["DPGA", "DPGB", "DPGC"]:
        print(f"Extra analyse {DPG}...")
        extra_analyse(DPG)
    print("Klaar.")
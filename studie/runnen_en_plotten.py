import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from main import pad, ALPHA, NA, IDX_1, TRAIN_BOOM_RIJ

#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (13 waarden) (al in code), R = 30 (al in code)
alpha = ALPHA
N_WAARDEN = [250, 1000, 4000]
SNR_WAARDEN = [1, 3, 10]
R = 30

#rijen in data_per_iteratie{DPG}{N}-{SNR}-{r}.csv (gemiddeld over de 100 bomen)
TEST_BOOM_RIJ = 8      # test-MSE boom
RIJ_DIEPTE = 8         # max_depth (zonder lin-knopen)
RIJ_MODEL_DIEPTE = 9   # max_model_depth (met lin-knopen)
MAX_DEPTH_INSTELLING = 20  # max_depth die in main.py aan RaFFLE wordt meegegeven


#algemene plots: (per N en SNR)
def plot(DPG):
    N = [250,1000,4000]
    SNR = [1, 3, 10]
    z = list(range(1, NA + 1))
    for j in range(3):
        for i in range(3):
            matrix = np.genfromtxt(pad(f'data{DPG}{N[j]}-{SNR[i]}.csv'), delimiter=',')
            trainMSE = matrix[4,]
            testMSE = matrix[5,]
            trainMSE_boom = matrix[6,]
            testMSE_boom = matrix[7,]

            plt.figure(3*j+i)
            plt.xticks(z,alpha)
            plt.plot(z, trainMSE, 'r-', label='TrainMSE in functie van alpha', markersize=5)
            plt.plot(z, testMSE, 'b-', label='TestMSE in functie van alpha', markersize=1)
            plt.plot(z, trainMSE_boom, 'o-', label='TrainboomMSE in functie van alpha', markersize=5)
            plt.plot(z, testMSE_boom, 'g-', label='TestboomMSE in functie van alpha', markersize=1)
            plt.xlabel('alpha', fontsize=12)
            plt.ylabel('relatieve MSE', fontsize=12)
            plt.title(f'MSE in functie van alpha voor {DPG}{N[j]}-{SNR[i]}', fontsize=10)
            plt.legend(loc='best', fontsize=10)
            plt.savefig(pad(f'MSE_{DPG}{N[j]}-{SNR[i]}.png'), bbox_inches='tight')
            plt.show()
    return


def laad_rij(DPG, rij, N=None, SNR=None, R=R):
    # gekozen rij per replicatie en alpha, over alle gevraagde (N, SNR)-cellen; vorm (aantal, NA)
    ns = N_WAARDEN if N is None else [N]
    snrs = SNR_WAARDEN if SNR is None else [SNR]
    return np.vstack([
        np.genfromtxt(pad(f'data_per_iteratie{DPG}{n}-{s}-{r}.csv'),
                      delimiter=',')[rij]
        for n in ns for s in snrs for r in range(R)])


def laad_train_boom(DPG, N, SNR, R=R):
    # train-MSE boom (eigen bootstrapsample) per replicatie en alpha, vorm (R, NA)
    return laad_rij(DPG, TRAIN_BOOM_RIJ, N, SNR, R)


def curven(DPG, N=None, SNR=None, rij=TRAIN_BOOM_RIJ, norm_idx=IDX_1):
    # alle replicatiecurven, elk genormaliseerd op gekozen index (standaard alpha = 1); vorm (aantal, NA)
    Y = laad_rij(DPG, rij, N, SNR)
    return Y / Y[:, [norm_idx]]


def plot_train_boom_drie(DPGS=("DPGA", "DPGB", "DPGC"), N=None, SNR=None,
                         bestand="trainMSE_boom_gemiddeld_alle_DPG.png"):
    z = np.arange(1, NA + 1)  # rangindex: het alpha-rooster is niet gelijkmatig
    labels = [str(a).replace('.', ',') for a in ALPHA]
    kleuren = ['#2b6cb0', '#c53030', '#2f855a']

    fig, assen = plt.subplots(1, len(DPGS), figsize=(4.6 * len(DPGS), 4.2), sharey=True)

    for ax, DPG, kleur in zip(assen, DPGS, kleuren):
        Y = curven(DPG, N, SNR, rij=TRAIN_BOOM_RIJ, norm_idx=IDX_1)
        gem, sd = Y.mean(axis=0), Y.std(axis=0)

        ax.fill_between(z, gem - sd, gem + sd, color=kleur, alpha=0.2, lw=0,
                        label=r'$\pm 1$ standaardafwijking')
        ax.plot(z, gem, '-o', color=kleur, lw=2, ms=5, markerfacecolor='white',
                markeredgewidth=1.5, label='gemiddelde')
        ax.axhline(1, color='grey', ls=':', lw=1)
        ax.axvline(z[IDX_1], color='grey', ls=':', lw=1)

        ax.set_title(DPG.replace('DPG', 'DGP '), fontweight='bold')
        ax.set_xticks(z)
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_xlabel(r'$\alpha$')
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    assen[0].set_ylabel(r'train-MSE boom / train-MSE bij $\alpha = 1$')
    assen[0].legend(frameon=False, fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(pad(bestand), bbox_inches='tight', dpi=300)
    plt.show()


def plot_test_boom_drie(DPGS=("DPGA", "DPGB", "DPGC"), N=None, SNR=None,
                        bestand="testMSE_boom_gemiddeld_alle_DPG.png"):
    z = np.arange(1, NA + 1)  # rangindex: het alpha-rooster is niet gelijkmatig
    labels = [str(a).replace('.', ',') for a in ALPHA]
    kleuren = ['#2b6cb0', '#c53030', '#2f855a']

    fig, assen = plt.subplots(1, len(DPGS), figsize=(4.6 * len(DPGS), 4.2))

    for ax, DPG, kleur in zip(assen, DPGS, kleuren):
        Y = curven(DPG, N, SNR, rij=TEST_BOOM_RIJ, norm_idx=IDX_1)
        gem, sd = Y.mean(axis=0), Y.std(axis=0)

        ax.fill_between(z, gem - sd, gem + sd, color=kleur, alpha=0.2, lw=0,
                        label=r'$\pm 1$ standaardafwijking')
        ax.plot(z, gem, '-o', color=kleur, lw=2, ms=5, markerfacecolor='white',
                markeredgewidth=1.5, label='gemiddelde')
        ax.axhline(1, color='grey', ls=':', lw=1)
        ax.axvline(z[IDX_1], color='grey', ls=':', lw=1)

        ax.set_title(DPG.replace('DPG', 'DGP '), fontweight='bold')
        ax.set_xticks(z)
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'test-MSE boom / test-MSE bij $\alpha = 1$')
        ax.grid(True, alpha=0.25, lw=0.6)
        ax.spines[['top', 'right']].set_visible(False)

    assen[0].legend(frameon=False, fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(pad(bestand), bbox_inches='tight', dpi=300)
    plt.show()


def _band(ax, z, Y, kleur, label, ls='-'):
    # gemiddelde +- 1 standaardafwijking over alle replicaties (en cellen)
    gem, sd = Y.mean(axis=0), Y.std(axis=0)
    ax.fill_between(z, gem - sd, gem + sd, color=kleur, alpha=0.18, lw=0)
    ax.plot(z, gem, ls, marker='o', color=kleur, lw=2, ms=5,
            markerfacecolor='white', markeredgewidth=1.5, label=label)


def plot_diepte(DPGS=("DPGA", "DPGB", "DPGC"), N=None, SNR=None,
                bestand="diepte_boom.png"):
            #gemiddelde maximale diepte van de bomen, op dezelfde alpha-as
    z = np.arange(1, NA + 1)  # rangindex: het alpha-rooster is niet gelijkmatig
    labels = [str(a).replace('.', ',') for a in ALPHA]
    kleuren = ['#2b6cb0', '#c53030', '#2f855a']

    fig, assen = plt.subplots(2, len(DPGS), figsize=(4.6 * len(DPGS), 7), sharex=True)

    for kol, (DPG, kleur) in enumerate(zip(DPGS, kleuren)):
        ax = assen[0, kol]
        _band(ax, z, laad_rij(DPG, RIJ_DIEPTE, N, SNR), kleur, 'max. diepte (zonder lin)')
        _band(ax, z, laad_rij(DPG, RIJ_MODEL_DIEPTE, N, SNR), 'black',
              'max. diepte (met lin)', ls='--')
        ax.axhline(MAX_DEPTH_INSTELLING, color='grey', ls=':', lw=1)  # max_depth = 20
        ax.set_xticks(z)
        ax.set_xticklabels(labels, rotation=60, fontsize=8)
        ax.set_xlabel(r'$\alpha$')

        for rij in (0, 1):
            a = assen[rij, kol]
            a.grid(True, alpha=0.25, lw=0.6)
            a.spines[['top', 'right']].set_visible(False)
            # zelfde zone markeren als in de tekst (0,35 tot 0,7)
            a.axvspan(z[ALPHA.index(0.35)], z[ALPHA.index(0.7)],
                      color='grey', alpha=0.08, lw=0)

    assen[0, 0].set_ylabel(r'train-MSE boom / train-MSE bij $\alpha = 1$')
    assen[1, 0].set_ylabel('gemiddelde maximale diepte')
    assen[0, 0].legend(frameon=False, fontsize=8, loc='best')
    assen[1, 0].legend(frameon=False, fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(pad(bestand), bbox_inches='tight', dpi=300)
    plt.show()


if __name__ == "__main__":
    #plot_train_boom_drie()
    plot_test_boom_drie()
    #plot_diepte()
    # plot('DPGA')
    # plot('DPGB')
    # plot('DPGC')
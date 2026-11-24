import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
from main import pad, ALPHA, NA, IDX_1, TRAIN_BOOM_RIJ

#SNR ∈ {10,3,1}, N ∈ {250,1000,4000}, α ∈ A (9 waarden) (al in code), R = 30 (al in code)
alpha = ALPHA
N_WAARDEN = [250, 1000, 4000]
SNR_WAARDEN = [1, 3, 10]
R = 30

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



def laad_train_boom(DPG, N, SNR, R=R):
    # train-MSE boom (eigen bootstrapsample) per replicatie en alpha, vorm (R, NA)
    return np.stack([
        np.genfromtxt(pad(f'data_per_iteratie{DPG}{N}-{SNR}-{r}.csv'),
                      delimiter=',')[TRAIN_BOOM_RIJ]
        for r in range(R)])


def curven(DPG, N=None, SNR=None):
    # alle replicatiecurven, elk genormaliseerd op alpha = 1; vorm (aantal, NA)
    ns = N_WAARDEN if N is None else [N]
    snrs = SNR_WAARDEN if SNR is None else [SNR]
    Y = np.vstack([laad_train_boom(DPG, n, s) for n in ns for s in snrs])
    return Y / Y[:, [IDX_1]]


def plot_train_boom_drie(DPGS=("DPGA", "DPGB", "DPGC"), N=None, SNR=None,
                         bestand="trainMSE_boom_gemiddeld_alle_DPG.png"):
    z = np.arange(1, NA + 1)  # rangindex: het alpha-rooster is niet gelijkmatig
    labels = [str(a).replace('.', ',') for a in ALPHA]
    kleuren = ['#2b6cb0', '#c53030', '#2f855a']

    fig, assen = plt.subplots(1, len(DPGS), figsize=(4.6 * len(DPGS), 4.2), sharey=True)

    for ax, DPG, kleur in zip(assen, DPGS, kleuren):
        Y = curven(DPG, N, SNR)
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


if __name__ == "__main__":
    plot_train_boom_drie()



if __name__ == "__main__":
    plot('DPGA')
    plot('DPGB')
    plot('DPGC')
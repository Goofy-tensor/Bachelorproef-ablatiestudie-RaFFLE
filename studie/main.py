from pilot import RaFFLE, DEFAULT_DF_SETTINGS
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from scipy import stats
import csv
 
#------------------------------------------------------------
# INSTELLINGEN: data-map en alpha-grid (alle andere bestanden halen dit uit main.py)
#------------------------------------------------------------
from pathlib import Path
 
# Alle csv's en png's staan in de map DATA_MAP naast de scripts. Het pad wordt berekend vanaf de
# locatie van dit bestand, dus het werkt op elke laptop en vanuit elke werkmap.
DATA_MAP = "data"
 
BASIS = Path(__file__).resolve().parent
DATA = BASIS / DATA_MAP
DATA.mkdir(exist_ok=True)
 
def pad(bestandsnaam):
    """Geeft het volledige pad van een bestand in de data-map terug (als string)."""
    return str(DATA / bestandsnaam)
 
# Het oorspronkelijke grid links van 0.5, plus extra punten tussen 0.5 en 1
ALPHA_TOT_1 = [0.0, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
 

 
ALPHA = ALPHA_TOT_1
assert ALPHA == sorted(ALPHA) and len(set(ALPHA)) == len(ALPHA), "alpha moet oplopend en uniek zijn"
assert 0.5 in ALPHA and 1.0 in ALPHA, "0.5 en 1.0 moeten in het grid zitten (referentiepunten)"
 
NA = len(ALPHA)                          # aantal alpha-waarden
K = list(range(1, NA + 1))               # rangindex 1..NA (voor de kwadratische fit)
IDX_05 = ALPHA.index(0.5)                # kolom van alpha = 0.5
IDX_1 = ALPHA.index(1.0)                 # kolom van alpha = 1 (standaard PILOT)
#------------------------------------------------------------
 
#Met model.predict(X, individual=True) krijg je de voorspelling van elke boom apart => in matrix
#dus fout per boom, forestfout en correlatie uit één fit
#model.estimators[m].tree_summary(feature_names=...) geeft per knoop het modeltype, de diepte en de variabele
#model.estimators_samples_ geeft de bootstrapindices van elke boom (nodig voor de train-MSE op de eigen bootstrapsample)
 
#Welke rij gebruiken we voor de train-MSE van de boom in de Spearman-toets (H1)?
#6  = train-MSE op de volledige trainingsset, 13 = train-MSE op de eigen bootstrapsample van elke boom
TRAIN_BOOM_RIJ = 13
 
#Raffle model fitten
def RaFFLE_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k=42,alpha=1):
    #model maken
    df = {k: 1 + alpha * (v - 1) for k, v in DEFAULT_DF_SETTINGS.items()}
    df["blin"] = -1
 
    #oob_score=True zodat estimators_samples_ (bootstrapindices per boom) zeker beschikbaar is
    model = RaFFLE(n_estimators=100,random_state=k,alpha=alpha,n_features_node=1,n_features_tree=1,df_settings=df,max_depth=20,max_model_depth=100,min_sample_fit=10,min_sample_alpha=5,min_sample_leaf=5,rel_tolerance=0.0000001,oob_score=True)
    model.fit(train_X, train_ruis_y)
 
    #Waarden die we willen weten
    predictions_train = model.predict(train_X)
    predictions_test = model.predict(test_X)
    train_R2 = r2_score(train_ruis_y,predictions_train)
    test_R2 = r2_score(test_ruis_y,predictions_test)
    train_MSE = mean_squared_error(train_ruis_y,predictions_train)
    test_MSE = mean_squared_error(test_ruis_y,predictions_test)
 
    y = predictions_test - test_y
    ruisvrij_error = 1/len(test_X)*(np.dot(y,y)) #ruisvrije error dus kijken naar effectieve f(x) waarden
 
    train_M = model.predict(train_X, individual=True) #predictions individuele bomen
    test_M = model.predict(test_X, individual=True)
 
    train_boom_R2 = boom(train_M,train_ruis_y,r2_score)
    test_boom_R2 = boom(test_M,test_ruis_y,r2_score)
    train_MSE_boom = boom(train_M, train_ruis_y,mean_squared_error)
    test_MSE_boom = boom(test_M,test_ruis_y, mean_squared_error)
    foutcorr = foutcorrelatie(test_M,test_y) #we bekijken hier effectieve f(x)
    ambigu = ambiguiteit(test_M,predictions_test)
    #train-MSE van elke boom op zijn eigen bootstrapsample (de punten die de boom effectief gezien heeft)
    train_MSE_boom_inbag = boom_inbag(train_M,train_ruis_y,model.estimators_samples_)
 
    #Eigenschappen model
    max_depth, max_model_depth = summary(model)
    return train_R2,test_R2,train_boom_R2,test_boom_R2,train_MSE,test_MSE,train_MSE_boom,test_MSE_boom ,max_depth, max_model_depth, ruisvrij_error,foutcorr,ambigu,train_MSE_boom_inbag
 
#Eigenschappen van de invividuele bomen
def boom(M,Y,fun):
    res = np.zeros(100)
    for i in range(100):
        pred_X = M[:,i] #elke Xi is een rij dus predictions i-de boom in de i-de kolom
        res[i] = fun(Y, pred_X)
    return np.average(res)
 
#MSE van elke boom op zijn eigen bootstrapsample (in-bag), gemiddeld over de bomen
def boom_inbag(M,Y,samples):
    res = np.zeros(len(samples))
    for i, idx in enumerate(samples): #samples[i] = bootstrapindices van boom i (met herhalingen)
        idx = np.asarray(idx)
        if idx.dtype == bool: #voor het geval het een masker is in plaats van indices
            idx = np.where(idx)[0]
        res[i] = mean_squared_error(Y[idx], M[idx,i]) #herhaalde indices wegen automatisch zwaarder mee
    return np.average(res)
 
#Foutcorrelatie, hoeveel gelijkaardige fouten de bomen maken
def foutcorrelatie(M,Y):
    L = np.zeros((len(Y),100))
    for i in range(100):
        z = M[:,i] - Y
        L[:,i] = z/np.sqrt((z.dot(z)))
    gramm = L.T.dot(L)
    intermediate = np.ones(100).dot(gramm) #elke kolom sommeren
    return (np.dot(intermediate,np.ones(100))-100)/(100*99)
 
#Ambiguïteit, hoeveel boom gemiddeld afwijkt
def ambiguiteit(M,Y):
    L = np.zeros(100)
    for i in range(100):
        z = M[:, i] - Y
        L[i] = np.dot(z,z)
    return sum(L)/(100 * len(Y))
 
#Raffle summary
def summary(model):
    max_depth = np.zeros(100)
    max_model_depth = np.zeros(100)
    #node_types = np.zeros((100,4)) (even weglaten mogelijks achteraf nog toevoegen)
    for i, estimator in enumerate(model.estimators):
        m = estimator.tree_summary() #summary of individual Pilot Tree
        max_depth[i] = max(m.to_numpy()[:, 0])  # zonder lin nodes
        max_model_depth[i] = max(m.to_numpy()[:, 1])  # nu met lin nodes er in
        #node_types[i,] = totaltypes(m)
    return np.average(max_depth), np.average(max_model_depth)
 
 
def iterate_predictions(fun1,fun2,SIGMA,DPG,N,SNR):
    info = np.zeros((14,NA)) #14 outputs van RaFFLE_prediction en NA alpha waarden
    alle = np.zeros((30,14,NA)) #matrix van elke replicatie bewaren (nodig voor leave-one-out optimum)
    rs_trainMSE = np.zeros(30) #gem over boom
    rs_G = np.zeros(30) #nu gat enkel voor forest
    rs_rho = np.zeros(30)
    opt_alpha_boom = np.zeros(30)
    opt_alpha_forest = np.zeros(30)
    c_forest = np.zeros(30)
    min_inwendig_forest = np.zeros(30)
    c_boom = np.zeros(30)
    min_inwendig_boom = np.zeros(30)
    aantal_omkeringen = np.zeros(30)
    treeMSE_d0 = np.zeros(30) #MSE_r(alpha=0) - MSE_r(alpha*_-r), met alpha*_-r bepaald op de overige 29 replicaties
    treeMSE_d1 = np.zeros(30) #idem maar voor alpha=1
 
    alpha = ALPHA #grid staat bovenaan in main.py
    #node_types = np.zeros((4,30))  # 4 modellen [CON,PCON,LIN,PLIN]
    for i in range(30):
        train_X, train_y, train_ruis_y, test_X, test_y, test_ruis_y = fun2(fun1, N, SIGMA,i)
        matrix = alpha_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,i) #dim = (14,9)
        alle[i] = matrix
        with open(pad(f'data_per_iteratie{DPG}{N}-{SNR}-{i}.csv'), 'w+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for j in range(14):
                spamwriter.writerow(matrix[j,])
 
        rs_trainMSE[i] = stats.spearmanr(alpha,matrix[TRAIN_BOOM_RIJ,]).statistic #train-MSE boom (rij zie TRAIN_BOOM_RIJ)
        aantal_omkeringen[i] = omkeringen(matrix[TRAIN_BOOM_RIJ,])
 
        G = matrix[5,:] - matrix[4,:] #generalisatiegat per alpha (voor forestMSEs)
        rs_G[i] = stats.spearmanr(alpha,G).statistic
 
        rs_rho[i] = stats.spearmanr(alpha,matrix[11,:], alternative = "two-sided").statistic #geeft zo onderlijningen omdat het int verwacht maar in voorbeelden in de documentatie
 
        pos_minMSE_forest = np.argmin(matrix[5,:]) #over testwaarden
        pos_minMSE_boom = np.argmin(matrix[7, :])
        opt_alpha_boom[i] = pos_minMSE_boom
        opt_alpha_forest[i] = pos_minMSE_forest #voor de hypothesetesten gebruiken we de rangindex
 
        p_forest = np.polyfit(K,matrix[5,:],2)
        c_forest[i] = p_forest[0]
        min_inwendig_forest[i] = in_min(p_forest)
 
        p_boom = np.polyfit(K, matrix[7, :], 2)
        c_boom[i] = p_boom[0]
        min_inwendig_boom[i] = in_min(p_boom)
 
    #U-vormtoets boom: optimum alpha*_-r bepalen op de overige 29 replicaties (replicatie r zelf telt niet mee)
    boom_test = alle[:,7,:] #(30,9): test-MSE boom per replicatie en alpha
    totaal = boom_test.sum(axis=0)
    for r in range(30):
        gem_zonder_r = (totaal - boom_test[r])/29 #gemiddelde test-MSE van de overige 29 replicaties
        k_opt = np.argmin(gem_zonder_r) #rangindex van alpha*_-r
        treeMSE_d0[r] = boom_test[r,0] - boom_test[r,k_opt]
        treeMSE_d1[r] = boom_test[r,IDX_1] - boom_test[r,k_opt] #alpha = 1 (standaard PILOT)
 
    info = alle.sum(axis=0)
    return info/30, rs_trainMSE, aantal_omkeringen, rs_G, rs_rho, opt_alpha_forest - opt_alpha_boom, c_forest, min_inwendig_forest, c_boom, min_inwendig_boom, treeMSE_d0, treeMSE_d1
 
def omkeringen(l):
    count = 0
    for i in range(len(l)-1):
        if l[i+1] < l[i]:
            count += 1
    return count
 
def in_min(p): #ligt het min van de kwadratische fit, -b/(2c) tussen alhpa = 0,resp 1
    #LET OP: deze oude codering wordt enkel nog in de hypothese*.csv (rij 6 en 8) bewaard.
    #Voor het rapporteren van de vorm: gebruik vorm_forest / vorm_boom uit Extra_hypothesetesten.py
    if p[0] > 0:
        if 1 < -p[1]/(2*p[0]) < NA: #p[0] = c, p[1] = b, p[2] = a
            return 1
        else: #monotoom als de x-waarde van het minimum niet tussen 1 of 9 ligt
            return 2
    elif p[0] == 0:#dan is het monotoom (hoewel ik vermoed dat dit geen optie is als je een 2e graadsfit vraagt)
        return 2
    else: #p[0] < 0
        if 1 < -p[1]/(2*p[0]) < NA:
            return 0
        else:
            return 2
 
#Nagaan welke modellen er gebruikt worden
def totaltypes(model): #ik denk dat dit niet zo efficiënt is
    l = model.to_numpy()[:,4]
    types = np.zeros(4)
    for x in l:
        if x == 'con':
            types[0] += 1
        if x == 'pcon':
            types[1] += 1
        if x == 'lin':
            types[2] += 1
        if x == 'plin':
            types[3] += 1
    return types
 
#Echte predictions
def alpha_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k):
    alpha_results = np.zeros((14,NA))
    #alpha_nodes = np.zeros((4,9))
    for i,alpha in enumerate(ALPHA): #grid staat bovenaan in main.py
        alpha_results[:,i] = RaFFLE_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k,alpha=alpha)
    return alpha_results #matrix met kolommen de alpha waarden en rijen de eigenschappen
 
 
#------------------------------------------------------------
# Welke DGP's opnieuw simuleren? Wat niet in de lijst staat, gebruikt de bestaande csv's.
#   ["DPGC"]                  = enkel C simuleren
#   ["DPGA","DPGB","DPGC"]    = alles simuleren
#   []                        = niets simuleren, enkel analyse + plots
#------------------------------------------------------------
RUN_DPGS = []
 
# Welke DGP's analyseren (hypothesetesten + plots)?
ANALYSE_DPGS = ["DPGA","DPGB","DPGC"]
 
if __name__ == "__main__":
    #imports hier en niet bovenaan: DPGA importeert zelf uit main (anders cirkelimport)
    if "DPGA" in RUN_DPGS:
        import DPGA
        print("Run DPGA...")
        DPGA.get_output("DPGA", DPGA.f, DPGA.generate_data, DPGA.estimate_sigma(DPGA.mean, DPGA.cov, DPGA.f))
    if "DPGB" in RUN_DPGS:
        import DPGB
        print("Run DPGB...")
        DPGB.get_output("DPGB", DPGB.f, DPGB.generate_data, DPGB.estimate_sigma(DPGB.f))
    if "DPGC" in RUN_DPGS:
        import DPGC
        print("Run DPGC...")
        DPGC.get_output("DPGC", DPGC.f, DPGC.generate_data, DPGC.estimate_sigma(DPGC.f))
    if not RUN_DPGS:
        print("Simulaties overgeslagen, bestaande csv's worden gebruikt.")
 
    #de rest draait altijd vanzelf
    import hypothesetesten
    import runnen_en_plotten
    import Extra_hypothesetesten #nieuw: vormtellingen en tweezijdige rho-toets
    for DPG in ANALYSE_DPGS:
        print(f"Hypothesetesten {DPG}...")
        P = hypothesetesten.bereken_pwaarden(DPG)
        hypothesetesten.holmcorrectie(P, DPG)
        hypothesetesten.u_vorm_boom(DPG)
        print(f"Extra analyse {DPG}...") #nieuw
        Extra_hypothesetesten.extra_analyse(DPG) #nieuw
        print(f"Plots {DPG}...")
        #runnen_en_plotten.plot(DPG)
        #runnen_en_plotten.plot_train_boom_drie()
        #runnen_en_plotten.plot_diepte()
        #runnen_en_plotten.plot_test_boom_drie()
        runnen_en_plotten.plot_generalisatiegat_drie()
    print("Klaar.")
 
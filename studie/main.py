from pilot import RaFFLE, DEFAULT_DF_SETTINGS
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from scipy import stats
import csv

#Met model.predict(X, individual=True) krijg je de voorspelling van elke boom apart => in matrix
#dus fout per boom, forestfout en correlatie uit één fit
#model.estimators[m].tree_summary(feature_names=...) geeft per knoop het modeltype, de diepte en de variabele


#Raffle model fitten
def RaFFLE_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k=42,alpha=1):
    #model maken
    df = {k: 1 + alpha * (v - 1) for k, v in DEFAULT_DF_SETTINGS.items()}
    df["blin"] = -1

    model = RaFFLE(n_estimators=100,random_state=k,alpha=alpha,n_features_node=1,n_features_tree=1,df_settings=df,max_depth=20,max_model_depth=100,min_sample_fit=10,min_sample_alpha=5,min_sample_leaf=5)
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

    #Eigenschappen model
    max_depth, max_model_depth = summary(model)
    return train_R2,test_R2,train_boom_R2,test_boom_R2,train_MSE,test_MSE,train_MSE_boom,test_MSE_boom ,max_depth, max_model_depth, ruisvrij_error,foutcorr,ambigu

#Eigenschappen van de invividuele bomen
def boom(M,Y,fun):
    res = np.zeros(100)
    for i in range(100):
        pred_X = M[:,i] #elke Xi is een rij dus predictions i-de boom in de i-de kolom
        res[i] = fun(Y, pred_X)
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
    info = np.zeros((13,9)) #13 outputs van RaFFLE_prediction en 9 alpha waarden
    rs_trainMSE = np.zeros(30) #gem over boom
    rs_G = np.zeros(30) #nu gat enkel voor forest
    rs_rho = np.zeros(30)
    minMSE_half = np.zeros(30) #laagste MSE waarde vergeleken met die bij alpha = 0.5
    opt_alpha_boom = np.zeros(30)
    opt_alpha_forest = np.zeros(30)
    c_forest = np.zeros(30)
    min_inwendig_forest = np.zeros(30)
    c_boom = np.zeros(30)
    min_inwendig_boom = np.zeros(30)
    aantal_omkeringen = np.zeros(30)
    treeMSE_d0 = np.zeros(30) #treeMSE bij alpha=0 - minMSE (gekeken naar de index)
    treeMSE_d1 = np.zeros(30)

    #node_types = np.zeros((4,30))  # 4 modellen [CON,PCON,LIN,PLIN]
    for i in range(30):
        train_X, train_y, train_ruis_y, test_X, test_y, test_ruis_y = fun2(fun1, N, SIGMA,i)
        matrix = alpha_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,i) #dim = (13,9)
        with open(f'data_per_iteratie{DPG}{N}-{SNR}-{i}.csv', 'w+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for j in range(13):
                spamwriter.writerow(matrix[j,])

        alpha = [0.0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1.0]
        rs_trainMSE[i] = stats.spearmanr(alpha,matrix[6,]).statistic #rij 6 want we bekijken voor boom
        aantal_omkeringen[i] = omkeringen(matrix[6,])

        G = matrix[5,:] - matrix[4,:] #generalisatiegat per alpha (voor forestMSEs)
        rs_G[i] = stats.spearmanr(alpha,G).statistic

        rs_rho[i] = stats.spearmanr(alpha,matrix[11,:]).statistic #geeft zo onderlijningen omdat het int verwacht maar in voorbeelden in de documentatie

        pos_minMSE_forest = np.argmin(matrix[5,:]) #over testwaarden
        pos_minMSE_boom = np.argmin(matrix[7, :])
        opt_alpha_boom[i] = pos_minMSE_boom
        opt_alpha_forest[i] = pos_minMSE_forest #voor de hypothesetesten gebruiken we de rangindex

        minMSE_forest = matrix[5,pos_minMSE_forest]
        minMSE_tree = matrix[7,pos_minMSE_boom]
        minMSE_half[i] = matrix[5,6] - minMSE_forest #6 is rangindex van alpha=0.5

        p_forest = np.polyfit([1,2,3,4,5,6,7,8,9],matrix[5,:],2)
        c_forest[i] = p_forest[0]
        min_inwendig_forest[i] = in_min(p_forest)

        p_boom = np.polyfit([1, 2, 3, 4, 5, 6, 7, 8, 9], matrix[7, :], 2)
        c_boom[i] = p_boom[0]
        min_inwendig_boom[i] = in_min(p_boom)

        treeMSE_d0[i] = matrix[7,0] - minMSE_tree
        treeMSE_d1[i] = matrix[7,-1] - minMSE_tree

        info += matrix
    return info/30, rs_trainMSE, aantal_omkeringen, rs_G, rs_rho, minMSE_half, opt_alpha_forest - opt_alpha_boom, c_forest, min_inwendig_forest, c_boom, min_inwendig_boom, treeMSE_d0, treeMSE_d1

def omkeringen(l):
    count = 0
    for i in range(len(l)-1):
        if l[i+1] < l[i]:
            count += 1
    return count

def in_min(p): #ligt het min van de kwadratische fit, -b/(2c) tussen alhpa = 0,resp 1
    if p[0] > 0:
        if 1 < -p[1]/(2*p[0]) < 9: #p[0] = c, p[1] = b, p[2] = a
            return 1
        else: #monotoom als de x-waarde van het minimum niet tussen 1 of 9 ligt
            return 2
    else:
        return 0

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
    alpha_results = np.zeros((13,9))
    #alpha_nodes = np.zeros((4,9))
    for i,alpha in enumerate([0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1]): #0; 0,01; 0,05; 0,1; 0,2; 0,35; 0,5; 0,75; 1
        alpha_results[0,i], alpha_results[1,i], alpha_results[2,i], alpha_results[3,i],alpha_results[4,i], alpha_results[5,i], alpha_results[6,i], alpha_results[7,i], alpha_results[8,i], alpha_results[9,i], alpha_results[10,i], alpha_results[11,i], alpha_results[12,i] = RaFFLE_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k,alpha=alpha)
    return alpha_results #matrix met kolommen de alpha waarden en rijen de eigenschappen
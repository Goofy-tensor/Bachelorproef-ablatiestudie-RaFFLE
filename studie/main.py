from pilot import RaFFLE, DEFAULT_DF_SETTINGS
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
from scipy import stats
import csv

#Met model.predict(X, individual=True) krijg je de voorspelling van elke boom apart => in matrix
#dus fout per boom, forestfout en correlatie uit één fit
#model.estimators[m].tree_summary(feature_names=...) geeft per knoop het modeltype, de diepte en de variabele


#Raffle model fitten
def RaFFLE_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,k=42,alpha=1):#n is op hoeveel delen van de dataset je test (eig gwn hoeveel keer)
    #model maken
    df = {k: 1 + alpha * (v - 1) for k, v in DEFAULT_DF_SETTINGS.items()}
    #Vraag promotor: Wat doet dit juist? dacht dat je gwn alpha kon ingeven of is alphe wat erbij je BIC komt?
    df["blin"] = -1

    #start = perf_counter() (geen rekentijd)

    #zet hieronder nog alle parameters expliciet (zie effe RaFFLE paper)
    model = RaFFLE(n_estimators=100,random_state=k,alpha=alpha,n_features_node=1,n_features_tree=1,df_settings=df,max_depth=20,max_model_depth=100,min_sample_fit=10,min_sample_alpha=5,min_sample_leaf=5) #n_features_node = 1 net zoals in dRaFFLE van paper
    model.fit(train_X, train_ruis_y)

    #end = perf_counter()
    # time = (end - start)

    #Waarden die we willen weten eruit halen
    predictions_train = model.predict(train_X)
    predictions_test = model.predict(test_X)
    train_R2 = r2_score(train_ruis_y,predictions_train)
    test_R2 = r2_score(test_ruis_y,predictions_test)
    train_MSE = mean_squared_error(train_ruis_y,predictions_train)
    test_MSE = mean_squared_error(test_ruis_y,predictions_test)

    y = predictions_test - test_y
    ruisvrij_error = 1/len(test_X)*(np.dot(y,y))

    train_M = model.predict(train_X, individual=True)
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

#GEMIDDELDE RSS van de invividuele bomen
def boom(M,Y,fun):
    res = np.zeros(100)
    for i in range(100):
        pred_X = M[:,i] # elke Xi is een rij dus predicions i-de boom in de i-de kolom
        res[i] = fun(Y, pred_X)
    return np.average(res)

#Foutcorrelatie, hoe veel gelijkaardige fouten de bomen maken
def foutcorrelatie(M,Y):
    L = np.zeros((len(Y),100))
    for i in range(100):
        z = M[:,i] - Y
        L[:,i] = z/np.sqrt((z.dot(z)))
    gramm = L.T.dot(L)
    intermediate = np.ones(100).dot(gramm) #elke kolom sommeren
    return (np.dot(intermediate,np.ones(100).T)-100)/(100*99)

#Abbiguïteit, hoeveel boom gemiddeld afwijkt
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


def iterate_predictions(train_X,test_X,train_ruis_y,test_ruis_y,test_y,DPG,N,S):
    info = np.zeros((13,9)) #30 want we doen per type 30 predictions
    rs_trainMSE = np.zeros(30) #gem over boom
    rs_G = np.zeros(30) #nu gat enkel voor forest
    rs_rho = np.zeros(30)
    minMSE_0 = np.zeros(30) #laagste MSE waarde vergeleken met die bij alpha = 0
    minMSE_1 = np.zeros(30)
    versch_ambigu = np.zeros(30) #verschil in ambiguiteit
    opt_alpha_boom = np.zeros(30)
    opt_alpha_forest = np.zeros(30)
    c = np.zeros(30)
    #node_types = np.zeros((4,30))  # 4 modellen [CON,PCON,LIN,PLIN]
    for i in range(30):
        matrix = alpha_prediction(train_X,test_X,train_ruis_y,test_ruis_y,test_y,i) #dim = (13,9)
        with open(f'data_per_iteratie{DPG}{N}-{S}{i}.csv', 'w+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            for j in range(13):
                spamwriter.writerow(matrix[j,])

        alpha = [0.0,0.01,0.05,0.1,0.2,0.35,0.5,0.75,1.0]
        rs_trainMSE[i] = stats.spearmanr(alpha,matrix[6,]).statistic #of rij 4 als je voor forest wilt
        G = matrix[5,:] - matrix[4,:] #generalisatiegat per alpha (voor forestMSEs)
        rs_G[i] = stats.spearmanr(alpha,G).statistic
        rs_rho[i] = stats.spearmanr(alpha,matrix[11,:]).statistic #geeft zo onderlijningen omdat het int verwahct maar in voorbeelden in de documentatie

        pos_minMSE_forest = np.argmin(matrix[5,:]) #over testwaarden
        pos_minMSE_boom = np.argmin(matrix[7, :])
        opt_alpha_boom[i] = alpha[pos_minMSE_boom]
        opt_alpha_forest[i] = alpha[pos_minMSE_forest]

        minMSE = matrix[5,pos_minMSE_forest]
        minMSE_0[i] = matrix[5,0] - minMSE
        minMSE_1[i] = matrix[5,-1] - minMSE
        versch_ambigu[i] = matrix[-1,0] - matrix[-1,-1]
        p = np.polyfit([1,2,3,4,5,6,7,8,9],matrix[5,:],2)
        c[i] = p[-1]

        info += matrix
    return info/30, rs_trainMSE, rs_G, rs_rho, minMSE_0, minMSE_1, versch_ambigu, opt_alpha_forest - opt_alpha_boom, c

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
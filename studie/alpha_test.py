import numpy as np
from pilot import RaFFLE, DEFAULT_DF_SETTINGS
import sys, pilot

#alpha-test: wordt alpha in RaFFLE intern toegepast op de vrijheidsgraden, of enkel via onze eigen schaling in main.py?
#test 1: A == B? dan schaalt het pakket zelf (en passen we alpha in main.py dubbel toe)
#        A == C? dan heeft alpha in het pakket geen effect op de vrijheidsgraden (onze eigen schaling is de enige)
#test 2: worden niet-gehele df_settings stilletjes afgekapt of afgerond? dan meten de kleine alpha's niet wat we denken

#controleren dat we de juiste omgeving en versie van pilot gebruiken
print(sys.executable)
print(pilot.__file__)

#kleine testdataset, geen noodzaak om de echte DGP's te gebruiken
rng = np.random.default_rng(0)
X = rng.normal(size=(300, 10))
y = X[:, 0] + 0.5 * X[:, 1]**2 + rng.normal(size=300)

#basis = standaard df-waarden met blin uitgeschakeld (-1), zoals in main.py
basis = dict(DEFAULT_DF_SETTINGS)
basis["blin"] = -1
#zelf geschaald met alpha = 0.5: 1 + alpha*(v-1), de -1 (uitgeschakeld) blijft ongemoeid
geschaald = {k: (v if v == -1 else 1 + 0.5*(v - 1)) for k, v in basis.items()}

def fit(alpha, df):
    #zelfde instellingen als in main.py, enkel minder bomen zodat het snel draait
    m = RaFFLE(n_estimators=20, random_state=1, alpha=alpha, df_settings=df,
               n_features_node=1, n_features_tree=1, max_depth=20, max_model_depth=100,
               min_sample_fit=10, min_sample_alpha=5, min_sample_leaf=5)
    m.fit(X, y)
    return m.predict(X)

A = fit(0.5, basis)       #alpha door het pakket laten schalen (df niet aangepast)
B = fit(1.0, geschaald)   #alpha zelf geschaald, alpha = 1 in het pakket
C = fit(1.0, basis)       #geen alpha-effect (referentie)

#test 1
print("A == B:", np.allclose(A, B))
print("A == C:", np.allclose(A, C))

#test 2: heeft onze eigen schaling effect? B == C betekent dat df_settings genegeerd of afgekapt wordt
print("B == C (eigen schaling zonder effect?):", np.allclose(B, C))

def df_met(factor_lin, factor_pcon, factor_plin):
    #basis-df met drie waarden aangepast, de rest blijft staan
    d = dict(basis)
    d["lin"] = factor_lin
    d["pcon"] = factor_pcon
    d["plin"] = factor_plin
    return d

D_fractioneel = fit(1.0, df_met(1.5, 3.4, 4.6))   #niet-gehele waarden
D_afgekapt    = fit(1.0, df_met(1, 3, 4))         #naar beneden afgekapt
D_afgerond    = fit(1.0, df_met(2, 3, 5))         #naar boven afgerond

#als fractioneel == afgekapt of afgerond, dan worden niet-gehele df stilletjes omgezet
print("fractioneel == afgekapt:", np.allclose(D_fractioneel, D_afgekapt))
print("fractioneel == afgerond:", np.allclose(D_fractioneel, D_afgerond))
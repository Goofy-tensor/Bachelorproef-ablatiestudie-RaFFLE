from pilot import PILOT
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [50, 40], [60, 10], [70, 25], [80, 5], [90, 30], [100, 15],
    [110, 50], [120, 20], [130, 8], [140, 35], [150, 12], [160, 45]
])
y = np.array([130, 170, 183, 215, 218, 248, 227, 265, 284, 276, 298, 285])

# Model trainen
model = PILOT(max_depth=5)
model.train(X, y)

# Voorspellingen
predictions = model.predict(X)

# Boomstructuur tonen
print(model.tree_summary())

# Fout op de trainingsdata
mse = np.mean((y - predictions) ** 2)
print(f"MSE op trainingsdata: {mse:.2f}")

# --- Grafieken ---
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

# 1. Voorspeld versus werkelijk
axes[0].scatter(y, predictions, color="tab:blue")
lims = [min(y.min(), predictions.min()), max(y.max(), predictions.max())]
axes[0].plot(lims, lims, "k--", label="perfecte voorspelling")
axes[0].set_xlabel("Werkelijke prijs (x 1000 euro)")
axes[0].set_ylabel("Voorspelde prijs (x 1000 euro)")
axes[0].set_title("Voorspeld vs. werkelijk")
axes[0].legend()

# 2. Prijs in functie van oppervlakte (werkelijk en voorspeld)
order = np.argsort(X[:, 0])
axes[1].scatter(X[:, 0], y, color="tab:gray", label="werkelijk")
axes[1].scatter(X[:, 0], predictions, color="tab:red", label="voorspeld")
axes[1].plot(X[order, 0], predictions[order], color="tab:red", alpha=0.5)
axes[1].set_xlabel("Woonoppervlakte (m$^2$)")
axes[1].set_ylabel("Prijs (x 1000 euro)")
axes[1].set_title("Prijs vs. oppervlakte")
axes[1].legend()

# 3. Voorspelling op een fijn rooster, met X2 vast op zijn gemiddelde
#grid = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
#X_grid = np.column_stack([grid, np.full_like(grid, X[:, 1].mean())])
#axes[2].scatter(X[:, 0], y, color="tab:gray", label="werkelijk")
#axes[2].plot(grid, model.predict(X_grid), color="tab:red",
             #label="PILOT (X2 = gemiddelde)")
#axes[2].set_xlabel("Woonoppervlakte (m$^2$)")
#axes[2].set_ylabel("Prijs (x 1000 euro)")
#axes[2].set_title("Voorspellingscurve")
#axes[2].legend()

plt.tight_layout()
plt.savefig("pilot_voorspellingen.png", dpi=200)
plt.show()
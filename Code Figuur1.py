import matplotlib.pyplot as plt
import numpy as np

#dit is even om te zien of github werkt

# Stijlinstellingen     
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
})

np.random.seed(42)
x_l = np.linspace(0, 0.5, 30)
x_r = np.linspace(0.5, 1, 30)
x_full = np.linspace(0, 1, 60)

# Maak een figuur met 2 rijen en 3 kolommen
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
axes = axes.flatten()

# Kleuren en stijlen
model_color = "tab:blue"
split_color = "tab:red"

y_l_const = np.ones_like(x_l) * 0.7
y_r_const = np.ones_like(x_r) * 0.3

# 1. CON: Constant model
axes[0].hlines(0.5, 0, 1, color=model_color, linewidth=2.5)
axes[0].set_title("CON (Constant)")

# 2. PCON: Piecewise Constant model
axes[1].hlines(0.7, 0, 0.5, color=model_color, linewidth=2.5)
axes[1].hlines(0.3, 0.5, 1, color=model_color, linewidth=2.5)
axes[1].axvline(0.5, color=split_color, linestyle="--", alpha=0.7)
axes[1].set_title("PCON (Piecewise Constant)")

# 3. LIN: Lineaire regressie
axes[2].plot(x_full, 0.2 + 0.6 * x_full, color=model_color, linewidth=2.5)
axes[2].set_title("LIN (Linear)")

# 4. BLIN: Broken Linear model 
axes[3].plot(x_l, 0.2 + 0.8 * x_l, color=model_color, linewidth=2.5)
# Laat rechterkant aansluiten in het punt (0.5, y_split)
y_split_blin = 0.2 + 0.8 * 0.5
axes[3].plot(x_r, y_split_blin - 0.5 * (x_r - 0.5), color=model_color, linewidth=2.5)
axes[3].axvline(0.5, color=split_color, linestyle="--", alpha=0.7)
axes[3].set_title("BLIN (Broken Linear - Cont.)")

# 5. PLIN: Two-Piece Linear model
axes[4].plot(x_l, 0.2 + 0.6 * x_l, color=model_color, linewidth=2.5)
axes[4].plot(x_r, 0.7 - 0.4 * (x_r - 0.5), color=model_color, linewidth=2.5)
axes[4].axvline(0.5, color=split_color, linestyle="--", alpha=0.7)
axes[4].set_title("PLIN (Two-Piece Linear)")

fig.delaxes(axes[5])

# Opmaak voor alle actieve subplots
for i in range(5):
    axes[i].set_xlim(-0.05, 1.05)
    axes[i].set_ylim(-0.05, 1.05)
    # Verberg de getallen/labels op de assen
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    # Rooster aanhouden
    axes[i].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()


plt.show()
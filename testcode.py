from pilot import PILOT, RaFFLE
import numpy as np

# Test RaFFLE
X = np.random.randn(100, 5)
y = X[:, 0] + np.random.randn(100) * 0.1

model = RaFFLE(n_estimators=10, max_depth=3, random_state=42)
model.fit(X, y)
predictions = model.predict(X)

print("Installation successful!")
print(f"R² score: {1 - np.mean((y - predictions)**2) / np.var(y):.3f}")
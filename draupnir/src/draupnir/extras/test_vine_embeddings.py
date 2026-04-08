#Todo: implement vine or hyperbolic

import numpy as np
import matplotlib.pyplot as plt

# Your observed data
X_obs = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]]) # Example timestamps
y_obs = np.array([1.2, 1.5, 1.1, 0.9, 1.3])           # Example values

# Create dense test points (e.g., 200 points between 0 and 2)
X_test = np.linspace(0, 2, 200).reshape(-1, 1)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Note: scikit-learn uses 'length_scale' directly.
# We set it to 1.0 as your model converged there.
kernel = C(2.0) * RBF(length_scale=1.0)

gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1) # alpha is noise variance
gp.fit(X_obs, y_obs)

# Predict
y_pred_mean, y_pred_std = gp.predict(X_test, return_std=True)

plt.figure(figsize=(10, 6))

# Plot observed data
plt.scatter(X_obs, y_obs, color='black', label='Observed Data', zorder=5)

# Plot posterior mean
plt.plot(X_test, y_pred_mean, 'b-', label='Posterior Mean', linewidth=2)

# Plot uncertainty band (Mean +/- 2 Std Dev)
plt.fill_between(X_test.flatten(),
                 y_pred_mean - 2 * y_pred_std,
                 y_pred_mean + 2 * y_pred_std,
                 color='blue', alpha=0.2, label='95% Confidence Interval')

plt.xlabel('Time (scaled)')
plt.ylabel('Value')
plt.title(f'Posterior Predictive Distribution (Length Scale = 1.0)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
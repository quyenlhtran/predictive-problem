import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Explicitly set the backend
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed
np.random.seed(42)

# Parameters
n_samples = 1000
p_success = 0.5

# Generate data
Y = np.random.binomial(1, p_success, n_samples)
x1 = np.random.normal(155, 6, n_samples)
x2 = np.random.normal(175, 10, n_samples)
final_values = Y * x1 + (1 - Y) * x2

# Histogram
plt.hist(final_values, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black', label='Histogram')

# Bimodal density curve
x_range = np.linspace(130, 210, 500)
y_mixture = 0.5 * norm.pdf(x_range, 155, 6) + 0.5 * norm.pdf(x_range, 175, 10)
plt.plot(x_range, y_mixture, color='red', linewidth=2, label='Mixture Density Curve')

# Labels
plt.xlabel('Generated Values')
plt.ylabel('Density')
plt.title('Bi-Modal Distribution with Custom Density Curve')
plt.legend()
plt.show()

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Fix PyCharm backend issue
import matplotlib.pyplot as plt

# Basket values
x_values = np.array([10, 18, 21, 30, 36])

# Define the penalty function
def penalty_half(t):
    return np.sum(np.abs(x_values - t) ** 0.5)

# Grid search between 19 and 23
t_values = np.linspace(19, 23, 400)
penalties = [penalty_half(t) for t in t_values]

# Find minimizer
min_index = np.argmin(penalties)
min_t = t_values[min_index]
min_penalty = penalties[min_index]

# Output
print(f"Minimum penalty sum: {min_penalty:.4f}")
print(f"Minimizer (nu_1/2) in [19, 23]: {min_t:.4f}")

# Plot
plt.plot(t_values, penalties, label=r"$\sum |x_i - t|^{1/2}$")
plt.axvline(min_t, color='red', linestyle='--', label=f"Min ≈ {min_t:.2f}")
plt.xlabel("t")
plt.ylabel("Penalty")
plt.title(r"Grid Search for $\nu_{1/2}$ in [19, 23]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

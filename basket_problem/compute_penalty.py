import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Basket of values
x_values = np.array([18, 20, 21, 23, 27, 29])
num_trials = 1000

# Grid search over t = 20 + 0.01 * i for i = 1 to 500
i_values = np.arange(1, 501)
t_values = 20 + 0.01 * i_values

# Define penalty powers and labels
powers = [0.5, 1, 1.5, 2]
rule_names = {
    0.5: "Rule 3 (power 1/2)",
    1:   "Rule 1 (power 1)",
    1.5: "Rule 4 (power 3/2)",
    2:   "Rule 2 (power 2)"
}

# Store best results per rule
min_results = {}

for power in powers:
    expected_penalties = []
    std_penalties = []

    for t in t_values:
        penalties = []
        for _ in range(num_trials):
            y = np.random.choice(x_values)
            penalty = abs(y - t) ** power
            penalties.append(penalty)
        penalties = np.array(penalties)
        expected_penalties.append(np.mean(penalties))
        std_penalties.append(np.std(penalties))

    expected_penalties = np.array(expected_penalties)
    std_penalties = np.array(std_penalties)
    min_index = np.argmin(expected_penalties)
    best_i = i_values[min_index]
    best_t = t_values[min_index]
    best_penalty = expected_penalties[min_index]
    best_std = std_penalties[min_index]

    min_results[power] = {
        "rule": rule_names[power],
        "best_i": best_i,
        "best_t": best_t,
        "mean_penalty": best_penalty,
        "std_penalty": best_std
    }

    # Plot
    plt.plot(i_values, expected_penalties, label=f"{rule_names[power]}")
    plt.axvline(best_i, color='red', linestyle='--', linewidth=0.8)

# Plot formatting
plt.xlabel(r"$i$ (where $t = 20 + 0.01 \cdot i$)")
plt.ylabel("Expected Penalty")
plt.title(r"Expected Penalty vs $i$ under Different Penalty Rules")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print report
for power in powers:
    result = min_results[power]
    print(f"{result['rule']}:")
    print(f"  Optimal i: {result['best_i']}")
    print(f"  Optimal t: {result['best_t']:.2f}")
    print(f"  Mean Penalty: {result['mean_penalty']:.4f}")
    print(f"  Std Dev Penalty: {result['std_penalty']:.4f}")
    print()

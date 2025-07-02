import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# Set random seed for reproducibility
np.random.seed(42)

# Basket values
x_values = np.array([10, 18, 21, 30, 36])
num_trials = 1000

# Compute mean and median
mean_val = np.mean(x_values)
median_val = np.median(x_values)

# Function to find minimizer for a given power
def minimize_power(power):
    loss_fn = lambda t: np.sum(np.abs(x_values - t) ** power)
    result = minimize_scalar(loss_fn, bounds=(min(x_values), max(x_values)), method='bounded')
    return result.x

# Compute nu_{1/2} and nu_{3/2}
nu_half = minimize_power(0.5)
nu_three_half = minimize_power(1.5)

# Define prediction strategies
strategies = {
    f"Mean ({mean_val:.2f})": lambda: mean_val,
    f"Median ({median_val:.2f})": lambda: median_val,
    "Flip Mean/Median": lambda: np.random.choice([mean_val, median_val]),
    "Random Choice": lambda: np.random.choice(x_values),
    f"nu_1/2 ({nu_half:.2f})": lambda: nu_half,
    f"nu_3/2 ({nu_three_half:.2f})": lambda: nu_three_half,
}

# Define penalty rules
penalty_rules = {
    "Rule 1": lambda y, yhat: abs(y - yhat),
    "Rule 2": lambda y, yhat: (y - yhat) ** 2,
    "Rule 3": lambda y, yhat: np.sqrt(abs(y - yhat)),
    "Rule 4": lambda y, yhat: abs(y - yhat) ** 1.5,
}

# Run simulation
results = {strategy: {rule: 0.0 for rule in penalty_rules} for strategy in strategies}

for _ in range(num_trials):
    y = np.random.choice(x_values)
    for strategy_name, predictor in strategies.items():
        yhat = predictor()
        for rule_name, penalty_fn in penalty_rules.items():
            results[strategy_name][rule_name] += penalty_fn(y, yhat)

# Create DataFrame and format table
df = pd.DataFrame(results).T.round(2)
df_display = df.applymap(lambda x: f"{x:>8}")
print(df_display)

# Print computed minimizers
print(f"\nComputed minimizers:")
print(f"  nu_1/2 ≈ {nu_half:.2f}")
print(f"  nu_3/2 ≈ {nu_three_half:.2f}")

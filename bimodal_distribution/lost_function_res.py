import numpy as np
import scipy.stats as stats

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
n_samples = 1000  # Total number of height samples
n_simulations = 1000  # Number of simulations
p_success = 0.5  # Probability of choosing x1


# Define loss functions
def mean_absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)  # Rule 1


def mean_squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2  # Rule 2


def mean_sqrt_error(y_true, y_pred):
    return np.sqrt(np.abs(y_true - y_pred))  # Rule 3


def mean_cubic_root_error(y_true, y_pred):
    return np.abs(y_true - y_pred) ** (3 / 2)  # Rule 4


# Initialize loss storage
loss_results = {name: {'MAE': [], 'MSE': [], 'MSRE': [], 'MCRE': []} for name in [
    "Mean Prediction",
    "Median Prediction",
    "Random Normal Prediction",
    "Random Middle 50% Prediction",
    "Trimmed Mean Prediction"
]}

# Run simulations
for _ in range(n_simulations):
    # Generate binomial variable Y
    Y = np.random.binomial(1, p_success, n_samples)

    # Generate x1 and x2 from normal distributions
    x1 = np.random.normal(155, 6, n_samples)
    x2 = np.random.normal(175, 10, n_samples)

    # Compute actual values
    heights = Y * x1 + (1 - Y) * x2

    # Compute predictive strategies
    mean_prediction = np.mean(heights)
    median_prediction = np.median(heights)
    random_normal_prediction = np.random.normal(mean_prediction, np.std(heights))
    q25, q75 = np.percentile(heights, [25, 75])
    random_middle_50 = np.random.uniform(q25, q75)
    trimmed_mean_prediction = stats.trim_mean(heights, proportiontocut=0.25)

    strategies = {
        "Mean Prediction": mean_prediction,
        "Median Prediction": median_prediction,
        "Random Normal Prediction": random_normal_prediction,
        "Random Middle 50% Prediction": random_middle_50,
        "Trimmed Mean Prediction": trimmed_mean_prediction
    }

    # Compute loss for each strategy
    for name, pred in strategies.items():
        loss_results[name]['MAE'].append(mean_absolute_error(heights, pred).mean())
        loss_results[name]['MSE'].append(mean_squared_error(heights, pred).mean())
        loss_results[name]['MSRE'].append(mean_sqrt_error(heights, pred).mean())
        loss_results[name]['MCRE'].append(mean_cubic_root_error(heights, pred).mean())

# Compute the average losses across all simulations
avg_loss = {name: {metric: np.mean(losses) for metric, losses in loss_dict.items()}
            for name, loss_dict in loss_results.items()}

# Print final loss results
for strategy, metrics in avg_loss.items():
    print(f"{strategy}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Fix for PyCharm backend issue
matplotlib.use('TkAgg')  # Use TkAgg for interactive plots

# Set seed for reproducibility
np.random.seed(42)

# Simulation parameters
num_samples = 1000
prop_small_donors = 0.8  # 80% small donors, 20% generous donors

# Gamma distribution parameters
shape_small, scale_small = 10, 10  # Mean = shape * scale = 100
shape_large, scale_large = 8, 50  # Mean = shape * scale = 400

# Generate donations
small_donors = np.random.gamma(shape_small, scale_small, int(num_samples * prop_small_donors))
large_donors = np.random.gamma(shape_large, scale_large, int(num_samples * (1 - prop_small_donors)))
donations = np.concatenate((small_donors, large_donors))

# Shuffle donations to simulate sequential arrival
np.random.shuffle(donations)

from scipy.stats import gamma

# Define the x range for the density curve
x = np.linspace(0, max(donations), 1000)

# Define the gamma PDFs for each group
pdf_small = gamma.pdf(x, a=10, scale=10)  # shape=10, scale=10
pdf_large = gamma.pdf(x, a=8, scale=50)   # shape=8, scale=50

# Weighted average to create the mixture density (80% small, 20% large)
mixture_pdf = 0.8 * pdf_small + 0.2 * pdf_large

# Plot
plt.figure(figsize=(10, 5))
sns.histplot(donations, bins=50, stat="density", label="Donation Distribution")
plt.plot(x, mixture_pdf, color="red", label="Theoretical Density (Mixture Gamma)")
plt.xlabel("Donation Amount")
plt.ylabel("Density")
plt.title("Histogram and Density of Simulated Charitable Donations")
plt.legend()
plt.show()



# --- Predictive Strategies ---
def mean_prediction(observed_data):
    return np.mean(observed_data)


def median_prediction(observed_data):
    return np.median(observed_data)


def middle_50_prediction(observed_data):
    q1, q3 = np.percentile(observed_data, [25, 75])
    return np.random.uniform(q1, q3)  # Randomly choose within the middle 50%


def random_guess_prediction(observed_data):
    return np.random.choice(observed_data)  # Randomly select from observed values

def trimmed_mean_prediction(observed_data, trim_percent=25):
    lower = np.percentile(observed_data, trim_percent)
    upper = np.percentile(observed_data, 100 - trim_percent)
    middle_values = [x for x in observed_data if lower <= x <= upper]
    return np.mean(middle_values) if middle_values else np.mean(observed_data)


def fixed_range_prediction_factory(use_median=False, bin_width=200, activate_after=200):
    fixed_prediction = [None]  # Use a list for mutability in closure

    def strategy(observed_data):
        if len(observed_data) < activate_after:
            return np.mean(observed_data) if not use_median else np.median(observed_data)

        if fixed_prediction[0] is None:
            # Compute once after threshold
            counts, bin_edges = np.histogram(observed_data, bins=np.arange(0, max(observed_data) + bin_width, bin_width))
            max_bin_index = np.argmax(counts)
            range_start = bin_edges[max_bin_index]
            range_end = bin_edges[max_bin_index + 1]

            # Get values in that bin
            in_range_values = [x for x in observed_data if range_start <= x < range_end]

            # Compute the fixed prediction
            if not in_range_values:  # fallback
                fixed_prediction[0] = np.mean(observed_data) if not use_median else np.median(observed_data)
            else:
                fixed_prediction[0] = np.median(in_range_values) if use_median else np.mean(in_range_values)

        return fixed_prediction[0]

    return strategy




# --- Penalty Functions ---
def compute_penalties(true_value, predicted_value):
    return {
        "Absolute Error": np.abs(true_value - predicted_value),
        "Squared Error": (true_value - predicted_value) ** 2,
        "Square Root Error": np.sqrt(np.abs(true_value - predicted_value)),
        "Higher-Order Absolute Error": np.abs(true_value - predicted_value) ** 1.5
    }


# Store predictions and penalties
strategies = {
    "Mean": mean_prediction,
    "Median": median_prediction,
    "Middle 50%": middle_50_prediction,
    "Random Guess": random_guess_prediction,
    "Most Frequent Bin → Fixed Mean": fixed_range_prediction_factory(use_median=False),
    "Most Frequent Bin → Fixed Median": fixed_range_prediction_factory(use_median=True),
    "Trimmed Mean": trimmed_mean_prediction
}


# Initialize penalty storage
penalties = {method: {"Absolute Error": 0, "Squared Error": 0, "Square Root Error": 0, "Higher-Order Absolute Error": 0}
             for method in strategies}

# Sequential Prediction with Accumulated Penalty Calculation
observed_data = []
for i, true_value in enumerate(donations):
    if len(observed_data) == 0:
        observed_data.append(true_value)
        continue  # Skip first prediction (no prior data)

    predictions = {method: strategies[method](observed_data) for method in strategies}

    for method in strategies:
        penalty_values = compute_penalties(true_value, predictions[method])
        for rule in penalties[method]:
            penalties[method][rule] += penalty_values[rule]

    observed_data.append(true_value)  # Update observed data

# Convert penalty results into DataFrame for analysis
penalty_summary = pd.DataFrame.from_dict(penalties, orient='index') / (num_samples - 1)  # Average penalties

# Round values for cleaner display
penalty_summary_rounded = penalty_summary.round(2)

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


# Display penalty summary in console
print("\n=== Predictive Strategy Performance Comparison ===\n")
print(penalty_summary_rounded.sort_values(by="Absolute Error"))

# Optional: Show DataFrame visually in PyCharm popup
try:
    import ace_tools as tools
    tools.display_dataframe_to_user(name="Penalty Comparison Results", dataframe=penalty_summary)
except ImportError:
    pass  # If ace_tools not available, continue silently

print("Standard Deviation of Small Donors:", np.std(small_donors, ddof=1))
print("Standard Deviation of Large Donors:", np.std(large_donors, ddof=1))


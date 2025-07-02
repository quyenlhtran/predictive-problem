import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
n = 2000
num_simulations = 1000
p = 0.7

# Strategies (matching previous code)
strategies = [
    "Always Stick to Heads",
    "Always Stick to Tails",
    "Delay Estimate-Based Fixed",
    "Repeat Adherence",
    "Repeat Avoidance",
    "Randomize Always",
    "Estimate-Based Adaptive"
]

# Color map (matching previous plotting)
strategy_colors = {
    "Always Stick to Heads": "lightblue",
    "Always Stick to Tails": "royalblue",
    "Delay Estimate-Based Fixed": "green",
    "Repeat Adherence": "red",
    "Repeat Avoidance": "purple",
    "Randomize Always": "gray",
    "Estimate-Based Adaptive": "gold"
}

# Accuracy storage
final_accuracies = {strategy: [] for strategy in strategies}

# Bernoulli simulation function
def simulate_bernoulli(p, n):
    return np.random.choice([0, 1], size=n, p=[1 - p, p])

# Accuracy calculator
def calculate_cumulative_accuracy(predictions, outcomes):
    correct = (predictions == outcomes).astype(int)
    return np.cumsum(correct) / np.arange(1, len(outcomes) + 1)

# Run simulations
for _ in range(num_simulations):
    outcomes = simulate_bernoulli(p, n)

    # Always Stick to Heads
    pred_heads = np.ones(n)
    acc_heads = calculate_cumulative_accuracy(pred_heads, outcomes)
    final_accuracies["Always Stick to Heads"].append(acc_heads[-1])

    # Always Stick to Tails
    pred_tails = np.zeros(n)
    acc_tails = calculate_cumulative_accuracy(pred_tails, outcomes)
    final_accuracies["Always Stick to Tails"].append(acc_tails[-1])

    # Delay Estimate-Based Fixed
    k = 100
    pred_fixed = np.zeros(n)
    pred_fixed[k:] = (np.mean(outcomes[:k]) >= 0.5).astype(int)
    acc_fixed = calculate_cumulative_accuracy(pred_fixed, outcomes)
    final_accuracies["Delay Estimate-Based Fixed"].append(acc_fixed[-1])

    # Estimate-Based Adaptive
    pred_adaptive = np.zeros(n)
    for i in range(1, n):
        pred_adaptive[i] = 1 if np.mean(outcomes[:i]) >= 0.5 else 0
    acc_adaptive = calculate_cumulative_accuracy(pred_adaptive, outcomes)
    final_accuracies["Estimate-Based Adaptive"].append(acc_adaptive[-1])

    # Repeat Preference
    pred_repeat = np.zeros(n)
    pred_repeat[1:] = outcomes[:-1]
    acc_repeat = calculate_cumulative_accuracy(pred_repeat, outcomes)
    final_accuracies["Repeat Adherence"].append(acc_repeat[-1])

    # Repeat Avoidance
    pred_avoid = np.ones(n)
    pred_avoid[1:] = 1 - outcomes[:-1]
    acc_avoid = calculate_cumulative_accuracy(pred_avoid, outcomes)
    final_accuracies["Repeat Avoidance"].append(acc_avoid[-1])

    # Randomize Always
    pred_random = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    acc_random = calculate_cumulative_accuracy(pred_random, outcomes)
    final_accuracies["Randomize Always"].append(acc_random[-1])

summary = {
    strategy: {
        "Mean": np.mean(scores),
        "Standard Deviation": np.std(scores)
    }
    for strategy, scores in final_accuracies.items()
}

# Convert to DataFrame
summary_df = pd.DataFrame(summary).T.round(4)

# Display results
print(summary_df)

# Plot histograms
plt.figure(figsize=(12, 8))

# Set custom y-coordinate for each strategy
arrow_y_positions = {
    "Repeat Adherence": -3,
    "Repeat Avoidance": -3,
    "Randomize Always": -3
}

for strategy in strategies:
    data = final_accuracies[strategy]
    mean = np.mean(data)
    std = np.std(data)
    hist, bins = np.histogram(data, bins=30, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    max_height = max(hist)

    is_inverted = "Always Stick to" in strategy
    bar_height = -hist if is_inverted else hist

    # Arrow and text positions
    arrow_gap = 2
    text_gap = 1
    y_arrow = -(max_height + arrow_gap) if is_inverted else (max_height + arrow_gap)
    y_text = y_arrow - 2*text_gap if is_inverted else y_arrow + text_gap

    # Display label
    display_label = f"{strategy} (Inverted)" if is_inverted else strategy

    # Plot bars
    plt.bar(bin_centers, bar_height, width=(bins[1] - bins[0]),
            label=display_label, color=strategy_colors[strategy], edgecolor='black', alpha=0.7 if is_inverted else 0.5)

    # Plot arrows for mean ± std
    plt.annotate('', xy=(mean + std, y_arrow), xytext=(mean, y_arrow),
                 arrowprops=dict(arrowstyle='->', color='black'))
    plt.annotate('', xy=(mean - std, y_arrow), xytext=(mean, y_arrow),
                 arrowprops=dict(arrowstyle='->', color='black'))

    # Text above the arrow
    plt.text(mean, y_text, f"μ={mean:.2f}", ha='center', fontsize=8)

plt.axhline(0, color='black', linewidth=0.8)
plt.xlabel("Final Cumulative Accuracy")
plt.ylabel("Density (some inverted)")
plt.title("Distributions of accuracy after 2000 predictions based on 1000 simulations ($p = 0.7$)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_cumulative_accuracy.png")
plt.show()
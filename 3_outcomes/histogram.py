import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
n = 2000
num_simulations = 1000
pA, pB, pC = 0.5, 0.3, 0.2

strategies = [
    "Always Stick to A",
    "Delay Estimate-Based Fixed",
    "Repeat-Adherence Strategy",
    "Randomize All The Time",
    "Repeat-Avoidance Strategy",
    "Estimate-Based Adaptive"
]

strategy_colors = {
    "Always Stick to A": "blue",
    "Delay Estimate-Based Fixed": "green",
    "Repeat-Adherence Strategy": "red",
    "Randomize All The Time": "purple",
    "Repeat-Avoidance Strategy": "lightblue",
    "Estimate-Based Adaptive": "orange"
}

# Initialize storage
final_accuracies = {s: [] for s in strategies}

def simulate_three_outcome(pA, pB, pC, n):
    return np.random.choice(["A", "B", "C"], size=n, p=[pA, pB, pC])

def calculate_cumulative_accuracy(predictions, outcomes):
    correct = (np.array(predictions) == np.array(outcomes)).astype(int)
    return np.cumsum(correct) / np.arange(1, len(outcomes) + 1)

# Simulations
for _ in range(num_simulations):
    outcomes = simulate_three_outcome(pA, pB, pC, n)

    # Strategy 1: Always Stick to A
    preds = ["A"] * n
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Always Stick to A"].append(acc[-1])

    # Strategy 2: Delay Estimate-Based Fixed
    k = 100
    counts = {"A": 0, "B": 0, "C": 0}
    preds = []
    for i in range(n):
        if i < k:
            p = "A"
            counts[outcomes[i]] += 1
        else:
            p = max(counts, key=counts.get)
        preds.append(p)
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Delay Estimate-Based Fixed"].append(acc[-1])

    # Strategy 3: Estimate-Based Adaptive
    counts = {"A": 0, "B": 0, "C": 0}
    preds = []
    for i in range(n):
        if i > 0:
            probs = {k: counts[k] / i for k in counts}
            p = max(probs, key=probs.get)
        else:
            p = np.random.choice(["A", "B", "C"])
        preds.append(p)
        counts[outcomes[i]] += 1
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Estimate-Based Adaptive"].append(acc[-1])

    # Strategy 4: Repeat-Adherence Strategy
    preds = ["A"] + list(outcomes[:-1])
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Repeat-Adherence Strategy"].append(acc[-1])

    # Strategy 5: Randomize All The Time
    preds = [np.random.choice(["A", "B", "C"]) for _ in range(n)]
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Randomize All The Time"].append(acc[-1])

    # Strategy 6: Repeat-Avoidance Strategy
    preds = []
    for i in range(n):
        if i == 0:
            p = np.random.choice(["A", "B", "C"])
        else:
            p = np.random.choice([x for x in ["A", "B", "C"] if x != outcomes[i - 1]])
        preds.append(p)
    acc = calculate_cumulative_accuracy(preds, outcomes)
    final_accuracies["Repeat-Avoidance Strategy"].append(acc[-1])

df = pd.DataFrame({
    strategy: {
        "Mean": np.mean(scores),
        "Standard Deviation": np.std(scores)
    } for strategy, scores in final_accuracies.items()
}).T

print(df)

# Define equal-width bin edges
breaks = np.arange(0.25, 0.55 + 0.0025, 0.0025)

plt.figure(figsize=(14, 8))

for strategy in strategies:
    data = final_accuracies[strategy]
    mean = np.mean(data)

    # Histogram with consistent binning
    hist, bins = np.histogram(data, bins=breaks, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    max_height = max(hist)

    is_inverted = "Always Stick to" in strategy
    bar_height = -hist if is_inverted else hist

    display_label = f"{strategy} (Inverted)" if is_inverted else strategy

    # Plot bars
    plt.bar(bin_centers, bar_height,
            width=(bins[1] - bins[0]),
            label=display_label,
            color=strategy_colors[strategy],
            edgecolor='black',
            alpha=0.7 if is_inverted else 0.5)

    # === Control vertical spacing per strategy ===
    arrow_gap = 0.2  # space between bar top and arrow
    text_gap = 0.1   # space between arrow and label

    if strategy == "Delay Estimate-Based Fixed":
        y_arrow = max_height + arrow_gap if not is_inverted else -(max_height + arrow_gap)
    elif strategy == "Estimate-Based Adaptive":
        y_arrow = max_height + 3 * arrow_gap if not is_inverted else -(max_height + 3 * arrow_gap)
    else:
        y_arrow = max_height + 2 * arrow_gap if not is_inverted else -(max_height + 2 * arrow_gap)

    y_text = y_arrow + 3*text_gap if not is_inverted else y_arrow - 3*text_gap
    # =============================================

    # Horizontal double-headed arrow centered at mean
    arrow_style = dict(arrowstyle='<|-|>', color='black', linewidth=1.2)
    arrow_half_length = 0.01
    plt.annotate('', xy=(mean + arrow_half_length, y_arrow),
                 xytext=(mean - arrow_half_length, y_arrow),
                 arrowprops=arrow_style)

    # Text for mean
    plt.text(mean, y_text,
             f"μ = {mean:.3f}",
             ha='center', fontsize=9)

# Final formatting
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Final Cumulative Accuracy", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.title(f"Distribution of Final Cumulative Accuracy Over {num_simulations} Simulations", fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("histogram_mean_arrow_horizontal_custom_spacing.png")
plt.show()

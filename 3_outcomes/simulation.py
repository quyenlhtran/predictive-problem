import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # GUI backend
import matplotlib.pyplot as plt

# Parameters
n = 2000  # Number of trials

def simulate_three_outcome(pA, pB, pC, n):
    assert abs(pA + pB + pC - 1.0) < 1e-6, "Probabilities must sum to 1"
    return np.random.choice(["A", "B", "C"], size=n, p=[pA, pB, pC])

def calculate_cumulative_accuracy(predictions, outcomes):
    correct_predictions = (np.array(predictions) == np.array(outcomes)).astype(int)
    return np.cumsum(correct_predictions) / np.arange(1, len(outcomes) + 1)

def plot_three_outcome_linegraphs():
    np.random.seed(42)  # Reproducibility
    pA, pB, pC = 0.5, 0.3, 0.2
    outcomes = simulate_three_outcome(pA, pB, pC, n)

    # Strategy 1: Always Stick (to A)
    always_stick_predictions = ["A"] * n
    always_stick_cumulative_accuracy = calculate_cumulative_accuracy(always_stick_predictions, outcomes)

    # Strategy 2: Delay Estimate-Based Fixed
    k = 100
    delay_fixed_predictions = []
    burnin_counts = {"A": 0, "B": 0, "C": 0}
    for i in range(n):
        if i < k:
            prediction = "A"  # arbitrary during burn-in
            burnin_counts[outcomes[i]] += 1
        else:
            locked = max(burnin_counts, key=burnin_counts.get)
            prediction = locked
        delay_fixed_predictions.append(prediction)
    delay_fixed_cumulative_accuracy = calculate_cumulative_accuracy(delay_fixed_predictions, outcomes)

    # Strategy 3: Estimate-Based Adaptive
    adaptive_predictions = []
    counts = {"A": 0, "B": 0, "C": 0}
    for i in range(n):
        if i > 0:
            estimated_probs = {k: counts[k] / i for k in counts}
            prediction = max(estimated_probs, key=estimated_probs.get)
        else:
            prediction = np.random.choice(["A", "B", "C"])
        adaptive_predictions.append(prediction)
        counts[outcomes[i]] += 1
    adaptive_cumulative_accuracy = calculate_cumulative_accuracy(adaptive_predictions, outcomes)

    # Strategy 4: Repeat-Adherence Strategy (Myopic)
    myopic_predictions = ["A"] + list(outcomes[:-1])  # First is arbitrary
    myopic_cumulative_accuracy = calculate_cumulative_accuracy(myopic_predictions, outcomes)

    # Strategy 5: Randomize All The Time
    always_randomize_predictions = [np.random.choice(["A", "B", "C"]) for _ in range(n)]
    always_randomize_cumulative_accuracy = calculate_cumulative_accuracy(always_randomize_predictions, outcomes)

    # Strategy 6: Repeat-Avoidance Strategy (Anti-Persistence)
    anti_persistence_predictions = []
    for i in range(n):
        if i == 0:
            prediction = np.random.choice(["A", "B", "C"])
        else:
            prev = outcomes[i - 1]
            choices = [x for x in ["A", "B", "C"] if x != prev]
            prediction = np.random.choice(choices)
        anti_persistence_predictions.append(prediction)
    anti_persistence_cumulative_accuracy = calculate_cumulative_accuracy(anti_persistence_predictions, outcomes)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(always_stick_cumulative_accuracy, label="Always Stick to A", color='blue', linewidth=2.5)
    plt.plot(delay_fixed_cumulative_accuracy, label="Delay Estimate-Based Fixed", color='green')
    plt.plot(myopic_cumulative_accuracy, label="Repeat-Adherence Strategy", color='red')
    plt.plot(always_randomize_cumulative_accuracy, label="Randomize All The Time", color='purple')
    plt.plot(anti_persistence_cumulative_accuracy, label="Repeat-Avoidance Strategy", color='lightblue')
    plt.plot(adaptive_cumulative_accuracy, label="Estimate-Based Adaptive", color='orange')

    plt.xlabel("Number of Predictions", fontsize=12)
    plt.ylabel("Cumulative Accuracy", fontsize=12)
    plt.title(r"Three-Outcome Problem ($p_A = 0.5,\ p_B = 0.3,\ p_C = 0.2$)", fontsize=14)

    # Legend in top-right
    plt.legend(loc='upper right', fontsize=10)

    # Explanatory note at the bottom
    plt.figtext(0.5, -0.05,
                "Estimate-based strategies use empirical frequencies to adapt prediction behavior.",
                wrap=True, horizontalalignment='center', fontsize=10)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run the function
plot_three_outcome_linegraphs()

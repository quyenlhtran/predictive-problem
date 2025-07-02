import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

# Parameters
n = 2000
p_values = [0.7]

# Bernoulli simulation
def simulate_bernoulli(p, n):
    return np.random.choice([0, 1], size=n, p=[1 - p, p])

# Accuracy calculator
def calculate_cumulative_accuracy(predictions, outcomes):
    correct_predictions = (predictions == outcomes).astype(int)
    return np.cumsum(correct_predictions) / np.arange(1, len(outcomes) + 1)

# Plotting
def plot_bernoulli_linegraphs():
    for p in p_values:
        outcomes = simulate_bernoulli(p, n)

        # Always Stick to Heads
        stick_heads = np.ones(n)
        acc_heads = calculate_cumulative_accuracy(stick_heads, outcomes)

        # Always Stick to Tails
        stick_tails = np.zeros(n)
        acc_tails = calculate_cumulative_accuracy(stick_tails, outcomes)

        # Delay Estimate-Based Fixed (burn-in then fixed decision)
        k = 100
        delay_fixed_predictions = np.zeros(n)
        delay_fixed_predictions[k:] = (np.mean(outcomes[:k]) >= 0.5).astype(int)
        acc_delay_fixed = calculate_cumulative_accuracy(delay_fixed_predictions, outcomes)

        # Estimate-Based Adaptive (update every step)
        adaptive_predictions = np.zeros(n)
        for i in range(1, n):
            adaptive_predictions[i] = 1 if np.mean(outcomes[:i]) >= 0.5 else 0
        acc_adaptive = calculate_cumulative_accuracy(adaptive_predictions, outcomes)

        # Repeat Preference (predict previous)
        repeat_pref = np.zeros(n)
        repeat_pref[1:] = outcomes[:-1]
        acc_repeat_pref = calculate_cumulative_accuracy(repeat_pref, outcomes)

        # Repeat Avoidance (predict opposite of previous)
        repeat_avoid = np.ones(n)
        repeat_avoid[1:] = 1 - outcomes[:-1]
        acc_repeat_avoid = calculate_cumulative_accuracy(repeat_avoid, outcomes)

        # Randomize Always
        random_guess = np.random.choice([0, 1], size=n, p=[0.5, 0.5])
        acc_random = calculate_cumulative_accuracy(random_guess, outcomes)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(acc_heads, label="Always Stick to Heads", color='lightblue')
        plt.plot(acc_tails, label="Always Stick to Tails", color='royalblue')
        plt.plot(acc_delay_fixed, label="Delay Estimate-Based Fixed", color='green')
        plt.plot(acc_repeat_pref, label="Repeat Adherence", color='red')
        plt.plot(acc_repeat_avoid, label="Repeat Avoidance", color='purple')
        plt.plot(acc_random, label="Randomize Always", color='gray')
        plt.plot(acc_adaptive, label="Estimate-Based Adaptive", color='gold', linewidth=2.5)  # Gold on top

        plt.xlabel("Number of Predictions")
        plt.ylabel("Cumulative Accuracy")
        plt.title(f"Bernoulli Problem Simulation (p={p})")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

# Run it
plot_bernoulli_linegraphs()

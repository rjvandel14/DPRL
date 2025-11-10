# assignment_dp.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# Problem primitives
T = 150
h = 0.1
p_t = np.arange(1, T + 1) / 150.0  # demand prob at each t

def solve_dp(Xmax: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-horizon DP via backward recursion (start-of-period holding).
    Returns:
        V: shape (T+2, Xmax+1), with V[T+1, :] = 0 terminal values
        pi: shape (T+1, Xmax+1), optimal actions in {0,1}
    """
    V = np.zeros((T + 2, Xmax + 1), dtype=float)  # V[T+1,:]=0 already
    pi = np.zeros((T + 1, Xmax + 1), dtype=int)

    for t in range(T, 0, -1):
        p = p_t[t - 1]
        for x in range(Xmax + 1):
            Q = np.zeros(2, dtype=float)
            for a in (0, 1):
                q = 0.5 * a  # arrival prob if we order

                # Expected sales; start-of-period holding cost
                ES = p if x >= 1 else q * p
                reward = ES - h * x

                # Next-state value E[V_{t+1}(X_{t+1})]
                if x >= 1:
                    pr_xm1 = p * (1 - q)
                    pr_x   = p * q + (1 - p) * (1 - q)
                    pr_xp1 = (1 - p) * q
                    y_vals = np.array([x - 1, x, min(x + 1, Xmax)], dtype=int)
                    probs  = np.array([pr_xm1, pr_x, pr_xp1], dtype=float)
                    expected_next = probs @ V[t + 1, y_vals]
                else:
                    if a == 0:
                        expected_next = V[t + 1, 0]
                    else:
                        expected_next = 0.5 * (1 - p) * V[t + 1, 1] + (0.5 + 0.5 * p) * V[t + 1, 0]

                Q[a] = reward + expected_next

            a_star = int(Q[1] > Q[0])  # tie-break to 0
            V[t, x] = Q[a_star]
            pi[t, x] = a_star

    return V, pi

# Simulation 
rng = np.random.default_rng(seed=42)

def simulate(policy: np.ndarray, Xmax: int = 25, runs: int = 1_000) -> tuple[float, np.ndarray]:
    rewards = np.zeros(runs, dtype=float)
    for r in range(runs):
        x = 5
        total = 0.0
        for t in range(1, T + 1):
            a = policy[t, x]
            y = 1 if (a == 1 and rng.random() < 0.5) else 0  # arrival before demand
            x_tilde = x + y
            d = 1 if (rng.random() < p_t[t - 1]) else 0
            s = min(x_tilde, d)
            total += s - h * x  # start-of-period holding
            x = max(x_tilde - d, 0)
            x = min(x, Xmax)
        rewards[r] = total
    return float(np.mean(rewards)), rewards

# Quick run & report-ready plots 
if __name__ == "__main__":
    XMAX = 25
    V, pi = solve_dp(XMAX)

    print(f"DP expected reward V_1(5): {V[1, 5]:.4f}")

    avg_sim, samples = simulate(pi, XMAX, runs=1000)
    print(f"Simulation average over 1000 runs: {avg_sim:.4f}")

    # Plot styling
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 300,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    })

    # Figure 1: Optimal policy
    cmap = ListedColormap(["#2c3e50", "#f1c40f"])   # 0=no order, 1=order
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    xmax_for_plot = min(XMAX - 1, pi.shape[1] - 2)  # hide top boundary row
    policy_slice = pi[1:T+1, :xmax_for_plot+1].T

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    im = ax.imshow(policy_slice, aspect="auto", origin="lower",
                   cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Inventory $x$")
    ax.set_title("Optimal policy $\\pi_t(x)$")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, xmax_for_plot)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["0 = no order", "1 = order"])

    fig.tight_layout()
    fig.savefig("fig_policy.png", bbox_inches="tight")

    # Figure 2: Reward distribution
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.hist(samples, bins=30, edgecolor="black", linewidth=0.3)

    ax.set_xlabel("Total reward")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of total reward over 1000 runs")

    sim_mean = float(np.mean(samples))
    dp_value = float(V[1, 5])

    line_sim = ax.axvline(sim_mean, color="orange", linewidth=2.0, label="Simulation mean")
    line_dp  = ax.axvline(dp_value, color="yellow", linewidth=2.0, label=r"$V_1(5)$")

    ax.legend(handles=[line_sim, line_dp], loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig("fig_rewards_hist.png", bbox_inches="tight")

    plt.show()


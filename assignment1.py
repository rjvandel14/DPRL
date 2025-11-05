# assignment_dp.py
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# ----- Problem primitives -----
T = 150
h = 0.1
p_t = np.arange(1, T + 1) / 150.0  # demand prob at each t

def solve_dp(Xmax: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Finite-horizon DP via backward recursion.
    Returns:
        V: shape (T+2, Xmax+1), with V[T+1, :] = 0 terminal values
        pi: optimal action policy, shape (T+1, Xmax+1), actions in {0,1}
    """
    V = np.zeros((T + 2, Xmax + 1), dtype=float)  # V[T+1,:]=0 already
    pi = np.zeros((T + 1, Xmax + 1), dtype=int)

    for t in range(T, 0, -1):
        p = p_t[t - 1]
        for x in range(Xmax + 1):
            # Evaluate Q(x,a) for a=0 and a=1
            Q = np.zeros(2, dtype=float)
            for a in (0, 1):
                q = 0.5 * a  # arrival prob if we order
                # Reward r_t(x,a) = E[S_t] - h*x
                if x >= 1:
                    ES = p
                else:
                    ES = 0.5 * a * p
                reward = ES - h * x

                # Transitions: X_{t+1} = max(x + Y - D, 0)
                if x >= 1:
                    # Probabilities to x-1, x, x+1
                    pr_xm1 = p * (1 - q)
                    pr_x   = p * q + (1 - p) * (1 - q)
                    pr_xp1 = (1 - p) * q

                    y_vals = np.array([x - 1, x, min(x + 1, Xmax)], dtype=int)
                    probs  = np.array([pr_xm1, pr_x, pr_xp1], dtype=float)

                    # If x==Xmax, x+1 is folded into Xmax (already handled by min)
                    expected_next = probs @ V[t + 1, y_vals]

                else:  # x == 0
                    if a == 0:
                        expected_next = V[t + 1, 0]
                    else:
                        # a=1: Pr(x'=1)=0.5*(1-p), Pr(x'=0)=0.5+0.5*p
                        expected_next = 0.5 * (1 - p) * V[t + 1, 1] + (0.5 + 0.5 * p) * V[t + 1, 0]

                Q[a] = reward + expected_next

            # Greedy action, tie-break to 0 (or 1—your choice) if equal
            a_star = int(Q[1] > Q[0])
            V[t, x] = Q[a_star]
            pi[t, x] = a_star

    return V, pi

# ----- Simulation under a given policy -----
rng = np.random.default_rng(seed=42)

def simulate(policy: np.ndarray, Xmax: int = 20, runs: int = 1_000) -> tuple[float, np.ndarray]:
    """
    Simulate 'runs' sample paths under the optimal policy.
    Returns:
        avg_reward, per_run_rewards
    """
    rewards = np.zeros(runs, dtype=float)
    for r in range(runs):
        x = 5  # initial stock per assignment
        total = 0.0
        for t in range(1, T + 1):
            a = policy[t, x]
            # Arrival before sales
            y = 1 if (a == 1 and rng.random() < 0.5) else 0
            x_tilde = x + y
            # Demand
            d = 1 if (rng.random() < p_t[t - 1]) else 0
            # Sales and reward
            s = min(x_tilde, d)
            total += s - h * x
            # Next state
            x = max(x_tilde - d, 0)
            x = min(x, Xmax)  # cap at boundary
        rewards[r] = total
    return float(np.mean(rewards)), rewards

# ----- Quick run & basic visuals -----
if __name__ == "__main__":
    XMAX = 20
    V, pi = solve_dp(XMAX)

    # Report DP expected reward from initial condition (t=1, x=5)
    print(f"DP expected reward V_1(5): {V[1, 5]:.4f}")

    # Simulate under optimal policy
    avg_sim, samples = simulate(pi, XMAX, runs=1000)
    print(f"Simulation average over 1000 runs: {avg_sim:.4f}")

    # Policy plot: time vs inventory (action as discrete values)
    plt.figure()
    plt.imshow(pi[1:T+1, :].T, aspect="auto", origin="lower")
    plt.xlabel("time t")
    plt.ylabel("inventory x")
    cbar = plt.colorbar()
    cbar.set_label("action (0 = no order, 1 = order)")
    plt.title("Optimal policy π_t(x)")
    plt.tight_layout()

    # Histogram of simulated rewards
    plt.figure()
    plt.hist(samples, bins=30)
    plt.xlabel("total reward")
    plt.ylabel("count")
    plt.title("Distribution of total reward over runs")
    plt.tight_layout()
    plt.show()

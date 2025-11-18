import numpy as np

prob_det = 0.1 # deterioration probability


# ONLY CORRECTIVE
## Exercise B1: simulate 1000000 time units and give average reward
def next_state_and_reward(d1, d2, rng):
    """
    One step of the system under 'no preventive repair'.
    Input: current deterioration levels d1, d2 in {1,...,10}.
    Output: (d1_next, d2_next, reward).
    """
    # Decide action a = alpha(x)
    if d1 <= 9 and d2 <= 9:
        a = 0
    elif d1 == 10 and d2 <= 9:
        a = 1
    elif d1 <= 9 and d2 == 10:
        a = 2
    else:  # d1 == 10 and d2 == 10
        a = 3

    # Reward
    if a == 0:
        reward = 1.0
    else:
        n_failed = (d1 == 10) + (d2 == 10)
        reward = -25.0 * n_failed

    # State update
    if a == 0:
        # No repair, each working component deteriorates independently
        if d1 <= 9 and rng.random() < prob_det:
            d1_next = d1 + 1
        else:
            d1_next = d1

        if d2 <= 9 and rng.random() < prob_det:
            d2_next = d2 + 1
        else:
            d2_next = d2

    else:
        # Corrective repair: failed components -> 1, others stay
        d1_next = 1 if d1 == 10 else d1
        d2_next = 1 if d2 == 10 else d2

    return d1_next, d2_next, reward


def simulate_no_preventive(T=1_000_000, seed=42):
    rng = np.random.default_rng(seed)
    d1, d2 = 1, 1  # start in 'as good as new'
    total_reward = 0.0

    for _ in range(T):
        d1, d2, r = next_state_and_reward(d1, d2, rng)
        total_reward += r

    return total_reward / T  # \hat{\varphi}_T

avg_reward_estimate = simulate_no_preventive()
print("B1: Estimated long-run average reward (no preventive repair):", avg_reward_estimate)


## Exercise B2: determining the stationary distribution in a forward recursive manner

states_det = 10 # deterioration states 1,...,10

def state_to_index(d1: int, d2: int) -> int:
    """Map (d1,d2) with d_i in {1,...,10} to index in {0,...,99}."""
    return (d1 - 1) * states_det + (d2 - 1)


def index_to_state(idx: int) -> tuple[int, int]:
    """Inverse of state_to_index."""
    d1 = idx // states_det + 1
    d2 = idx % states_det + 1
    return d1, d2


def build_P_and_r_no_preventive():
    """
    Build transition matrix P and reward vector r for
    the 'no preventive repair' policy.
    P has shape (100,100), r has length 100.
    """
    n_states = states_det * states_det
    P = np.zeros((n_states, n_states))
    r = np.zeros(n_states)

    for d1 in range(1, states_det + 1):
        for d2 in range(1, states_det + 1):
            i = state_to_index(d1, d2)

            # Policy alpha(x): no preventive repair
            if d1 <= 9 and d2 <= 9:
                a = 0
            elif d1 == 10 and d2 <= 9:
                a = 1
            elif d1 <= 9 and d2 == 10:
                a = 2
            else:  # d1 == 10 and d2 == 10
                a = 3

            # Reward r(x) = r(x, alpha(x))
            if a == 0:
                r[i] = 1.0
            else:
                n_failed = (d1 == 10) + (d2 == 10)
                r[i] = -25.0 * n_failed

            # Transitions P[i, :]
            if a == 0:
                # No repair, both components working (d1,d2 <= 9)
                # Each component deteriorates independently with prob 0.1.
                for flag1 in (0, 1):   # 0 = stay, 1 = deteriorate
                    for flag2 in (0, 1):
                        p = ((1 - prob_det) if flag1 == 0 else prob_det) * \
                            ((1 - prob_det) if flag2 == 0 else prob_det)

                        d1_next = d1 + flag1
                        d2_next = d2 + flag2
                        # at most 10
                        d1_next = min(d1_next, 10)
                        d2_next = min(d2_next, 10)

                        j = state_to_index(d1_next, d2_next)
                        P[i, j] += p
            else:
                # Corrective repair: failed components -> 1, others stay
                d1_next = 1 if d1 == 10 else d1
                d2_next = 1 if d2 == 10 else d2
                j = state_to_index(d1_next, d2_next)
                P[i, j] = 1.0

    return P, r


def stationary_distribution_forward(P: np.ndarray,
                                    tol: float = 1e-12,
                                    max_iter: int = 1_000_000) -> np.ndarray:
    """
    Compute stationary distribution pi* by forward recursion:
        pi_{t+1}^T = pi_t^T P
    starting from pi_0 concentrated in state (1,1).
    """
    n_states = P.shape[0]
    pi = np.zeros(n_states)
    # start in state (1,1)
    start_idx = state_to_index(1, 1)
    pi[start_idx] = 1.0

    for _ in range(max_iter):
        pi_next = pi @ P
        diff = np.max(np.abs(pi_next - pi))
        pi = pi_next
        if diff < tol:
            break

    # normalise just in case of numerical noise
    pi /= pi.sum()
    return pi


## Exercise B3: Poisson equation using value iteration
def solve_poisson_no_preventive(P: np.ndarray,
                                r: np.ndarray,
                                tol: float = 1e-10,
                                max_iter: int = 1_000_000):
    """
    Solve the Poisson equation
        V + phi * e = r + P V
    for the 'no preventive repair' Markov reward chain.

    Returns:
        phi  : long-run average reward
        V    : bias function with normalization V[0] = 0
    """
    n_states = len(r)
    V = np.zeros(n_states)  # V_0 = 0

    for _ in range(max_iter):
        # Backward recursion: V_{t+1} = r + P V_t
        V_new = r + P @ V

        # Normalization as in slides: fix one state, e.g. V(0)=0
        V_new -= V_new[0]

        # Check convergence using span(V_{t+1} - V_t)
        diff = V_new - V
        span = diff.max() - diff.min()

        V = V_new

        if span < tol:
            # V_{t+1} - V_t → phi * e, so every component ≈ phi
            phi = diff.mean()
            return phi, V

    raise RuntimeError("Poisson value iteration did not converge within max_iter")


## All results B
if __name__ == "__main__":
    # B1 – simulation
    avg_reward_sim = simulate_no_preventive(T=1_000_000, seed=42)
    print("B1 – simulated average reward (no preventive):", avg_reward_sim)

    # B2 – stationary distribution
    P, r = build_P_and_r_no_preventive()
    pi_star = stationary_distribution_forward(P)
    avg_reward_stationary = pi_star @ r
    print("B2 – average reward via stationary distribution:", avg_reward_stationary)

    # B3 – Poisson equation (value iteration)
    phi_poisson, V_poisson = solve_poisson_no_preventive(P, r)
    print("B3 – average reward via Poisson equation:", phi_poisson)

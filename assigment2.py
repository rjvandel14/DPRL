import numpy as np

prob_det = 0.1 # deterioration probability
states_det = 10 # deterioration states 1,...,10


# Exercise B: only corrective repair
## Exercise B1: simulate 1000000 time units and give average reward
def next_state_and_reward(d1, d2, rng):
    """
    No preventive repairs.
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
        reward = -25.0 * n_failed  # corrective repair costs 25 per component

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



## Exercise B2: determining the stationary distribution in a forward recursive manner

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
            if d1 <= 9 and d2 <= 9: # no repairs
                a = 0
            elif d1 == 10 and d2 <= 9: # repair component 1
                a = 1
            elif d1 <= 9 and d2 == 10: # repair component 2
                a = 2
            else:  # d1 == 10 and d2 == 10 # repair both components
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
    Compute stationary distribution pi* by forward recursion
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
    Solve V + phi * e = r + P V for fixed policy.
    """
    n_states = len(r)
    V = np.zeros(n_states)  # V_0 = 0

    for _ in range(max_iter):
        # Backward recursion
        V_new = r + P @ V

        # Use unnormalised increment to estimate phi
        diff = V_new - V
        span = diff.max() - diff.min()

        V = V_new

        if span < tol:
            phi = diff.mean()
            # normalisation V(0)=0
            V -= V[0]
            return phi, V

    raise RuntimeError("Poisson value iteration did not converge within max_iter")




## Exercise C: preventive repair when the other has failed

def actions_C(d1: int, d2: int) -> list[int]:
    """
    Feasible actions A(x):
    - both working: only a=0 (no preventive if both OK)
    - one failed  : repair failed only (1 or 2) or repair both (3)
    - both failed : repair both (3)
    """
    if d1 <= 9 and d2 <= 9:
        return [0]
    if d1 == 10 and d2 <= 9:
        return [1, 3]
    if d1 <= 9 and d2 == 10:
        return [2, 3]
    # d1 == 10 and d2 == 10
    return [3]


def reward_xa(d1: int, d2: int, a: int) -> float:
    """
    Immediate reward r(x,a) with x=(d1,d2).

    +1 if system works (d1,d2 <= 9) and a=0.
    -5 for each preventive repair (d_i <= 9),
    -25 for each corrective repair (d_i = 10).
    """
    r = 0.0

    # functioning reward
    if d1 <= 9 and d2 <= 9 and a == 0:
        r += 1.0

    # which components are repaired under action a?
    S = []
    if a in (1, 3):
        S.append(1)
    if a in (2, 3):
        S.append(2)

    # preventive/corrective costs
    for comp in S:
        di = d1 if comp == 1 else d2
        if di <= 9:
            r -= 5.0  # preventive
        elif di == 10:
            r -= 25.0  # corrective
        else:
            raise ValueError("Invalid deterioration level > 10.")

    return r


def transitions_C(d1: int, d2: int, a: int):
    """
    Return list of (next_state_index, probability) for given (d1,d2,a).
    """
    next_probs: dict[int, float] = {}

    if a == 0:
        # No repair; both components must be working here (by actions_C)
        assert d1 <= 9 and d2 <= 9

        # Each component independently deteriorates with prob prob_det
        nexts1 = [(d1, 1 - prob_det), (d1 + 1, prob_det)]
        nexts2 = [(d2, 1 - prob_det), (d2 + 1, prob_det)]

        for j1, p1 in nexts1:
            for j2, p2 in nexts2:
                j1_clamped = min(j1, states_det)
                j2_clamped = min(j2, states_det)
                idx = state_to_index(j1_clamped, j2_clamped)
                next_probs[idx] = next_probs.get(idx, 0.0) + p1 * p2

    elif a == 1:
        # Repair component 1 (corrective); component 2 is frozen
        assert d1 == 10 and d2 <= 9
        j1, j2 = 1, d2
        next_probs[state_to_index(j1, j2)] = 1.0

    elif a == 2:
        # Repair component 2 (corrective); component 1 is frozen
        assert d2 == 10 and d1 <= 9
        j1, j2 = d1, 1
        next_probs[state_to_index(j1, j2)] = 1.0

    elif a == 3:
        # Repair both components (corrective and possibly preventive)
        j1, j2 = 1, 1
        next_probs[state_to_index(j1, j2)] = 1.0

    else:
        raise ValueError(f"Unknown action {a}")

    return list(next_probs.items())


def value_iteration_C(
    tol: float = 1e-8,
    max_iter: int = 1_000_000,
):
    """
    Average-reward value iteration

    Returns:
        phi    : estimated optimal long-run average reward
        V      : bias function (normalised so V(1,1)=0)
        policy : optimal action for each state index
    """
    n_states = states_det * states_det
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    ref_index = state_to_index(1, 1)  # reference state x0 = (1,1)

    for _ in range(max_iter):
        V_new = np.empty_like(V)
        policy_new = np.empty_like(policy)

        # Bellman update per state
        for i in range(n_states):
            d1, d2 = index_to_state(i)
            actions = actions_C(d1, d2)

            Q_values = []
            for a in actions:
                r = reward_xa(d1, d2, a)
                exp_next = 0.0
                for j, p in transitions_C(d1, d2, a):
                    exp_next += p * V[j]
                Q_values.append(r + exp_next)

            # choose best action (tie-breaker = smallest index)
            best_idx = int(np.argmax(Q_values))
            best_a = actions[best_idx]

            V_new[i] = Q_values[best_idx]
            policy_new[i] = best_a

        # increment and span for stopping criterion
        diff = V_new - V
        span = diff.max() - diff.min()

        # estimate average reward from increment
        phi = float(diff.mean())

        # normalise so that V_new(ref_index) = 0
        V = V_new - V_new[ref_index]
        policy = policy_new

        if span < tol:
            return phi, V, policy

    raise RuntimeError("Value iteration did not converge within max_iter")



## Exercise D: preventive repair of 1 or 2 components in any state

def actions_D(d1: int, d2: int) -> list[int]:
    """
    Feasible actions A(x):
    - both working: a in {0,1,2,3} (preventive on 1 or 2 components allowed)
    - one failed  : repair failed only (1 or 2) or repair both (3)
    - both failed : repair both (3)
    """
    if d1 <= 9 and d2 <= 9:
        return [0, 1, 2, 3]
    if d1 == 10 and d2 <= 9:
        return [1, 3]
    if d1 <= 9 and d2 == 10:
        return [2, 3]
    # d1 == 10 and d2 == 10
    return [3]


def transitions_D(d1: int, d2: int, a: int):
    """
    Return list of (next_state_index, probability) for given (d1,d2,a)
    """
    next_probs: dict[int, float] = {}

    if a == 0:
        # No repair. Each non-failed component deteriorates with prob prob_det.
        if d1 <= 9:
            nexts1 = [(d1, 1 - prob_det), (min(d1 + 1, states_det), prob_det)]
        else:
            nexts1 = [(d1, 1.0)]  # already failed, stays failed

        if d2 <= 9:
            nexts2 = [(d2, 1 - prob_det), (min(d2 + 1, states_det), prob_det)]
        else:
            nexts2 = [(d2, 1.0)]

        for j1, p1 in nexts1:
            for j2, p2 in nexts2:
                idx = state_to_index(j1, j2)
                next_probs[idx] = next_probs.get(idx, 0.0) + p1 * p2

    elif a == 1:
        # Repair component 1 (preventive or corrective); component 2 is frozen.
        j1, j2 = 1, d2
        next_probs[state_to_index(j1, j2)] = 1.0

    elif a == 2:
        # Repair component 2 (preventive or corrective); component 1 is frozen.
        j1, j2 = d1, 1
        next_probs[state_to_index(j1, j2)] = 1.0

    elif a == 3:
        # Repair both components (preventive/corrective).
        j1, j2 = 1, 1
        next_probs[state_to_index(j1, j2)] = 1.0

    else:
        raise ValueError(f"Unknown action {a}")

    return list(next_probs.items())


def value_iteration_D(
    tol: float = 1e-8,
    max_iter: int = 1_000_000,
):
    """
    Average-reward value iteration

    Returns:
        phi    : estimated optimal long-run average reward
        V      : bias function (normalised so V(1,1)=0)
        policy : optimal action for each state index
    """
    n_states = states_det * states_det
    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    ref_index = state_to_index(1, 1)  # reference state x0 = (1,1)

    for _ in range(max_iter):
        V_new = np.empty_like(V)
        policy_new = np.empty_like(policy)

        # Bellman update per state
        for i in range(n_states):
            d1, d2 = index_to_state(i)
            actions = actions_D(d1, d2)

            Q_values = []
            for a in actions:
                r_ = reward_xa(d1, d2, a)  # reuse reward from C
                exp_next = 0.0
                for j, p in transitions_D(d1, d2, a):
                    exp_next += p * V[j]
                Q_values.append(r_ + exp_next)

            # choose best action (tie-breaker = smallest index)
            best_idx = int(np.argmax(Q_values))
            best_a = actions[best_idx]

            V_new[i] = Q_values[best_idx]
            policy_new[i] = best_a

        # increment and span for stopping criterion
        diff = V_new - V
        span = diff.max() - diff.min()

        # estimate average reward from increment
        phi = float(diff.mean())

        # normalise so that V_new(ref_index) = 0
        V = V_new - V_new[ref_index]
        policy = policy_new

        if span < tol:
            return phi, V, policy

    raise RuntimeError("Value iteration (D) did not converge within max_iter")



## All results

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

    # C – optimal policy with preventive repair when the other has failed
    phi_C, V_C, policy_C = value_iteration_C()
    print("C – optimal average reward (with preventive repair):", phi_C)

    # D – optimal policy with preventive repair in any state
    phi_D, V_D, policy_D = value_iteration_D()
    print("D – optimal average reward (preventive in any state):", phi_D)


import numpy as np


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """
    Args:
        env: Gymnasium environment with env.P
        policy: [nS, nA] np policy matrix with policy[s][a] = p(a|s)
        gamma: discount factor
        theta: convergence threshold
    """

    nS = env.nS
    nA = env.nA

    # Start with 0 value for each state
    V = np.zeros(nS)

    while True:
        delta = 0

        # Over all states
        for s in range(nS):
            v = 0

            # Over all actions for s
            for a in range(nA):
                action_prob = policy[s][a]

                # Over all transitions for a in s
                for prob, next_state, reward, terminated, _ in env.P[s][a]:
                    v += action_prob * prob * (
                        reward + gamma * V[next_state]
                    )

            delta = max(delta, abs(V[s] - v))
            V[s] = v

        # Convergence
        if delta < theta:
            break

    return V
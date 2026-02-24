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
    V = np.zeros(nS)

    while True:
        delta = 0
        for s in range(nS):
            if s in env.pits or s == env.goal_pos:
                V[s] = 0
                continue
                
            v = 0
            for a in range(nA):
                action_prob = policy[s][a]
                for prob, next_state, reward, terminated, _ in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v
        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.99):
    """
    Args:
        env: Gymnasium environment with env.P
        V: State-value function [nS]
        gamma: discount factor

    Returns:
        new_policy: deterministic policy [nS, nA]
    """
    nS = env.nS
    nA = env.nA
    new_policy = np.zeros((nS, nA))

    for s in range(nS):
        Q_s = np.zeros(nA)
        for a in range(nA):
            for prob, next_state, reward, terminated, _ in env.P[s][a]:
                Q_s[a] += prob * (reward + gamma * V[next_state])
        best_a = np.argmax(Q_s)
        new_policy[s][best_a] = 1.0  # deterministic p = 1

    return new_policy


def policy_iteration(env, gamma=0.99, theta=1e-8, max_iterations=1000):
    """
    Args:
        env: Gymnasium environment with env.P
        gamma: discount factor
        theta: evaluation convergence threshold
        max_iterations: limit preventing possible infinite loops

    Returns:
        policy: optimal deterministic policy [nS, nA]
        V: value function for optimal policy [nS]
    """
    nS = env.nS
    nA = env.nA

    # Random policy with p(a|s) = 1/nA for all s, a
    policy = np.ones((nS, nA)) / nA

    for i in range(max_iterations):
        # Policy Evaluation
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)

        # Policy Improvement
        new_policy = policy_improvement(env, V, gamma=gamma)

        # Stable policy check
        if np.array_equal(policy, new_policy):
            print(f"Policy converged after {i+1} iterations")
            break

        policy = new_policy

    return policy, V
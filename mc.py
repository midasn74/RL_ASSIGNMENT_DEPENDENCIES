import numpy as np
from collections import defaultdict

def mc_control_epsilon_greedy(
    env,
    num_episodes=20000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
):
    """
    Monte Carlo without Exploring Starts

    Args:
        env: Gymnasium environment with env.P
        num_episodes: how many episodes/trials to run
        gamma: discount factor
        epsilon: exploration probability for ε-greedy

    Returns:
        Q: action-value function Q[s][a]
        policy: final NON-deterministic policy [nS, nA]
        V: value function based on greedy (deterministic) policy: V[s] = max_a Q[s][a]
    """

    nS = env.nS
    nA = env.nA

    Q = defaultdict(lambda: np.zeros(nA))
    returns_count = defaultdict(lambda: np.zeros(nA))

    policy = np.ones((nS, nA)) / nA

    epsilon = epsilon_start
    returns_history = []

    for ep in range(num_episodes):

        episode = []
        state, _ = env.reset()
        done = False
        total_return = 0

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(nA)
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            episode.append((state, action, reward))
            total_return += reward

            done = terminated or truncated
            state = next_state

        returns_history.append(total_return)

        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))

                returns_count[s][a] += 1
                Q[s][a] += (G - Q[s][a]) / returns_count[s][a]

        for s in range(nS):
            best_a = np.argmax(Q[s])
            policy[s] = epsilon / nA
            policy[s][best_a] += 1.0 - epsilon

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    greedy_policy = np.zeros_like(policy)
    for s in range(nS):
        best_a = np.argmax(Q[s])
        greedy_policy[s][best_a] = 1.0

    V = np.zeros(nS)
    for s in range(nS):
        V[s] = np.max(Q[s])

    return Q, greedy_policy, policy, V, returns_history
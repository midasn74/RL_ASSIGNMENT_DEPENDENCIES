import numpy as np

def evaluate_policy(env, policy, episodes=200):
    """
    Evaluate a policy by running multiple episodes and collecting total rewards.

    Args:
        env: Gymnasium environment
        policy: Deterministic policy [nS, nA] (will take argmax for action)
        episodes: Number of episodes to run

    Returns:
        rewards: List of total rewards for each episode
    """
    rewards = []
    for _ in range(episodes):
        state, _ = env.reset()
        total = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = np.argmax(policy[state])
            state, reward, done, truncated, info = env.step(action)
            total += reward
        rewards.append(total)
    return rewards
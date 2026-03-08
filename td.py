import numpy as np
from collections import defaultdict

def sarsa(
    env,
    num_episodes=20000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    step_size = 0.01
):

    nS = env.nS
    nA = env.nA

    # Initialize Q(s,a) arbritrarily, ensuring that Q(terminal, *) = 0
    Q = defaultdict(lambda: np.zeros(nA))

    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Initialize S
        state, _ = env.reset() 
        done = False
        
        # Choose action from state using policy derived from Q (e-greedy)
        if np.random.rand() < epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(Q[state])

        # Generate episode
        while not done:
            # Take action A observe #, S'
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Choose A' from S' using policy derived from Q
            if np.random.rand() < epsilon:
                next_action = np.random.randint(nA)
            else:
                next_action = np.argmax(Q[next_state])

            # Update Q
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

            done = terminated or truncated
    
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # extract deterministic policy
    policy = env.get_empty_policy()
    for s in range(nS):
        best_a = np.argmax(Q[s])
        policy[s][best_a] = 1.0

    # extract value of each state
    V = np.zeros(nS)
    for s in range(nS):
        V[s] = np.max(Q[s])

    return Q, policy, V
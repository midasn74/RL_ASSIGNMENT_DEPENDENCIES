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
    """
    On policy TD Control: Sarsa

    Args:
        env: Gymnasium environment with env.nS and env.nA
        num_episodes: how many episodes/trials to run
        gamma: discount factor
        epsilon: exploration probability for ε-greedy
        epsilon_decay: the rate at which epsilon decays
        step_size: how fast we update Q

    Returns:
        Q: action-value function Q[s][a]
        policy: final NON-deterministic policy [nS, nA]
        V: value function based on greedy (deterministic) policy: V[s] = max_a Q[s][a]
    """

    nS = env.nS
    nA = env.nA

    # Initialize Q(s,a) arbritrarily, ensuring that Q(terminal, *) = 0
    Q = defaultdict(lambda: np.zeros(nA))

    policy = np.ones((nS, nA)) / nA

    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Initialize S
        state, _ = env.reset() 
        done = False
        
        # Policy derived from Q (e-greedy)
        best_action = np.argmax(Q[state])
        for action in range(nA):
            if action == best_action:
                policy[state][action] = epsilon / nA + 1 - epsilon
            else:
                policy[state][action] = epsilon / nA

        # Choose action from state using policy
        action = np.random.choice(len(policy[state]), p=policy[state])

        # Generate episode
        while not done:
            # Take action A observe #, S'
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Policy derived from Q (e-greedy)
            best_next_action = np.argmax(Q[next_state])
            for action in range(nA):
                if action == best_next_action:
                    policy[next_state][action] = epsilon / nA + 1 - epsilon
                else:
                    policy[next_state][action] = epsilon / nA

            # Choose action from state using policy
            next_action = np.random.choice(len(policy[next_state]), p=policy[next_state])

            # Update Q
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

            done = terminated or truncated
    
        # Update epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # # extract deterministic policy
    # policy = env.get_empty_policy()
    # for s in range(nS):
    #     best_a = np.argmax(Q[s])
    #     policy[s][best_a] = 1.0

    # extract value of each state
    V = np.zeros(nS)
    for s in range(nS):
        V[s] = np.max(Q[s])

    return Q, policy, V

def q_learning(env,
    num_episodes=20000,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    step_size = 0.01
):
    """
    Off policy TD Control: Q-Learning

    Args:
        env: Gymnasium environment with env.nS and env.nA
        num_episodes: how many episodes/trials to run
        gamma: discount factor
        epsilon: exploration probability for ε-greedy
        epsilon_decay: the rate at which epsilon decays
        step_size: how fast we update Q

    Returns:
        Q: action-value function Q[s][a]
        policy: deterministic policy [nS, nA]
        V: value function based on greedy (deterministic) policy: V[s] = max_a Q[s][a]
    """
    
    nS = env.nS
    nA = env.nA

    # Initialize Q(s,a) arbritrarily, ensuring that Q(terminal, *) = 0
    Q = defaultdict(lambda: np.zeros(nA))

    epsilon = epsilon_start

    for ep in range(num_episodes):
        # Initialize S
        state, _ = env.reset() 
        done = False
        
        # Loop for each step of episode:
        while not done:
            # Choose action from state using policy derived from Q (e-greedy)
            if np.random.rand() < epsilon:
                action = np.random.randint(nA)
            else:
                action = np.argmax(Q[state])

            # Take action A observe R, S'
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Update Q
            best_a = np.argmax(Q[state])
            Q[state][action] += step_size * (reward + gamma * Q[next_state][best_a] - Q[state][action])
            state = next_state

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
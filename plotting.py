import matplotlib.pyplot as plt
import numpy as np

def plot_value_function(V, title="Value Function"):
    """
    Plot the value function as a bar chart.

    Args:
        V: Value function array [nS]
        title: Title for the plot
    """
    plt.figure(figsize=(10,4))
    plt.bar(np.arange(len(V)), V, color="skyblue", edgecolor="black")
    plt.xlabel("State")
    plt.ylabel("V(s)")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.show()

def plot_policy(policy, title="Policy"):
    """
    Plot the policy as a sequence of action arrows.

    Args:
        policy: Policy array [nS, nA]
        title: Title for the plot
    """
    actions = {0:"←", 1:"→", 2:"⇑"}
    greedy = np.argmax(policy, axis=1)

    plt.figure(figsize=(10,2))
    plt.title(title)
    plt.xticks(np.arange(len(greedy)), [actions[a] for a in greedy], fontsize=14)
    plt.yticks([])
    plt.show()

def plot_returns(returns, title="Returns per Episode"):
    """
    Plot the returns across episodes.

    Args:
        returns: List of returns for each episode
        title: Title for the plot
    """
    plt.figure(figsize=(10,4))
    plt.plot(returns, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()

def plot_learning_curve(rewards, title="Learning Curve"):
    """
    Plot the learning curve showing total reward per episode.

    Args:
        rewards: List of total rewards for each episode
        title: Title for the plot
    """
    plt.figure(figsize=(10,4))
    plt.plot(rewards, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.show()
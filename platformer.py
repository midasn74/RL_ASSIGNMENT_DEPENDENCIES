import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np
import matplotlib.pyplot as plt

class PlatformerEnv(gym.Env):
    """
    Custom 1D Platformer environment.
    The agents goal is to walk right while avoiding pits, and reach the goal at the right most position.
    """
    @staticmethod
    def get_actions():
        return {
            0: "Left",
            1: "Right",
            2: "Jump Right"
        }

    def __init__(self, 
                 length=15,
                 walk_cost=-1.0,
                 jump_cost=-2.5,
                 pit_penalty=-20.0,
                 goal_reward=25.0
    ):
        super(PlatformerEnv, self).__init__()

        self.length = length

        self.walk_cost = walk_cost
        self.jump_cost = jump_cost
        self.pit_penalty = pit_penalty
        self.goal_reward = goal_reward
        
        # Action space -> 0: Left, 1: Right, 2: Jump Right (moves 2 spaces)
        self.action_space = spaces.Discrete(3)
        
        # Observation space -> agents position integer from 0 to length-1
        self.observation_space = spaces.Discrete(self.length)
        
        # Pit location
        self.pits = [4, 8, 12]
        
        self.reset()
    
    def get_empty_policy(self):
        # Returns a starting/sample probabalistic policy (all zeros)
        return np.zeros((self.observation_space.n, self.action_space.n))
    
    @property
    def nS(self):
        return self.observation_space.n
    
    @property
    def nA(self):
        return self.action_space.n
        
    @property
    def P(self):
        P = {}
        for s in range(self.nS):
            P[s] = {}
            for a in range(self.nA):
                if s in self.pits or s == self.goal_pos:
                    P[s][a] = [(1.0, s, 0.0, True, {})]
                    continue

                if a == 0:    # Left
                    next_state = max(0, s - 1)
                    reward = self.walk_cost
                elif a == 1:  # Right
                    next_state = min(self.length - 1, s + 1)
                    reward = self.walk_cost
                elif a == 2:  # Jump Right
                    next_state = min(self.length - 1, s + 2)
                    reward = self.jump_cost

                terminated = False
                if next_state in self.pits:
                    reward = self.pit_penalty
                    terminated = True
                elif next_state == self.goal_pos:
                    reward = self.goal_reward
                    terminated = True

                P[s][a] = [(1.0, next_state, reward, terminated, {})]
        return P
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 0
        self.goal_pos = self.length - 1
        self.steps = 0
        return self.agent_pos, {}

    def step(self, action):
        if action == 0: # Left
            self.agent_pos = max(0, self.agent_pos - 1)
            reward = self.walk_cost
        elif action == 1: # Right
            self.agent_pos = min(self.length - 1, self.agent_pos + 1)
            reward = self.walk_cost
        elif action == 2: # Jump Right
            self.agent_pos = min(self.length - 1, self.agent_pos + 2)
            reward = self.jump_cost

        terminated = False
        truncated = False

        if self.agent_pos in self.pits:
            reward = self.pit_penalty
            terminated = True
        elif self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self, V=None, policy=None):
        """
        Args:
            V: State-value function [nS]
            policy: Policy [nS, nA], deterministic or not
        """
        nS = self.length
        corridor = np.array(["_"] * nS, dtype=object)

        corridor[self.pits] = "U"
        corridor[self.goal_pos] = "G"
        corridor[self.agent_pos] = "A"

        fig, ax = plt.subplots(figsize=(nS * 0.7, 4))

        x = np.arange(nS)

        if V is not None:
            values = np.array(V)
        else:
            values = np.zeros(nS)

        bars = ax.bar(
            x,
            values,
            color="lightgray",
            edgecolor="black",
            width=0.8,
            linewidth=2
        )

        for i in range(nS):
            if i in self.pits:
                bars[i].set_color("red")
            elif i == self.goal_pos:
                bars[i].set_color("green")
            elif i == self.agent_pos:
                bars[i].set_color("orange")

        if policy is not None:

            action_colors = ["blue", "purple", "gold"]
            action_labels = ["← Left", "→ Right", "⇑ Jump"]

            for i in range(nS):
                if i in self.pits or i == self.goal_pos:
                    continue

                probs = policy[i]

                probs = probs / np.sum(probs)

                base_height = values[i]

                bottom = base_height

                for a in range(self.nA):
                    height = probs[a] * 2.0

                    ax.bar(
                        i,
                        height,
                        bottom=bottom,
                        width=0.5,
                        color=action_colors[a],
                        alpha=0.8
                    )

                    bottom += height

        ax.set_xticks(x)
        ax.set_xticklabels(corridor, fontsize=12)
        ax.set_ylabel("State Value")
        ax.set_title("Platformer: Value Function + Action Probabilities")

        ax.axhline(0, linewidth=2)

        ax.set_ylim(min(values) - 5, max(values) + 6)

        if policy is not None:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="blue", label="← Left"),
                Patch(facecolor="purple", label="→ Right"),
                Patch(facecolor="gold", label="⇑ Jump"),
            ]
            ax.legend(handles=legend_elements, loc="upper left")

        plt.tight_layout()
        plt.show()


register(
    id='Platformer-v0',
    entry_point='platformer:PlatformerEnv', 
)
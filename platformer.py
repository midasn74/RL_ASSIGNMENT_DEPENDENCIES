import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

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
                 pit_penalty=-10.0,
                 goal_reward=10.0
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

                if a == 0:  # Left
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

        self.steps += 1
        terminated = False
        truncated = False

        if self.agent_pos in self.pits:
            reward = self.pit_penalty
            terminated = True

        elif self.agent_pos == self.goal_pos:
            reward = self.goal_reward
            terminated = True

        if self.steps >= 5 * self.length:
            truncated = True

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        corridor = ["_"] * self.length
        
        for p in self.pits:
            corridor[p] = "U" 
            
        corridor[self.goal_pos] = "G"
        corridor[self.agent_pos] = "A" 
        
        print("\n" + " ".join(corridor))


register(
    id='Platformer-v0',
    entry_point='platformer:PlatformerEnv', 
)
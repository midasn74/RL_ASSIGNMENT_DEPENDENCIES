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

    def __init__(self, length=15):
        super(PlatformerEnv, self).__init__()
        
        self.length = length
        
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
    
    def nS(self):
        return self.observation_space.n
    
    def nA(self):
        return self.action_space.n

    def P(self):
        P = {}
        for s in range(self.nS()):
            P[s] = {}
            for a in range(self.nA()):
                if a == 0:
                    next_state = max(0, s - 1)
                    reward = -1
                elif a == 1:
                    next_state = min(self.length - 1, s + 1)
                    reward = -1
                elif a == 2:
                    next_state = min(self.length - 1, s + 2)
                    reward = -2.5
                terminated = next_state in self.pits or next_state == self.goal_pos
                P[s][a] = [(1.0, next_state, reward, terminated, {})]
            return P

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 0
        self.goal_pos = self.length - 1
        self.steps = 0
        return self.agent_pos, {}

    def step(self, action):
        reward = -1 # Cost of walking
        
        if action == 0:   # Move Left
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1: # Move Right
            self.agent_pos = min(self.length - 1, self.agent_pos + 1)
        elif action == 2: # Jump Right
            self.agent_pos = min(self.length - 1, self.agent_pos + 2)
            reward = -2.5   # Cost of jumping
            
        self.steps += 1
        terminated = False
        truncated = False
        
        if self.agent_pos in self.pits:
            reward = -10 # Fell in pit
            terminated = True 
        elif self.agent_pos == self.goal_pos:
            reward = 10  # Reached the the goal
            terminated = True
            
        # Truncate game if agents is too bad
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
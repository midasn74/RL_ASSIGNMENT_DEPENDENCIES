import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class CorridorEnv(gym.Env):
    """
    Custom 1D Corridor environment,
    inheriting from gymnasium.Env.
    """
    def __init__(self, length=10):
        super(CorridorEnv, self).__init__()
        
        self.length = length
        
        # Action space -> 0: Left, 1: Right
        self.action_space = spaces.Discrete(2)
        
        # Observation space -> the agent's position on the line [0-length)
        self.observation_space = spaces.Discrete(self.length)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Agent starts on the most left position
        self.agent_pos = 0
        
        # Goal is the most right position
        self.goal_pos = self.length - 1
        
        self.steps = 0
        
        return self.agent_pos, {}

    def step(self, action):
        if action == 0:
            self.agent_pos = max(0, self.agent_pos - 1)
        elif action == 1:
            self.agent_pos = min(self.length - 1, self.agent_pos + 1)
            
        self.steps += 1
        
        reward = -1
        terminated = False
        truncated = False
        
        if self.agent_pos == self.goal_pos:
            reward = 10
            terminated = True
            
        if self.steps >= 3 * self.length:
            truncated = True

        return self.agent_pos, reward, terminated, truncated, {}

    def render(self):
        corridor = ["."] * self.length
        
        corridor[self.goal_pos] = "G"
        corridor[self.agent_pos] = "A"
        
        print("\n" + " ".join(corridor))

register(
    id='Corridor-v0',
    entry_point='corridor:CorridorEnv', # Change 'corridor' to the actual name of your python file
)
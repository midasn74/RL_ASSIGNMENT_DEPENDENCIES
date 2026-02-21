import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register

class BattleshipEnv(gym.Env):
    """
    Custom Battleship (zeeslag) environment,
    inheriting from gymnasium.Env.
    """
    def __init__(self, board_size=5, ship_lengths=[3, 2]):
        super(BattleshipEnv, self).__init__()
        
        self.board_size = board_size
        self.ship_lengths = ship_lengths
        
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # Observation space -> possible observations: 0: Unknown, 1: Miss, 2: Hit
        self.observation_space = spaces.Box(
            low=0, 
            high=2, 
            shape=(board_size, board_size), 
            dtype=np.int8
        )
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Set everything to zero (0: Unknown)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        # Hidden layer, agent will now know the ship locations beforehand
        # 0: Water, 1: Ship
        self.ships = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self._place_ships()
        
        self.steps = 0
        self.hits = 0
        self.total_ship_cells = sum(self.ship_lengths)
        
        return self.board, {}

    def step(self, action):
        # Convert action to coordinates:
        # Left to right, then next row
        row = action // self.board_size
        col = action % self.board_size
        
        reward = 0
        terminated = False
        truncated = False
        
        if self.board[row, col] != 0: # Already shot here:
            reward = -5  # Penalty for wasting ammunition

        elif self.ships[row, col] == 1: # Hit!
            self.board[row, col] = 2 # Mark as Hit
            reward = 1 # Reward for hitting
            self.hits += 1

        else:
            self.board[row, col] = 1 # Miss 💩
            reward = -0.25 # Penalty for missing: Maybe play around with this value for more efficient learning
            
        self.steps += 1
        
        # Check win condition
        if self.hits == self.total_ship_cells:
            reward += 10 # Winning reward
            terminated = True
            
        # Truncate game if agent takes too many steps: 
        # if 2x the number of cells, assume agent is not learning efficiently
        if self.steps >= 2 * self.board_size**2:
            truncated = True

        return self.board.copy(), reward, terminated, truncated, {}

    def render(self):
        print("\n  " + " ".join([str(i) for i in range(self.board_size)]))
        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                if self.board[r, c] == 0: symbol = "."
                elif self.board[r, c] == 1: symbol = "O" # Miss
                elif self.board[r, c] == 2: symbol = "X" # Hit
                row_str += symbol + " "
            print(row_str)

    def _place_ships(self):
        """
        Random placement for the ships
        """
        for length in self.ship_lengths:
            placed = False
            while not placed:
                orientation = self.np_random.integers(0, 2) # boat is either: 0: Horizontal, 1: Vertical
                if orientation == 0:
                    r = self.np_random.integers(0, self.board_size)
                    c = self.np_random.integers(0, self.board_size - length + 1)
                    if np.all(self.ships[r, c:c+length] == 0):
                        self.ships[r, c:c+length] = 1
                        placed = True
                else:
                    r = self.np_random.integers(0, self.board_size - length + 1)
                    c = self.np_random.integers(0, self.board_size)
                    if np.all(self.ships[r:r+length, c] == 0):
                        self.ships[r:r+length, c] = 1
                        placed = True


register(
    id='Battleship-v0',
    entry_point='battleships:BattleshipEnv',
)
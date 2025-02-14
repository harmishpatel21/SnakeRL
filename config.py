from enum import Enum
import torch

class Config:
    # Game Settings
    BLOCK_SIZE = 20
    GRID_WIDTH = 32    # 640px / 20
    GRID_HEIGHT = 24   # 480px / 20
    SPEED = 40
    MAX_FRAME_ITERATIONS = 100

    # Colors
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLUE1 = (0, 0, 255)
    BLUE2 = (0, 100, 255)
    BLACK = (0, 0, 0)
    
    # AI Parameters
    STATE_SIZE = 11     # From game.get_state()
    ACTION_SIZE = 3     # [straight, right, left]
    
    # DQN Hyperparameters
    GAMMA = 0.95        # Discount factor
    EPSILON_START = 1.0 # Initial exploration rate
    EPSILON_MIN = 0.01  # Minimum exploration rate 
    EPSILON_DECAY = 0.995
    BATCH_SIZE = 32
    MEMORY_SIZE = 100000
    LEARNING_RATE = 0.001
    
    # Training Parameters
    EPISODES = 2000
    TARGET_UPDATE_FREQ = 100  # Update target network every X episodes
    
    # Neural Network Architecture
    HIDDEN_SIZE = 128
    
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reward System
    FOOD_REWARD = 10
    COLLISION_PENALTY = -10
    MOVE_PENALTY = -0.1

class Direction(Enum):
    RIGHT = 0
    LEFT = 1
    UP = 2
    DOWN = 3

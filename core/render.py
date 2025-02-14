
from game import SnakeGame
import pygame

game = SnakeGame()

actions = [
        [1, 0, 0], # Straight
        [0, 1, 0], # Right turn
        [0, 0, 1]  # Left turn
        ]

while True:
    for action in actions:    
        reward, done, score = game.play_step(action)
        if done:
            game.reset()
        pygame.time.wait(200)
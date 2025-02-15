from ai.agents import DQNAgent
from core.game import SnakeGame
from config import Config
import pygame

def run_trained_model():
    agent = DQNAgent(state_size=Config.STATE_SIZE, action_size=Config.ACTION_SIZE)
    game = SnakeGame()

    agent.load('best_model.pth')
    agent.epsilon = 0

    while True:
        # Get current state
        state = game.get_state()

        # Get action from model
        action_idx = agent.act(state)
        action = [0] * Config.ACTION_SIZE
        action[action_idx] = 1

        # perform the action in the environment
        reward, done, score = game.play_step(action)

        # check for exit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        if done:
            game.reset()

if __name__ == '__main__':
    run_trained_model()
    


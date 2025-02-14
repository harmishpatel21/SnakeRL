from ai.agents import DQNAgent
from core.game import SnakeGame

import matplotlib.pyplot as plt
import numpy as np 
from config import Config

def train():
    game = SnakeGame()
    agent = DQNAgent(state_size=Config.STATE_SIZE, action_size=Config.ACTION_SIZE)
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    
    for episode in range(Config.EPISODES):
        state = game.reset()
        state = game.get_state()
        score = 0
        
        while True:
            action_idx = agent.act(state)
            action = [0] * Config.ACTION_SIZE
            action[action_idx] = 1
            
            reward, done, current_score = game.play_step(action)
            next_state = game.get_state()
            
            agent.remember(state, action_idx, reward, next_state, done)
            agent.replay()
            
            state = next_state
            score += reward
            
            if done:
                break
                
        scores.append(score)
        total_score += score
        mean_score = total_score / (episode + 1)
        mean_scores.append(mean_score)
        
        if current_score > record:
            record = current_score
            agent.save('best_model.pth')
            
        print(f'Episode {episode+1}, Score: {current_score}, Mean: {mean_score:.2f}, Epsilon: {agent.epsilon:.2f}')
    
    # Plot training results
    plt.plot(scores, label='Score')
    plt.plot(mean_scores, label='Mean Score')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training.png')
    plt.show()

if __name__ == '__main__':
    train()
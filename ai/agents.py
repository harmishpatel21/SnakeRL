import numpy as np
import random 
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x 
    

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=Config.MEMORY_SIZE)
        self.gamma = Config.GAMMA    # Discount factor
        self.epsilon = Config.EPSILON_START   # Exploration rate
        self.epsilon_min = Config.EPSILON_MIN
        self.epsilon_decay = Config.EPSILON_DECAY
        self.batch_size = Config.BATCH_SIZE
        self.model = DQN(state_size, Config.HIDDEN_SIZE, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.device = Config.DEVICE
        self.model.to(Config.DEVICE)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            # Convert to tensors
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            
            # Get current Q values
            current_q = self.model(state)
            
            # Calculate target
            if done:
                target = reward
            else:
                next_q = self.model(next_state).detach()
                target = reward + self.gamma * torch.max(next_q)
            
            # Update only the action we took
            target_q = current_q.clone()
            target_q[action] = target
            
            states.append(state)
            targets.append(target_q)
        
        states = torch.stack(states)
        targets = torch.stack(targets)
        
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets)
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval() # Set the network to evaluation mode

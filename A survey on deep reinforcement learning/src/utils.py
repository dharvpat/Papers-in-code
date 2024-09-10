import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return torch.stack(states), torch.tensor(actions), torch.tensor(rewards), torch.stack(next_states), torch.tensor(dones)

    def __len__(self):
        return len(self.buffer)
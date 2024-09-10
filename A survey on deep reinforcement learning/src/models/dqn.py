import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_dqn(state_size, action_size):
    return DQN(state_size, action_size)

def train_dqn(model, optimizer, loss_fn, batch):
    states, actions, rewards, next_states, dones = batch
    q_values = model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        max_next_q_values = model(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * 0.99 * max_next_q_values
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
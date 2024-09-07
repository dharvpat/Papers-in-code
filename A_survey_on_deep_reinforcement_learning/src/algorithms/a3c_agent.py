import torch
import torch.optim as optim
from models.a3c import create_a3c, train_a3c

class A3CAgent:
    def __init__(self, state_size, action_size):
        self.model = create_a3c(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self, state):
        with torch.no_grad():
            policy_dist, _ = self.model(state)
            action = torch.multinomial(policy_dist, 1).item()
        return action

    def train(self, batch):
        train_a3c(self.model, self.optimizer, self.loss_fn, batch)
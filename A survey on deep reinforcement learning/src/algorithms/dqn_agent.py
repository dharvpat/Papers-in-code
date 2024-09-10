import torch
import torch.optim as optim
from models.dqn import create_dqn, train_dqn
from utils import ReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = create_dqn(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = torch.nn.MSELoss()
        self.replay_buffer = ReplayBuffer(10000)
        self.state_size = state_size
        self.action_size = action_size

    def select_action(self, state, epsilon=0.1):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            with torch.no_grad():
                return self.model(state).argmax().item()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)
        train_dqn(self.model, self.optimizer, self.loss_fn, batch)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
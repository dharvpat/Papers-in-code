import torch
import torch.nn as nn
import torch.optim as optim

class A3C(nn.Module):
    def __init__(self, state_size, action_size):
        super(A3C, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, action_size)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy_dist = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return policy_dist, value

def create_a3c(state_size, action_size):
    return A3C(state_size, action_size)

def train_a3c(model, optimizer, loss_fn, batch):
    states, actions, rewards, next_states, dones = batch
    policy_dist, value = model(states)
    log_probs = torch.log(policy_dist.gather(1, actions.unsqueeze(-1)).squeeze(-1))
    with torch.no_grad():
        _, next_value = model(next_states)
        target_value = rewards + (1 - dones) * 0.99 * next_value.squeeze(-1)
    advantage = target_value - value.squeeze(-1)
    actor_loss = -(log_probs * advantage.detach()).mean()
    critic_loss = loss_fn(value.squeeze(-1), target_value)
    loss = actor_loss + critic_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
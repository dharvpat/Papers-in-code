import gym
import torch
from algorithms.a3c_agent import A3CAgent

def train_a3c_agent(env_name, num_episodes):
    env = gym.make(env_name)
    agent = A3CAgent(env.observation_space.shape[0], env.action_space.n)
    
    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        total_reward = 0
        
        for _ in range(1000):  # max steps per episode
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            agent.train((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break
        
        print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

if __name__ == "__main__":
    train_a3c_agent(env_name='CartPole-v1', num_episodes=1000)
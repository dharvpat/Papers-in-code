import gym
import torch

def evaluate_model(agent, env_name, num_episodes):
    env = gym.make(env_name)
    total_rewards = []

    for episode in range(num_episodes):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        total_reward = 0
        
        for _ in range(1000):
            action = agent.select_action(state)
            state, reward, done, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            total_reward += reward
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f'Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}')

    print(f'Average Reward: {sum(total_rewards)/num_episodes}')

if __name__ == "__main__":
    from algorithms.dqn_agent import DQNAgent
    agent = DQNAgent(state_size=4, action_size=2)  # Use appropriate sizes
    evaluate_model(agent, env_name='CartPole-v1', num_episodes=100)
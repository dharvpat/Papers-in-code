import sys
import os
import torch

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from src.training.train_a3c import train_a3c_agent

def main():
    env_name = 'CartPole-v1'
    num_episodes = 500

    # Train an A3C agent on the CartPole-v1 environment
    print(f"Training A3C agent on {env_name} for {num_episodes} episodes...")
    train_a3c_agent(env_name, num_episodes)

if __name__ == "__main__":
    main()
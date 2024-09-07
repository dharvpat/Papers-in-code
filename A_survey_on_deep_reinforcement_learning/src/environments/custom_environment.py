import gym
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        self.state = None

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def step(self, action):
        reward = 1 if action == np.argmax(self.state) else 0
        done = np.random.rand() > 0.95
        self.state = self.observation_space.sample()
        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

def create_custom_env():
    return CustomEnv()
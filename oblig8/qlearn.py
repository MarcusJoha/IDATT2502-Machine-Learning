import gym
import numpy as np
import time, math, random
from typing import Tuple

from sklearn.preprocessing import KBinsDiscretizer


def rand():
    env = gym.make('CartPole-v1')

    for _ in range(10):
        env.reset()
        for _ in range(500):
            env.render()
            actions = env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            time.sleep(0.01)
            if done:
                break
        env.close()
        
        
rand()



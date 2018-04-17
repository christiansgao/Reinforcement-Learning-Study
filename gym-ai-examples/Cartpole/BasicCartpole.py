from __future__ import division

import gym
import numpy as np
import random
import math
from time import sleep

## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v0')

def simulate():

    num_streaks = 0

    for episode in range(100):

        # Reset the environment
        obv = env.reset()

        for t in range(100):
            env.render()

            action =  env.action_space.sample() # take a random action

            # Execute the action
            obv, reward, done, _ = env.step(action)

def trainQTable(obv, reward):
    print("Training");



if __name__ == "__main__":
    simulate()
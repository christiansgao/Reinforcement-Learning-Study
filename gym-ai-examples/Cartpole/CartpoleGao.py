from __future__ import division

import gym
import numpy as np
import random
import math
from time import sleep
## Initialize the "Cart-Pole" environment
env = gym.make('CartPole-v1')

EPISODE_LENGTH = 500;
FAIL_LENGTH = 50

#Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]

FIRST_BUCKETS = 3;
SECOND_BUCKETS = 3;
THIRD_BUCKETS = 3;
FOURTH_BUCKETS = 3;
NUM_ACTIONS = 2;

#3x3x3x3x2 table for each obs and action pair
q_table = np.zeros(FIRST_BUCKETS, SECOND_BUCKETS, THIRD_BUCKETS, FOURTH_BUCKETS, NUM_ACTIONS)


learning_rate = .1;

def simulate():

    num_streaks = 0

    for episode in range(100):

        # Reset the environment
        obv = env.reset()

        for t in range(EPISODE_LENGTH):

            env.render()

            action =  env.action_space.sample() # take a random action

            # Execute the action
            obv, reward, done, info = env.step(action)
            print(obv)
            if done & (t > FAIL_LENGTH):
                break

### MAIN ###
simulate()
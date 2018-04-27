from __future__ import division
import gym
import numpy as np
import random
import math
test = (1)/25;
test = math.log10((0+1)/25)
#print(test)

env = gym.make('CartPole-v1')

NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n # (left, right)

print (NUM_BUCKETS + (NUM_ACTIONS,))
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
print q_table

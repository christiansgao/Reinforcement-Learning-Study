import gym
import time

import numpy as np
from sklearn import datasets, linear_model

from sklearn.linear_model import LogisticRegression

####################################################

def update_model(hist):
    X = hist[:, [1, 2, 3, 4, 5]]
    y = hist[:, 0]
    logistic.fit(X, y)


def get_action(new_observation):
        # Make predictions using the testing set
        test = np.append(new_observation,1)
        test = test.reshape(1, -1)
        leftProb = logistic.predict_proba(test)[0,0]
        test = np.append(new_observation, 0)
        test = test.reshape(1, -1)
        rightProb = logistic.predict_proba(test)[0,0]
        if(leftProb > rightProb):
            return 1
        else:
            return 0

####################################################

env = gym.make('CartPole-v1')
observation, reward, done, info  = env.reset()

#historical data
hist = np.array([]).reshape(0,6)

# Create logistic regression object
logistic = LogisticRegression()

#Default Values
old_observation = np.array([]).reshape(0,4)
new_observation = None

for round in range(20000):

    if(round > 10000):
        update_model(hist)

    env.reset()

    for step in range(10000):
        env.render()
        if (round <= 10000):
            action = env.action_space.sample()
        else:
            action = get_action(new_observation)

        new_observation, reward, done, info  = env.step(action) # take an action

        if step !=0:
            temp =np.append(done, old_observation)
            temp =np.append(temp, action)
            hist = np.vstack([hist, temp])

        old_observation = new_observation

        if done:
            print("Episode finished after timesteps: " + str(step) + " round: " + str(round))
            #print(hist)
            break

# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
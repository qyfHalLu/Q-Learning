"""Toy environment launcher. See the docs for more details about this environment.

"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
from Toy_env import MyEnv as Toy_env
import deer.experiment.base_controllers as bc

from simpleNN import *

def Q_learning(env, n_epochs = 10, epoch_length = 100, gamma = 0.9, alpha = 0.1, epsilon = 0.1):
    '''
    alpha: learning rate
    gamma: discount factor
    epsilon-greedy
    '''
    n_actions = env.n_actions
    Q_table = np.zeros((env.n_states, n_actions))
    for i in range(n_epochs):
        for e in range(epoch_length):
            s = env._last_ponctual_observation
            a = np.random.randint(n_actions) if np.random.random() < epsilon else Q_table[s,].argmax()
            r = env.act(a)
            new_s = env._last_ponctual_observation
            Q_table[s, a] += alpha * (r + gamma * Q_table[new_s,].max() - Q_table[s, a])

    new_policy = Q_table.argmax(axis=1)
    return Q_table, new_policy


# --- Instantiate environment ---
env = Toy_env(n_states=10)

# --- solve with tabular Q-learning and epsilon greedy ---
Q_table, new_policy = Q_learning(env, n_epochs = 10, epoch_length = 1000, gamma = 0.9, alpha = .9, epsilon = 0.05)
print(Q_table, new_policy)

Q_table, new_policy = Q_learning(env, n_epochs = 10, epoch_length = 100, gamma = 0.9, alpha = .1, epsilon = 0.05)
print(Q_table, new_policy)

# --- Instantiate qnetwork ---

qnetwork = MyQNetwork(
    environment=env,
    neural_network=simpleNN)

# --- Instantiate agent ---
agent = NeuralAgent(
    env,
    qnetwork)

agent.setLearningRate(0.1)

# --- Bind controllers to the agent ---
# Before every training epoch, we want to print a summary of the agent's epsilon, discount and 
# learning rate as well as the training epoch number.
agent.attach(bc.VerboseController())

# During training epochs, we want to train the agent after every action it takes.
# Plus, we also want to display after each training episode (!= than after every training) the average bellman
# residual and the average of the V values obtained during the last episode.
agent.attach(bc.TrainerController())

# --- Run the experiment ---
agent.run(n_epochs=10, epoch_length=100)
for i in range(env.n_states):
    print(f'state {i}, Q value {agent._learning_algo.qValues([i])}, best action {agent._learning_algo.chooseBestAction([i])[0]}')
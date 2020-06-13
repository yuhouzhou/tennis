import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils.replayBuffer import ReplayBuffer
from utils.noise import OUNoise
from hyperparameter import *

def maddpg_wrapper(ddpgAgent):

    class MaddpgAgents():

        def __init__(self, state_size, action_size, num_agents, random_seed):
            self.num_agents = num_agents
            self.agents = [ddpgAgent(state_size, action_size, 1, random_seed) for _ in range(num_agents)]
            self.memory = []
            for i in range(num_agents):
                self.memory.append(ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed))
            for agent in self.agents:
                agent.noise = OUNoise(action_size, random_seed)
            self.t_step = 0
        
        def reset(self):
            """Resets OU Noise for each agent."""
            for agent in self.agents:
                agent.reset()
        
        def act(self, states, eps, add_noise=True):
            """Picks an action for each agent given."""
            actions = []
            for agent, state in zip(self.agents, states):
                action = agent.act(state, eps=eps, add_noise=add_noise)
                actions.append(action)
            return np.array(actions)
        
        def step(self, states, actions, rewards, next_states, dones):
            """Save experience in replay memory."""
            for i in range(self.num_agents):
                self.memory[i].add(states[i], actions[i], rewards[i], next_states[i], dones[i])

            # Learn, if enough samples are available in memory
            if len(self.memory[0]) > BATCH_SIZE and (self.t_step + 1) % UPDATE_EVERY == 0:
                for i, agent in enumerate(self.agents):
                    experiences = self.memory[i].sample()
                    agent.learn(experiences, gamma=GAMMA)

    return MaddpgAgents
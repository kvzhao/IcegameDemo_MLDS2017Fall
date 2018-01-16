from __future__ import division
from __future__ import print_function

import gym
from gym import error, spaces, utils, core
import sys, os
import json
import random
import numpy as np
from isinggame import IsingGame, INFO
import time

rnum = np.random.randint

class IsingGameEnv(core.Env):
    def __init__(self, L, kT, J):
        self.L = L
        self.kT = kT
        self.J = J
        self.N = L**2
        num_neighbors = 1
        num_replicas = 1
        num_mcsteps = 2000
        num_bins = 1
        num_thermalization = num_mcsteps
        tempering_period = 1
        self.mc_info = INFO(self.L, self.N, num_neighbors, num_replicas, \
                num_bins, num_mcsteps, tempering_period, num_thermalization)
        ### action space and state space
        self.action_space = spaces.Discrete(self.N)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L, self.L))
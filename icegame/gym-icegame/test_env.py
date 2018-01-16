import gym 
import gym_icegame

import itertools
import numpy as np
import os
import random
import sys

if "../" not in sys.path:
    sys.path.append("../build/src")

env = gym.make('IceGameEnv-v0')
env.step([0])
#env = gym.make('IcegameSSF_Env-v1')
print ("Load Icegame Done.")
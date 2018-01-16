import gym 
import gym_icegame

import itertools
import numpy as np
import os
import random
import sys

from collections import deque

if "../" not in sys.path:
    sys.path.append("../build/src")

env = gym.make('IceGameEnv-v0')
print ("Load IcegameEnv Done.")


state = env.reset()
print (state.shape)

for _ in range(10):
    s, _, _, info  = env.step(0)

state = env.reset()
print (state.shape)

"""
I = np.ones_like(state)
hist = deque([I, I, I, I])
print (len(hist))
hist.append(state)
hist.popleft()
print (hist)
for h in hist:
    print (h.shape)
print (len(hist))
"""

import os
import sys
import gym
import gym_icegame
import numpy as np

#TODO: dangerous relative path
sys.path.append("../icegame/build/src/")

def create_icegame_env(path, ID):
    if ID == "IceGameEnv-v3":
        print ("Create Env {}".format(ID))
    else:
        print ("Env {} is not suitable for this project.".format(ID))
    env = gym.make(ID)
    env.set_output_path(path)
    return env
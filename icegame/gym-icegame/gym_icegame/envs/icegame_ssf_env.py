from __future__ import division
import gym
from gym import error, spaces, utils, core

import os
import sys
import time
import json
import random
import numpy as np

from icegame_ssf import SQIceGameSSF, INFO
rnum = np.random.randint

class IcegameSSF_Env(core.Env):
    def __init__ (self, L, kT, J):
        # Well, in this game, L is fixed.
        self.L = L
        self.kT = kT
        self.J = J
        self.N = L**2
        num_neighbors = 2
        num_replicas = 1
        num_mcsteps = 2000
        num_bins = 1
        num_thermalization = num_mcsteps
        tempering_period = 1

        self.mc_info = INFO(self.L, self.N, num_neighbors, num_replicas, \
                num_bins, num_mcsteps, tempering_period, num_thermalization)

        self.sim = SQIceGameSSF(self.mc_info)
        self.sim.set_temperature (self.kT)
        self.sim.init_model()
        self.sim.mc_run(num_mcsteps)

        self.episode_terminate = False
        self.accepted_episode = False

        self.action_space = spaces.Discrete(self.N + 1)
        # action space and state space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L, self.L, 4))
        self.reward_range = (-1, 1)

        # output file
        self.ofilename = "loop_sites.log"
        # render file
        self.rfilename = "loop_renders.log"
        # save log to json for future analysis
        self.json_file = "env_history.json"
        self.stacked_axis = 2
        ## counts reset(), but this value is also records in icegame
        self.episode_counter = 0

    def step(self, action):
        """Step function.

            Only terminate when timeout which is ep_steps > num_spins.
        """
        reward = 0.0
        terminate = False
        obs = None
        info = None
        metropolis_executed = False

        if action == 0:
            self.sim.flip_trajectory()
            rets = self.sim.metropolis()
            metropolis_executed = True
        elif 0 < action <= self.N:
            rets = self.sim.ssf(action)
        else:
            pass
            # Exit with serious bug

        obs = self.get_obs()
        is_aceept, dEnergy, dDensity, dConfig = rets

        if self.sim.timeout():
            terminate = True

        if metropolis_executed:
            if is_aceept > 0 and dConfig > 0:
                """ Accept """
                self.sim.update_config()
                print ("[GAME_ENV] PROPOSAL ACCEPTED!")
                total_steps = self.sim.get_total_steps()
                ep_steps = self.sim.get_ep_step_counter()
                ep = self.sim.get_episode()
                loop_length = self.sim.get_accepted_length()[-1]
                loop_area = self.calculate_area()
                update_times = self.sim.get_updated_counter()
                reward = 1.0 * (loop_length / 4.0) # reward with different length by normalizing with len 4 elements

                # output to self.ofilename
                with open(self.ofilename, "a") as f:
                    f.write("1D: {}, \n(2D: {})\n".format(self.sim.get_trajectory(), self.convert_1Dto2D(self.sim.get_trajectory())))
                    print ("\tSave loop configuration to file: {}".format(self.ofilename))

                print ("\tTotal accepted number = {}".format(update_times))
                print ("\tAccepted loop length = {}, area = {}".format(loop_length, loop_area))
                print ("\tAgent walks {} steps in episode, action counters: {}".format(ep_steps, self.sim.get_ep_action_counters()))
                action_counters = self.sim.get_action_statistics()
                action_stats = [x / total_steps for x in action_counters]
                print ("\tStatistics of actions all episodes (ep={}, steps={}) : {}".format(ep, total_steps, action_stats))
                print ("\tAcceptance ratio (accepted/total Eps) = {}%".format(update_times * 100.0 / ep))

                # Write out info
                info = {
                    "AccRate" : update_times * 100.0 / ep,
                    "Updated" : update_times,
                    "Loop Size": loop_length,
                    "Loop Area": loop_area,
                    "Instant Reward" : reward,
                }
                self.render()
                self.dump_env_states()
                self.sim.clear_buffer()
            else:
                """ Reject """
                # should we clear canvas? Yes, for consistent.
                self.sim.clear_buffer()

                # reject penalty
                reward = -0.5
        else:
            # Metropolis is not executed.
            # How to calculate reward stepwise?
            """ Normal move. step-wise reward """
            # Use energy as penalty
            reward = - (dDensity + dEnergy) # ~ -1/N, put stress on each step

            if (rets[3] == 0.0):
                # punishment but not kill it.
                reward = -0.75

        return obs, reward, terminate, info


    # Start function used for agent learing
    def start(self, init_site=None):
        """
            Returns: same as step()
                obs, reward, terminate, rets
        """
        if init_site == None:
            init_agent_site = self.sim.start(rnum(self.N))
        else:
            init_agent_site = self.sim.start(init_site)
        assert(self.agent_site == init_agent_site)
        self.episode_terminate = False
        info = None
        return self.get_obs(), 0.0, False, info

    def reset(self, site=None):
        ## clear buffer and set new start of agent
        if site is None:
            site = rnum(self.N)
        init_site = self.sim.restart(site)
        assert(init_site == site)
        self.episode_terminate = False
        self.episode_counter += 1
        # actually, counter can be called by sim.get_episode()
        info = None
        return self.get_obs(), 0.0, False, info

    def set_output_path(self, path):
        self.ofilename = os.path.join(path, self.ofilename)
        self.rfilename = os.path.join(path, self.rfilename)
        self.json_file = os.path.join(path, self.json_file)
        print ("Set environment logging to {}".format(self.ofilename))
        print ("Set loop and sites logging to {}".format(self.rfilename))
        print ("Set results dumpping path to {}".format(self.json_file))

    def reward_function(self, rets):
        pass
        """ Different Reward Strategies Here
        """
    def convert_1Dto2D(self, input_1D):
        output_2D = None
        if type(input_1D) == int:
            output_2D = (int(input_1D/self.L), int(input_1D%self.L))
        elif type(input_1D) == list:
            output_2D = []
            for position in input_1D:
                output_2D.append((int(position/self.L), int(position%self.L)))
        return output_2D

    def calculate_area(self):
        """TODO:
            The periodic boundary condition is too naive that can be modified.
        """
        traj_2D = self.convert_1Dto2D(self.sim.get_trajectory())
        traj_2D_dict = {}
        for x, y in traj_2D:
            if x in traj_2D_dict:
                traj_2D_dict[x].append(y)
            else:
                traj_2D_dict[x] = [y]

        # check Max y_length
        y_position_list = []
        for y_list in traj_2D_dict.values():
            y_position_list = y_position_list + y_list
        y_position_list = list(set(y_position_list))
        max_y_length = len(y_position_list) -1

        area = 0.0
        for x in traj_2D_dict:
            diff = max(traj_2D_dict[x]) - min(traj_2D_dict[x])
            if diff > max_y_length:
                diff = max_y_length
            temp_area = diff - len(traj_2D_dict[x]) +1  ## avoid vertical straight line
            if temp_area > 0:
                area = area + temp_area

        return area

    def render(self, mapname ="traj", mode="ansi", close=False):
        #of = StringIO() if mode == "ansi" else sys.stdout
        #print ("Energy: {}, Defect: {}".format(self.sqice.cal_energy_diff(), self.sqice.cal_defect_density()))
        s = None
        if (mapname == "traj"):
            s = self._transf2d(self.sim.get_canvas_map())
        start = self.sim.get_start_point()
        start = (int(start/self.L), int(start%self.L))
        s[start] = 3
        screen = "\r"
        screen += "\n\t"
        screen += "+" + self.L * "---" + "+\n"
        for i in range(self.L):
            screen += "\t|"
            for j in range(self.L):
                p = (i, j)
                spin = s[p]
                if spin == -1:
                    screen += " o "
                elif spin == +1:
                    screen += " * "
                elif spin == 0:
                    screen += "   "
                elif spin == +2:
                    screen += " @ "
                elif spin == -2:
                    screen += " O "
                elif spin == +3:
                    # starting point
                    screen += " x "
            screen += "|\n"
        screen += "\t+" + self.L * "---" + "+\n"
        with open(self.rfilename, "a") as f:
            f.write("Episode: {}, global step = {}\n".format(self.episode_counter, self.sim.get_total_steps()))
            f.write("{}\n".format(screen))

    def get_obs(self):
        """
            Need more flexible in get_obs. There will may be config, sequence, scalar observed states.
        """
        config_map = self._transf2d(self.sim.get_state_t_map())
        canvas_map = self._transf2d(self.sim.get_canvas_map())
        energy_map = self._transf2d(self.sim.get_energy_map())
        defect_map = self._transf2d(self.sim.get_defect_map())

        return np.stack([config_map,
                        canvas_map,
                        energy_map,
                        defect_map
        ], axis=self.stacked_axis)

    @property
    def unwrapped(self):
        """Completely unwrap this env.
            Returns:
                gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def _transf2d(self, s):
        return np.array(s, dtype=np.float32).reshape([self.L, self.L])

    def _append_record(self, record):
        with open(self.json_file, "a") as f:
            json.dump(record, f)
            f.write(os.linesep)

    def dump_env_states(self):
        # get current timestamp
        total_steps = self.sim.get_total_steps()
        ep = self.sim.get_episode()
        # agent walk # steps in this episode
        ep_step_counters = self.sim.get_ep_step_counter()
        trajectory = self.sim.get_trajectory()
        if self.sim.get_accepted_length():
            loop_length = self.sim.get_accepted_length()[-1]
        else :
            loop_length = 0
        enclosed_area = self.calculate_area()
        update_times = self.sim.get_updated_counter()
        action_counters = self.sim.get_action_statistics()
        action_stats = [x / total_steps for x in action_counters]

        start_site = self.sim.get_start_point()
        acceptance = update_times * 100.0 / ep

        d = {
            "Episode": ep,
            "Steps"  : total_steps,
            "StartSite"  : start_site,
            "Trajectory": trajectory,
            "UpdateTimes": update_times,
            "AcceptanceRatio" : acceptance, 
            "LoopLength": loop_length,
            "EnclosedArea": enclosed_area,
            "ActionStats" : action_stats
        }
        
        self._append_record(d)
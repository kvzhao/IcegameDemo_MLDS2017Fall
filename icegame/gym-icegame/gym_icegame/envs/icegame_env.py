from __future__ import division
import gym
from gym import error, spaces, utils, core

#from six import StingIO
import sys, os
import json
import random
import numpy as np
from icegame import SQIceGame, INFO
import time
from collections import deque

rnum = np.random.randint

DEFAULT_LIVES = 5
LOOP_UNIT_REWARD = 12
HIST_LEN = 0
NUM_OBSERVATION_MAPS = 3 + (HIST_LEN+1) * 2

class IceGameEnv(core.Env):
    def __init__ (self, L, kT, J, is_cont=False):
        """IceGame
            is_cont (bool):
                Set True that action is continous variable; set Fasle using discrete action.
        """
        self.L = L
        self.kT = kT
        self.J = J
        self.N = L**2
        self.is_cont = is_cont
        num_neighbors = 2
        num_replicas = 1
        num_mcsteps = 2000
        num_bins = 1
        num_thermalization = num_mcsteps
        tempering_period = 1

        self.mc_info = INFO(self.L, self.N, num_neighbors, num_replicas, \
                num_bins, num_mcsteps, tempering_period, num_thermalization)

        self.sim = SQIceGame(self.mc_info)
        self.sim.set_temperature (self.kT)
        self.sim.init_model()
        self.sim.mc_run(num_mcsteps)

        self.episode_terminate = False
        self.accepted_episode = False

        self.last_update_step = 0

        """
            History FIFO
        """
        self.Imap = I = np.ones([self.L, self.L])
        self.Omap = O = np.zeros([self.L, self.L])
        if HIST_LEN > 0:
            self.canvas_hist = deque([O] * HIST_LEN)
            self.defect_hist = deque([O] * HIST_LEN)

        self.idx2act = dict({
                            0 :   "right",
                            1 :   "down",
                            2 :   "left",
                            3 :   "up",
                            4 :   "lower_next",
                            5 :   "upper_next",
                            6 :   "metropolis"
                            })

        self.act2idx  = {v: k for k, v in self.idx2act.items()}

        # action space and state space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L, self.L, NUM_OBSERVATION_MAPS))
        if is_cont:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self.idx2act)))
        else:
            self.action_space = spaces.Discrete(len(self.idx2act))

        #TODO: make more clear definition
        """
            Global Observations:
                *
            Local Observations:
                * neighboring spins up & down
                *
        """
        self.global_observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.L, self.L, 2))
        self.local_observation_space = spaces.Discrete(7)

        self.reward_range = (-1, 1)

        # output file
        self.ofilename = "loop_sites.log"
        # render file
        self.rfilename = "loop_renders.log"
        # save log to json for future analysis
        self.json_file = "env_history.json"

        self.stacked_axis = 2

        ## counts reset()
        self.episode_counter = 0
        self.lives = DEFAULT_LIVES

        ## legacy codes
        self.auto_metropolis = False
        # ray add list:
        #     1. log 2D (x, y) in self.ofilename
        #     2. add self.calculate_area() and loop_area
        #     3. auto_6 (uncompleted)

    def step(self, action):
        """step function with directional action
        """
        if self.is_cont:
            # actions are 7 continuous variables, pick the largest one
            action = np.argmax(action)

        terminate = False
        reward = 0.0 # -0.000975 # stepwise punishment.
        obs = None
        info = None
        rets = [0.0, 0.0, 0.0, 0.0]
        metropolis_executed = False

        ## execute different type of actions
        if (action == 6):
            self.sim.flip_trajectory()
            rets = self.sim.metropolis()
            metropolis_executed = True
        elif (0 <= action < 6) :
            rets = self.sim.draw(action)

        """ Results from icegame
            index 0 plays two roles:
                if action is walk:
                    rets[0] = is_icemove
                elif action is metropolis:
                    rets[0] = is_accept
        """
        is_accept, dEnergy, dDensity, dConfig = rets
        is_icemove = True if is_accept > 0.0 else False

        # metropolis judgement
        if (metropolis_executed):
            if is_accept > 0 and dConfig > 0:
                """ Updates Accepted
                    1. Calculate rewards
                    2. Save logs
                    3. Reset maps and buffers
                """
                self.sim.update_config()
                print ("[GAME_ENV] PROPOSAL ACCEPTED!")
                total_steps = self.sim.get_total_steps()
                ep_steps = self.sim.get_ep_step_counter()
                ep = self.sim.get_episode()
                loop_length = self.sim.get_accepted_length()[-1]
                loop_area = self.calculate_area()

                # get counters
                action_counters = self.sim.get_action_statistics()
                metropolis_times = self.sim.get_updating_counter()
                update_times = self.sim.get_updated_counter()

                # compute update interval
                update_interval = total_steps - self.last_update_step
                self.last_update_step = total_steps

                # acceptance rate
                total_acc_rate = self.sim.get_total_acceptance_rate() * 100.0
                effort =  update_times/total_steps * 100.0
                reward = 1.0 * (loop_length / LOOP_UNIT_REWARD) # reward with different length by normalizing with len 4 elements

                # TODO: Calculate recent # steps' acceptance rate

                # output to self.ofilename
                with open(self.ofilename, "a") as f:
                    f.write("1D: {}, \n(2D: {})\n".format(self.sim.get_trajectory(), self.convert_1Dto2D(self.sim.get_trajectory())))
                    print ("\tSave loop configuration to file: {}".format(self.ofilename))

                print ("\tTotal accepted number = {}".format(update_times))
                print ("\tAccepted loop length = {}, area = {}".format(loop_length, loop_area))
                print ("\tAgent walks {} steps in episode, action counters: {}".format(ep_steps, self.sim.get_ep_action_counters()))
                action_stats = [x / total_steps for x in action_counters]
                print ("\tStatistics of actions all episodes (ep={}, steps={}) : {}".format(ep, total_steps, action_stats))
                print ("\tAcceptance ratio (accepted/ # of metropolis) = {}%".format(
                                                                    update_times * 100.0 / metropolis_times))
                print ("\tAcceptance ratio (from icegame) = {}%".format(total_acc_rate))
                print ("\tRunning Effort = {}%".format(effort))

                # TODO: How to describe the loop?
                info = {
                    "Acceptance Ratio" : total_acc_rate,
                    "Running Effort": effort,
                    "Updated" : update_times,
                    "Loop Size": loop_length,
                    "Loop Area": loop_area,
                }

                # Stop rendering, it save huge log
                # self.render()
                self.dump_env_states()
                self.sim.clear_buffer()

                """ Terminate?
                    stop after accpetance, will increase the episode rewards.
                    But can we still running the program to increase the total rewards?

                    Or do not terminate, just reset the location?
                """
                # terminate = True
                self.sim.restart(rnum(self.N))

            else:
                self.sim.clear_buffer()
                self.lives -= 1
                """
                    Rejection
                        1. Keep updating with new canvas.
                            or
                        Early stop.
                        2. Wrong decision penalty

                """
                reward = -0.001
                #self.episode_terminate = True
                #terminate = True
                # Avoid running metropolis at start (Too hand-crafted method!)
                #if (rets[3] == 0.0):
                #    reward = -0.8

            # reset or update
        else:
            """Stepwise feedback:
                1. exploration
                2. icemove reards
                3. defect propagation guiding
                4. #more

                TODO: Write option in init arguments.
            """
            #reward = self._stepwise_weighted_returns(rets)
            # Check each scale (each of them stays in 0~1)
            #reward = 0.002 - (dEnergy + dDensity)
            #reward = -(dEnergy + dDensity) + dConfig
            if is_icemove:
                reward = .001
                #print ("is icemove: {}, {}".format(dEnergy, dDensity))
            else:
                reward = -.001
                #print ("not icemove: {}, {}".format(dEnergy, dDensity))
            # as usual

        obs = self.get_obs()
        #obs = self.get_hist_obs()

        ## add timeout mechanism?

        # Add the timeout counter
        if self.lives <= 0:
            terminate = True

        # Not always return info
        return obs, reward, terminate, info

    # Start function used for agent learning
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
        self.lives = DEFAULT_LIVES

        return self.get_obs()
        #return self.get_hist_obs()

    def reset(self, site=None):
        ## clear buffer and set new start of agent
        if site is None:
            site = rnum(self.N)
        init_site = self.sim.restart(site)
        assert(init_site == site)
        self.episode_terminate = False
        self.episode_counter += 1
        self.lives = DEFAULT_LIVES
        # actually, counter can be called by sim.get_episode()

        # Clear the fifo queue
        if HIST_LEN > 0:
            self.canvas_hist.clear()
            self.defect_hist.clear()
            for _ in range(HIST_LEN):
                self.canvas_hist.append(self.Omap)
                self.defect_hist.append(self.Omap)

        info = None

        return self.get_obs()
        #return self.get_hist_obs()

    def timeout(self):
        return self.sim.timeout()

    @property
    def game_status(self):
        """Return whether game is terminate"""
        return self.episode_terminate

    def set_output_path(self, path):
        self.ofilename = os.path.join(path, self.ofilename)
        self.rfilename = os.path.join(path, self.rfilename)
        self.json_file = os.path.join(path, self.json_file)
        print ("Set environment logging to {}".format(self.ofilename))
        print ("Set loop and sites logging to {}".format(self.rfilename))
        print ("Set results dumpping path to {}".format(self.json_file))

    @property
    def agent_site(self):
        return self.sim.get_agent_site()

    @property
    def action_name_mapping(self):
        return self.idx2act

    @property
    def name_action_mapping(self):
        return self.act2idx

    def reward_function(self, rets):
        pass
        """ Different Reward Strategies Here
        """

    def _stepwise_weighted_returns(self, rets):
        icemove_w = 0.000
        energy_w = -1.0
        defect_w = 0.0
        baseline = 0.009765625 ## 1 / 1024
        scaling = 2.0
        return (icemove_w * rets[0] + energy_w * rets[1] + defect_w * rets[2] + baseline) * scaling

    ## ray test  (for: int, list, np_list)
    def convert_1Dto2D(self, input_1D):
        output_2D = None
        if type(input_1D) == int:
            output_2D = (int(input_1D/self.L), int(input_1D%self.L))
        elif type(input_1D) == list:
            output_2D = []
            for position in input_1D:
                output_2D.append((int(position/self.L), int(position%self.L)))
        return output_2D

    ## ray test
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

    # TODO: Render on terminal.
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
        #TODO: Add choice write to terminal or file
        #sys.stdout.write(screen)
        with open(self.rfilename, "a") as f:
            f.write("Episode: {}, global step = {}\n".format(self.episode_counter, self.sim.get_total_steps()))
            f.write("{}\n".format(screen))

    def get_obs(self):
        """
            Need more flexible in get_obs. There will may be config, sequence, scalar observed states.
            TODO: add np.nan_to_num() to prevent ill value
        """
        config_map = self._transf2d(self.sim.get_state_t_map_color())
        #config_map = self._transf2d(self.sim.get_state_t_map())
        valid_map = self._transf2d(self.sim.get_valid_action_map())
        canvas_map = self._transf2d(self.sim.get_canvas_map())
        energy_map = self._transf2d(self.sim.get_energy_map())
        defect_map = self._transf2d(self.sim.get_defect_map())

        return np.stack([config_map,
                        valid_map,
                        canvas_map,
                        energy_map,
                        defect_map
        ], axis=self.stacked_axis)

    def get_hist_obs(self):
        config_map = self._transf2d(self.sim.get_state_t_map_color())
        valid_map = self._transf2d(self.sim.get_valid_action_map())
        canvas_map = self._transf2d(self.sim.get_canvas_map())
        energy_map = self._transf2d(self.sim.get_energy_map())
        defect_map = self._transf2d(self.sim.get_defect_map())

        self.canvas_hist.append(canvas_map)
        self.canvas_hist.popleft()
        self.defect_hist.append(defect_map)
        self.defect_hist.popleft()

        canvas_traj = np.stack([canvas for canvas in self.canvas_hist], axis=self.stacked_axis)
        defect_traj = np.stack([dmap for dmap in self.defect_hist], axis=self.stacked_axis)
        config_map = np.expand_dims(config_map, axis=self.stacked_axis)
        valid_map = np.expand_dims(valid_map, axis=self.stacked_axis)
        energy_map = np.expand_dims(energy_map, axis=self.stacked_axis)

        return np.concatenate([config_map,
                        valid_map,
                        energy_map,
                        canvas_traj,
                        defect_traj
                        ], axis=self.stacked_axis)

    def get_partial_obs(self):
        """Partial Observation:
            Get the multiple channel (different format) from the same states.

            Return:
                local: neighboring relation of the state_t
                global: whole maps of the state_t
        """
        pass

    @property
    def unwrapped(self):
        """Completely unwrap this env.
            Returns:
                gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def sysinfo(self):
        print ("")

    def _transf2d(self, s):
        # do we need zero mean here?
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

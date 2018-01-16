from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import logging

from envs import create_icegame_env
from model import LSTMPolicy
from worker import FastSaver

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def inference(args):
    indir = os.path.join(args.log_dir, 'train')
    outdir = os.path.join(args.log_dir, 'inference') if args.out_dir is None else args.out_dir

    with open(indir + "/checkpoint", "r") as f:
        first_line = f.readline().strip()
        print ("first_line is : {}".format(first_line))
    ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
    ckpt = ckpt.split('-')[-1]
    ckpt = indir + '/model.ckpt-' + ckpt

    print ("ckpt: {}".format(ckpt))

    # define environment
    env = create_icegame_env(outdir, args.env_id)
    num_actions = env.action_space.n

    with tf.device("/cpu:0"):
        # define policy network
        with tf.variable_scope("global"):
            policy = LSTMPolicy(env.observation_space.shape, num_actions)
            policy.global_step = tf.get_variable("global_step", [], 
                    tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainable=False)
        # Variable names that start with "local" are not saved in checkpoints.
        variables_to_restore = [v for v in tf.global_variables() if not v.name.startswith("local")]
        init_all_op = tf.global_variables_initializer()

        saver = FastSaver(variables_to_restore)

        # print trainable variables
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
        logger.info('Trainable vars:')
        for v in var_list:
            logger.info('  {} {}'.format(v.name, v.get_shape()))
        logger.info("Restored the trained model.")

        # summary of rewards
        action_writers = []
        summary_writer = tf.summary.FileWriter(outdir)
        for act_idx in range(num_actions):
            action_writers.append(tf.summary.FileWriter(
                os.path.join(outdir, "action_{}".format(act_idx))
            ))

        logger.info("Inference events directory: %s", outdir)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        with tf.Session() as sess:
            logger.info("Initializing all parameters.")
            sess.run(init_all_op)
            logger.info("Restoring trainable global parameters.")
            saver.restore(sess, ckpt)
            logger.info("Restored model was trained for %.2fM global steps", sess.run(policy.global_step)/1000000.)

            last_features = policy.get_initial_features()  # reset lstm memory
            length = 0
            rewards = 0
            loopsizes=[]

            # All Episodes records
            for ep in range(args.num_episodes):
                """TODO: policy sampling strategy
                    random, greedy and sampled policy.
                """

                last_state = env.reset()
                
                # Episode records 

                # running policy
                while True:
                    fetched = policy.act_inference(last_state, *last_features)
                    prob_action, action, value_, features = fetched[0], fetched[1], fetched[2], fetched[3:]

                    #TODO: policy sampling strategy

                    # Greedy
                    stepAct = action.argmax()
                    state, reward, terminal, info = env.step(stepAct)

                    # update stats
                    length += 1
                    rewards += reward
                    last_state = state
                    last_features = features

                    """TODO: Resonable Statistics are necessary
                    """

                    if info:
                        loopsize = info["Loop Size"]
                        looparea = info["Loop Area"]

                    # store summary
                    summary = tf.Summary()
                    summary.value.add(tag='ep_{}/reward'.format(ep), simple_value=reward)
                    summary.value.add(tag='ep_{}/netreward'.format(ep), simple_value=rewards)
                    summary.value.add(tag='ep_{}/value'.format(ep), simple_value=float(value_[0]))

                    if info:
                        summary.value.add(tag='ep_{}/loop_size'.format(ep), simple_value=loopsize)
                        summary.value.add(tag='ep_{}/loop_area'.format(ep), simple_value=looparea)
                        loopsizes.append(loopsize)

                    summary_writer.add_summary(summary, length)
                    summary_writer.flush()

                    summary = tf.Summary()
                    for ac_id in range(num_actions):
                        summary.value.add(tag='ep_{}/a_{}'.format(ep, ac_id), simple_value=float(prob_action[ac_id]))
                        action_writers[ac_id].add_summary(summary, length)
                        action_writers[ac_id].flush()

                    """TODO:
                        1. Need more concrete idea for playing the game when interfering.
                        2. Save these values for post processing.
                    """
                    if terminal:
                        #if length >= timestep_limit:
                        #    last_state, _, _, _ = env.reset()

                        last_features = policy.get_initial_features()  # reset lstm memory
                        print("Episode finished. Sum of rewards: %.2f. Length: %d." % (rewards, length))

                        length = 0
                        rewards = 0
                        break

        logger.info('Finished %d true episodes.', args.num_episodes)

        # Count loop topology
        unique, counts = np.unique(loopsizes, return_counts=True)
        loopstatistics = dict(zip(unique, counts))
        print (loopstatistics)
        env.close()


def main(_):
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="worker", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/icegame", help='Log directory path')
    parser.add_argument('--out-dir', default=None, help='output log directory. Default: log_dir/inference/')
    parser.add_argument('--env-id', default="IceGameEnv-v0", help='Environment id')

    parser.add_argument("--num-episodes", default=1000, type=int, help="Number of episodes to run.")
    #parser.add_argument("--render", action="store_true", help="Set true to rendering")

    args = parser.parse_args()

    inference(args)

if __name__ == "__main__":
    tf.app.run()
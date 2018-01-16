import sys
import os
import itertools
import collections
import argparse
import numpy as np
import tensorflow as tf
import time
import signal

from model import LSTMPolicy
from worker import FastSaver
from worker import cluster_spec

from envs import create_icegame_env
from configs import hparams

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PolicyMonitor(object):
    """
        Helps evaluating a policy by running an episode in an environment,
            saving a video, and plotting summaries to Tensorboard.

            Sync from the global policy network and run eval.
    Args:
        env: environment to run in
        summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """
    def __init__(self, env, args):

        #self.video_dir = os.path.join(summary_writer.get_logdir(), "../videos")
        #self.video_dir = os.path.abspath(args.video_dir)

        self.args = args
        self.env = env
        self.summary_writer = None 

        # define environment
        ob_space = env.observation_space.shape
        ac_space = env.action_space.n

        worker_device = "/job:worker/task:{}/cpu:0".format(args.task)
        with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
            with tf.variable_scope("global"):
                self.network = LSTMPolicy(ob_space, ac_space)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32),
                                                trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope("local"):
                self.policy = pi = LSTMPolicy(ob_space, ac_space)
                pi.global_step = self.global_step

        # copy weights from the parameter server to the local model
        self.sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(pi.var_list, self.network.var_list)])

    def set_writer(self, summary_writer):
        self.summary_writer = summary_writer

    def eval(self, sess, num_episodes=200):
        """Same as process in worker
        """
        logger.info("Start eval policy")
        sess.run(self.sync)  # copy weights from shared to local

        # Run an episode
        #for _ in range(num_episodes):
        #    pass
        done = False

        last_state = self.env.reset()
        last_features = self.policy.get_initial_features()

        total_reward = 0.0
        episode_length = 0
        policy_hist = np.zeros(self.env.action_space.n)

        while not done:
            fetched = self.policy.act(last_state, *last_features)
            action_probs, value_, features = fetched[0], fetched[1], fetched[2:]

            # Greedy action when testing
            action = action_probs.argmax()
            state, reward, done, info = self.env.step(action)
            # policy counter
            policy_hist[action] += 1

            episode_length += 1
            total_reward += reward

            last_state = state
            last_features = features

        policy_hist /= np.sum(policy_hist)
        # Add summaries
        # https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary
        episode_summary = tf.Summary()
        episode_summary.value.add(simple_value=total_reward, tag="eval/total_reward")
        episode_summary.value.add(simple_value=episode_length, tag="eval/episode_length")
        for actidx, prob_a in enumerate(policy_hist):
            episode_summary.value.add(simple_value=prob_a, tag="eval/prob_a{}".format(actidx))
        # add histogram here
        self.summary_writer.add_summary(episode_summary, self.global_step.eval())
        self.summary_writer.flush()

        logger.info ("Eval results at step {}: total_reward {}, episode_length {}".format(self.global_step.eval(), total_reward, episode_length))

        return total_reward, episode_length

def run_monitor(args, server):
    logger.info("Execute run monitor")
    env = create_icegame_env(args.log_dir, args.env_id)
    monitor = PolicyMonitor(env, args)

    variables_to_save = [v for v in tf.global_variables() if not v.name.startswith("local")]
    init_op = tf.variables_initializer(variables_to_save)
    init_all_op = tf.global_variables_initializer()

    # print trainable variables
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    for v in var_list:
        logger.info('  %s %s', v.name, v.get_shape())

    def init_fn(ses):
        logger.info("Initializing all parameters.")
        ses.run(init_all_op)
        
    config = tf.ConfigProto(device_filters=["/job:ps", "/job:worker/task:{}/cpu:0".format(args.task)])

    logdir = os.path.join(args.log_dir, 'eval')
    summary_writer = tf.summary.FileWriter(logdir)

    monitor.set_writer(summary_writer)

    sv = tf.train.Supervisor(is_chief=(args.task == 0),
                            logdir=logdir,
                            summary_op=None,
                            init_op=init_op,
                            init_fn=init_fn,
                            summary_writer=summary_writer,
                            ready_op=tf.report_uninitialized_variables(variables_to_save),
                            global_step=monitor.global_step)

    logger.info(
        "Starting session. If this hangs, we're mostly likely waiting to connect to the parameter server. " +
        "One common cause is that the parameter server DNS name isn't resolving yet, or is misspecified.")

    with sv.managed_session(server.target, config=config) as sess, sess.as_default():
        logger.info("PE Session Entered")
        sess.run(monitor.sync)
        global_step = sess.run(monitor.global_step)
        logger.info("Starting monitoring at step=%d", global_step)

        while not sv.should_stop():
            monitor.eval(sess)
            time.sleep(300)

    # Ask for all the services to stop.
    sv.stop()
    logger.info('reached %s steps. worker stopped.', global_step)

def main(_):
    """
        Tensorflow for monitoring trained policy
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    parser.add_argument('--task', default=0, type=int, help='Task index')
    parser.add_argument('--job-name', default="monitor", help='worker or ps')
    parser.add_argument('--num-workers', default=1, type=int, help='Number of workers')
    parser.add_argument('--log-dir', default="/tmp/icegame", help='Log directory path')
    parser.add_argument('--video-dir', default=None, help='Path to save video')
    parser.add_argument('--env-id', default="Pong-v0", help='Environment id')
    parser.add_argument('-r', '--remotes', default=None,
                        help='References to environments to create (e.g. -r 20), '
                            'or the address of pre-existing VNC servers and '
                            'rewarders to use (e.g. -r vnc://localhost:5900+15900,vnc://localhost:5901+15901)')

    # Add visualisation argument
    parser.add_argument('--visualise', action='store_true',
                        help="Visualise the gym environment by running env.render() between each timestep")

    args = parser.parse_args()

    spec = cluster_spec(args.num_workers, 1)
    cluster = tf.train.ClusterSpec(spec).as_cluster_def()

    def shutdown(signal, frame):
        logger.warn('Received signal %s: exiting', signal)
        sys.exit(128+signal)
    signal.signal(signal.SIGHUP, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Reset graph before allocating any?
    tf.reset_default_graph()

    if args.job_name == "monitor":
        server = tf.train.Server(cluster, job_name="worker", task_index=args.task,
                                config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
        run_monitor(args, server)

if __name__ == "__main__":
    tf.app.run()
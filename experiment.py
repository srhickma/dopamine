from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import random

from absl import logging, app, flags

from dopamine.agents.rainbow import rainbow_agent
from dopamine.discrete_domains import iteration_statistics
from dopamine.discrete_domains import run_experiment
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
import gin
import numpy as np
import tensorflow as tf
import tf_slim


def reload_checkpoint(checkpoint_path, session):
  """Load TF checkpoint and restore into agent session"""
  global_vars = set([x.name for x in tf.compat.v1.global_variables()])
  ckpt_vars = [
    '{}:0'.format(name)
    for name, _ in tf.train.list_variables(checkpoint_path)
  ]
  include_vars = list(global_vars.intersection(set(ckpt_vars)))
  variables_to_restore = tf_slim.get_variables_to_restore(include=include_vars)

  if variables_to_restore:
    reloader = tf.compat.v1.train.Saver(var_list=variables_to_restore)
    reloader.restore(session, checkpoint_path)
    logging.info('Done restoring from %s', checkpoint_path)
  else:
    logging.info('Nothing to restore!')


class MyRainbowAgent(rainbow_agent.RainbowAgent):
  def __init__(self, sess, num_actions, summary_writer=None):
    super(MyRainbowAgent, self).__init__(sess, num_actions, summary_writer=summary_writer)
    self.rewards = []

  def step(self, reward, observation):
    self.rewards.append(reward)
    return super(MyRainbowAgent, self).step(reward, observation)

  def reload_checkpoint(self, checkpoint_path):
    reload_checkpoint(checkpoint_path, self._sess)

  def get_probabilities(self):
    return self._sess.run(tf.squeeze(self._net_outputs.probabilities), {self.state_ph: self.state})

  def get_rewards(self):
    return [np.cumsum(self.rewards)]


class MyRunner(run_experiment.Runner):
  def __init__(self, base_dir, trained_agent_ckpt_path, create_agent_fn):
    self._trained_agent_ckpt_path = trained_agent_ckpt_path
    super(MyRunner, self).__init__(base_dir, create_agent_fn)

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._agent.reload_checkpoint(self._trained_agent_ckpt_path)
    self._start_iteration = 0

  def _run_one_iteration(self, iteration):
    statistics = iteration_statistics.IterationStatistics()
    logging.info('Starting iteration %d', iteration)
    _, _ = self._run_eval_phase(statistics)
    return statistics.data_lists

  def build_visualizer(self, record_path):
    atari_params = {'environment': self._environment}
    atari_plot = atari_plotter.AtariPlotter(parameter_dict=atari_params)

    return agent_visualizer.AgentVisualizer(
      record_path=record_path,
      plotters=[atari_plot],
      screen_width=atari_plot.parameters['width'],
      screen_height=atari_plot.parameters['height']
    )

  def run_game(self, seed, visualizer=None):
    step_count = 0
    total_reward = 0

    self._environment.environment.seed(seed)
    random.seed(seed)

    print(f'Running game with seed {seed}')

    initial_observation = self._environment.reset()
    action = self._agent.begin_episode(initial_observation)
    while True:
      observation, reward, is_terminal, _ = self._environment.step(action)
      step_count += 1
      total_reward += reward

      sys.stdout.write(f'Steps executed: {step_count} Score: {total_reward}\r')
      sys.stdout.flush()

      if visualizer is not None:
        visualizer.visualize()

      if self._environment.game_over:
        break
      elif is_terminal:
        self._agent.end_episode(reward)
        action = self._agent.begin_episode(observation)
      else:
        action = self._agent.step(reward, observation)

    self._end_episode(reward)

    print(f'Achieved a score of {total_reward} after {step_count} steps')

    if visualizer is not None:
      visualizer.generate_video()

    return total_reward

  def run_full_experiment(self, record_path):
    if not tf.io.gfile.exists(record_path):
      tf.io.gfile.makedirs(record_path)

    self._agent.eval_mode = True

    scores = []
    seeds = []

    for game_number in range(0, 4):
      print(f'Starting game {game_number}')

      seed = game_number
      score = self.run_game(seed)

      scores.append(score)
      seeds.append(seed)

    # TODO calculate 5 number summary here!

    print(f'Visualizing best game')

    best_seed = seeds[scores.index(max(scores))]
    self.run_game(best_seed, self.build_visualizer(record_path))


def create_rainbow_agent(sess, environment, summary_writer=None):
  return MyRainbowAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)


def create_runner(base_dir, trained_agent_ckpt_path, agent):
  create_agent = None
  if agent == 'rainbow':
    create_agent = create_rainbow_agent

  return MyRunner(base_dir, trained_agent_ckpt_path, create_agent)


def run(agent, root_dir, restore_ckpt):
  """Main entrypoint for running and generating visualizations.

  Args:
    agent: str, agent type to use.
    root_dir: str, root directory where files will be stored.
    restore_ckpt: str, path to the checkpoint to reload.
  """
  tf.compat.v1.reset_default_graph()
  config = """
  atari_lib.create_atari_environment.game_name = 'SpaceInvaders'
  WrappedReplayBuffer.replay_capacity = 300
  """
  base_dir = os.path.join(root_dir, 'results', agent)
  gin.parse_config(config)
  runner = create_runner(base_dir, restore_ckpt, agent)
  runner.run_full_experiment(os.path.join(base_dir, 'images'))


flags.DEFINE_string('agent', 'rainbow', 'Agent to visualize.')
flags.DEFINE_string('root_dir', '/tmp/dopamine/', 'Root directory.')
flags.DEFINE_string('restore_checkpoint', None,
                    'Path to checkpoint to restore for visualizing.')

FLAGS = flags.FLAGS


def main(_):
  run(
    agent=FLAGS.agent,
    root_dir=FLAGS.root_dir,
    restore_ckpt=FLAGS.restore_checkpoint
  )


if __name__ == '__main__':
  app.run(main)

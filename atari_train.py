import os
import argparse
import gym
import numpy as np
import cntk
from cntk import *
from cntk.layers import *
from cntk.logging.progress_print import TensorBoardProgressWriter

from agent import LearningAgent
from decay import *
from memory import Memory
from runner import AtariRunner
from wrapper import GymWrapper


def train(runner, agent, episodes, target_update, log_freq, chkpt_freq, chkpt_path, writer):
    neg_inf = float('-inf')
    pos_inf = float('inf')

    min_reward = pos_inf
    max_reward = neg_inf
    sum_reward = 0

    min_steps = pos_inf
    max_steps = neg_inf
    sum_steps = 0

    episode = 0
    while episode < episodes:
        steps, rewards = runner.run_episode()

        episode += 1
        agent.update_epsilon(episode)

        min_reward = min(min_reward, rewards)
        max_reward = max(max_reward, rewards)
        sum_reward += rewards

        min_steps = min(min_steps, steps)
        max_steps = max(max_steps, steps)
        sum_steps += steps

        if episode % target_update == 0:
            agent.update_target()

        if episode % log_freq == 0:
            denom = float(log_freq)
            ave_rewards = sum_reward / denom
            writer.write_value('rewards/min.', min_reward, episode)
            writer.write_value('rewards/ave.', ave_rewards, episode)
            writer.write_value('rewards/max.', max_reward, episode)

            ave_steps = sum_steps / denom
            writer.write_value('steps/min.', min_steps, episode)
            writer.write_value('steps/ave.', ave_steps, episode)
            writer.write_value('steps/max.', max_steps, episode)

            writer.write_value('epsilon', agent.epsilon, episode)
            agent.trainer.summarize_training_progress()
            writer.flush()

            # reset statistics
            min_reward = pos_inf
            max_reward = neg_inf
            sum_reward = 0

            min_steps = pos_inf
            max_steps = neg_inf
            sum_steps = 0

        if episode % chkpt_freq == 0:
            print('Checkpoint after {} episodes'.format(episode))
            agent.trainer.save_checkpoint(chkpt_path.format(episode))


def main(env_name,
         episodes,
         gamma,
         learning_rate,
         batch_size,
         mem_cap,
         target_update,
         action_repeat,
         stack_frames,
         replay_period,
         replay_start_size,
         use_exp,
         min_epsilon,
         decay_exp,
         decay_lin,
         model_dir,
         log_freq,
         chkpt_freq):
    # OpenAI gym environment to train against
    gym_env = gym.make(env_name)
    env = GymWrapper(gym_env, action_repeat, stack_frames)

    state_dim = (stack_frames, 40, 100)
    action_dim = gym_env.action_space.n

    # Updater for decaying the value of epsilon
    if use_exp:
        updater = ExponentialDecay(min_epsilon, decay_exp)
    else:
        updater = LinearDecay(min_epsilon, decay_lin)

    # Create the model for the agent
    state_var = input(state_dim, dtype=np.float32)
    action_var = input(action_dim, dtype=np.float32)

    with default_options(activation=relu, pad=False):
        model = Sequential([
            Convolution2D((8, 8), 8, strides=(4, 4), pad=True, name='conv1'),
            Convolution2D((4, 4), 16, strides=(2, 2), name='conv2'),
            Convolution2D((4, 4), 16, strides=(1, 1), name='conv3'),
            Dense(256, name='dense1'),
            Dense(action_dim, activation=None, name='out')
        ])(state_var)

    loss = reduce_mean(square(model - action_var), axis=0)
    lr_schedule = learning_rate_schedule(learning_rate, UnitType.sample)
    learner = sgd(model.parameters, lr_schedule)

    tb_writer = TensorBoardProgressWriter(log_dir=os.path.join(model_dir, 'log'), model=model)
    trainer = Trainer(model, loss, learner, [tb_writer])

    # Agent to train
    agent = LearningAgent((state_dim, action_dim), (state_var, action_var), (model, trainer), updater)

    # Create the buffer for storing agent experiences
    buffer = Memory(mem_cap)

    # Episode runner
    runner = AtariRunner(env, agent, buffer, gamma, batch_size, replay_period)
    runner.initialize_buffer(replay_start_size)

    # Train the agent
    try:
        chkpt_path = os.path.join(model_dir, env_name + '_{}.dqn')
        train(runner, agent, episodes, target_update, log_freq, chkpt_freq, chkpt_path, tb_writer)
    finally:
        model_path = os.path.join(model_dir, '{}.dqn'.format(env_name))
        print('Saving model to {}...'.format(model_path))
        agent.target_model.save_model(model_path)
        print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('env_name', help='The name of the Atari OpenAI Gym environment')

    # Core training arguments
    parser.add_argument('-e', '--episodes', type=int, default=10000, help='The number of episodes to train for')
    parser.add_argument('-g', '--gamma', type=float, default=0.99, help='The discount rate')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00025, help='The learning rate')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='The number of samples in training minibatch')
    parser.add_argument('-mc', '--mem_cap', type=int, default=50000, help='The replay buffer capacity')
    parser.add_argument('-tf', '--target_update', type=int, default=50,
                        help='The frequency of updates to the target network')
    parser.add_argument('-ar', '--action_repeat', type=int, default=1,
                        help='The number of frames to repeat each action for')
    parser.add_argument('-sf', '--stack_frames', type=int, default=4, help='The number of state frames to stack')
    parser.add_argument('-rp', '--replay_period', type=int, default=4,
                        help='The number of steps between experience replay')
    parser.add_argument('-rss', '--replay_start_size', type=int, default=10000,
                        help='Initial number of random transitions to populate the buffer with')

    # Decay arguments
    parser.add_argument('-exp', '--use_exp', action='store_true', help='Set to use exponential decay for epsilon')
    parser.add_argument('-mne', '--min_epsilon', type=float, default=0.1, help='The minimum value of epsilon')
    parser.add_argument('-de', '--decay_exp', type=float, default=3.838e-4, help='Exponential decay rate for epsilon')
    parser.add_argument('-dl', '--decay_lin', type=float, default=-1.125e-4, help='Linear decay rate for epsilon')

    # Checkpointing and logging
    parser.add_argument('-md', '--model_dir', default='chkpt', help='Directory for logs and checkpoints')
    parser.add_argument('-lf', '--log_freq', type=int, default=10, help='The number of episodes between progress logs')
    parser.add_argument('-cf', '--chkpt_freq', type=int, default=100, help='The number of episodes between checkpoints')

    args = parser.parse_args()

    # Select the right target device when this notebook is being tested
    if 'TEST_DEVICE' in os.environ:
        if os.environ['TEST_DEVICE'] == 'cpu':
            cntk.try_set_default_device(cntk.device.cpu())
        else:
            cntk.try_set_default_device(cntk.device.gpu(0))

    main(args.env_name,
         args.episodes,
         args.gamma,
         args.learning_rate,
         args.batch_size,
         args.mem_cap,
         args.target_update,
         args.action_repeat,
         args.stack_frames,
         args.replay_period,
         args.replay_start_size,
         args.use_exp,
         args.min_epsilon,
         args.decay_exp,
         args.decay_lin,
         args.model_dir,
         args.log_freq,
         args.chkpt_freq)

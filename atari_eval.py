import argparse
import gym
import numpy as np
from cntk import *
from wrapper import GymWrapper


def run(env, model, render):
    done = False
    reward = 0

    s = env.reset()
    while not done:
        if render:
            env.render()

        a = np.argmax(model.eval(s))
        s, r, done, info = env.step(a)
        reward += r
    return reward


def random(env, render):
    """
    Run an episode of the given environment using a random 
    :param env: The gym environment to run
    :return: The reward for the episode
    """
    done = False
    reward = 0

    s = env.reset()
    while not done:
        if render:
            env.render()

        a = env.random_action()
        s, r, done, info = env.step(a)
        reward += r

    return reward


def main(env_name,
         model_path,
         chkpt_model,
         episodes,
         render,
         use_random):
    gym_env = gym.make(env_name)
    env = GymWrapper(gym_env)

    rewards = 0

    if use_random:
        for i in range(episodes):
            rewards += random(env, render)

    else:
        model = load_model(model_path)
        if chkpt_model:
            model = combine([model.outputs[0].owner])

        for i in range(episodes):
            rewards += run(env, model, render)

    ave_reward = rewards / float(episodes)
    print('Average reward for {} episodes = {}'.format(episodes, ave_reward))

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', help='The name of the Atari OpenAI Gym environment')
    parser.add_argument('model_path', help='The path to the trained model')
    parser.add_argument('-c', '--chkpt_model', action='store_true', help='Set if model is from a checkpoint')
    parser.add_argument('-e', '--episodes', type=int, default=100, help='The number of episodes to evaluate over')
    parser.add_argument('-r', '--render', action='store_true', help='Set to render the agent-environment interaction')
    parser.add_argument('-rnd', '--random', action='store_true', help='True to evaluate a random agent')
    args = parser.parse_args()

    main(args.env_name,
         args.model_path,
         args.chkpt_model,
         args.episodes,
         args.render,
         args.random)

import scipy as sc
import scipy.misc
import numpy as np


class GymWrapper:
    def __init__(self, env, action_repeat=1, stack_length=4):
        """
        Creates a new wrapper around the OpenAI environment
        :param env: The OpenAI environment to wrap around
        :param action_repeat: The number of frames to apply each selected action for
        :param stack_length: The number of frames to stack as a single observation for the network input
        """
        self.env = env
        self.action_repeat = action_repeat
        self.stack_length = stack_length
        self.action_count = self.env.action_space.n
        self.state_stack = None

    def reset(self):
        """
        Resets the environment
        :return: preprocessed first state of next game
        """
        return self.preprocess(self.env.reset(), new_game=True)

    def preprocess(self, state, new_game=False):
        state = state.mean(axis=2)  # Convert to single channel

        # KungFuMaster specific
        state = state[95:155, 8:]
        state = sc.misc.imresize(state, (40, 100))  # Downsample

        state = state * (1. / 255)  # Normalize
        state = state.astype(np.float32)  # Required by CNTK
        state = state.reshape(1, state.shape[0], state.shape[1])

        if new_game or self.state_stack is None:
            self.state_stack = np.repeat(state, self.stack_length, axis=0)
        else:
            self.state_stack = np.append(state, self.state_stack[:self.stack_length - 1, :, :], axis=0)

        return self.state_stack

    def step(self, action):
        """
        Executes action on the next 'action_repeat' frames
        :param action: The action to execute
        :return: (s, r, done, info): next state, reward, terminated, debugging information
        """
        rewards = 0
        for _ in range(self.action_repeat):
            s, r, done, info = self.env.step(action)
            rewards += r
            if done:
                break
        return self.preprocess(s), rewards / 100.0, done, info

    def random_action(self):
        """        
        Returns a random action to execute
        """
        return self.env.action_space.sample()

    def render(self):
        """
        Renders the current state
        """
        self.env.render()

    def close(self):
        """
        Closes the rendering window
        """
        self.env.close()

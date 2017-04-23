import numpy as np
from cntk import *


class LearningAgent:
    def __init__(self, space, input, model, eps_updater):
        """        
        :param space: tuple (state_space, action_space) of the state and action dimensions
        :param input: tuple (state_var, action_var) of inputs to the network
        :param model: tuple (model, trainer) of the network to train and the trainer to use
        :param eps_updater: object for updating the value of epsilon with time
        """
        self.state_dim, self.action_dim = space
        self.state_var, self.action_var = input
        self.online_model, self.trainer = model
        self.eps_updater = eps_updater
        self.epsilon = 1.0

        # Create target network and initialize with same weights
        self.target_model = None
        self.update_target()

    def update_target(self):
        """
        Updates the target network using the online network weights
        """
        self.target_model = self.online_model.clone(CloneMethod.clone)

    def update_epsilon(self, episode):
        """
        Updates epsilon using exponential decay 
        """
        self.epsilon = self.eps_updater.update(episode)

    def predict(self, s, target=False):
        """
        Feeds a state through the model (our network) and obtains the values of each action
        """
        if target:
            return self.target_model.eval(s)
        else:
            return self.online_model.eval(s)

    def act(self, state):
        """
        Selects an action using the epoch-greedy approach
        """
        if np.random.randn(1) > self.epsilon:
            # exploit (greedy)
            return np.argmax(self.predict(state))
        else:
            # explore (random action)
            return np.random.randint(0, self.action_dim)

    def train(self, x, y):
        """
        Performs a single gradient descent step using the provided states and targets
        """
        self.trainer.train_minibatch({self.state_var: x, self.action_var: y})


class EvalAgent:
    def __init__(self, model_path, chkpt):
        self.model = load_model(model_path)

        # See: https://github.com/Microsoft/CNTK/wiki/Evaluate-a-saved-convolutional-network
        if chkpt:
            self.model = combine([self.model.outputs[0].owner])

    def act(self, s):
        return np.argmax(self.model.eval(s.astype(np.float32)))

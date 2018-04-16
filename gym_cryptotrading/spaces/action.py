import numpy as np

import gym
from gym import error

from gym_cryptotrading.strings import *

class ActionSpace(gym.Space):
    lookup = {
            0: NEUTRAL,
            1: LONG,
            2: SHORT
        } 

    def __init__(self):
        super(ActionSpace, self).__init__()

    def sample(self):
        return np.random.randint(0, 3)

    def contains(self, action):
        return action in ActionSpace.lookup.keys()

    def to_jsonable(self, sample_n):
        super(ActionSpace, self).to_jsonable(sample_n)

    def from_jsonable(self, sample_n):
        super(ActionSpace, self).from_jsonable(sample_n)

    @staticmethod
    def get_action_name(action):
        if action in ActionSpace.lookup.keys():
            return ActionSpace.lookup[action]
        else:
            raise error.InvalidAction()

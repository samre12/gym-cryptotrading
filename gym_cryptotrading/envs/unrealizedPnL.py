import numpy as np

from gym import error

from gym_cryptotrading.strings import *
from gym_cryptotrading.envs.cryptoenv import CryptoEnv

class UnRealizedPnLEnv(CryptoEnv):
    def __init__(self):
        super(UnRealizedPnLEnv, self).__init__()

    def _reset_params(self):
        self.long, self.short = 0, 0
        self.timesteps = 0

    def _take_action(self, action):
        if action not in CryptoEnv.action_space.lookup.keys():
            raise error.InvalidAction()
        else:
            if CryptoEnv.action_space.lookup[action] is LONG:
                self.long = self.long + 1
                
            elif CryptoEnv.action_space.lookup[action] is SHORT:
                self.short = self.short + 1
        
    def _get_reward(self):
        return (self.long - self.short) * self.unit * self.diffs[self.current]

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

    def step(self, action):
        if not self.episode_number or self.timesteps is self.horizon:
            raise error.ResetNeeded()

        state = self._get_new_state()
        self._take_action(action)
        reward = self._get_reward()

        message = "Timestep {}:==: Action: {} ; Reward: {}".format(
            self.timesteps, CryptoEnv.action_space.lookup[action], reward
        )
        self.logger.debug(message)
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            self.current = self.current + 1
            return state, reward, False, np.array([float(self.horizon - self.timesteps) / self.horizon])
        else:
            return state, reward, True, np.array([0.0])
    
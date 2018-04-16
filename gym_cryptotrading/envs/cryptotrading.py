import numpy as np

from gym import error, logger

from gym_cryptotrading.strings import *
from gym_cryptotrading.envs.basicenv import BaseEnv

class CryptoTradingEnv(BaseEnv):
    def __init__(self, history_length=100, horizon=5, unit=5e-4):
        super(CryptoTradingEnv, self).__init__(history_length, horizon, unit)

    def _reset_params(self):
        self.long, self.short = 0, 0
        self.timesteps = 0

    def _take_action(self, action):
        super(CryptoTradingEnv, self)._take_action(action)

        if BaseEnv.action_space.lookup[action] is LONG:
            self.long = self.long + 1
            
        elif BaseEnv.action_space.lookup[action] is SHORT:
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
            self.timesteps, BaseEnv.action_space.lookup[action], reward
        )
        logger.debug(message)
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            self.current = self.current + 1
            return state, reward, False, (self.horizon - self.timesteps)
        else:
            return state, reward, True, (self.horizon - self.timesteps)
    
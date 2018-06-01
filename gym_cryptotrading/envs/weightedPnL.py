import numpy as np

from collections import deque

from gym import error

from gym_cryptotrading.strings import *
from gym_cryptotrading.envs.cryptoenv import CryptoEnv

class ExponentiallyWeightedReward:
    def __init__(self, lag, decay_rate):
        self.lag = lag
        self.decay_rate = decay_rate

        '''
        `self.rewards`: deque containing unrealized PnL rewards in order of occurence
        `self.sum`: sum of all the rewards in deque weighted by their position
        `self.denominator`:  
            - sum of all the weights assigned to rewards
            - used for normalization of the weighted reward
        '''
        self.rewards = deque(np.zeros(self.lag, dtype=float))
        self.sum = 0.0
        self.denominator = 0.0
        for i in range(self.lag):
            self.denominator = self.denominator + np.exp(-1 * i * self.decay_rate)

    def insert(self, reward):
        stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * (self.lag - 1) * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)

    @property
    def reward(self):
        return self.sum / self.denominator
        
class WeightedPnLEnv(CryptoEnv):
    def __init__(self):
        super(WeightedPnLEnv, self).__init__()

        self.decay_rate = 1e-2
        self.lag = self.horizon
        
    def _set_env_specific_params(self, **kwargs):
        if DECAY_RATE in kwargs:
            if kwargs[DECAY_RATE] > 0:
                self.decay_rate = kwargs[DECAY_RATE]
            else:
                raise ValueError(INVALID_DECAY_RATE)

        if LAG in kwargs:
            if kwargs[LAG] > 0 and kwargs[LAG] <= self.horizon:
                self.lag = kwargs[LAG]
            else:
                raise ValueError(INVALID_LAG)

    def _reset_params(self):
        self.long, self.short = 0, 0
        self.timesteps = 0

        self.reward = ExponentiallyWeightedReward(self.lag, self.decay_rate)

    def _take_action(self, action):
        if action not in CryptoEnv.action_space.lookup.keys():
            raise error.InvalidAction()
        else:
            if CryptoEnv.action_space.lookup[action] is LONG:
                self.long = self.long + 1
                
            elif CryptoEnv.action_space.lookup[action] is SHORT:
                self.short = self.short + 1
        
    def _get_reward(self):
        reward = (self.long - self.short) * self.unit * self.diffs[self.current]
        self.reward.insert(reward)
        return self.reward.reward
        
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

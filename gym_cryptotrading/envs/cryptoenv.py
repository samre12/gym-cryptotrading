import numpy as np

import gym
from gym import error, logger

from abc import abstractmethod

from gym_cryptotrading.envs.basicenv import BaseEnv

from gym_cryptotrading.generator import Generator
from gym_cryptotrading.strings import *
from gym_cryptotrading.errors import *

from gym_cryptotrading.spaces.action import ActionSpace
from gym_cryptotrading.spaces.observation import ObservationSpace

class CryptoEnv(gym.Env, BaseEnv):
    action_space = ActionSpace()
    observation_space = ObservationSpace()
    metadata = {'render.modes': []}

    def __init__(self):
        super(CryptoEnv, self).__init__()
        self.generator = None

    def _get_new_state(self):
        return self.historical_prices[self.current]

    def _load_gen(self):
        if not self.generator:
            self.generator = Generator(self.history_length, self.horizon)

    def _new_random_episode(self):
        '''
        TODO: In the current setting, the selection of an episode does not follow pure uniform process. 
        Need to index every episode and then generate a random index rather than going on multiple levels
        of selection.
        '''
        self._load_gen()
        self._reset_params()
        message_list = []
        self.episode_number = self.episode_number + 1
        message_list.append("Starting a new episode numbered {}".format(self.episode_number))
        
        block_index = np.random.randint(0, len(self.generator.price_blocks) - 1)
        message_list.append("Block index selected for episode number {} is {}".format(
                self.episode_number, block_index
            )
        )

        self.diffs = self.generator.diff_blocks[block_index]
        self.historical_prices = self.generator.price_blocks[block_index]

        self.current = np.random.randint(self.history_length,  
                                        len(self.historical_prices) - self.horizon)
        message_list.append(
            "Starting index and timestamp point selected for episode number {} is {}:==:{}".format(
                self.episode_number, 
                self.current, 
                self.generator.timestamp_blocks[block_index][self.current]
            )
        )
        
        map(self.logger.debug, message_list)

        return self.historical_prices[self.current - self.history_length:self.current], np.array([1.0])


    def _reset_params(self):
        pass

    def _set_env_specific_params(self, **kwargs):
        pass

    def reset(self):
        return self._new_random_episode()

    def set_params(self, history_length, horizon, unit, **kwargs):
        if self.generator:
            raise EnvironmentAlreadyLoaded()

        if history_length < 0 or horizon < 1 or unit < 0:
            raise ValueError()
        
        else:
            self.history_length = history_length
            self.horizon = horizon
            self.unit = unit #units of Bitcoin traded each time

            self._set_env_specific_params(**kwargs)
    
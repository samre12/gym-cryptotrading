import numpy as np

import gym
from gym import error, logger

from abc import abstractmethod

from gym_cryptotrading.generator import Generator
from gym_cryptotrading.strings import *
from gym_cryptotrading.errors import *

from gym_cryptotrading.spaces.action import ActionSpace
from gym_cryptotrading.spaces.observation import ObservationSpace

class BaseEnv(gym.Env):
    action_space = ActionSpace()
    observation_space = ObservationSpace()
    metadata = {'render.modes': []}

    def __init__(self):
        self.episode_number = 0
        self.generator = None

        self.history_length = 100 
        self.horizon = 5 
        self.unit = 5e-4

    def set_params(self, history_length, horizon, unit):
        if self.generator:
            raise EnvironmentAlreadyLoaded()

        if history_length < 0 or horizon < 1 or unit < 0:
            raise ValueError()
        
        else:
            self.history_length = history_length
            self.horizon = horizon
            self.unit = unit #units of Bitcoin traded each time

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
        
        map(logger.debug, message_list)

        return self.historical_prices[self.current - self.history_length:self.current]
    
    @abstractmethod
    def _reset_params(self):
        pass

    @abstractmethod
    def _take_action(self, action):
        pass
    
    @abstractmethod
    def _get_reward(self):
        return 0

    def _get_new_state(self):
        return self.historical_prices[self.current]
    
    def reset(self):
        return self._new_random_episode()

    @abstractmethod
    def step(self, action):
        state = self._get_new_state()
        self._take_action(action)
        reward = self._get_reward()
        return state, reward, False, None
    
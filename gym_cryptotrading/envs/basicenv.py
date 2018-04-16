import numpy as np

import gym
from gym import error, logger

from gym_cryptotrading.generator import Generator
from gym_cryptotrading.strings import *

from gym_cryptotrading.spaces.action import ActionSpace
from gym_cryptotrading.spaces.observation import ObservationSpace

class BaseEnv(gym.Env):
    action_space = ActionSpace()
    observation_space = ObservationSpace()
    metadata = {'render.modes': []}

    def __init__(self, history_length=100, horizon=5, unit=5e-4):
        self.episode_number = 0
        self.timesteps = None
        self.history_length = history_length
        self.horizon = horizon
        self.unit = unit #units of Bitcoin traded each time

        self.timesteps = None

        self.generator = Generator(history_length, horizon)

    def _new_random_episode(self):
        '''
        TODO: In the current setting, the selection of an episode does not follow pure uniform process. 
        Need to index every episode and then generate a random index rather than going on multiple levels
        of selection.
        '''
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
    
    def _reset_params(self):
        pass

    def _take_action(self, action):
        if action not in BaseEnv.action_space.lookup.keys():
            raise error.InvalidAction()
        
    def _get_reward(self):
        return 0

    def _get_new_state(self):
        return self.historical_prices[self.current]
    
    def reset(self):
        return self._new_random_episode()

    def step(self, action):
        raise NotImplementedError()
    
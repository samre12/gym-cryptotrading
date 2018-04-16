import numpy as np

import gym
from gym import error, logger

from gym_cryptotrading.generator import Generator
from gym_cryptotrading.strings import *

class CryptoTradingEnv(gym.Env):
    action_dict = {
            0: NEUTRAL,
            1: LONG,
            2: SHORT
        } 

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
        message_list = []
        self.episode_number = self.episode_number + 1
        message_list.append("Starting a new episode numbered {}".format(self.episode_number))
        self.long, self.short = 0, 0
        self.timesteps = 0
        
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

    def _take_action(self, action):
        if action in CryptoTradingEnv.action_dict.keys():
            if CryptoTradingEnv.action_dict[action] is LONG:
                self.long = self.long + 1
                
            elif CryptoTradingEnv.action_dict[action] is SHORT:
                self.short = self.short + 1
        else:
            raise error.InvalidAction()
        

    def _get_reward(self):
        return (self.long - self.short) * self.unit * self.diffs[self.current]

    def _get_new_state(self):
        return self.historical_prices[self.current]
    
    def reset(self):
        return self._new_random_episode()

    def step(self, action):
        if not self.episode_number or self.timesteps is self.horizon:
            raise error.ResetNeeded()

        state = self._get_new_state()
        self._take_action(action)
        reward = self._get_reward()

        message = "Timestep {}:==: Action: {} ; Reward: {}".format(
            self.timesteps, self.action_dict[action], reward
        )
        logger.debug(message)
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            self.current = self.current + 1
            return state, reward, False, (self.horizon - self.timesteps)
        else:
            return state, reward, True, (self.horizon - self.timesteps)

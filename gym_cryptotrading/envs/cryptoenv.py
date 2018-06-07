from abc import abstractmethod

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from talib import abstract

import gym
from gym import error, logger
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

    groups = ['Overlap Studies', 'Momentum Indicators', 
                'Volatility Indicators', 'Volume Indicators']

    technical_indicators = {
        'Overlap Studies': [
            'SMA', 'WMA', 'EMA', 'DEMA', 'TEMA', 'T3', 'TRIMA', 'KAMA'
        ],

        'Momentum Indicators': [
            'ADXR', 'APO', 'AROONOSC', 'CCI', 'CMO', 'DX', 'MFI', 'MINUS_DI', 
            'MINUS_DM', 'MOM', 'PLUS_DI', 'PLUS_DM', 'PPO', 'RSI'
        ],

        'Volatility Indicators': [
            'ATR'
        ],

        'Volume Indicators': [
            'AD', 'ADOSC', 'OBV'
        ]
    }

    def __init__(self):
        super(CryptoEnv, self).__init__()
        self.generator = None

    def _get_new_state(self):
        return self._get_state(self.current)

    def _get_new_supp(self):
        return self._get_supp(self.current)

    def _get_state(self, index):
        return self.prices[index]

    def _get_supp(self, index):
        supp = [self.remaining_timesteps[index]]
        for group in self.groups:
            for indicator in self.technical_indicators[group]:
                supp.append(self.indicators[indicator][index])

        return np.array(supp).reshape(len(supp))

    def _load_gen(self):
        if not self.generator:
            self.generator = Generator(self.history_length, self.horizon)

    def _new_random_episode(self):
        '''
        TODO: In the current setting, the selection of an episode does not follow pure 
        uniform process. Need to index every episode and then generate a random index 
        rather than going on multiple levels of selection.
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

        price_block = self.generator.price_blocks[block_index]
        start = np.random.randint(0, 
                                len(price_block) - self.horizon - self.history_length)

        self.current = -1 * self.horizon
        
        message_list.append(
            "Starting index and timestamp point selected for episode number {} is {}:==:{}".format(
                self.episode_number, 
                start + self.history_length, 
                self.generator.timestamp_blocks[block_index][start + self.history_length]
            )
        )

        self.prices = price_block.iloc[start: start + self.history_length + self.horizon, :]

        inputs = {
            'close': self.prices['price_close'].values,
            'low': self.prices['price_low'].values,
            'high': self.prices['price_high'].values,
            'volume': self.prices['volume'].values
        }

        message_list.append('Setting up diffs of closing prices for reward calculation')
        self.diffs = np.subtract(self.prices.iloc[self.current:, 0].values, 
                                    self.prices.iloc[self.current - 1: -1, 0].values)

        standard_scalar = StandardScaler()

        message_list.append('Setting up normalized remaining steps')
        self.remaining_timesteps = [
            float(self.horizon - i) / self.horizon for i in range(self.horizon + 1)
        ]
        self.remaining_timesteps = np.array(self.remaining_timesteps).reshape(-1, 1)
        self.remaining_timesteps = standard_scalar.fit_transform(self.remaining_timesteps)
        
        message_list.append('Calculating and normalizing technical indicators by groups')
        self.indicators = {}
        for group in self.groups:
            for indicator in self.technical_indicators[group]:
                values = np.array(abstract.Function(indicator)(inputs))
                self.indicators[indicator] = standard_scalar.fit_transform(
                                                values[~np.isnan(values)].reshape(-1, 1)
                                            )

        message_list.append('Normalizing the price tensor and volume traded')
        self.prices = standard_scalar.fit_transform(self.prices)

        map(self.logger.debug, message_list)

        return self.prices[:self.current], self._get_supp(self.current - 1)

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

    def step(self, action):
        if not self.episode_number or self.timesteps is self.horizon:
            raise error.ResetNeeded()

        state = self._get_new_state()
        supp = self._get_new_supp()
        self._take_action(action)
        reward = self._get_reward()

        message = "Timestep {}:==: Action: {} ; Reward: {}".format(
            self.timesteps, CryptoEnv.action_space.lookup[action], reward
        )
        self.logger.debug(message)
        
        self.timesteps = self.timesteps + 1
        if self.timesteps is not self.horizon:
            self.current = self.current + 1
            return state, reward, False, supp
        else:
            return state, reward, True, supp
    
from abc import ABCMeta, abstractmethod

from gym import logger

class BaseEnv:
    '''
    Abstract Base Class for CryptoTrading Environments
    '''

    __metaclass__ = ABCMeta

    def __init__(self):
        self.episode_number = 0
        self.logger = logger

        self.history_length = 100 
        self.horizon = 5 
        self.unit = 5e-4

    @abstractmethod
    def _get_new_state(self):
        raise NotImplementedError

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError

    @abstractmethod
    def _new_random_episode(self):
        raise NotImplementedError
    
    @abstractmethod
    def _reset_params(self):
        raise NotImplementedError

    @abstractmethod
    def _set_env_specific_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _take_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def set_params(self, history_length, horizon, unit, **kwargs):
        raise NotImplementedError

    def set_logger(self, custom_logger):
        if custom_logger:
            self.logger = custom_logger
    
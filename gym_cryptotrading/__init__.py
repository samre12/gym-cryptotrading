import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='CryptoTrading-v0',
    entry_point='gym_cryptotrading.envs:CryptoTradingEnv',
    timestep_limit=10,
    nondeterministic = True
)

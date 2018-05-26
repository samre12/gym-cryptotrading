from gym.envs.registration import register

register(
    id='RealizedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:RealizedPnLEnv',
    timestep_limit=10,
    nondeterministic = True
)

register(
    id='UnRealizedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:UnRealizedPnLEnv',
    timestep_limit=10,
    nondeterministic = True
)

register(
    id='WeightedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:WeightedPnLEnv',
    timestep_limit=10,
    nondeterministic = True
)
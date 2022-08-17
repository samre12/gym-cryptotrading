from gym.envs.registration import register

register(
    id='RealizedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:RealizedPnLEnv',
    max_episode_steps=10,
    nondeterministic = True
)

register(
    id='UnRealizedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:UnRealizedPnLEnv',
    max_episode_steps=10,
    nondeterministic = True
)

register(
    id='WeightedPnLEnv-v0',
    entry_point='gym_cryptotrading.envs:WeightedPnLEnv',
    max_episode_steps=10,
    nondeterministic = True
)
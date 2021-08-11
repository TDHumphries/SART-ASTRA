from gym.envs.registration import register

register(
    id='CTEnv-v0',
    entry_point='SART-ASTRA.envs:CTEnv',
)
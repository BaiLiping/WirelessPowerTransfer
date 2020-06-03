from gym.envs.registration import register

register(id='WPTEnv-v0',
        entry_point='envs.wpt_env_dir:WPTEnv')

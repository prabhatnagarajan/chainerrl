import chainerrl.wrappers
from chainerrl.wrappers import atari_wrappers
import gym
from pdb import set_trace


def make_atari(env_id, max_frames=30 * 60 * 60, mask_render=False):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    # Unwrap TimeLimit wrapper because we use our own time limits
    env = env.env
    if max_frames:
        env = chainerrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=max_frames)
    if env_id == "SpaceInvadersNoFrameskip-v4":
        env = SpaceInvadersScoreMask(env, mask_render)
    else:
        assert False, "Not a supported env"
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    return env


class SpaceInvadersScoreMask(gym.Wrapper):
    def __init__(self, env, mask_render):
        """ Masked env
        """
        assert 'SpaceInvadersNoFrameskip' in env.spec.id
        gym.Wrapper.__init__(self, env)
        self.obs = None
        self.mask_render = mask_render

    def reset(self, **kwargs):
        obs =  self.env.reset(**kwargs)
        mask_obs = self.mask_obs(obs)
        self.obs = mask_obs
        return mask_obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        mask_obs = self.mask_obs(obs)
        self.obs = mask_obs
        return mask_obs, reward, done, info

    def render(self, mode='human', **kwargs):
        if self.mask_render:
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(self.obs)
            return self.viewer.isopen
        else:
            return self.env.render(mode)

    def mask_obs(self, obs):
        mask_obs = obs
        # mask out score
        # TODO: check whether spaceship comes
        mask_obs[0:20] = 0
        return mask_obs
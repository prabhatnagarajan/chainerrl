from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import gym

import chainerrl.wrappers
from chainerrl.wrappers import atari_wrappers

def make_atari(env_id, max_frames=30 * 60 * 60, mask_render=False):
    env = gym.make(env_id)
    assert 'NoFrameskip' in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    # Unwrap TimeLimit wrapper because we use our own time limits
    env = env.env
    if max_frames:
        env = chainerrl.wrappers.ContinuingTimeLimit(
            env, max_episode_steps=max_frames)
    env = ScoreMaskEnv(env, mask_render)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    return env


class AtariMask():
    def __init__(self, env, height=210, width=160):
        if "SpaceInvadersNoFrameskip" in env.spec.id:
            self.mask = self.mask_space_invaders
        elif "PongNoFrameskip" in env.spec.id:
            self.mask = self.mask_pong
        elif "BreakoutNoFrameskip" in env.spec.id:
            self.mask = self.mask_breakout
        elif "EnduroNoFrameskip" in env.spec.id:
            self.mask = self.mask_enduro
        elif "SeaquestNoFrameskip" in env.spec.id:
            self.mask = self.mask_seaquest
        elif "QbertNoFrameskip" in env.spec.id:
            self.mask = self.mask_qbert
        elif "BeamRiderNoFrameskip" in env.spec.id:
            self.mask = self.mask_beam_rider
        elif "HeroNoFrameskip" in env.spec.id:
            self.mask = self.mask_hero
        elif "MontezumaRevengeNoFrameskip" in env.spec.id:
            self.mask = self.mask_revenge
        elif "MsPacmanNoFrameskip" in env.spec.id:
            self.mask = self.mask_ms_pacman
        elif "VideoPinballNoFrameskip" in env.spec.id:
            self.mask = self.mask_pinball
        elif "FreewayNoFrameskip" in env.spec.id:
            self.mask = self.mask_freeway
        elif "AssaultNoFrameskip" in env.spec.id:
            self.mask = self.mask_assault
        elif "BoxingNoFrameskip" in env.spec.id:
            self.mask = self.mask_boxing
        elif "StarGunnerNoFrameskip" in env.spec.id:
            self.mask = self.mask_star_gunner
        elif "ZaxxonNoFrameskip" in env.spec.id:
            self.mask = self.mask_zaxxon
        elif "SkiingNoFrameskip" in env.spec.id:
            self.mask = self.mask_skiing
        elif "DemonAttack" in env.spec.id:
            self.mask = self.mask_demon_attack
        elif "Asteroids" in env.spec.id:
            self.mask = self.mask_asteroids
        elif "Tennis" in env.spec.id:
            self.mask = self.mask_tennis
        elif "IceHockey" in env.spec.id:
            self.mask = self.mask_ice_hockey
        else:
            assert False, env.spec.id + " is not a supported env"

        self.h_ratio = 210/height
        self.w_ratio = 160/width

    def __call__(self, x):
        return self.mask(x)

    def mask_space_invaders(self, obs):
        mask_obs = obs
        # mask out score
        # TODO: check whether spaceship comes
        mask_obs[0:int(20/self.h_ratio)] = 0
        return mask_obs

    def mask_pong(self, obs):
        mask_obs = obs
        mask_obs[0:int(21/self.h_ratio)] = 0
        return mask_obs

    def mask_breakout(self, obs):
        mask_obs = obs
        mask_obs[0:int(15/self.h_ratio)] = 0
        return mask_obs

    def mask_enduro(self, obs):
        mask_obs = obs
        mask_obs[int(178//self.h_ratio):] = 0
        return mask_obs

    def mask_seaquest(self, obs):
        mask_obs = obs
        mask_obs[0:int(17/self.h_ratio)] = 0
        return mask_obs

    def mask_qbert(self, obs):
        mask_obs = obs
        mask_obs[0:int(13/self.h_ratio)] = 0
        return mask_obs

    def mask_beam_rider(self, obs):
        mask_obs = obs
        mask_obs[0:int(18/self.h_ratio)] = 0
        return mask_obs

    def mask_hero(self, obs):
        mask_obs = obs
        mask_obs[int(179/self.h_ratio):] = 0
        return mask_obs

    def mask_revenge(self, obs):
        mask_obs = obs
        mask_obs[0:int(14/self.h_ratio)] = 0
        return mask_obs

    def mask_ms_pacman(self, obs):
        mask_obs = obs
        mask_obs[int(187/self.h_ratio):] = 0
        return mask_obs

    def mask_pinball(self, obs):
        mask_obs = obs
        mask_obs[int(29/self.h_ratio):int(39/self.h_ratio),
                 int(64/self.w_ratio):int(156/self.w_ratio)] = 0
        return mask_obs

    def mask_freeway(self, obs):
        mask_obs = obs
        mask_obs[0:int(13/self.h_ratio)] = 0
        return mask_obs

    def mask_assault(self, obs):
        mask_obs = obs
        mask_obs[int(39/self.h_ratio):int((47)/self.h_ratio)] = 0
        return mask_obs 

    def mask_boxing(self, obs):
        mask_obs = obs
        mask_obs[0:int((12)/self.h_ratio)] = 0
        return mask_obs 

    def mask_star_gunner(self, obs):
        mask_obs = obs
        mask_obs[0:int((21)/self.h_ratio)] = 0
        return mask_obs

    def mask_zaxxon(self, obs):
        mask_obs = obs
        mask_obs[0:int((18)/self.h_ratio)] = 0
        return mask_obs

    def mask_skiing(self, obs):
        mask_obs = obs
        mask_obs[0:int((48)/self.h_ratio)] = 0
        return mask_obs

    def mask_demon_attack(self, obs):
        mask_obs = obs
        mask_obs[0:int((16)/self.h_ratio)] = 0
        return mask_obs

    def mask_asteroids(self, obs):
        mask_obs = obs
        mask_obs[0:int((16)/self.h_ratio), 0:int((80)/self.w_ratio)] = 0
        return mask_obs

    def mask_tennis(self, obs):
        mask_obs = obs
        mask_obs[0:int((38)/self.h_ratio)] = 0
        return mask_obs

    def mask_ice_hockey(self, obs):
        mask_obs = obs
        mask_obs[int((14)/self.h_ratio):int((21)/self.h_ratio)] = 0
        return mask_obs


class ScoreMaskEnv(gym.Wrapper):
    def __init__(self, env, mask_render):
        """ Masked env
        """
        gym.Wrapper.__init__(self, env)
        self.obs = None
        self.mask_render = mask_render
        self.mask = AtariMask(env)

    def reset(self, **kwargs):
        obs =  self.env.reset(**kwargs)
        mask_obs = self.mask(obs)
        self.obs = mask_obs
        return mask_obs

    def step(self, ac):
        obs, reward, done, info = self.env.step(ac)
        mask_obs = self.mask(obs)
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

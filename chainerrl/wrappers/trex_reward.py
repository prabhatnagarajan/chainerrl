from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import os
import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import optimizers
from chainer import serializers 
import gym
import numpy as np

from chainerrl.envs import MultiprocessVectorEnv
from chainerrl.misc.batch_states import batch_states
from chainerrl.wrappers import VectorFrameStack


def subseq(seq, subseq_len, start):
    return seq[start: start + subseq_len]


class TREXNet(chainer.ChainList):
    """TREX's architecture: https://arxiv.org/abs/1904.06387"""


    def __init__(self):
        layers = [
            L.Convolution2D(4, 16, 7, stride=3),
            L.Convolution2D(16, 16, 5, stride=2),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Convolution2D(16, 16, 3, stride=1),
            L.Linear(784, 64),
            L.Linear(64, 1)
        ]

        super(TREXNet, self).__init__(*layers)

    def __call__(self, trajectory):
        h = trajectory
        for layer in self:
            h = F.leaky_relu(layer(h))
        return h

class TREXReward():
    """Implements Trajectory-ranked Reward EXtrapolation (TREX):

    https://arxiv.org/abs/1904.06387.

    Args:
        env: an Env
        ranked_demos (RankedDemoDataset): A list of ranked demonstrations
        steps: number of gradient steps
        sub_traj_len: a tuple containing (min, max) traj length to sample
        traj_batch_size: num trajectory pairs to use per update
        opt: optimizer
        network: A reward network to train

    Attributes:
        demos: A list of demonstrations
        trex_network: Reward network

    """

    def __init__(self,
                 ranked_demos,
                 steps=30000,
                 num_sub_trajs=12800,
                 sub_traj_len=(50,100),
                 traj_batch_size=16,
                 opt=optimizers.Adam(alpha=0.00005),
                 sample_live=True,
                 network=TREXNet(),
                 train_network=True,
                 gpu=None,
                 outdir=None,
                 phi=lambda x: x,
                 save_network=False):
        self.ranked_demos = ranked_demos
        self.steps = steps
        self.trex_network = network
        self.train_network = train_network
        self.training_observations = []
        self.training_labels = []
        self.prev_reward = None
        self.traj_batch_size = traj_batch_size
        self.min_sub_traj_len = sub_traj_len[0]
        self.max_sub_traj_len = sub_traj_len[1]
        self.num_sub_trajs = num_sub_trajs
        self.sample_live = sample_live
        self.outdir = outdir
        self.examples = []      
        self.phi = phi 
        if self.train_network:
            self.opt = opt
            self.opt.setup(self.trex_network)
            if gpu is not None and gpu >= 0:
                cuda.get_device(gpu).use()
                self.trex_network.to_gpu(device=gpu)
            self.save_network = save_network
            self._train()
        self.xp = self.trex_network.xp


    def create_example(self):
        '''Creates a training example.'''

        ranked_trajs = self.ranked_demos.episodes
        indices = np.arange(len(ranked_trajs)).tolist()
        traj_indices = np.random.choice(indices, size=2, replace=False)
        i = traj_indices[0]
        j = traj_indices[1]
        min_ep_len = min(len(ranked_trajs[i]), len(ranked_trajs[j]))
        sub_traj_len = np.random.randint(self.min_sub_traj_len,
                                         self.max_sub_traj_len)
        traj_1 = ranked_trajs[i]
        traj_2 = ranked_trajs[j]
        if i < j:
            i_start = np.random.randint(min_ep_len - sub_traj_len + 1)
            j_start = np.random.randint(i_start, len(traj_2) - sub_traj_len + 1)
        else:
            j_start = np.random.randint(min_ep_len - sub_traj_len + 1)
            i_start = np.random.randint(j_start, len(traj_1) - sub_traj_len + 1)
        sub_traj_i = subseq(traj_1, sub_traj_len, start=i_start)
        sub_traj_j = subseq(traj_2, sub_traj_len, start=j_start)
        # if trajectory i is better than trajectory j
        if i > j:
            label = 0
        else:
            label = 1
        return sub_traj_i, sub_traj_j, label

    def create_training_dataset(self):
        self.examples = []
        self.index = 0
        for _ in range(self.num_sub_trajs):
            self.examples.append(self.create_example())


    def get_training_batch(self):
        if not self.examples:
            self.create_training_dataset()
        if self.index + self.traj_batch_size > len(self.examples):
            self.index = 0
            if not self.sample_live:
                random.shuffle(self.examples)
            else:
                self.create_training_dataset()
        batch = self.examples[self.index:self.index + self.traj_batch_size]
        return batch

    def _compute_loss(self, batch):
        xp = self.trex_network.xp
        preprocessed = {
            'i' : [batch_states([transition["obs"] for transition in example[0]], xp, self.phi)
                               for example in batch],
            'j' : [batch_states([transition["obs"] for transition in example[1]], xp, self.phi)
                                           for example in batch],
            'label' : xp.array([example[2] for example in batch])
        }
        rewards_i = [F.sum(self.trex_network(preprocessed['i'][i])) for i in range(len(preprocessed['i']))]
        rewards_j = [F.sum(self.trex_network(preprocessed['j'][i])) for i in range(len(preprocessed['j']))]
        rewards_i = F.expand_dims(F.stack(rewards_i), 1)
        rewards_j = F.expand_dims(F.stack(rewards_j), 1)
        predictions = F.concat((rewards_i, rewards_j))
        mean_loss = F.mean(F.softmax_cross_entropy(predictions,
                                                   preprocessed['label']))
        return mean_loss

    def _train(self):
        for step in range(1, self.steps + 1):
            # get batch of traj pairs
            batch = self.get_training_batch()
            # do updates
            loss = self._compute_loss(batch)
            self.trex_network.cleargrads()
            loss.backward()
            self.opt.update()
            if step % int(self.steps / min(self.steps, 100)) == 0:
                print("Performed update " + str(step) + "/" + str(self.steps))
        print("Finished training TREX network.")
        if self.save_network:
            serializers.save_npz(os.path.join(self.outdir, 'reward_net.model'),
                                 self.trex_network)
    def __call__(self, x):
        return self.trex_network(x)

class TREXRewardEnv(gym.Wrapper):
    """Environment Wrapper for neural network reward:

    Args:
        env: an Env
        network: A reward Network

    Attributes:
        trex_network: Reward network

    """

    def __init__(self, env,
                 trex_network):
        super().__init__(env)
        self.trex_network = trex_network

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        obs = batch_states([observation], self.trex_network.xp,
                          self.trex_network.phi)
        with chainer.no_backprop_mode():
            inverse_reward = F.sigmoid(self.trex_network(obs)).item()
        info["true_reward"] = reward
        return observation, inverse_reward, done, info

class TREXMultiprocessRewardEnv(MultiprocessVectorEnv):
    """Environment Wrapper for neural network reward:

    Args:
        env: an Env
        network: A reward Network

    Attributes:
        trex_network: Reward network

    """

    def __init__(self, env_fns,
                 trex_network):
        super().__init__(env_fns)
        self.trex_network = trex_network


    def step(self, actions):
        self._assert_not_closed()
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        self.last_obs, rews, dones, infos = zip(*results)
        obs = batch_states(self.last_obs, self.trex_network.xp,
                          self.trex_network.phi)
        trex_rewards = F.sigmoid(self.trex_network(obs))
        trex_rewards = tuple(trex_rewards.array[:,0].tolist())
        for i in range(len(rews)):
            infos[i]["true_reward"] = rews[i]
        return self.last_obs, trex_rewards, dones, infos

class TREXVectorEnv(VectorFrameStack):
    """Environment Wrapper for vector of environments

    to replace with a neural network reward.

    Args:
        env: a MultiProcessVectorEnv
        k: Num frames to stack
        stack_axis: axis to stack frames
        trex_network: A reward network

    Attributes:
        trex_network: Reward network

    """

    def __init__(self, env, k, stack_axis,
                 trex_network):
        super().__init__(env, k, stack_axis)
        self.trex_network = trex_network


    def step(self, actions):
        batch_ob, rewards, dones, infos = self.env.step(actions)
        for frames, ob in zip(self.frames, batch_ob):
            frames.append(ob)
        obs = self._get_ob()
        processed_obs = batch_states(obs, self.trex_network.xp,
                                     self.trex_network.phi)
        trex_rewards = F.sigmoid(self.trex_network(processed_obs))
        # convert variable([[r1],[r2], ...]) to tuple
        trex_rewards = tuple(trex_rewards.array[:,0].tolist())
        for i in range(len(rewards)):
            infos[i]["true_reward"] = rewards[i]
        return obs, trex_rewards, dones, infos

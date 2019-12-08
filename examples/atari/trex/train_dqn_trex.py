from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA

import argparse
import json
import operator
from operator import xor
import os


import chainer
from chainer import cuda
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import gym
import gym.wrappers
import numpy as np


import chainerrl
from chainerrl.action_value import DiscreteActionValue
from chainerrl import agents
from chainerrl import demonstration
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl.q_functions import DuelingDQN
from chainerrl import replay_buffer
from chainerrl.wrappers import atari_wrappers
from chainerrl.wrappers import score_mask_atari
from chainerrl.wrappers.trex_reward import TREXNet
from chainerrl.wrappers.trex_reward import TREXReward

import demo_parser


class SingleSharedBias(chainer.Chain):
    """Single shared bias used in the Double DQN paper.

    You can add this link after a Linear layer with nobias=True to implement a
    Linear layer with a single shared bias parameter.

    See http://arxiv.org/abs/1509.06461.
    """

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=1)

    def __call__(self, x):
        return x + F.broadcast_to(self.bias, x.shape)


def parse_arch(arch, n_actions):
    if arch == 'nature':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions),
            DiscreteActionValue)
    elif arch == 'doubledqn':
        return links.Sequence(
            links.NatureDQNHead(),
            L.Linear(512, n_actions, nobias=True),
            SingleSharedBias(),
            DiscreteActionValue)
    elif arch == 'nips':
        return links.Sequence(
            links.NIPSDQNHead(),
            L.Linear(256, n_actions),
            DiscreteActionValue)
    elif arch == 'dueling':
        return DuelingDQN(n_actions)
    else:
        raise RuntimeError('Not supported architecture: {}'.format(arch))


def parse_agent(agent):
    return {'DQN': agents.DQN,
            'DoubleDQN': agents.DoubleDQN,
            'PAL': agents.PAL}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str,
                        default='SpaceInvadersNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--load-trex', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6,
                        help='Timesteps after which we stop ' +
                        'annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01,
                        help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--arch', type=str, default='doubledqn',
                        choices=['nature', 'nips', 'dueling', 'doubledqn'],
                        help='Network architecture to use.')
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                        'performing gradient updates.')
    parser.add_argument('--target-update-interval',
                        type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which ' +
                        'the target network is updated.')
    parser.add_argument('--update-interval', type=int, default=4,
                        help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=200)
    parser.add_argument('--no-clip-delta',
                        dest='clip_delta', action='store_false')
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--agent', type=str, default='DoubleDQN',
                        choices=['DQN', 'DoubleDQN', 'PAL'])
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Use prioritized experience replay.')
    parser.add_argument('--mask-render', action='store_true', default=False,
                        help='Mask when you render.')
    parser.add_argument('--trex-steps', type=int, default=30000,
                        help='Number of TREX updates.')
    parser.add_argument('--gc-loc', type=str,
                        help='Atari Grand Challenge Data location.')
    parser.add_argument('--load-demos', type=str, 
                        help='Location of demonstrations pickle file.')
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in ChainerRL.
    misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    assert xor(bool(args.gc_loc), bool(args.load_demos)), \
        "Must specify exactly one of the location of Atari Grand " + \
        "Challenge dataset or the location of demonstrations " + \
        "stored in a pickle file."
    if args.gc_loc:
        assert args.env in ['SpaceInvadersNoFrameskip-v4',
                            'MontezumaRevengeNoFrameskip-v4',
                            'MsPacmanNoFrameskip-v4',
                            'QbertNoFrameskip-v4',
                            'VideoPinballNoFrameskip-v4'
                            ]
    else:
        # TODO: add asserts for the other envs
        assert True

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        else:
            if args.gc_loc:
                demo_extractor = demo_parser.AtariGrandChallengeParser(
                    args.gc_loc, env, args.outdir)
            else:
                demo_extractor = demo_parser.ChainerRLAtariDemoParser(
                    args.load_demos, env, 12)
            episodes = demo_extractor.episodes
            # Sort episodes by ground truth ranking
            # episodes contain transitions of (obs, a, r, new_obs, done, info)
            # redundance for sanity - demoparser should return sorted
            ranked_episodes = sorted(episodes,
                                     key=lambda ep:sum([ep[i]['reward'] for i in range(len(ep))]))
            episode_rewards = [sum([episode[i]['reward']  \
                               for i in range(len(episode))]) \
                               for episode in ranked_episodes]
            demo_dataset = demonstration.RankedDemoDataset(ranked_episodes)
            assert sorted(episode_rewards) == episode_rewards
            network = TREXNet()
            if args.load_trex:
                from chainer import serializers
                serializers.load_npz(args.load_trex, network)
            env = TREXReward(env=env,
                             ranked_demos=demo_dataset,
                             steps=args.trex_steps,
                             network=TREXNet(),
                             gpu=args.gpu,
                             outdir=args.outdir,
                             save_network=True)
        if args.monitor:
            env = gym.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    n_actions = env.action_space.n
    q_func = parse_arch(args.arch, n_actions)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func, sigma_scale=args.noisy_net_sigma)
        # Turn off explorer
        explorer = explorers.Greedy()
    else:
        explorer = explorers.LinearDecayEpsilonGreedy(
            1.0, args.final_epsilon,
            args.final_exploration_frames,
            lambda: np.random.randint(n_actions))

    # Draw the computational graph and save it in the output directory.
    chainerrl.misc.draw_computational_graph(
        [q_func(np.zeros((4, 84, 84), dtype=np.float32)[None])],
        os.path.join(args.outdir, 'model'))

    # Use the Nature paper's hyperparameters
    opt = optimizers.RMSpropGraves(
        lr=args.lr, alpha=0.95, momentum=0.0, eps=1e-2)

    opt.setup(q_func)

    # Select a replay buffer to use
    if args.prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.6,
            beta0=0.4, betasteps=betasteps,
            num_steps=args.num_step_return)
    else:
        rbuf = replay_buffer.ReplayBuffer(10 ** 6, args.num_step_return)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = parse_agent(args.agent)
    agent = Agent(q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
                  explorer=explorer, replay_start_size=args.replay_start_size,
                  target_update_interval=args.target_update_interval,
                  clip_delta=args.clip_delta,
                  update_interval=args.update_interval,
                  batch_accumulator='sum',
                  phi=phi)

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        stats = experiments.evaluator.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames/4,
            logger=None)
        with open(os.path.join(args.outdir, 'bestscores.json'), 'w') as f:
            # temporary hack to handle python 2/3 support issues.
            # json dumps does not support non-string literal dict keys
            json_stats = json.dumps(stats)
            print(str(json_stats), file=f)
        print("The results of the best scoring network:")
        for stat in stats:
            print(str(stat) + ":" + str(stats[stat]))


if __name__ == '__main__':
    main()

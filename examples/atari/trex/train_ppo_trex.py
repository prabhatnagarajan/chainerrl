"""An example of training PPO against OpenAI Gym Atari Envs.

This script is an example of training a PPO agent on Atari envs.

To train PPO for 10M timesteps on Breakout, run:
    python train_ppo_ale.py

To train PPO using a recurrent model on a flickering Atari env, run:
    python train_ppo_ale.py --recurrent --flicker --no-frame-stack
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import operator
from operator import xor
import os

import chainer
from chainer import functions as F
from chainer import links as L
import numpy as np

import chainerrl
from chainerrl.agents import PPO
from chainerrl import demonstration
from chainerrl import experiments
from chainerrl import misc
from chainerrl.wrappers import atari_wrappers
from chainerrl.wrappers import score_mask_atari
from chainerrl.wrappers.trex_reward import TREXNet
from chainerrl.wrappers.trex_reward import TREXReward

import demo_parser

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='SpaceInvadersNoFrameskip-v4',
                        help='Gym Env ID.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID. Set to -1 to use CPUs only.')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Number of env instances run in parallel.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--steps', type=int, default=10 ** 7,
                        help='Total time steps for training.')
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                        help='Learning rate.')
    parser.add_argument('--eval-interval', type=int, default=100000,
                        help='Interval (in timesteps) between evaluation'
                             ' phases.')
    parser.add_argument('--eval-n-runs', type=int, default=10,
                        help='Number of episodes ran in an evaluation phase.')
    parser.add_argument('--demo', action='store_true', default=False,
                        help='Run demo episodes, not training.')
    parser.add_argument('--load', type=str, default='',
                        help='Directory path to load a saved agent data from'
                             ' if it is a non-empty string.')
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--update-interval', type=int, default=128 * 8,
                        help='Interval (in timesteps) between PPO iterations.')
    parser.add_argument('--batchsize', type=int, default=32 * 8,
                        help='Size of minibatch (in timesteps).')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Number of epochs used for each PPO iteration.')
    parser.add_argument('--log-interval', type=int, default=10000,
                        help='Interval (in timesteps) of printing logs.')
    parser.add_argument('--recurrent', action='store_true', default=False,
                        help='Use a recurrent model. See the code for the'
                             ' model definition.')
    parser.add_argument('--flicker', action='store_true', default=False,
                        help='Use so-called flickering Atari, where each'
                             ' screen is blacked out with probability 0.5.')
    parser.add_argument('--no-frame-stack', action='store_true', default=False,
                        help='Disable frame stacking so that the agent can'
                             ' only see the current screen.')
    parser.add_argument('--checkpoint-frequency', type=int,
                        default=None,
                        help='Frequency at which agents are stored.')
    # TREX arguments
    parser.add_argument('--load-trex', type=str, default=None)
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

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    print('Output files are saved in {}'.format(args.outdir))

    assert xor(bool(args.gc_loc), bool(args.load_demos)), \
    "Must specify exactly one of the location of Atari Grand " + \
    "Challenge dataset or the location of demonstrations " + \
    "stored in a pickle file."

    env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=True,
            clip_rewards=True,
            flicker=args.flicker,
            frame_stack=not args.no_frame_stack,
        )
    if args.gc_loc:
        demo_extractor = demo_parser.AtariGrandChallengeParser(
                        args.gc_loc, env, args.outdir)
    else:
        demo_extractor = demo_parser.ChainerRLAtariDemoParser(
                        args.load_demos, env, 12, args.outdir)
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
    if args.load_trex:
        from chainer import serializers
        serializers.load_npz(args.load_trex, trex_network)
    env = TREXReward(env=env,
                     ranked_demos=demo_dataset,
                     steps=args.trex_steps,
                     network=TREXNet(),
                     train_network=(False if args.load_trex),
                     gpu=args.gpu,
                     outdir=args.outdir,
                     save_network=True)
    trex_network = env.trex_network

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=not test,
            clip_rewards=not test,
            flicker=args.flicker,
            frame_stack=not args.no_frame_stack,
        )
        env.seed(env_seed)
        if not test:
            env = TREXReward(env=env,
                             ranked_demos=demo_dataset,
                             steps=args.trex_steps,
                             network=trex_network,
                             train_network=False,
                             gpu=args.gpu,
                             outdir=args.outdir)
            # trex_network = env.trex_network
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        return chainerrl.envs.MultiprocessVectorEnv(
            [(lambda: make_env(idx, test))
             for idx, env in enumerate(range(args.num_envs))])

    sample_env = make_env(0, test=False)
    print('Observation space', sample_env.observation_space)
    print('Action space', sample_env.action_space)
    n_actions = sample_env.action_space.n

    winit_last = chainer.initializers.LeCunNormal(1e-2)
    if args.recurrent:
        model = chainerrl.links.StatelessRecurrentSequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            L.NStepGRU(1, 512, 512, 0),
            chainerrl.links.Branched(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1),
            )
        )
    else:
        model = chainer.Sequential(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            L.Linear(None, 512),
            F.relu,
            chainerrl.links.Branched(
                chainer.Sequential(
                    L.Linear(None, n_actions, initialW=winit_last),
                    chainerrl.distribution.SoftmaxDistribution,
                ),
                L.Linear(None, 1),
            )
        )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros(
        sample_env.observation_space.shape, dtype=np.float32)[None]
    if args.recurrent:
        fake_out, _ = model(fake_obss, None)
    else:
        fake_out = model(fake_obss)
    chainerrl.misc.draw_computational_graph(
        [fake_out], os.path.join(args.outdir, 'model'))

    opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
    opt.setup(model)
    opt.add_hook(chainer.optimizer.GradientClipping(0.5))

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    agent = PPO(
        model,
        opt,
        gpu=args.gpu,
        phi=phi,
        update_interval=args.update_interval,
        minibatch_size=args.batchsize,
        epochs=args.epochs,
        clip_eps=0.1,
        clip_eps_vf=None,
        standardize_advantages=True,
        entropy_coef=1e-2,
        recurrent=args.recurrent,
    )
    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs)
        print('n_runs: {} mean: {} median: {} stdev: {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        step_hooks = []

        # Linearly decay the learning rate to zero
        def lr_setter(env, agent, value):
            agent.optimizer.alpha = value

        step_hooks.append(
            experiments.LinearInterpolationHook(
                args.steps, args.lr, 0, lr_setter))

        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(False),
            eval_env=make_batch_env(True),
            outdir=args.outdir,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            checkpoint_freq=args.checkpoint_frequency,
            eval_interval=args.eval_interval,
            log_interval=args.log_interval,
            save_best_so_far_agent=False,
            step_hooks=step_hooks,
        )


if __name__ == '__main__':
    main()

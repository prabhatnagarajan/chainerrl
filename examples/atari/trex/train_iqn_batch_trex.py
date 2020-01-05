import argparse
import functools
import json
import os

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers
import numpy as np

import chainerrl
from chainerrl import agents
from chainerrl import demonstration
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import replay_buffer

from chainerrl.wrappers import atari_wrappers
from chainerrl.wrappers import score_mask_atari
from chainerrl.wrappers.trex_reward import TREXNet
from chainerrl.wrappers.trex_reward import TREXReward
from chainerrl.wrappers.trex_reward import TREXVectorEnv

import demo_parser


def parse_agent(agent):
    return {'IQN': chainerrl.agents.IQN,
            'DoubleIQN': chainerrl.agents.DoubleIQN}[agent]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--final-exploration-frames',
                        type=int, default=10 ** 6)
    parser.add_argument('--final-epsilon', type=float, default=0.01)
    parser.add_argument('--eval-epsilon', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=5 * 10 ** 7)
    parser.add_argument('--max-frames', type=int,
                        default=30 * 60 * 60,  # 30 minutes with 60 fps
                        help='Maximum number of frames for each episode.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4)
    parser.add_argument('--target-update-interval',
                        type=int, default=10 ** 4)
    parser.add_argument('--agent', type=str, default='DoubleIQN',
                        choices=['IQN', 'DoubleIQN'])
    parser.add_argument('--prioritized', action='store_true', default=False,
                        help='Flag to use a prioritized replay buffer')
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--batch-accumulator', type=str, default='mean',
                        choices=['mean', 'sum'])
    # IQN arguments
    parser.add_argument('--quantile-thresholds-N', type=int, default=64)
    parser.add_argument('--quantile-thresholds-N-prime', type=int, default=64)
    parser.add_argument('--quantile-thresholds-K', type=int, default=32)
    parser.add_argument('--n-best-episodes', type=int, default=200)
    # Batch arguments
    parser.add_argument('--num-envs', type=int, default=1)

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

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    env = atari_wrappers.wrap_deepmind(
            score_mask_atari.make_atari(args.env, max_frames=args.max_frames,
                                        mask_render=args.mask_render),
            episode_life=True,
            clip_rewards=True,
            frame_stack=True,
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
    trex_network = TREXNet()
    if args.load_trex:
        from chainer import serializers
        serializers.load_npz(args.load_trex, trex_network)
    trex_reward = TREXReward(ranked_demos=demo_dataset,
                     steps=args.trex_steps,
                     network=trex_network,
                     train_network=(False if args.load_trex else True),
                     gpu=args.gpu,
                     outdir=args.outdir,
                     phi=phi,
                     save_network=True)

    def make_env(idx, test):
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
            episode_life=not test,
            clip_rewards=not test,
            frame_stack=False,
        )
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = chainerrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        env.seed(env_seed)
        if args.monitor:
            env = chainerrl.wrappers.Monitor(
                env, args.outdir,
                mode='evaluation' if test else 'training')
        if args.render:
            env = chainerrl.wrappers.Render(env)
        return env

    def make_batch_env(test):
        vec_env = chainerrl.envs.MultiprocessVectorEnv(
                  [functools.partial(make_env, idx, test)
                   for idx, env in enumerate(range(args.num_envs))])
        if test:
            vec_env = chainerrl.wrappers.VectorFrameStack(vec_env, 4)
        else:
            vec_env = TREXVectorEnv(vec_env, 4, 0, trex_reward)
        return vec_env

    sample_env = make_env(0, test=False)

    n_actions = sample_env.action_space.n
    q_func = chainerrl.agents.iqn.ImplicitQuantileQFunction(
        psi=chainerrl.links.Sequence(
            L.Convolution2D(None, 32, 8, stride=4),
            F.relu,
            L.Convolution2D(None, 64, 4, stride=2),
            F.relu,
            L.Convolution2D(None, 64, 3, stride=1),
            F.relu,
            functools.partial(F.reshape, shape=(-1, 3136)),
        ),
        phi=chainerrl.links.Sequence(
            chainerrl.agents.iqn.CosineBasisLinear(64, 3136),
            F.relu,
        ),
        f=chainerrl.links.Sequence(
            L.Linear(None, 512),
            F.relu,
            L.Linear(None, n_actions),
        ),
    )

    # Draw the computational graph and save it in the output directory.
    fake_obss = np.zeros((4, 84, 84), dtype=np.float32)[None]
    fake_taus = np.zeros(32, dtype=np.float32)[None]
    chainerrl.misc.draw_computational_graph(
        [q_func(fake_obss)(fake_taus)],
        os.path.join(args.outdir, 'model'))

    # Use the same hyper parameters as https://arxiv.org/abs/1710.10044
    opt = chainer.optimizers.Adam(5e-5, eps=1e-2 / args.batch_size)
    opt.setup(q_func)

    # Select a replay buffer to use
    if args.prioritized:
        betasteps = args.steps / args.update_interval
        rbuf = replay_buffer.PrioritizedReplayBuffer(
            10 ** 6, alpha=0.5, beta0=0.4, betasteps=betasteps,
            num_steps=args.num_step_return)
    else:
        rbuf = replay_buffer.ReplayBuffer(
            10 ** 6,
            num_steps=args.num_step_return)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0, args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions))

    Agent = parse_agent(args.agent)
    agent = Agent(
        q_func, opt, rbuf, gpu=args.gpu, gamma=0.99,
        explorer=explorer, replay_start_size=args.replay_start_size,
        target_update_interval=args.target_update_interval,
        update_interval=args.update_interval,
        batch_accumulator=args.batch_accumulator,
        phi=phi,
        quantile_thresholds_N=args.quantile_thresholds_N,
        quantile_thresholds_N_prime=args.quantile_thresholds_N_prime,
        quantile_thresholds_K=args.quantile_thresholds_K,
    )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_batch_with_evaluation(
            agent=agent,
            env=make_batch_env(test=False),
            eval_env=make_batch_env(test=True),
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            log_interval=1000,
        )

        dir_of_best_network = os.path.join(args.outdir, "best")
        agent.load(dir_of_best_network)

        # run 200 evaluation episodes, each capped at 30 mins of play
        stats = experiments.evaluator.eval_performance(
            env=make_batch_env(test=True),
            agent=agent,
            n_steps=None,
            n_episodes=args.n_best_episodes,
            max_episode_len=args.max_frames / 4,
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

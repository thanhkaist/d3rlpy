import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

from torch import multiprocessing as mp

import numpy as np

import gym
from tqdm import tqdm
import time
import copy
import os


from d3rlpy.adversarial_training.utility import make_checkpoint_list, EvalLogger
from d3rlpy.adversarial_training.eval_utility import eval_clean_env, eval_env_under_attack


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=100)

SUPPORTED_TRANSFORMS = ['random', 'adversarial_training']
parser.add_argument('--transform', type=str, default='random', choices=SUPPORTED_TRANSFORMS)

SUPPORTED_ATTACKS = ['random', 'critic_normal', 'actor_mad']
parser.add_argument('--attack_type', type=str, default='random', choices=SUPPORTED_ATTACKS)
parser.add_argument('--attack_epsilon', type=float, default=None)
parser.add_argument('--attack_type_list', type=str, default='random', nargs='+')
parser.add_argument('--attack_epsilon_list', type=float, default=1e-4, nargs='+')
parser.add_argument('--attack_iteration', type=int, default=10)

parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--n_seeds_want_to_test', type=int, default=1)
parser.add_argument('--ckpt_steps', type=str, default="model_500000.pt")

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

parser.add_argument('--eval_logdir', type=str, default='eval_results')
args = parser.parse_args()


def eval_multiprocess_wrapper(algo, func, env_list, params):
    n_trials_per_each = int(params.n_eval_episodes / params.n_processes)
    n_trials_for_last = n_trials_per_each if params.n_eval_episodes % params.n_processes == 0 else \
        n_trials_per_each + params.n_eval_episodes % params.n_processes

    args_list = []
    for i in range(params.n_processes):
        params_tmp = copy.deepcopy(params)

        if i == params_tmp.n_processes - 1:  # last iteration
            params_tmp.n_eval_episodes = n_trials_for_last
        else:
            params_tmp.n_eval_episodes = n_trials_per_each

        start_seed = n_trials_per_each * i + 1
        args_list.append((i, algo, env_list[i], start_seed, params_tmp))

    with mp.Pool(params.n_processes) as pool:
        unorm_score = pool.map(func, args_list)
        unorm_score = np.mean(unorm_score)

    return unorm_score


def eval_func(algo, writer, env, attack_type, attack_epsilon, disable_clean):
    args_clone = copy.deepcopy(args)
    args_clone.attack_type = attack_type
    args_clone.attack_epsilon = attack_epsilon
    args_clone.disable_clean = disable_clean

    if attack_type in ['critic_normal']:
        args_clone.attack_iteration = 5
    elif attack_type in ['actor_mad']:
        args_clone.attack_iteration = 5
    else:
        args_clone.attack_iteration = 1

    unorm_score, norm_score, unorm_score_noise, norm_score_noise = None, None, None, None
    if not args_clone.mp:
        print('[INFO] Normally evaluating...')
        func_args = (0, algo, env, args_clone.seed, args_clone)  # algo, env, start_seed, args

        if not args_clone.disable_clean:
            unorm_score = eval_clean_env(func_args)
        unorm_score_noise = eval_env_under_attack(func_args)

    else:
        print('[INFO] Multiple-processing evaluating...')
        # start = time.time()
        env_list = []
        env_list.append(env)
        for i in range(args_clone.n_processes - 1):
            _env = gym.make(args_clone.dataset)
            _env.seed(args_clone.seed)
            env_list.append(_env)
        if not args_clone.disable_clean:
            unorm_score = eval_multiprocess_wrapper(algo, eval_clean_env, env_list, args_clone)
        unorm_score_noise = eval_multiprocess_wrapper(algo, eval_env_under_attack, env_list, args_clone)

        del env_list

    if not args_clone.disable_clean:
        norm_score = env.env.wrapped_env.get_normalized_score(unorm_score) * 100
        writer.log(attack_type="clean", attack_epsilon=attack_epsilon,
                   attack_iteration=args_clone.attack_iteration,
                   unorm_score=unorm_score, norm_score=norm_score)
    norm_score_noise = env.env.wrapped_env.get_normalized_score(unorm_score_noise) * 100

    writer.log(attack_type=attack_type, attack_epsilon=attack_epsilon,
               attack_iteration=args_clone.attack_iteration,
               unorm_score=unorm_score_noise, norm_score=norm_score_noise)

    print("***** Env: %s - method: %s *****" % (args_clone.dataset, args_clone.ckpt.split('/')[-3]))
    if unorm_score is not None:
        print("Clean env: unorm = %.3f, norm = %.2f" % (unorm_score, norm_score))
    print("Noise env: unorm = %.3f, norm = %.2f" % (unorm_score_noise, norm_score_noise))


def main(args):
    if not os.path.exists(args.eval_logdir):
        os.makedirs(args.eval_logdir)

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    ### Initialize algorithm
    td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu, env_name=args.dataset)
    td3.build_with_env(env)  # Create policy/critic w.r.t. env

    ### Convert dataset to list of transitions to compute mean & std
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions
    td3._scaler.fit(transitions)  # Compute mean & std of dataset

    ckpt_list = make_checkpoint_list(args)

    # Scan through all checkpoints
    for checkpoint in ckpt_list[:args.n_seeds_want_to_test]:
        args.ckpt = checkpoint
        print('Evaluating: ', args.ckpt)
        writer = EvalLogger(args)

        td3.load_model(checkpoint)

        # if args.attack_type.startswith('sarsa'):
        #     td3 = train_sarsa(td3, env)

        start = time.time()

        for attack_type in args.attack_type_list:
            for i, attack_epsilon in enumerate(args.attack_epsilon_list):
                if i == 0:
                    disable_clean = args.disable_clean
                else:
                    disable_clean = True

                eval_func(td3, writer, env, attack_type, attack_epsilon, disable_clean)

        writer.close()
        print("=> Time(s) for evaluation (1 seed): %.3f" % (time.time() - start))


if __name__ == '__main__':
    if args.mp:
        mp.set_start_method("spawn")
    main(args)


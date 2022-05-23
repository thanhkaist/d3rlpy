import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

import multiprocessing as mp
from torch import multiprocessing
from tqdm import tqdm
import numpy as np
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=100)

parser.add_argument('--noise_test', type=str, default='uniform')
parser.add_argument('--noise_test_eps', type=float, default=1e-4)

SUPPORTED_TRANSFORMS = ['gaussian', 'adversarial_training']
parser.add_argument('--transform', type=str, default='gaussian', choices=SUPPORTED_TRANSFORMS)
parser.add_argument('--epsilon', type=float, default=3e-4)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--step_size', type=float, default=2.5e-5)
parser.add_argument('--norm_min_max', action='store_true')
parser.add_argument('--adv_version', type=str, default='a1_d1')

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)
args = parser.parse_args()


def eval_clean_env(args_):
    algo, env, start_seed, args = args_
    n_trials = args.n_eval_episodes

    episode_rewards = []
    for i in tqdm(range(n_trials)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        observation = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([observation])[0]

            observation, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_noise_env_v1(args_):
    algo, env, start_seed, args = args_
    n_trials = args.n_eval_episodes

    def add_noise(o, noise_type, eps_noise):
        if noise_type == 'gaussian':
            o = o + np.random.randn(o.shape[0]) * eps_noise
        elif noise_type == 'uniform':
            o = o + np.random.uniform(-eps_noise, eps_noise, size=o.shape[0])
        else:
            raise NotImplementedError
        return o

    episode_rewards = []
    for i in tqdm(range(n_trials)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        observation = env.reset()
        observation = add_noise(observation, args.noise_test, args.noise_test_eps)
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([observation])[0]

            observation, reward, done, _ = env.step(action)
            observation = add_noise(observation, args.noise_test, args.noise_test_eps)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_multiprocess_wrapper(algo, func, env_list, args):
    n_trials_per_each = int(args.n_eval_episodes / args.n_processes)
    n_trials_for_last = n_trials_per_each if args.n_eval_episodes % args.n_processes == 0 else \
        n_trials_per_each + args.n_eval_episodes % args.n_processes

    args_list = []
    for i in range(args.n_processes):
        args_ = copy.deepcopy(args)

        if i == args.n_processes - 1:  # last iteration
            args_.n_eval_episodes = n_trials_for_last
        else:
            args_.n_eval_episodes = n_trials_per_each

        start_seed = n_trials_per_each * i + 1
        args_list.append((algo, env_list[i], start_seed, args_))

    with mp.Pool(args.n_processes) as pool:
        unorm_score = pool.map(func, args_list)
        unorm_score = np.mean(unorm_score)

    return unorm_score


def main(args):
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    ### Initialize algorithm
    td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu)
    td3.build_with_env(env)  # Create policy/critic w.r.t. env
    td3.load_model(args.ckpt)

    ### Convert dataset to list of transitions to compute mean & std
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions

    td3._scaler.fit(transitions)  # Compute mean & std of dataset

    if not args.mp:
        print('[INFO] Normally evaluating...')
        start = time.time()
        func_args = (td3, env, args.seed, args)      # algo, env, start_seed, args

        if not args.disable_clean:
            unorm_score = eval_clean_env(func_args)
        unorm_score_noise = eval_noise_env_v1(func_args)

    else:
        print('[INFO] Multiple-processing evaluating...')
        start = time.time()
        env_list = []
        env_list.append(env)
        for i in range(args.n_processes - 1):
            _, _env = d3rlpy.datasets.get_dataset(args.dataset)
            _env.seed(args.seed)
            env_list.append(_env)
        if not args.disable_clean:
            unorm_score = eval_multiprocess_wrapper(td3, eval_clean_env, env_list, args)
        unorm_score_noise = eval_multiprocess_wrapper(td3, eval_noise_env_v1, env_list, args)

    if not args.disable_clean:
        norm_score = env.env.wrapped_env.get_normalized_score(unorm_score) * 100
    norm_score_noise = env.env.wrapped_env.get_normalized_score(unorm_score_noise) * 100

    print("***** Env: %s - method: %s *****" % (args.dataset, args.ckpt.split('/')[-3]))
    if not args.disable_clean:
        print("Clean env: unorm = %.3f, norm = %.2f" % (unorm_score, norm_score))
    print("Noise env: unorm = %.3f, norm = %.2f" % (unorm_score_noise, norm_score_noise))
    print("=> Time(s) for evaluation: %.3f" % (time.time() - start))



if __name__ == '__main__':
    if args.mp:
        multiprocessing.set_start_method("spawn")

    main(args)


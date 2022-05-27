import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp

import numpy as np

import gym
from tqdm import tqdm
import time
import copy
import os

from d3rlpy.models.torch.policies import WrapperBoundDeterministicPolicy
from d3rlpy.models.torch.q_functions.ensemble_q_function import WrapperBoundEnsembleContinuousQFunction
from d3rlpy.adversarial_training.attackers import critic_normal_attack, actor_mad_attack, random_attack
from d3rlpy.adversarial_training.utility import tensor


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--gpu', type=int)
parser.add_argument('--n_eval_episodes', type=int, default=100)

SUPPORTED_TRANSFORMS = ['gaussian', 'adversarial_training']
parser.add_argument('--transform', type=str, default='gaussian', choices=SUPPORTED_TRANSFORMS)
parser.add_argument('--epsilon', type=float, default=3e-4)
parser.add_argument('--num_steps', type=int, default=5)
parser.add_argument('--step_size', type=float, default=2.5e-5)
parser.add_argument('--norm_min_max', action='store_true')
parser.add_argument('--adv_version', type=str, default='a1_d1')

SUPPORTED_ATTACKS = ['random', 'critic', 'action', 'sarsa']
parser.add_argument('--attack_type', type=str, default='random', choices=SUPPORTED_ATTACKS)
parser.add_argument('--attack_epsilon', type=float, default=1e-4)
parser.add_argument('--attack_iteration', type=int, default=10)

parser.add_argument('--disable_clean', action='store_true')
parser.add_argument('--ckpt', type=str, default='.')
parser.add_argument('--mp', action='store_true')
parser.add_argument('--n_processes', type=int, default=5)

args = parser.parse_args()


"""
##### Functions used to evaluate
"""
def eval_clean_env(args_):
    algo, env, start_seed, args = args_
    n_trials = args.n_eval_episodes

    episode_rewards = []
    for i in tqdm(range(n_trials)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

    unorm_score = float(np.mean(episode_rewards))
    return unorm_score


def eval_env_under_attack(args_):
    algo, env, start_seed, args = args_
    n_trials = args.n_eval_episodes

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    attack_type = args.attack_type
    attack_epsilon = args.attack_epsilon
    attack_iteration = args.attack_iteration
    attack_stepsize = attack_epsilon / attack_iteration
    print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
          (args.attack_type.upper(), attack_epsilon, attack_iteration, attack_stepsize))

    state_min, state_max = algo._impl._obs_min.cpu().numpy(), algo._impl._obs_max.cpu().numpy()

    def attack(state, type, attack_epsilon=None, attack_iteration=None, attack_stepsize=None):
        if type in ['random']:
            ori_state_tensor = tensor(state, algo._impl.device)
            perturb_state = random_attack(
                ori_state_tensor, attack_epsilon,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        elif type in ['critic_normal']:
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            perturb_state = critic_normal_attack(
                ori_state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        elif type in ['actor_mad']:
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            perturb_state = actor_mad_attack(
                ori_state_tensor, algo._impl._policy, algo._impl._q_func,
                attack_epsilon, attack_iteration, attack_stepsize,
                algo._impl._obs_min, algo._impl._obs_max,
                algo.scaler
            )
            perturb_state = perturb_state.cpu().numpy()

        else:
            raise NotImplementedError
        return perturb_state

    episode_rewards = []
    for i in tqdm(range(n_trials)):
        if start_seed is None:
            env.seed(i)
        else:
            env.seed(start_seed + i)
        state = env.reset()

        state = attack(state, attack_type, attack_epsilon, attack_iteration, attack_stepsize)
        episode_reward = 0.0

        while True:
            # take action
            action = algo.predict([state])[0]

            state, reward, done, _ = env.step(action)
            state = attack(state, attack_type, attack_epsilon, attack_iteration, attack_stepsize)
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


def make_bound_for_network(algo):
    # For convex relaxation: This is tested for TD3, not guarantee to work with other algorithms
    algo._impl._q_func = WrapperBoundEnsembleContinuousQFunction(
        q_func=algo._impl._q_func,
        observation_shape=algo.observation_shape[0],
        action_shape=algo.action_size,
        device=algo._impl.device
    )
    algo._impl._targ_q_func = WrapperBoundEnsembleContinuousQFunction(
        q_func=algo._impl._targ_q_func,
        observation_shape=algo.observation_shape[0],
        action_shape=algo.action_size,
        device=algo._impl.device
    )

    algo._impl._policy = WrapperBoundDeterministicPolicy(
        policy=algo._impl._policy,
        observation_shape=algo.observation_shape[0],
        device=algo._impl.device
    )
    algo._impl._targ_policy = WrapperBoundDeterministicPolicy(
        policy=algo._impl._targ_policy,
        observation_shape=algo.observation_shape[0],
        device=algo._impl.device
    )

    return algo


def train_sarsa(algo, env, buffer=None, n_sarsa_steps=1000, n_warmups=1000):

    logdir_sarsa = os.path.join(args.ckpt[:args.ckpt.rfind('/')], 'sarsa_model')
    model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))


    algo = make_bound_for_network(algo)

    # We need to re-initialize the critic, not using the old one (following SA-DDPG)
    algo._impl._q_func.reset_weight()
    algo._impl._targ_q_func.reset_weight()

    if not os.path.exists(logdir_sarsa):
        os.mkdir(logdir_sarsa)
    if os.path.exists(model_path):
        print('Found pretrained SARSA: ', model_path)
        algo.load_model(model_path)
    else:
        print('Not found pretrained SARSA: ', model_path)
        algo.fit_sarsa(env, buffer, n_sarsa_steps, n_warmups)
        algo.save_model(model_path)

    return algo


def main(args):
    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    ### Initialize algorithm
    td3 = d3rlpy.algos.TD3PlusBC(scaler="standard", use_gpu=args.gpu, env_name=args.dataset)
    td3.build_with_env(env)  # Create policy/critic w.r.t. env
    td3.load_model(args.ckpt)


    ### Convert dataset to list of transitions to compute mean & std
    transitions = []
    for episode in dataset.episodes:
        transitions += episode.transitions

    td3._scaler.fit(transitions)  # Compute mean & std of dataset

    if args.attack_type.startswith('sarsa'):
        td3 = train_sarsa(td3, env)


    if not args.mp:
        print('[INFO] Normally evaluating...')
        start = time.time()
        func_args = (td3, env, args.seed, args)      # algo, env, start_seed, args

        if not args.disable_clean:
            unorm_score = eval_clean_env(func_args)
        unorm_score_noise = eval_env_under_attack(func_args)

    else:
        print('[INFO] Multiple-processing evaluating...')
        start = time.time()
        env_list = []
        env_list.append(env)
        for i in range(args.n_processes - 1):
            _env = gym.make(args.dataset)
            _env.seed(args.seed)
            env_list.append(_env)
        if not args.disable_clean:
            unorm_score = eval_multiprocess_wrapper(td3, eval_clean_env, env_list, args)
        unorm_score_noise = eval_multiprocess_wrapper(td3, eval_env_under_attack, env_list, args)

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
        mp.set_start_method("spawn")
    main(args)


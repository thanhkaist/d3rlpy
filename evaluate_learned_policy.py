import argparse
import d3rlpy
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F
from torch import multiprocessing
import multiprocessing as mp
import numpy as np

import gym
from tqdm import tqdm
import time
import copy
import os

from auto_LiRPA.bound_ops import BoundParams


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
##### Utility
"""
def standardization(x, mean, std, eps=1e-3):
    return (x - mean) / (std + eps)

def reverse_standardization(x, mean, std, eps=1e-3):
    return ((std + eps) * x) + mean

def tensor(x, device='cpu'):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x


def clamp(x, vec_min, vec_max):
    if isinstance(vec_min, list):
        vec_min = torch.Tensor(vec_min).to(x.device)
    if isinstance(vec_max, list):
        vec_max = torch.Tensor(vec_max).to(x.device)

    assert isinstance(vec_min, torch.Tensor) and isinstance(vec_max, torch.Tensor)
    x = torch.max(x, vec_min)
    x = torch.min(x, vec_max)
    return x



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
        if type == 'random':
            perturb_state = standardization(state, algo.scaler._mean, algo.scaler._std)
            noise = np.random.uniform(-attack_epsilon, attack_epsilon, size=state.shape[0])
            perturb_state = perturb_state + noise
            perturb_state = reverse_standardization(perturb_state, algo.scaler._mean, algo.scaler._std)
            perturb_state = np.clip(perturb_state, state_min, state_max)

        elif type == 'critic':
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            adv_x = ori_state_tensor.clone().detach()

            adv_x = algo.scaler.transform(adv_x)                        # normalized
            noise = torch.zeros_like(adv_x).uniform_(-attack_epsilon, attack_epsilon)
            adv_x = adv_x + noise                                       # already normalized

            ori_state_tensor = algo.scaler.transform(ori_state_tensor)  # normalize original state
            for _ in range(attack_iteration):
                adv_x_clone = adv_x.clone().detach()    # normalized
                adv_x.requires_grad = True

                # adv_x = algo.scaler.transform(adv_x)
                action = algo._impl._policy(adv_x)
                qval = algo._impl._q_func(ori_state_tensor, action, "none")[0]

                cost = -qval.mean()

                grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

                # adv_x = adv_x_clone + attack_stepsize * torch.sign(grad)  # Not good enough
                adv_x = adv_x_clone + attack_stepsize * grad

                delta = torch.clamp(adv_x - adv_x_clone, min=-attack_epsilon, max=attack_epsilon)
                adv_x = adv_x_clone + delta     # This is adversarial example

                # This clamp is performed in ORIGINAL scale
                adv_x = algo.scaler.reverse_transform(adv_x)
                adv_x = clamp(adv_x, algo._impl._obs_min, algo._impl._obs_max)
                adv_x = algo.scaler.transform(adv_x)

            perturb_state = algo.scaler.reverse_transform(adv_x).cpu().numpy()

        elif type == 'action':
            ori_state_tensor = tensor(state, algo._impl.device)         # original, unnormalized
            ori_state_tensor = algo.scaler.transform(ori_state_tensor)  # normalized

            with torch.no_grad():
                gt_action = algo._impl._policy(ori_state_tensor).clone().detach()  # ground truth

            adv_x = ori_state_tensor.clone().detach()                   # already normalized

            noise = torch.zeros_like(adv_x).uniform_(-attack_epsilon, attack_epsilon)
            adv_x = adv_x + noise

            for _ in range(attack_iteration):
                adv_x_clone = adv_x.clone().detach()    # normalized
                adv_x.requires_grad = True

                adv_a = algo._impl._policy(adv_x)

                cost = F.mse_loss(adv_a, gt_action)

                grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

                adv_x = adv_x_clone + attack_stepsize * grad

                delta = torch.clamp(adv_x - adv_x_clone, min=-attack_epsilon, max=attack_epsilon)
                adv_x = adv_x_clone + delta     # This is adversarial example

                # This clamp is performed in ORIGINAL scale
                adv_x = algo.scaler.reverse_transform(adv_x)
                adv_x = clamp(adv_x, algo._impl._obs_min, algo._impl._obs_max)
                adv_x = algo.scaler.transform(adv_x)

            perturb_state = algo.scaler.reverse_transform(adv_x).cpu().numpy()

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


def train_sarsa(algo, env, buffer=None, n_sarsa_steps=30000, n_warmups=100000):
    import torch.nn as nn

    logdir_sarsa = os.path.join(args.ckpt[:args.ckpt.rfind('/')], 'sarsa_model')
    model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))

    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
        if isinstance(m, BoundParams):
            params = m.forward_value
            if params.ndim == 2:
                torch.nn.init.kaiming_uniform_(params, a=np.sqrt(5))
            else:
                torch.nn.init.normal_(params)

    # We need to re-initialize the critic, not using the old one (following SA-DDPG)
    algo._impl._q_func.apply(weight_reset)
    algo._impl._targ_q_func.apply(weight_reset)

    # if not os.path.exists(logdir_sarsa):
    #     os.mkdir(logdir_sarsa)
    # if os.path.exists(model_path):
    #     algo.load_model(model_path)
    # else:

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

    exit()

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
            _env = env = gym.make(args.dataset)
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
        multiprocessing.set_start_method("spawn")

    main(args)


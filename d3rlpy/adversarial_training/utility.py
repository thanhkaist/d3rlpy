import os
import json
import time
from tqdm import tqdm

import torch

import numpy as np


from d3rlpy.models.torch.policies import WrapperBoundDeterministicPolicy
from d3rlpy.models.torch.q_functions.ensemble_q_function import WrapperBoundEnsembleContinuousQFunction
from d3rlpy.adversarial_training.attackers import critic_normal_attack, actor_mad_attack, random_attack

ENV_OBS_RANGE = {
    'walker2d-v0': dict(
        max=[1.8164345026016235, 0.999911367893219, 0.5447346568107605, 0.7205190062522888,
             1.5128496885299683, 0.49508699774742126, 0.6822911500930786, 1.4933640956878662,
             9.373093605041504, 5.691765308380127, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        min=[0.800006091594696, -0.9999997019767761, -3.006617546081543, -2.9548180103302,
             -1.72023344039917, -2.9515464305877686, -3.0064914226531982, -1.7654582262039185,
             -6.7458906173706055, -8.700752258300781, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0,
             -10.0]
    ),
    'hopper-v0': dict(
        max=[1.7113906145095825, 0.1999576985836029, 0.046206455677747726, 0.10726844519376755,
             0.9587112665176392, 5.919354438781738, 3.04956316947937, 6.732881546020508,
             7.7671966552734375, 10.0, 10.0],
        min=[0.7000009417533875, -0.1999843567609787, -1.540910243988037, -1.1928397417068481,
             -0.9543644189834595, -1.6949318647384644, -5.237359523773193, -6.2852582931518555,
             -10.0, -10.0, -10.0]
    ),
    'halfcheetah-v0': dict(
        max=[1.600443959236145,22.812137603759766, 1.151809811592102, 0.949776291847229,
             0.9498141407966614, 0.8997246026992798, 1.1168793439865112, 0.7931482791900635,
             16.50477409362793, 5.933143138885498, 13.600515365600586, 27.84033203125,
             30.474760055541992, 30.78533935546875, 30.62249755859375, 37.273799896240234,
             31.570491790771484],
        min=[-0.6028550267219543, -3.561767339706421, -0.7100794315338135, -1.0610754489898682,
             -0.6364201903343201, -1.2164583206176758, -1.2236766815185547, -0.7376371026039124,
             -3.824833869934082, -5.614060878753662, -12.930273056030273, -29.38336944580078,
             -31.534399032592773, -27.823902130126953, -32.996246337890625, -30.887380599975586,
             -30.41145896911621]
    ),
}

ENV_NAME_MAPPING = {
    'walker2d-random-v0': 'w2d-r',
    'walker2d-medium-v0': 'w2d-m',
    'walker2d-medium-replay-v0': 'w2d-m-re',
    'walker2d-medium-expert-v0': 'w2d-m-e',
    'walker2d-expert-v0': 'w2d-e',
    'hopper-random-v0': 'hop-r',
    'hopper-medium-v0': 'hop-m',
    'hopper-medium-replay-v0': 'hop-m-re',
    'hopper-medium-expert-v0': 'hop-m-e',
    'hopper-expert-v0': 'hop-e',
    'halfcheetah-random-v0': 'che-r',
    'halfcheetah-medium-v0': 'che-m',
    'halfcheetah-medium-replay-v0': 'che-m-re',
    'halfcheetah-medium-expert-v0': 'che-m-e',
    'halfcheetah-expert-v0': 'che-e',
    'unknown': 'unknown'
}



def standardization(x, mean, std, eps=1e-3):
    return (x - mean) / (std + eps)

def reverse_standardization(x, mean, std, eps=1e-3):
    return ((std + eps) * x) + mean

def normalize(x, min, max):
    x = (x - min)/(max - min)
    return x


def denormalize(x, min, max):
    x = x * (max - min) + min
    return x

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


def make_checkpoint_list(main_args):
    if os.path.isfile(main_args.ckpt):
        assert main_args.n_seeds_want_to_test == 1
        ckpt_list = [main_args.ckpt]
    elif os.path.isdir(main_args.ckpt):
        entries = os.listdir(main_args.ckpt)
        entries.sort()
        ckpt_list = []
        for entry in entries:
            ckpt_file = os.path.join(main_args.ckpt, entry, main_args.ckpt_steps)
            assert os.path.isfile(ckpt_file), \
                "Cannot file checkpoint {} in {}".format(main_args.ckpt_steps, ckpt_file)
            ckpt_list.append(ckpt_file)
    else:
        print("Path doesn't exist: ", main_args.ckpt)
        raise ValueError

    return ckpt_list


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


"""
##### Functions used to evaluate
"""
def eval_clean_env(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0)):
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


def eval_env_under_attack(params):
    rank, algo, env, start_seed, params = params
    n_trials = params.n_eval_episodes

    # Set seed
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    attack_type = params.attack_type
    attack_epsilon = params.attack_epsilon
    if attack_type in ['critic_normal']:
        attack_iteration = 5
    elif attack_type in ['actor_mad']:
        attack_iteration = 5
    else:
        attack_iteration = 1
    attack_stepsize = attack_epsilon / attack_iteration
    if rank == 0:
        print("[INFO] Using %s attack: eps=%f, n_iters=%d, sz=%f" %
              (params.attack_type.upper(), attack_epsilon, attack_iteration, attack_stepsize))

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
        return perturb_state.squeeze()

    episode_rewards = []
    for i in tqdm(range(n_trials), disable=(rank != 0)):
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


# def train_sarsa(algo, env, buffer=None, n_sarsa_steps=1000, n_warmups=1000):
#
#     logdir_sarsa = os.path.join(args.ckpt[:args.ckpt.rfind('/')], 'sarsa_model')
#     model_path = os.path.join(logdir_sarsa, 'sarsa_ntrains{}_warmup{}.pt'.format(n_sarsa_steps, n_warmups))
#
#
#     algo = make_bound_for_network(algo)
#
#     # We need to re-initialize the critic, not using the old one (following SA-DDPG)
#     algo._impl._q_func.reset_weight()
#     algo._impl._targ_q_func.reset_weight()
#
#     if not os.path.exists(logdir_sarsa):
#         os.mkdir(logdir_sarsa)
#     if os.path.exists(model_path):
#         print('Found pretrained SARSA: ', model_path)
#         algo.load_model(model_path)
#     else:
#         print('Not found pretrained SARSA: ', model_path)
#         algo.fit_sarsa(env, buffer, n_sarsa_steps, n_warmups)
#         algo.save_model(model_path)
#
#     return algo


class EvalLogger():
    def __init__(self, args):

        # Extract required information
        checkpoint_step = args.ckpt.split('/')[-1]
        exp_path = args.ckpt.replace(checkpoint_step, '')

        f = open(os.path.join(exp_path, "params.json"))
        exp_params = json.load(f)

        timestamp = time.localtime()
        timestamp = time.strftime("%m_%d-%H_%M_%S", timestamp)

        self.ckpt = args.ckpt
        self.exp_params = exp_params

        self.exp_name = args.ckpt.split('/')[-3]
        self.exp_name_with_seed = args.ckpt.split('/')[-2]
        self.checkpoint_step = checkpoint_step

        if 'env_name' in exp_params.keys():
            self.env_name = exp_params['env_name']
        else:
            self.env_name = 'unknown'

        self.filename = "eval_" + ENV_NAME_MAPPING[self.env_name] + '_' + self.exp_name + '_' + timestamp + '.txt'
        self.logfile = os.path.join(args.eval_logdir, self.filename)

        self.writer = open(self.logfile, "w")
        self.write_header()

    def write_header(self):
        self.writer.write("********* ATTACK EVALUATION *********\n\n")

        self.writer.write("Environment: %s\n" % (self.env_name))
        self.writer.write("Experiment name: %s\n" % (self.exp_name))
        self.writer.write("Experiment name with timestamp: %s\n" % (self.exp_name_with_seed))
        self.writer.write("Checkpoint step: %s\n" % (self.checkpoint_step))
        self.writer.write("Full path: %s\n" % (self.ckpt))
        self.writer.write("\n")
        if 'transform_params' in self.exp_params:
            self.writer.write("TRAINING PARAMS: \n")
            self.writer.write("Attack type: %s\n" % (self.exp_params['transform_params']['attack_type']))
            self.writer.write("\t epsilon: %s\n" % (self.exp_params['transform_params']['epsilon']))
            self.writer.write("\t num_steps: %s\n" % (self.exp_params['transform_params']['num_steps']))
            self.writer.write("\t step_size: %s\n" % (self.exp_params['transform_params']['step_size']))
            self.writer.write("Robust type: %s\n" % (self.exp_params['transform_params']['robust_type']))
            self.writer.write("\t critic_reg_coef: %s\n" % (self.exp_params['transform_params']['critic_reg_coef']))
            self.writer.write("\t actor_reg_coef: %s\n" % (self.exp_params['transform_params']['actor_reg_coef']))
            self.writer.write("\n")
            self.writer.write("\n")

    def log(self, attack_type, attack_epsilon, attack_iteration, unorm_score, norm_score):
        if attack_type == 'clean':
            self.writer.write("Clean performance: unorm_score=%.3f, norm = %.2f\n" %
                              (unorm_score, norm_score)
                              )
        else:
            self.writer.write("Attack=%s - epsilon=%.4f, n_iters=%d: unorm_score=%.3f, norm = %.2f\n"
                              % (attack_type.upper(), attack_epsilon, attack_iteration,
                                 unorm_score, norm_score)
                              )

    def close(self):
        self.writer.close()


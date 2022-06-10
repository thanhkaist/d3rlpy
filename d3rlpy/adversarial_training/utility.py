import os
import json
import time
import shutil
from d4rl import infos

import torch

import numpy as np


from ..models.torch.policies import WrapperBoundDeterministicPolicy
from ..models.torch.q_functions.ensemble_q_function import WrapperBoundEnsembleContinuousQFunction


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
    dtype = torch.uint8 if x.dtype == np.uint8 else torch.float32
    tensor = torch.tensor(data=x, dtype=dtype, device=device)
    return tensor.float()


def clamp(x, vec_min, vec_max):
    if isinstance(vec_min, list):
        vec_min = torch.Tensor(vec_min).to(x.device)
    if isinstance(vec_max, list):
        vec_max = torch.Tensor(vec_max).to(x.device)

    assert isinstance(vec_min, torch.Tensor) and isinstance(vec_max, torch.Tensor)
    x = torch.max(x, vec_min)
    x = torch.min(x, vec_max)
    return x


def copy_file(src, des):
    try:
        shutil.copy(src, des)
        print("File copied successfully.")

    # If source and destination are same
    except shutil.SameFileError:
        print("Source and destination represents the same file.")

    # If there is any permission issue
    except PermissionError:
        print("Permission denied.")

    # For other errors
    except:
        print("Error occurred while copying file.")


def make_checkpoint_list(ckpt_path, n_seeds_want_to_test, ckpt_steps):
    print("[INFO] Finding checkpoints...")
    if os.path.isfile(ckpt_path):
        assert n_seeds_want_to_test == 1
        ckpt_list = [ckpt_path]
        print("\tEvaluating with single checkpoint.")
    elif os.path.isdir(ckpt_path):
        entries = os.listdir(ckpt_path)
        entries.sort()
        print("\tFound %d experiments." % (len(entries)))
        ckpt_list = []
        for entry in entries:
            ckpt_file = os.path.join(ckpt_path, entry, ckpt_steps)
            if not os.path.isfile(ckpt_file):
                print("\tCannot find checkpoint {} in {}".format(ckpt_steps, ckpt_file))
            else:
                ckpt_list.append(ckpt_file)

        print('\tFound {} checkpoints.'.format(len(ckpt_list)))
        if len(ckpt_list) < n_seeds_want_to_test:
            print("\tWARNING: Number of found checkpoints less than requirement")
    else:
        print("\tPath doesn't exist: ", ckpt_path)
        raise ValueError

    return ckpt_list


def count_num_seeds_in_path(ckpt_path, ckpt_steps):
    assert os.path.isdir(ckpt_path)
    entries = os.listdir(ckpt_path)
    entries.sort()
    ckpt_list = []
    for entry in entries:
        ckpt_file = os.path.join(ckpt_path, entry, ckpt_steps)
        if not os.path.isfile(ckpt_file):
            print("[WARNING] Cannot find checkpoint {} in {}".format(ckpt_steps, ckpt_file))
        else:
            ckpt_list.append(ckpt_file)

    return len(ckpt_list)


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


class EvalLogger():
    def __init__(self, ckpt, eval_logdir, prefix="eval"):

        # Extract required information
        assert os.path.isfile(ckpt)

        self.init_info_from_ckpt(ckpt)

        # Construct new writer
        timestamp = time.localtime()
        timestamp = time.strftime("%m_%d-%H_%M_%S", timestamp)
        self.filename = prefix + '_' +  ENV_NAME_MAPPING[self.env_name] + \
                        '_' + self.exp_name + '_' + timestamp + '.txt'
        self.logfile = os.path.join(eval_logdir, self.filename)
        self.writer = open(self.logfile, "w")

        self.write_header()

    def init_info_from_ckpt(self, ckpt):
        checkpoint_step = ckpt.split('/')[-1]
        exp_path = ckpt.replace(checkpoint_step, '')

        f = open(os.path.join(exp_path, "params.json"))
        exp_params = json.load(f)

        self.ckpt = ckpt
        self.exp_params = exp_params

        self.exp_name = ckpt.split('/')[-3]
        self.exp_name_with_seed = ckpt.split('/')[-2]
        self.checkpoint_step = checkpoint_step

        if 'env_name' in exp_params.keys():
            self.env_name = exp_params['env_name']
        else:
            self.env_name = None
            # Attempt to get name of env from checkpoint's path
            _ckpt_paths = ckpt.split('/')
            for _ckpt in _ckpt_paths:
                if _ckpt in list(infos.DATASET_URLS.keys()):
                    self.env_name = _ckpt
            # If still cannot find name of env
            if self.env_name is None:
                self.env_name = 'unknown'

    def write_header(self):
        self.writer.write("\n\n********* ATTACK EVALUATION *********\n\n")

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
    def print(self, text):
        print(text)
        self.writer.write(text)

    def append_to_previous_writer(self, ckpt):
        self.init_info_from_ckpt(ckpt)
        self.write_header()

    def close(self):
        self.writer.close()


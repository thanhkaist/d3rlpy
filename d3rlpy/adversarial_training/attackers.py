import torch
import torch.nn.functional as F

import numpy as np

from .utility import clamp

def preprocess_state(x):
    if len(x.shape) == 1:
        x = x.reshape((1,) + x.shape)

    assert len(x.shape) == 2, "Currently only support the low-dimensional state"
    return x

def random_attack(x, epsilon, _obs_min_norm, _obs_max_norm, clip=True, use_assert=True):
    """" NOTE: x must be normalized """""
    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())
    noise = torch.zeros_like(ori_x).uniform_(-epsilon, epsilon)
    adv_x = ori_x + noise

    # This clamp is performed in normalized scale
    if clip:
        adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm)

    perturbed_state = adv_x  # already normalized

    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"

    return adv_x


def critic_normal_attack(x, _policy, _q_func, epsilon, num_steps, step_size,
                         _obs_min_norm, _obs_max_norm,
                         q_func_id=0, optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                   # already normalized

    adv_x = ori_x.clone().detach()               # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            action = _policy(adv_x)
            qval = _q_func(ori_x, action, "none")[q_func_id]

            cost = -qval.mean()

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta       # This is adversarial example

            # This clamp is performed in normalized scale
            if clip:
                adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError
    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
            epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state


def critic_mqd_attack(x, a, _policy, _q_func, epsilon, num_steps, step_size, _obs_min_norm, _obs_max_norm,
                      q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                              # already normalized

    with torch.no_grad():
        gt_qval = _q_func(ori_x, a, "none")[q_func_id].detach()

    adv_x = ori_x.clone().detach()                          # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            qval = _q_func(adv_x, a, "none")[q_func_id]

            cost = F.mse_loss(qval, gt_qval)

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta

            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    return perturbed_state


def actor_mad_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min_norm, _obs_max_norm,
                     optimizer='pgd', clip=True, use_assert=True):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x.clone().detach())                           # already normalized

    adv_x = ori_x.clone().detach()  # already normalized

    with torch.no_grad():
        gt_action = _policy(ori_x).detach()              # ground truth

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                          # Add noise in `normalized space`

    adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            adv_a = _policy(adv_x)

            cost = F.mse_loss(adv_a, gt_action)

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta         # This is adversarial example

            if clip:
                # This clamp is performed in normalized scale
                adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError
    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    if use_assert:
        assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (
                epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}\n Origin: {ori_x.cpu()}, perturb: {perturbed_state.cpu()}"
    return perturbed_state

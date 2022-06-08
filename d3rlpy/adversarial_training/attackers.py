import torch
import torch.nn.functional as F

import numpy as np

from .utility import clamp


def random_attack(x, epsilon, _obs_min, _obs_max, scaler):
    """" NOTE: x must be un-normalized """""
    assert isinstance(x, torch.Tensor), "input x must be tensor."
    adv_x = scaler.transform(x)     # normalize original state
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise

    # This clamp is performed in ORIGINAL scale
    adv_x = scaler.reverse_transform(adv_x)
    adv_x = clamp(adv_x, _obs_min, _obs_max).detach()
    # adv_x = scaler.transform(adv_x)   # No need

    return adv_x


def critic_normal_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                         scaler=None):
    """" NOTE: x must be un-normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_state_tensor = x.clone().detach()
    ori_state_tensor = scaler.transform(ori_state_tensor)   # normalize original state

    adv_x = ori_state_tensor.clone().detach()               # already normalized

    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    # TODO: Optimized with PGD
    for _ in range(num_steps):
        adv_x_clone = adv_x.clone().detach()
        adv_x.requires_grad = True

        action = _policy(adv_x)
        qval = _q_func(ori_state_tensor, action, "none")[0]

        cost = -qval.mean()

        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        # adv_x = adv_x_clone + step_size * torch.sign(grad)
        adv_x = adv_x_clone + step_size * grad

        delta = torch.clamp(adv_x - ori_state_tensor, min=-epsilon, max=epsilon)
        adv_x = adv_x_clone + delta

        # This clamp is performed in ORIGINAL scale
        adv_x = scaler.reverse_transform(adv_x)
        adv_x = clamp(adv_x, _obs_min, _obs_max).detach()
        adv_x = scaler.transform(adv_x)

    perturbed_state = scaler.reverse_transform(adv_x)  # un-normalized
    return perturbed_state


def critic_mqd_attack(x, a, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                      scaler=None):
    """" NOTE: x must be un-normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_state_tensor = x.clone().detach()
    ori_state_tensor = scaler.transform(ori_state_tensor)   # normalize original state

    with torch.no_grad():
        gt_qval = _q_func(ori_state_tensor, a, "none").detach()[0]  # Take q1

    adv_x = ori_state_tensor.clone().detach()               # already normalized

    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    # TODO: Optimized with PGD
    for _ in range(num_steps):
        adv_x_clone = adv_x.clone().detach()
        adv_x.requires_grad = True

        qval = _q_func(adv_x, a, "none")[0]     # Take q1

        cost = F.mse_loss(qval, gt_qval)

        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        adv_x = adv_x_clone + step_size * grad

        delta = torch.clamp(adv_x - ori_state_tensor, min=-epsilon, max=epsilon)
        adv_x = adv_x_clone + delta

        # This clamp is performed in ORIGINAL scale
        adv_x = scaler.reverse_transform(adv_x)
        adv_x = clamp(adv_x, _obs_min, _obs_max).detach()
        adv_x = scaler.transform(adv_x)

    perturbed_state = scaler.reverse_transform(adv_x)  # un-normalized
    return perturbed_state


def actor_mad_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                     scaler=None):
    """" NOTE: x must be un-normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_state_tensor = x.clone().detach()
    ori_state_tensor = scaler.transform(ori_state_tensor)           # normalize original state

    with torch.no_grad():
        gt_action = _policy(ori_state_tensor).clone().detach()      # ground truth

    adv_x = ori_state_tensor.clone().detach()                       # already normalized

    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                          # Add noise in `normalized space`

    # TODO: Optimized with PGD
    for _ in range(num_steps):
        adv_x_clone = adv_x.clone().detach()    # normalized
        adv_x.requires_grad = True

        adv_a = _policy(adv_x)

        cost = F.mse_loss(adv_a, gt_action)

        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        adv_x = adv_x_clone + step_size * grad

        delta = torch.clamp(adv_x - ori_state_tensor, min=-epsilon, max=epsilon)
        adv_x = adv_x_clone + delta         # This is adversarial example

        # This clamp is performed in ORIGINAL scale
        adv_x = scaler.reverse_transform(adv_x)
        adv_x = clamp(adv_x, _obs_min, _obs_max).detach()
        adv_x = scaler.transform(adv_x)

    perturbed_state = scaler.reverse_transform(adv_x)  # un-normalized
    return perturbed_state

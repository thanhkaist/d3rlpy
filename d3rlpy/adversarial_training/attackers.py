import torch
import torch.nn.functional as F

import numpy as np

from .utility import clamp


def random_attack(x, epsilon, _obs_min, _obs_max, scaler):
    """" NOTE: x must be normalized """""
    assert isinstance(x, torch.Tensor), "input x must be tensor."
    adv_x = x.clone().detach()
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise

    # This clamp is performed in ORIGINAL scale
    adv_x = scaler.reverse_transform(adv_x)
    adv_x = clamp(adv_x, _obs_min, _obs_max)
    adv_x = scaler.transform(adv_x)   # No need

    return adv_x


def critic_normal_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                         scaler, q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = x.clone().detach()                   # already normalized

    adv_x = ori_x.clone().detach()               # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    adv_x = scaler.reverse_transform(adv_x)
    adv_x = clamp(adv_x, _obs_min, _obs_max)
    adv_x = scaler.transform(adv_x).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            action = _policy(adv_x)
            qval = _q_func(ori_x, action, "none")[q_func_id]

            cost = -qval.mean()

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            # adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())
            adv_x = adv_x.detach() + step_size * grad.detach()

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta       # This is adversarial example

            # This clamp is performed in ORIGINAL scale
            adv_x = scaler.reverse_transform(adv_x)
            adv_x = clamp(adv_x, _obs_min, _obs_max)
            adv_x = scaler.transform(adv_x).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    return perturbed_state


def critic_mqd_attack(x, a, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                      scaler, q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = x.clone().detach()                              # already normalized

    with torch.no_grad():
        gt_qval = _q_func(ori_x, a, "none")[q_func_id].detach()

    adv_x = ori_x.clone().detach()                          # already normalized

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                   # Add noise in `normalized space`

    adv_x = scaler.reverse_transform(adv_x)
    adv_x = clamp(adv_x, _obs_min, _obs_max)
    adv_x = scaler.transform(adv_x).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            qval = _q_func(adv_x, a, "none")[q_func_id]

            cost = F.mse_loss(qval, gt_qval)

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x.detach() + step_size * grad.detach()

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta

            # This clamp is performed in ORIGINAL scale
            adv_x = scaler.reverse_transform(adv_x)
            adv_x = clamp(adv_x, _obs_min, _obs_max)
            adv_x = scaler.transform(adv_x).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    return perturbed_state


def actor_mad_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                     scaler, q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = x.clone().detach()                           # already normalized

    adv_x = ori_x.clone().detach()  # already normalized

    with torch.no_grad():
        gt_action = _policy(ori_x).detach()              # ground truth

    # Starting at a uniformly random point
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise                                          # Add noise in `normalized space`

    adv_x = scaler.reverse_transform(adv_x)
    adv_x = clamp(adv_x, _obs_min, _obs_max)
    adv_x = scaler.transform(adv_x).detach()

    if optimizer == 'pgd':
        for _ in range(num_steps):
            adv_x.requires_grad = True

            adv_a = _policy(adv_x)

            cost = F.mse_loss(adv_a, gt_action)

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            # adv_x = adv_x.detach() + step_size * torch.sign(grad.detach())
            adv_x = adv_x.detach() + step_size * grad.detach()

            delta = torch.clamp(adv_x - ori_x, min=-epsilon, max=epsilon)
            adv_x = ori_x + delta         # This is adversarial example

            # This clamp is performed in ORIGINAL scale
            adv_x = scaler.reverse_transform(adv_x)
            adv_x = clamp(adv_x, _obs_min, _obs_max)
            adv_x = scaler.transform(adv_x).detach()

    elif optimizer == 'sgld':
        raise NotImplementedError

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    return perturbed_state

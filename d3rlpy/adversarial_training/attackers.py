from re import M
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


def actor_mad_attack(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max,
                     scaler=None, optimizer="PGD"):
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

    if optimizer in ["PGD",]:
        for _ in range(num_steps):
            adv_x_clone = adv_x.clone().detach()    # normalized
            adv_x.requires_grad = True

            adv_a = _policy(adv_x)

            cost = F.mse_loss(adv_a, gt_action)

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            adv_x = adv_x_clone + step_size * grad

            delta = torch.clamp(adv_x - ori_state_tensor, min=-epsilon, max=epsilon)
            adv_x = adv_x_clone + delta         # This is adversarial example
    elif optimizer in ["SGLD",]:

        ##########################################
        step_eps = epsilon/num_steps
        # upper and lower bounds for clipping
        adv_ub = ori_state_tensor + epsilon
        adv_lb = ori_state_tensor - epsilon
        # add uniform noise beween +/- scaled_robust_eps
        # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
        beta = 1e-5
        noise_factor = torch.sqrt(2 * step_eps) * beta
        noise = torch.randn_like(ori_state_tensor) * noise_factor
        # First SGLD step, the gradient is 0, so only need to add noise. Project to Linf box.
        adv_x = (ori_state_tensor.clone() + noise.sign() * step_eps).detach().requires_grad_()
        # and clip into the upper and lower bounds (not necessary for now as we use uniform noise)
        # adv_phi = torch.max(adv_phi, adv_lb)
        # adv_phi = torch.min(adv_phi, adv_ub)
        for i in range(num_steps):
            # Find a nearby state adv_phi that maximize the difference
            adv_loss = (_policy(adv_x) - gt_action).pow(2).mean()
            # Need to clear gradients before the backward() for policy_loss
            grad = torch.autograd.grad(adv_loss, adv_x, retain_graph=False, create_graph=False)[0]
            # Reduce noise at every step. We start at step 2.
            noise_factor = torch.sqrt(2 * step_eps) * beta / (i+2)
            # Project noisy gradient to step boundary.
            update = (grad + noise_factor * torch.randn_like(adv_x)).sign() * step_eps
            adv_x = adv_x + update
            # clip into the upper and lower bounds
            adv_x = torch.max(adv_x, adv_lb)
            adv_x = torch.min(adv_x, adv_ub).detach().requires_grad_()
            # see how much the difference is
            # self._meter.update('sgld_act_diff', (adv_phi - phi).abs().sum().item())
            # We want to minimize the loss
            # action_reg_loss = (self.network.actor(adv_phi) - action).pow(2).mean()

            # This clamp is performed in ORIGINAL scale
            adv_x = scaler.reverse_transform(adv_x)
            adv_x = clamp(adv_x, _obs_min, _obs_max).detach()
            adv_x = scaler.transform(adv_x)
        

    perturbed_state = scaler.reverse_transform(adv_x)  # un-normalized
    return perturbed_state

import torch
import torch.nn.functional as F

import numpy as np

from .utility import clamp

def preprocess_state(x):
    if len(x.shape) == 1:
        x = x.reshape((1,) + x.shape)

    assert len(x.shape) == 2, "Currently only support the low-dimensional state"
    return x

def random_attack(x, epsilon, _obs_min_norm, _obs_max_norm):
    """" NOTE: x must be normalized """""
    assert isinstance(x, torch.Tensor), "input x must be tensor."
    adv_x = preprocess_state(x).clone().detach()
    noise = torch.zeros_like(adv_x).uniform_(-epsilon, epsilon)
    adv_x = adv_x + noise

    # This clamp is performed in normalized scale
    adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm)

    return adv_x


def critic_normal_attack(x, _policy, _q_func, epsilon, num_steps, step_size,
                         _obs_min_norm, _obs_max_norm,
                         q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x).clone().detach()                   # already normalized

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
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    elif optimizer == 'sgld':
        step_eps = step_size

        adv_ub = ori_x + epsilon
        adv_lb = ori_x - epsilon

        beta = 1e-5
        noise_factor = torch.sqrt(2 * step_eps) * beta

        # First SGLD step, the gradient is 0, so only need to add noise. Project to Linf box.
        adv_x = (ori_x.clone() + (noise_factor * torch.randn_like(ori_x)).sign() * step_eps).detach()

        for i in range(num_steps):
            adv_x.requires_grad = True

            # Find a nearby state adv_phi that maximize the difference
            action = _policy(adv_x)
            qval = _q_func(ori_x, action, "none")[q_func_id]

            cost = -qval.mean()

            grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

            # Reduce noise at every step. We start at step 2.
            noise_factor = torch.sqrt(2 * step_eps) * beta / (i + 2)

            # Project noisy gradient to step boundary.
            update = (grad + noise_factor * torch.randn_like(ori_x)).sign() * step_eps
            adv_x = adv_x + update

            # clip into the upper and lower bounds
            adv_x = torch.max(adv_x, adv_lb)
            adv_x = torch.min(adv_x, adv_ub)

            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    return perturbed_state


def critic_mqd_attack(x, a, _policy, _q_func, epsilon, num_steps, step_size, _obs_min_norm, _obs_max_norm,
                      q_func_id=0, optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x).clone().detach()                              # already normalized

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
                     optimizer='pgd'):
    """" NOTE: x must be normalized """""

    assert isinstance(x, torch.Tensor), "input x must be tensor."
    ori_x = preprocess_state(x).clone().detach()                           # already normalized

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

            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    elif optimizer == 'sgld':

        ##########################################
        step_eps = epsilon / num_steps
        # upper and lower bounds for clipping
        adv_ub = ori_x + epsilon
        adv_lb = ori_x - epsilon
        # add uniform noise beween +/- scaled_robust_eps
        # SGLD noise factor. We set (inverse) beta=1e-5 as gradients are relatively small here.
        beta = 1e-5
        noise_factor = np.sqrt(2 * step_eps) * beta
        noise = torch.randn_like(ori_x) * noise_factor
        # First SGLD step, the gradient is 0, so only need to add noise. Project to Linf box.
        adv_x = (ori_x.clone() + noise.sign() * step_eps).detach().requires_grad_()
        # and clip into the upper and lower bounds (not necessary for now as we use uniform noise)
        # adv_phi = torch.max(adv_phi, adv_lb)
        # adv_phi = torch.min(adv_phi, adv_ub)
        for i in range(num_steps):
            adv_x.requires_grad = True
            # Find a nearby state adv_phi that maximize the difference
            adv_loss = (_policy(adv_x) - gt_action).pow(2).mean()
            # Need to clear gradients before the backward() for policy_loss
            grad = torch.autograd.grad(adv_loss, adv_x, retain_graph=False, create_graph=False)[0]
            # Reduce noise at every step. We start at step 2.
            noise_factor = np.sqrt(2 * step_eps) * beta / (i + 2)
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

            # This clamp is performed in normalized scale
            adv_x = clamp(adv_x, _obs_min_norm, _obs_max_norm).detach()

    else:
        raise NotImplementedError

    perturbed_state = adv_x     # already normalized
    assert np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1)) < (epsilon + 1e-4), f"Perturbed state go out of epsilon {np.max(np.linalg.norm(perturbed_state.cpu() - ori_x.cpu(), np.inf, 1))}"
    return perturbed_state

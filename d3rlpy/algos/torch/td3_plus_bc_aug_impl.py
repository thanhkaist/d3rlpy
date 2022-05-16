# pylint: disable=too-many-ancestors

from typing import Optional, Sequence
import copy

import torch
import numpy as np

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .td3_impl import TD3Impl


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


def clamp(x, vec_min, vec_max):
    if isinstance(vec_min, list):
        vec_min = torch.Tensor(vec_min).to(x.device)
    if isinstance(vec_max, list):
        vec_max = torch.Tensor(vec_max).to(x.device)

    assert isinstance(vec_min, torch.Tensor) and isinstance(vec_max, torch.Tensor)
    x = torch.max(x, vec_min)
    x = torch.min(x, vec_max)
    return x


def normalize(x, min, max):
    x = (x - min)/(max - min)
    return x


def denormalize(x, min, max):
    x = x * (max - min) + min
    return x

def generate_adv_example(x, _policy, _q_func, epsilon, num_steps, step_size, _obs_min, _obs_max):
    adv_x = x.clone().detach()
    # Starting at a uniformly random point
    adv_x = adv_x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
    adv_x = clamp(adv_x, _obs_min, _obs_max)

    with torch.no_grad():
        # Using action from current learned policy as ground truth
        action = _policy(x)
        action = action.detach()

    for _ in range(num_steps):
        adv_x.requires_grad = True

        outputs = _q_func(adv_x, action, "none")[0]

        cost = -outputs.mean()

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_x, retain_graph=False, create_graph=False)[0]

        adv_x = adv_x.detach() + step_size * grad.detach()
        delta = torch.clamp(adv_x - x, min=-epsilon, max=epsilon)
        adv_x = clamp(x + delta, _obs_min, _obs_max).detach()

    return adv_x


class TD3PlusBCAugImpl(TD3Impl):

    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        transform: str = 'gaussian',
        transform_params: dict = None,
        env_name: str = '',
        custom_scaler: Optional[Scaler] = None,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

        self._transform = transform
        self._transform_params = transform_params
        self._custom_scaler = custom_scaler

        env_name_ = env_name.split('-')
        env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        return lam * -q_t.mean() + ((batch.actions - action) ** 2).mean()

    def do_augmentation(self, batch: TorchMiniBatch):
        batch_aug = copy.deepcopy(batch)
        # Transforming the copied batch
        if self._transform in ['gaussian']:
            assert self._transform_params is not None, "Cannot find params for {} transform.".format(self._transform)
            epsilon = self._transform_params.get('epsilon', None)
            norm_min_max = self._transform_params.get('norm_min_max', False)
            assert epsilon is not None, "Please provide the epsilon to perform {} transform.".format(self._transform)

            if norm_min_max:
                noise = torch.randn_like(batch_aug.observations, dtype=batch_aug.observations.dtype,
                                         device=batch_aug.observations.device) * epsilon
                batch_aug._observations = normalize(batch_aug._observations, self._obs_min,
                                                    self._obs_max)
                batch_aug._observations += noise
                batch_aug._observations = torch.clamp(batch_aug._observations, 0, 1)
                batch_aug._observations = denormalize(batch_aug._observations, self._obs_min,
                                                      self._obs_max)

                noise = torch.randn_like(batch_aug.observations, dtype=batch_aug.observations.dtype,
                                         device=batch_aug.observations.device) * epsilon
                batch_aug._next_observations = normalize(batch_aug._next_observations,
                                                         self._obs_min, self._obs_max)
                batch_aug._next_observations += noise
                batch_aug._next_observations = torch.clamp(batch_aug._next_observations, 0, 1)
                batch_aug._next_observations = denormalize(batch_aug._next_observations,
                                                           self._obs_min, self._obs_max)
            else:
                noise = torch.randn_like(batch_aug.observations, dtype=batch_aug.observations.dtype,
                                         device=batch_aug.observations.device) * epsilon
                batch_aug._observations += noise
                batch_aug._observations = clamp(batch_aug._observations,
                                                self._obs_min, self._obs_max)

                noise = torch.randn_like(batch_aug.observations, dtype=batch_aug.observations.dtype,
                                         device=batch_aug.observations.device) * epsilon
                batch_aug._next_observations += noise
                batch_aug._next_observations = clamp(batch_aug._next_observations,
                                                     self._obs_min, self._obs_max)
        elif self._transform in ['adversarial_training']:
            #### Using PGD with Linf-norm
            epsilon = self._transform_params.get('epsilon', None)
            num_steps = self._transform_params.get('num_steps', None)
            step_size = self._transform_params.get('step_size', None)
            assert epsilon is not None and num_steps is not None and step_size is not None, \
                "Please provide the epsilon to perform {} transform.".format(self._transform)

            adv_x = generate_adv_example(batch_aug._observations, self._policy, self._q_func,
                                         epsilon, num_steps, step_size,
                                         self._obs_min, self._obs_max)
            batch_aug._observations = adv_x

            adv_x = generate_adv_example(batch_aug._next_observations, self._policy, self._q_func,
                                         epsilon, num_steps, step_size,
                                         self._obs_min, self._obs_max)
            batch_aug._next_observations = adv_x
        else:
            raise NotImplementedError
        return batch, batch_aug

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        ###### TODO: Augment state here
        batch, batch_aug = self.do_augmentation(batch)

        batch._observations = self._custom_scaler.transform(batch._observations)
        batch._next_observation = self._custom_scaler.transform(batch._next_observation)
        batch_aug._observations = self._custom_scaler.transform(batch_aug._observations)
        batch_aug._next_observation = self._custom_scaler.transform(batch_aug._next_observation)
        q_tpn = self.compute_target(batch)          # Compute target for clean data
        q_aug_tpn = self.compute_target(batch_aug)  # Compute target for augmented data
        q_tpn = (q_tpn + q_aug_tpn) / 2

        loss = self.compute_critic_loss(batch, q_tpn)
        loss += self.compute_critic_loss(batch_aug, q_tpn)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

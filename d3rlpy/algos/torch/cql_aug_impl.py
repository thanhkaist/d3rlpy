import math
from typing import Optional, Sequence
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ...gpu import Device
from ...models.builders import create_parameter
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...models.torch import Parameter
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch, torch_api, train_api
from .dqn_impl import DoubleDQNImpl
from .sac_impl import SACImpl


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
    assert (0.0 <= x).all() and (x <= 1.0).all()
    return x


def denormalize(x, min, max):
    x = x * (max - min) + min
    return x


class CQLAugImpl(SACImpl):

    _alpha_learning_rate: float
    _alpha_optim_factory: OptimizerFactory
    _initial_alpha: float
    _alpha_threshold: float
    _conservative_weight: float
    _n_action_samples: int
    _soft_q_backup: bool
    _log_alpha: Optional[Parameter]
    _alpha_optim: Optional[Optimizer]

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        transform: str = 'gaussian',
        transform_params: dict = None,
        env_name: str = ''
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            actor_learning_rate=actor_learning_rate,
            critic_learning_rate=critic_learning_rate,
            temp_learning_rate=temp_learning_rate,
            actor_optim_factory=actor_optim_factory,
            critic_optim_factory=critic_optim_factory,
            temp_optim_factory=temp_optim_factory,
            actor_encoder_factory=actor_encoder_factory,
            critic_encoder_factory=critic_encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            tau=tau,
            n_critics=n_critics,
            initial_temperature=initial_temperature,
            use_gpu=use_gpu,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._conservative_weight = conservative_weight
        self._n_action_samples = n_action_samples
        self._soft_q_backup = soft_q_backup

        self._transform = transform
        self._transform_params = transform_params

        env_name_ = env_name.split('-')
        env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[env_name]['max']).to('cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[env_name]['min']).to('cuda:{}'.format(self._use_gpu.get_id()))

        # initialized in build
        self._log_alpha = None
        self._alpha_optim = None

    def build(self) -> None:
        self._build_alpha()
        super().build()
        self._build_alpha_optim()

    def _build_alpha(self) -> None:
        initial_val = math.log(self._initial_alpha)
        self._log_alpha = create_parameter((1, 1), initial_val)

    def _build_alpha_optim(self) -> None:
        assert self._log_alpha is not None
        self._alpha_optim = self._alpha_optim_factory.create(
            self._log_alpha.parameters(), lr=self._alpha_learning_rate
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, batch_aug: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ###### TODO: Augment state here
        assert batch_aug is not None
        loss = super().compute_critic_loss(batch, q_tpn)        # Compute TD error for clean data
        loss += super().compute_critic_loss(batch_aug, q_tpn)   # Compute TD error for augmented data
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        return loss + conservative_loss

    @train_api
    @torch_api()
    def update_alpha(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._alpha_optim.zero_grad()

        # the original implementation does scale the loss value
        loss = -self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )

        loss.backward()
        self._alpha_optim.step()

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    def _compute_policy_is_values(
        self, policy_obs: torch.Tensor, value_obs: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        with torch.no_grad():
            policy_actions, n_log_probs = self._policy.sample_n_with_log_prob(
                policy_obs, self._n_action_samples
            )

        obs_shape = value_obs.shape

        repeated_obs = value_obs.expand(self._n_action_samples, *obs_shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs_shape[1:])
        # (batch, n, action) -> (batch * n, action)
        flat_policy_acts = policy_actions.reshape(-1, self.action_size)

        # estimate action-values for policy actions
        policy_values = self._q_func(flat_obs, flat_policy_acts, "none")
        policy_values = policy_values.view(
            self._n_critics, obs_shape[0], self._n_action_samples
        )
        log_probs = n_log_probs.view(1, -1, self._n_action_samples)

        # importance sampling
        return policy_values - log_probs

    def _compute_random_is_values(self, obs: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None

        repeated_obs = obs.expand(self._n_action_samples, *obs.shape)
        # (n, batch, observation) -> (batch, n, observation)
        transposed_obs = repeated_obs.transpose(0, 1)
        # (batch, n, observation) -> (batch * n, observation)
        flat_obs = transposed_obs.reshape(-1, *obs.shape[1:])

        # estimate action-values for actions from uniform distribution
        # uniform distribution between [-1.0, 1.0]
        flat_shape = (obs.shape[0] * self._n_action_samples, self._action_size)
        zero_tensor = torch.zeros(flat_shape, device=self._device)
        random_actions = zero_tensor.uniform_(-1.0, 1.0)
        random_values = self._q_func(flat_obs, random_actions, "none")
        random_values = random_values.view(
            self._n_critics, obs.shape[0], self._n_action_samples
        )
        random_log_probs = math.log(0.5**self._action_size)

        # importance sampling
        return random_values - random_log_probs

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        loss = logsumexp.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        if self._soft_q_backup:
            target_value = super().compute_target(batch)
        else:
            target_value = self._compute_deterministic_target(batch)
        return target_value

    def _compute_deterministic_target(
        self, batch: TorchMiniBatch
    ) -> torch.Tensor:
        assert self._policy
        assert self._targ_q_func
        with torch.no_grad():
            action = self._policy.best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )

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
                import pdb; pdb.set_trace()
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
                batch_aug._next_observations = torch.clamp(batch._next_observations, 0, 1)
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
                batch_aug._next_observations = clamp(batch._next_observations,
                                                     self._obs_min, self._obs_max)
        elif self._transform in ['adversarial_training']:
            # Using PGD with Linf-norm
            pass
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
        q_tpn = self.compute_target(batch)  # Compute target for clean data
        q_aug_tpn = self.compute_target(batch_aug)  # Compute target for augmented data
        q_tpn = (q_tpn + q_aug_tpn) / 2

        loss = self.compute_critic_loss(batch, q_tpn, batch_aug)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()


class DiscreteCQLAugImpl(DoubleDQNImpl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        learning_rate: float,
        optim_factory: OptimizerFactory,
        encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        gamma: float,
        n_critics: int,
        alpha: float,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            learning_rate=learning_rate,
            optim_factory=optim_factory,
            encoder_factory=encoder_factory,
            q_func_factory=q_func_factory,
            gamma=gamma,
            n_critics=n_critics,
            use_gpu=use_gpu,
            scaler=scaler,
            reward_scaler=reward_scaler,
        )
        self._alpha = alpha

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        loss = super().compute_loss(batch, q_tpn)
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions.long()
        )
        return loss + self._alpha * conservative_loss

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        # compute logsumexp
        policy_values = self._q_func(obs_t)
        logsumexp = torch.logsumexp(policy_values, dim=1, keepdim=True)

        # estimate action-values under data distribution
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        data_values = (self._q_func(obs_t) * one_hot).sum(dim=1, keepdim=True)

        return (logsumexp - data_values).mean()

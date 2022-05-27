# pylint: disable=too-many-ancestors

from typing import Optional, Sequence, Tuple
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
from ...adversarial_training import (
    clamp,
    ENV_OBS_RANGE,
)
from ...adversarial_training.attackers import critic_normal_attack, actor_mad_attack, random_attack


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

        env_name_ = env_name.split('-')
        env_name = env_name_[0] + '-' + env_name_[-1]
        self._obs_max = torch.Tensor(ENV_OBS_RANGE[env_name]['max']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))
        self._obs_min = torch.Tensor(ENV_OBS_RANGE[env_name]['min']).to(
            'cuda:{}'.format(self._use_gpu.get_id()))

    def compute_actor_loss(self, batch: TorchMiniBatch) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        actor_loss = lam * -q_t.mean()
        bc_loss = ((batch.actions - action) ** 2).mean()
        total_loss = actor_loss + bc_loss
        return total_loss, actor_loss, bc_loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                clipped_action,
                reduction="min",
            )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    @train_api
    @torch_api()
    def update_critic(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._critic_optim is not None
        robust_type = self._transform_params.get('robust_type', None)

        if robust_type in ['critic_reg']:
            batch._observations = self._scaler.reverse_transform(batch._observations)
            batch._next_observations = self._scaler.reverse_transform(batch._next_observations)

            batch, batch_aug = self.do_augmentation(batch)

            with torch.no_grad():
                q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach()
                q2_pred = q_prediction[1].cpu().detach()

                q_prediction_adv = self._q_func(batch_aug.observations, batch_aug.actions, reduction="none")
                q1_pred_adv_diff = (q_prediction_adv[0].cpu().detach() - q1_pred).numpy().mean()
                q2_pred_adv_diff = (q_prediction_adv[1].cpu().detach() - q2_pred).numpy().mean()
                q1_pred = q1_pred.numpy().mean()
                q2_pred = q2_pred.numpy().mean()

            batch._observations = self._scaler.transform(batch._observations)
            batch._next_observations = self._scaler.transform(batch._next_observations)
            batch_aug._observations = self._scaler.transform(batch_aug._observations)
            batch_aug._next_observations = self._scaler.transform(batch_aug._next_observations)

            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)          # Compute target for clean data
            q_aug_tpn = self.compute_target(batch_aug)  # Compute target for augmented data
            q_tpn = (q_tpn + q_aug_tpn) / 2

            loss = (self.compute_critic_loss(batch, q_tpn) +
                    self.compute_critic_loss(batch_aug, q_tpn)) / 2

            loss.backward()
            self._critic_optim.step()
        else:
            q1_pred_adv_diff, q2_pred_adv_diff = 0.0, 0.0
            with torch.no_grad():
                q_prediction = self._q_func(batch.observations, batch.actions, reduction="none")
                q1_pred = q_prediction[0].cpu().detach().numpy().mean()
                q2_pred = q_prediction[1].cpu().detach().numpy().mean()

            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)

            loss = self.compute_critic_loss(batch, q_tpn)

            loss.backward()
            self._critic_optim.step()

        return loss.cpu().detach().numpy(), q_tpn.cpu().detach().numpy().mean(), q1_pred, q2_pred, \
               q1_pred_adv_diff, q2_pred_adv_diff

    @train_api
    @torch_api()
    def update_actor(self, batch: TorchMiniBatch) \
        -> Tuple[np.ndarray, Tuple]:
        assert self._q_func is not None
        assert self._actor_optim is not None
        robust_type = self._transform_params.get('robust_type', None)

        if robust_type in ['actor_mad']:
            actor_reg_coef = self._transform_params.get('actor_reg_coef', 0)

            batch._observations = self._scaler.reverse_transform(batch._observations)
            batch._next_observations = self._scaler.reverse_transform(batch._next_observations)

            batch, batch_aug = self.do_augmentation(batch)

            batch._observations = self._scaler.transform(batch._observations)
            batch._next_observations = self._scaler.transform(batch._next_observations)
            batch_aug._observations = self._scaler.transform(batch_aug._observations)
            batch_aug._next_observations = self._scaler.transform(batch_aug._next_observations)

            # Q function should be inference mode for stability
            self._q_func.eval()

            self._actor_optim.zero_grad()

            loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

            if actor_reg_coef > 0:
                action_reg_loss =  ((self._policy(batch_aug.observations) - batch.actions) ** 2).mean()
                loss += action_reg_loss * action_reg_loss
            else:
                action_reg_loss = 0

            loss.backward()
            self._actor_optim.step()

            action_reg_loss = action_reg_loss.item() if action_reg_loss > 0 else 0
            extra_logs = (actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy(),
                          action_reg_loss)

        else:

            # Q function should be inference mode for stability
            self._q_func.eval()

            self._actor_optim.zero_grad()

            loss, actor_loss, bc_loss = self.compute_actor_loss(batch)

            loss.backward()
            self._actor_optim.step()

            extra_logs = (actor_loss.cpu().detach().numpy(), bc_loss.cpu().detach().numpy())

        return loss.cpu().detach().numpy(), extra_logs

    def do_augmentation(self, batch: TorchMiniBatch):
        #### Always assuming obs, next_obs in original space, i.e.: without normalized, standardized

        batch_aug = copy.deepcopy(batch)
        # Transforming the copied batch
        if self._transform in ['random']:
            assert self._transform_params is not None, "Cannot find params for random transform."
            epsilon = self._transform_params.get('epsilon', None)
            assert epsilon is not None, "Please provide the epsilon for random transform."

            adv_x = random_attack(batch_aug._observations, epsilon,
                                  self._obs_min, self._obs_max,
                                  self.scaler)
            batch_aug._observations = adv_x

            adv_x = random_attack(batch_aug._next_observations, epsilon,
                                  self._obs_min, self._obs_max,
                                  self.scaler)
            batch_aug._next_observations = adv_x

        elif self._transform in ['adversarial_training']:
            #### Using PGD with Linf-norm
            epsilon = self._transform_params.get('epsilon', None)
            num_steps = self._transform_params.get('num_steps', None)
            step_size = self._transform_params.get('step_size', None)
            attack_type = self._transform_params.get('attack_type', None)
            assert (epsilon is not None) and (num_steps is not None) and \
                   (step_size is not None) and (attack_type is not None)

            if attack_type in ['critic_normal']:
                adv_x = critic_normal_attack(batch_aug._observations,
                                      self._policy, self._q_func,
                                      epsilon, num_steps, step_size,
                                      self._obs_min, self._obs_max,
                                      self._scaler)
                batch_aug._observations = adv_x

                adv_x = critic_normal_attack(batch_aug._next_observations,
                                                self._policy, self._q_func,
                                                epsilon, num_steps, step_size,
                                                self._obs_min, self._obs_max,
                                                self._scaler)
                batch_aug._next_observations = adv_x

            elif attack_type in ['actor_mad']:
                adv_x = actor_mad_attack(batch_aug._observations,
                                         self._policy, self._q_func,
                                         epsilon, num_steps, step_size,
                                         self._obs_min, self._obs_max,
                                         self._scaler)
                batch_aug._observations = adv_x

                adv_x = actor_mad_attack(batch_aug._next_observations,
                                         self._policy, self._q_func,
                                         epsilon, num_steps, step_size,
                                         self._obs_min, self._obs_max,
                                         self._scaler)
                batch_aug._next_observations = adv_x

            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return batch, batch_aug


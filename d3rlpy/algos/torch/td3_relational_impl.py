# pylint: disable=too-many-ancestors

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from ...gpu import Device
from ...models.encoders import EncoderFactory
from ...models.optimizers import OptimizerFactory
from ...models.q_functions import QFunctionFactory
from ...preprocessing import ActionScaler, RewardScaler, Scaler
from ...torch_utility import TorchMiniBatch
from .td3_impl import TD3Impl


class TD3PlusRelationImpl(TD3Impl):

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
        self.log_metrics ={}

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None


        temperature_k =0.04
        temperature_q =0.1

        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        lam = self._alpha / (q_t.abs().mean()).detach()


        self.log_metrics ={}
        self.log_metrics.update({"absQmean":(q_t.abs().mean()).detach().cpu()})

        logits_q = torch.einsum("nc,kc->nk", [action, batch.actions])
        logits_k = torch.einsum("nc,kc->nk", [batch.actions, batch.actions])
        enable_BC = False

        lossRelation = -torch.sum(
            F.softmax(logits_k.detach() / temperature_k, dim=1)
            * F.log_softmax(logits_q / temperature_q, dim=1),
            dim=1,
        ).mean()


        self.log_metrics.update({"RelationLoss":lossRelation.detach().cpu()})
        self.log_metrics.update({"TD3Loss":-q_t.mean().detach().cpu()})
        self.log_metrics.update({"BCLoss":((batch.actions - action.detach()) ** 2).mean().cpu()})
        return lam * -q_t.mean() + lossRelation +enable_BC*((batch.actions - action) ** 2).mean()

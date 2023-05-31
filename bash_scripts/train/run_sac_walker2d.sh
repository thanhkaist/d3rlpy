#!/bin/bash
SERVER="s144"
ROOT_DIR="${SERVER}_d3rlpy_logs"

GPU=1

ENV_NAME="Walker2d-v2"
ENV_NAME_DIR="walker2d"

N_STEPS=3000000


RL_ALGO='SAC'
EXP_NAME="${RL_ALGO}_baseline"
LOG_DIR="${ROOT_DIR}/online/${ENV_NAME_DIR}/${EXP_NAME}"

#SEEDS=(1 2 3)
SEEDS=(1)
for SEED in ${SEEDS[*]}; do
CUDA_VISIBLE_DEVICES=${GPU} python run_rl_online.py --dataset ${ENV_NAME} --gpu 0 --logdir ${LOG_DIR} --exp ${EXP_NAME} \
  --algo ${RL_ALGO} --n_steps ${N_STEPS} --standardization \
  --seed ${SEED} --wandb
done


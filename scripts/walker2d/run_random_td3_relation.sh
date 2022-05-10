#!/bin/bash
ROOT_DIR="d3rlpy_logs"

ENV_NAME="walker2d-random-v0"
GPU=1
EXP_NAME="TD3_Relational"
LOG_DIR="${ROOT_DIR}/${ENV_NAME}/${EXP_NAME}"

python ReTD.py --dataset ${ENV_NAME} --gpu ${GPU} --logdir ${LOG_DIR} --seed 1
python ReTD.py --dataset ${ENV_NAME} --gpu ${GPU} --logdir ${LOG_DIR} --seed 2
python ReTD.py --dataset ${ENV_NAME} --gpu ${GPU} --logdir ${LOG_DIR} --seed 3
python ReTD.py --dataset ${ENV_NAME} --gpu ${GPU} --logdir ${LOG_DIR} --seed 4

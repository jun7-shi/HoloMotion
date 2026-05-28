#!/usr/bin/env bash
# Project HoloMotion

set -euo pipefail

source train.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

config_name="train_g1_29dof_quiet_goal_velocity_mlp"
num_envs="${NUM_ENVS:-4096}"

COMMON_ARGS=(
    "holomotion/src/training/train.py"
    "--config-name=training/velocity_tracking/train_g1_29dof_quiet_goal_velocity_mlp"
    "experiment_name=${config_name}"
    "num_envs=${num_envs}"
    "headless=true"
)

"${Train_CONDA_PREFIX}/bin/accelerate" launch "${COMMON_ARGS[@]}"

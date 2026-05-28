#!/usr/bin/env bash
# Project HoloMotion

set -euo pipefail

source train.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

checkpoint="${1:?checkpoint path is required}"
num_envs="${NUM_ENVS:-1}"

"${Train_CONDA_PREFIX}/bin/python" \
    holomotion/src/evaluation/eval_velocity_tracking.py \
    --config-name=evaluation/eval_quiet_goal_velocity \
    project_name="HoloMotionQuietGoalVelocityG1" \
    experiment_name="eval_quiet_goal_velocity" \
    num_envs="${num_envs}" \
    headless=false \
    checkpoint="${checkpoint}"

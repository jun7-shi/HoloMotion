# Repository Guidelines

## Project Structure & Module Organization

HoloMotion is a Python robotics/control repository with shell entry points for data preparation, training, evaluation, and deployment. Core source lives in `holomotion/src/`: `algo/`, `env/`, `modules/`, `training/`, `evaluation/`, `motion_retargeting/`, `data_curation/`, and `utils/`. Hydra-style YAML configs live in `holomotion/config/`; keep new configs near the matching domain, robot, or task. Workflow scripts are under `holomotion/scripts/`. Tests are in `tests/`, with deployment-specific tests in `deployment/unitree_g1_ros2_29dof/tests/`. Robot models and sample assets belong in `assets/`; external dependencies and vendored code live in `thirdparties/`.

## Build, Test, and Development Commands

- `conda env create -f environments/environment_train_isaaclab_cu118.yaml`: create the training environment. Use `environment_train_isaaclab_cu128.yaml` for CUDA 12.8 GPUs.
- `conda env create -f environments/environment_deploy.yaml`: create the deployment environment.
- `source train.env`: load `Train_CONDA_PREFIX` and related paths used by training/evaluation scripts.
- `make lint`: run Ruff checks on `holomotion/`.
- `make format`: format Python files with Ruff and apply safe lint fixes.
- `pytest -q tests`: run the main unit tests.
- `pytest -q deployment/unitree_g1_ros2_29dof/tests`: run deployment runtime tests.
- `bash holomotion/scripts/training/train_motion_tracking.sh`: start training after editing `CUDA_VISIBLE_DEVICES`, `config_name`, and dataset paths.

## Coding Style & Naming Conventions

Python targets 3.11 style in Ruff, with 4-space indentation, 79-character lines, double quotes, sorted imports, and no relative imports. Prefer `snake_case` for functions, variables, YAML keys, and test files; use `PascalCase` for classes. Keep Hydra config names task-scoped, such as `train_g1_29dof_velocity_tracking_mlp.yaml`. Shell scripts should source `train.env` or `deploy.env` instead of hard-coding conda paths.

## Testing Guidelines

Use pytest. Add tests as `test_*.py` in the relevant suite: general logic in `tests/`, robot deployment behavior in `deployment/unitree_g1_ros2_29dof/tests/`. Keep tests deterministic and avoid GPUs, IsaacLab simulation, real robot hardware, or large local datasets unless explicitly documented.

## Agent-Specific Instructions

For agent-run Python commands, use the existing conda environment named `holomotion`, for example `conda run -n holomotion python ...`. Do not install new packages or modify dependencies. If a package is missing, stop and report it instead of running `pip install`, `conda install`, or environment updates.

## Commit & Pull Request Guidelines

Recent history uses conventional prefixes such as `feat(scope): ...`, `hotfix(scope): ...`, `docs(scope): ...`, and `chore(scope): ...`; follow that pattern with concise, imperative subjects. Pull requests should describe the changed workflow or module, list validation commands run, link issues when applicable, and include screenshots/videos only for visualization, MuJoCo, teleop, or deployment UI changes. Do not commit generated logs, checkpoints, ONNX exports, local datasets, or machine-specific `.env` edits.

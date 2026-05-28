# Quiet Goal-Velocity Training

This guide describes the staged training schedule for the quiet, navigable WBC policy. Use the existing `holomotion` conda environment and do not install new packages during this workflow.

## Stage 1: Style Acquisition

- Commands: `vx [-0.3, 0.5]`, `vy [-0.15, 0.15]`, `yaw [-0.4, 0.4]`.
- Task mix: imitation probability `1.0`.
- Rewards: moderate reference imitation, mild quiet penalties, standard velocity tracking.
- Data dependency: requires retargeted quiet reference clips from HUM-82.
- Exit metric: stable 20 s flat-terrain rollouts with `velocity_error_mean <= 0.20 m/s`.

Example:

```bash
bash holomotion/scripts/training/train_quiet_goal_velocity.sh \
  env.config.commands.base_velocity.params.ranges.lin_vel_x='[-0.3,0.5]' \
  env.config.commands.base_velocity.params.ranges.lin_vel_y='[-0.15,0.15]' \
  env.config.commands.base_velocity.params.ranges.ang_vel_z='[-0.4,0.4]' \
  env.config.commands.base_velocity.params.imitation_prob_initial=1.0 \
  env.config.commands.base_velocity.params.imitation_prob_final=1.0
```

## Stage 2: Quiet Dynamics

- Commands: `vx [-0.4, 0.8]`, `vy [-0.25, 0.25]`, `yaw [-0.6, 0.6]`.
- Task mix: imitation probability annealed from `1.0` to `0.5`.
- Rewards: increase touchdown velocity and contact force-rate penalties after Stage 1 is stable.
- Exit metric: lower `touchdown_vz_mean` and `force_rate_mean` than the current velocity baseline at matched commands.

Example:

```bash
bash holomotion/scripts/training/train_quiet_goal_velocity.sh \
  env.config.commands.base_velocity.params.imitation_prob_initial=1.0 \
  env.config.commands.base_velocity.params.imitation_prob_final=0.5
```

## Stage 3: Generalization

- Commands: `vx [-0.6, 1.0]`, `vy [-0.35, 0.35]`, `yaw [-0.8, 0.8]`.
- Task mix: imitation probability `0.5`.
- Rewards: retain quiet penalties and velocity tracking while broadening terrain and command coverage.
- Exit metric: no completion-rate regression versus Stage 2 on flat and rough terrain.

Example:

```bash
bash holomotion/scripts/training/train_quiet_goal_velocity.sh \
  env.config.commands.base_velocity.params.ranges.lin_vel_x='[-0.6,1.0]' \
  env.config.commands.base_velocity.params.ranges.lin_vel_y='[-0.35,0.35]' \
  env.config.commands.base_velocity.params.ranges.ang_vel_z='[-0.8,0.8]' \
  env.config.commands.base_velocity.params.imitation_prob_initial=0.5 \
  env.config.commands.base_velocity.params.imitation_prob_final=0.5
```

## Evaluation

Compare each stage against the existing velocity baseline at matched commands:

- `velocity_error_mean`
- `touchdown_vz_mean`
- `peak_normal_force_p95`
- `force_rate_mean`
- `foot_slip_mean`
- `root_jerk_mean`
- completion rate

Do not report Stage 1 or Stage 2 as real quiet-style training until HUM-82 provides retargeted reference clips and metadata.

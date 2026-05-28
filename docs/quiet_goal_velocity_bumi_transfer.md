# Quiet Goal-Velocity Bumi Transfer

This checklist prepares transfer of the quiet goal-velocity WBC workflow from G1 to Noetix Bumi. Do not start Bumi implementation until the robot profile and interface data are confirmed.

## Preconditions

- G1 quiet goal-velocity policy passes Stage 3 evaluation.
- Bumi asset, joint names, actuator config, contact body names, and default pose are available.
- Bumi ROS or low-level command interface data is confirmed through HUM-43 and HUM-44.
- No new packages are installed during transfer work; missing robot tooling is reported before execution.

## Required Bumi Inputs

- Robot asset and kinematic tree.
- Joint order, limits, default pose, and actuator gains.
- Foot/contact body names for quiet rewards.
- Action scale and control decimation.
- Observation and command adapter shape for velocity control.
- Retargeting map from G1 lower-body reference names to Bumi joints and feet.

## Transfer Steps

1. Create Bumi robot config under `holomotion/config/robot/noetix/bumi/`.
2. Add Bumi body and joint name mappings for quiet reference metadata.
3. Reuse `quiet_goal_velocity_tracking`, `obs_quiet_goal_velocity`, and `rew_quiet_goal_velocity` when action and observation contracts match.
4. Start from the G1 checkpoint only if action dimensions, joint semantics, and default pose are compatible.
5. Train from scratch with the same reference-guided task mix if dimensions or joint semantics do not match.
6. Evaluate matched commands on flat terrain before rough terrain.
7. Keep deployment switching hard-gated until a unified manipulation-capable velocity policy is intentionally scoped.

## Required Metrics

- `velocity_error_mean`
- `touchdown_vz_mean`
- `peak_normal_force_p95`
- `force_rate_mean`
- `foot_slip_mean`
- `root_jerk_mean`
- completion rate

## Related Linear Work

- HUM-39: Bumi robot profile and adapter requirements.
- HUM-43: Bumi ROS interface data collection.
- HUM-44: Bumi profile and command adapter implementation.

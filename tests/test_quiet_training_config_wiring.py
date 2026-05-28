from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read_yaml(relative_path: str) -> dict:
    return yaml.safe_load((ROOT / relative_path).read_text())


def test_quiet_training_config_uses_dedicated_env_and_obs():
    cfg = _read_yaml(
        "holomotion/config/training/velocity_tracking/"
        "train_g1_29dof_quiet_goal_velocity_mlp.yaml"
    )

    assert cfg["defaults"] == [
        {"/training": "train_base"},
        {"/algo": "ppo"},
        {"/robot": "unitree/G1/29dof/29dof_training_isaaclab"},
        {"/env": "quiet_goal_velocity_tracking"},
        {"/env/terminations": "termination_velocity_tracking"},
        {"/env/observations": "velocity_tracking/obs_quiet_goal_velocity"},
        {"/env/rewards": "velocity_tracking/rew_quiet_goal_velocity"},
        {"/env/domain_randomization": "domain_rand_quiet_medium"},
        {"/env/terrain": "isaaclab_rough"},
        {"/modules": "velocity_tracking/velocity_tracking_mlp"},
    ]
    assert cfg["project_name"] == "HoloMotionQuietGoalVelocityG1"


def test_quiet_training_config_defines_unused_motion_roots():
    cfg = _read_yaml(
        "holomotion/config/training/velocity_tracking/"
        "train_g1_29dof_quiet_goal_velocity_mlp.yaml"
    )

    assert cfg["train_hdf5_roots"] == []


def test_quiet_env_targets_dedicated_wrapper():
    cfg = _read_yaml("holomotion/config/env/quiet_goal_velocity_tracking.yaml")

    env_cfg = cfg["env"]
    assert (
        env_cfg["_target_"]
        == "holomotion.src.env.quiet_goal_velocity_tracking."
        "QuietGoalVelocityTrackingEnv"
    )
    assert (
        env_cfg["config"]["commands"]["base_velocity"]["type"]
        == "QuietGoalVelocityCommandCfg"
    )
    assert env_cfg["config"]["task_names"] == [
        "quiet_style_imitation",
        "goal_velocity_generalization",
    ]


def test_velocity_env_dispatches_quiet_command_builder():
    source = (ROOT / "holomotion/src/env/velocity_tracking.py").read_text()

    assert "build_quiet_goal_velocity_commands_config" in source
    assert 'command_type == "QuietGoalVelocityCommandCfg"' in source


def test_quiet_reward_config_uses_grouped_task_rewards():
    cfg = _read_yaml(
        "holomotion/config/env/rewards/velocity_tracking/"
        "rew_quiet_goal_velocity.yaml"
    )

    assert set(cfg["rewards"]) == {
        "common",
        "quiet_style_imitation",
        "goal_velocity_generalization",
    }
    assert "feet_contact_force_rate_l1" in cfg["rewards"]["common"]
    assert (
        "quiet_reference_joint_pose_l2"
        in cfg["rewards"]["quiet_style_imitation"]
    )


def test_quiet_reward_scene_entity_params_are_instantiable():
    cfg = _read_yaml(
        "holomotion/config/env/rewards/velocity_tracking/"
        "rew_quiet_goal_velocity.yaml"
    )

    scene_entity_target = "isaaclab.managers.scene_entity_cfg.SceneEntityCfg"
    assert cfg["rewards"]["common"]["feet_slide"]["params"]["sensor_cfg"][
        "_target_"
    ] == scene_entity_target
    assert cfg["rewards"]["common"]["feet_slide"]["params"]["asset_cfg"][
        "_target_"
    ] == scene_entity_target
    assert cfg["rewards"]["goal_velocity_generalization"]["feet_air_time_v4"][
        "params"
    ]["sensor_cfg"]["_target_"] == scene_entity_target


def test_quiet_domain_randomization_keeps_contact_sensitive_physics():
    cfg = _read_yaml(
        "holomotion/config/env/domain_randomization/"
        "domain_rand_quiet_medium.yaml"
    )
    domain_rand = cfg["domain_rand"]

    assert "rigid_body_material" in domain_rand
    assert "action_delay" in domain_rand
    assert "obs_noise" in domain_rand
    assert domain_rand["rigid_body_material"]["params"][
        "restitution_range"
    ] == [0.0, 0.02]


def test_quiet_scripts_source_train_env_and_use_expected_configs():
    train_script = (
        ROOT / "holomotion/scripts/training/train_quiet_goal_velocity.sh"
    )
    eval_script = (
        ROOT / "holomotion/scripts/evaluation/eval_quiet_goal_velocity.sh"
    )
    eval_cfg = (
        ROOT / "holomotion/config/evaluation/eval_quiet_goal_velocity.yaml"
    )

    train_text = train_script.read_text()
    eval_text = eval_script.read_text()
    eval_cfg_text = eval_cfg.read_text()

    assert "source train.env" in train_text
    assert "source train.env" in eval_text
    assert "${Train_CONDA_PREFIX}" in train_text
    assert "${Train_CONDA_PREFIX}" in eval_text
    assert (
        "--config-name=training/velocity_tracking/"
        "train_g1_29dof_quiet_goal_velocity_mlp"
        in train_text
    )
    assert "--config-name=evaluation/eval_quiet_goal_velocity" in eval_text
    assert "/env: quiet_goal_velocity_tracking" in eval_cfg_text

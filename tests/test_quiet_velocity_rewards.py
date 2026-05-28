import sys
from types import ModuleType, SimpleNamespace

import torch

from test_root_rel_rewards import _load_rewards_module


class _Scene:
    def __init__(self, sensor, asset):
        self.sensors = {"contact_forces": sensor}
        self._asset = asset

    def __getitem__(self, name):
        assert name == "robot"
        return self._asset


def _load_quiet_rewards_module(monkeypatch):
    fake_frame_utils = ModuleType("holomotion.src.utils.frame_utils")
    fake_frame_utils.positions_world_to_env_frame = lambda pos, _env: pos
    fake_frame_utils.root_relative_positions_from_env_frame = (
        lambda pos, root_pos, root_quat: pos - root_pos[:, None, :]
    )
    fake_frame_utils.root_relative_positions_from_mixed_position_frames = (
        lambda pos, root_pos, root_quat: pos - root_pos[:, None, :]
    )
    monkeypatch.setitem(
        sys.modules,
        "holomotion.src.utils.frame_utils",
        fake_frame_utils,
    )
    return _load_rewards_module(monkeypatch)


def test_contact_force_rate_penalty_uses_history_difference(monkeypatch):
    rewards = _load_quiet_rewards_module(monkeypatch)
    forces = torch.tensor(
        [[[[0.0, 0.0, 10.0]], [[0.0, 0.0, 30.0]], [[0.0, 0.0, 70.0]]]]
    )

    penalty = rewards._contact_force_rate_from_history(forces, body_ids=[0])

    assert torch.allclose(penalty, torch.tensor([60.0]))


def test_touchdown_vertical_velocity_uses_new_contacts_only(monkeypatch):
    rewards = _load_quiet_rewards_module(monkeypatch)
    prev_contact = torch.tensor([[False, True]])
    cur_contact = torch.tensor([[True, True]])
    body_vel_z = torch.tensor([[-0.4, -1.0]])

    penalty = rewards._touchdown_vertical_speed(
        prev_contact,
        cur_contact,
        body_vel_z,
    )

    assert torch.allclose(penalty, torch.tensor([0.4]))


def test_feet_touchdown_vertical_velocity_reads_sensor_and_asset(monkeypatch):
    rewards = _load_quiet_rewards_module(monkeypatch)
    forces = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
                [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
            ]
        ]
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    body_vel = torch.tensor([[[0.0, 0.0, -0.4], [0.0, 0.0, -1.0]]])
    asset = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=body_vel))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=_Scene(sensor=sensor, asset=asset),
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1])
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0, 1])

    penalty = rewards.feet_touchdown_vertical_velocity_l1(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )

    assert torch.allclose(penalty, torch.tensor([0.4]))


def test_feet_contact_force_rate_reads_sensor(monkeypatch):
    rewards = _load_quiet_rewards_module(monkeypatch)
    forces = torch.tensor(
        [[[[0.0, 0.0, 10.0]], [[0.0, 0.0, 30.0]], [[0.0, 0.0, 70.0]]]]
    )
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene=SimpleNamespace(sensors={"contact_forces": sensor}),
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0])

    penalty = rewards.feet_contact_force_rate_l1(env, sensor_cfg=sensor_cfg)

    assert torch.allclose(penalty, torch.tensor([60.0]))


def test_quiet_reference_rewards_return_zero_without_reference_state(
    monkeypatch,
):
    rewards = _load_quiet_rewards_module(monkeypatch)
    env = SimpleNamespace(num_envs=2, device="cpu")

    joint_penalty = rewards.quiet_reference_joint_pose_l2(
        env,
        lower_body_joint_pattern=".*",
    )
    velocity_penalty = rewards.quiet_reference_base_velocity_l2(env)

    assert torch.equal(joint_penalty, torch.zeros(2))
    assert torch.equal(velocity_penalty, torch.zeros(2))


def test_quiet_reference_rewards_delegate_to_reference_state(monkeypatch):
    rewards = _load_quiet_rewards_module(monkeypatch)

    class _ReferenceState:
        def joint_pose_error(self, env, lower_body_joint_pattern, asset_cfg):
            assert env.num_envs == 2
            assert lower_body_joint_pattern == ".*ankle.*"
            assert asset_cfg.name == "robot"
            return torch.tensor([0.1, 0.2])

        def base_velocity_error(self, env):
            assert env.num_envs == 2
            return torch.tensor([0.3, 0.4])

    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        quiet_reference_state=_ReferenceState(),
    )

    joint_penalty = rewards.quiet_reference_joint_pose_l2(
        env,
        lower_body_joint_pattern=".*ankle.*",
    )
    velocity_penalty = rewards.quiet_reference_base_velocity_l2(env)

    assert torch.equal(joint_penalty, torch.tensor([0.1, 0.2]))
    assert torch.equal(velocity_penalty, torch.tensor([0.3, 0.4]))

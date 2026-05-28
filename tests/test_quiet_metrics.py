import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
QUIET_REFERENCE_PATH = (
    ROOT
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "quiet_reference.py"
)
QUIET_METRICS_PATH = (
    ROOT / "holomotion" / "src" / "evaluation" / "quiet_metrics.py"
)


def _load_quiet_reference_module():
    spec = importlib.util.spec_from_file_location(
        "quiet_reference_under_test",
        QUIET_REFERENCE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_quiet_metrics_module():
    spec = importlib.util.spec_from_file_location(
        "quiet_metrics_under_test",
        QUIET_METRICS_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quiet_reference_index_filters_by_velocity(tmp_path: Path):
    path = tmp_path / "metadata.yaml"
    path.write_text(
        """
clips:
  - path: data/quiet_refs/a.h5
    robot: g1_29dof
    mean_vx: 0.20
    mean_vy: 0.00
    mean_yaw_rate: 0.10
    quiet_score: 0.85
  - path: data/quiet_refs/b.h5
    robot: g1_29dof
    mean_vx: 0.90
    mean_vy: 0.00
    mean_yaw_rate: 0.00
    quiet_score: 0.30
""",
        encoding="utf-8",
    )

    quiet_reference = _load_quiet_reference_module()
    index = quiet_reference.QuietReferenceIndex.from_yaml(path)
    matches = index.match(
        vx=0.25,
        vy=0.0,
        yaw_rate=0.1,
        max_lin_error=0.2,
        max_yaw_error=0.2,
    )

    assert [clip.path for clip in matches] == ["data/quiet_refs/a.h5"]


def test_quiet_reference_index_sorts_matches_by_quiet_score(tmp_path: Path):
    path = tmp_path / "metadata.yaml"
    path.write_text(
        """
clips:
  - path: data/quiet_refs/noisier.h5
    robot: g1_29dof
    mean_vx: 0.20
    mean_vy: 0.00
    mean_yaw_rate: 0.00
    quiet_score: 0.20
  - path: data/quiet_refs/quieter.h5
    robot: g1_29dof
    mean_vx: 0.22
    mean_vy: 0.00
    mean_yaw_rate: 0.00
    quiet_score: 0.90
""",
        encoding="utf-8",
    )

    quiet_reference = _load_quiet_reference_module()
    index = quiet_reference.QuietReferenceIndex.from_yaml(path)
    matches = index.match(
        vx=0.2,
        vy=0.0,
        yaw_rate=0.0,
        max_lin_error=0.1,
        max_yaw_error=0.1,
    )

    assert [clip.path for clip in matches] == [
        "data/quiet_refs/quieter.h5",
        "data/quiet_refs/noisier.h5",
    ]


def _write_reference_csv(path: Path, joint_degrees: list[float]) -> None:
    rows = [
        "Frame,root_translateX,root_translateY,root_translateZ,"
        "root_rotateX,root_rotateY,root_rotateZ,"
        "left_hip_pitch_joint_dof,right_knee_joint_dof",
    ]
    for frame, value in enumerate(joint_degrees):
        rows.append(
            f"{frame},0,0,80,0,0,0,{value},{value * 2}"
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def test_quiet_reference_state_loads_csv_joint_positions_in_radians(
    tmp_path: Path,
):
    csv_path = tmp_path / "walk.csv"
    _write_reference_csv(csv_path, [90.0, 180.0])
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        f"""
clips:
  - path: {csv_path}
    robot: g1_29dof
    mean_vx: 0.50
    mean_vy: 0.00
    mean_yaw_rate: 0.00
    quiet_score: 0.80
""",
        encoding="utf-8",
    )

    quiet_reference = _load_quiet_reference_module()
    state = quiet_reference.QuietReferenceState.from_yaml(
        metadata_path,
        num_envs=1,
        device="cpu",
    )

    trajectory = state.trajectories[0]
    assert trajectory.joint_names == (
        "left_hip_pitch_joint",
        "right_knee_joint",
    )
    assert trajectory.joint_positions.shape == (2, 2)
    assert trajectory.joint_positions[0, 0].item() == pytest.approx(
        torch.pi / 2
    )
    assert trajectory.joint_positions[1, 0].item() == pytest.approx(torch.pi)


def test_quiet_reference_state_assigns_matches_and_computes_errors(
    tmp_path: Path,
):
    csv_path = tmp_path / "walk.csv"
    _write_reference_csv(csv_path, [90.0])
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        f"""
clips:
  - path: {csv_path}
    robot: g1_29dof
    mean_vx: 0.50
    mean_vy: 0.00
    mean_yaw_rate: 0.10
    quiet_score: 0.80
""",
        encoding="utf-8",
    )

    quiet_reference = _load_quiet_reference_module()
    state = quiet_reference.QuietReferenceState.from_yaml(
        metadata_path,
        num_envs=2,
        device="cpu",
    )
    state.assign_by_command(
        env_ids=torch.tensor([0, 1]),
        commands_b=torch.tensor([[0.52, 0.0, 0.12], [-0.20, 0.0, 0.0]]),
        max_lin_error=0.1,
        max_yaw_error=0.1,
    )

    assert torch.equal(state.assigned_clip_ids, torch.tensor([0, -1]))

    asset = type("Asset", (), {})()
    asset.joint_names = ["left_hip_pitch_joint", "right_knee_joint"]
    asset.data = type("AssetData", (), {})()
    asset.data.joint_pos = torch.tensor(
        [[torch.pi / 2 + 0.1, torch.pi], [0.0, 0.0]]
    )
    asset.data.root_lin_vel_b = torch.tensor(
        [[0.40, 0.0, 0.0], [-0.20, 0.0, 0.0]]
    )
    asset.data.root_ang_vel_b = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )

    class _Scene:
        def __getitem__(self, name):
            assert name == "robot"
            return asset

    env = type("Env", (), {})()
    env.num_envs = 2
    env.device = "cpu"
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.scene = _Scene()

    asset_cfg = type("Cfg", (), {"name": "robot"})()
    joint_error = state.joint_pose_error(
        env,
        lower_body_joint_pattern=".*(hip|knee).*",
        asset_cfg=asset_cfg,
    )
    velocity_error = state.base_velocity_error(env)

    assert torch.allclose(joint_error, torch.tensor([0.005, 0.0]))
    assert torch.allclose(velocity_error, torch.tensor([0.02, 0.0]))


def test_summarize_quiet_metrics_reports_contact_and_tracking():
    quiet_metrics = _load_quiet_metrics_module()

    metrics = quiet_metrics.summarize_quiet_metrics(
        commanded_velocity=torch.tensor([[0.2, 0.0, 0.0]]),
        measured_velocity=torch.tensor([[0.1, 0.0, 0.0]]),
        touchdown_vz=torch.tensor([0.3, 0.1]),
        peak_normal_force=torch.tensor([120.0, 80.0]),
        force_rate=torch.tensor([40.0, 60.0]),
        foot_slip=torch.tensor([0.02, 0.01]),
        root_jerk=torch.tensor([1.5, 2.0]),
    )

    assert metrics["velocity_error_mean"] == pytest.approx(0.1)
    assert metrics["touchdown_vz_mean"] == pytest.approx(0.2)
    assert metrics["peak_normal_force_p95"] == pytest.approx(118.0)
    assert metrics["force_rate_mean"] == pytest.approx(50.0)
    assert metrics["foot_slip_mean"] == pytest.approx(0.015)
    assert metrics["root_jerk_mean"] == pytest.approx(1.75)

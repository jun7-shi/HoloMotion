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

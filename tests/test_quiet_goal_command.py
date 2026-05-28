import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_quiet_goal_command.py"
)


class _FakeVelocityCommand:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.num_envs = env.num_envs
        self.device = env.device
        self.parent_resample_called = False

    def _resample_command(self, _env_ids):
        self.parent_resample_called = True


class _FakeVelocityCommandCfg:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _convert_ranges_dict_to_object(ranges_dict):
    return SimpleNamespace(
        **{
            key: tuple(value) if isinstance(value, list) else value
            for key, value in ranges_dict.items()
        }
    )


def _load_quiet_command_module(monkeypatch):
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    velocity_module = ModuleType(
        "holomotion.src.env.isaaclab_components."
        "isaaclab_velocity_tracking_command"
    )
    velocity_module.HoloMotionUniformVelocityCommand = _FakeVelocityCommand
    velocity_module.HoloMotionUniformVelocityCommandCfg = (
        _FakeVelocityCommandCfg
    )
    velocity_module._convert_ranges_dict_to_object = (
        _convert_ranges_dict_to_object
    )

    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, velocity_module.__name__, velocity_module)

    spec = importlib.util.spec_from_file_location(
        "isaaclab_quiet_goal_command_under_test",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_task_state_initializes_named_task_ids(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)

    state = module.QuietGoalTaskState(num_envs=3, device="cpu")

    assert state.task_name_to_id == {
        "quiet_style_imitation": 0,
        "goal_velocity_generalization": 1,
    }
    assert torch.equal(state.task_ids, torch.ones(3, dtype=torch.long))
    assert torch.equal(state.quiet_mode, torch.ones(3))


def test_task_state_samples_imitation_probability(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)
    state = module.QuietGoalTaskState(num_envs=4, device="cpu")

    state.assign(
        env_ids=torch.tensor([0, 1, 2, 3]),
        imitation_mask=torch.tensor([True, False, True, False]),
    )

    assert torch.equal(state.task_ids, torch.tensor([0, 1, 0, 1]))


def test_task_state_samples_quiet_mode(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)
    state = module.QuietGoalTaskState(num_envs=4, device="cpu")

    state.assign_quiet_mode(
        env_ids=torch.tensor([0, 1, 2, 3]),
        quiet_mask=torch.tensor([True, False, True, False]),
    )

    assert torch.equal(state.quiet_mode, torch.tensor([1.0, 0.0, 1.0, 0.0]))


def test_builder_accepts_quiet_goal_velocity_command(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)

    cfg = module.build_quiet_goal_velocity_commands_config(
        {
            "base_velocity": {
                "type": "QuietGoalVelocityCommandCfg",
                "params": {
                    "asset_name": "robot",
                    "ranges": {
                        "lin_vel_x": [-0.4, 0.8],
                        "lin_vel_y": [-0.25, 0.25],
                        "ang_vel_z": [-0.6, 0.6],
                        "heading": [-3.14, 3.14],
                    },
                },
            }
        }
    )

    assert isinstance(cfg.base_velocity, module.QuietGoalVelocityCommandCfg)
    assert cfg.base_velocity.quiet_mode_prob == 1.0
    assert cfg.base_velocity.ranges.lin_vel_x == (-0.4, 0.8)

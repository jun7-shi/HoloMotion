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
        self.vel_command_b = torch.zeros(self.num_envs, 3)
        self.parent_resample_called = False
        self.parent_update_called = False

    def _resample_command(self, env_ids):
        self.parent_resample_called = True
        self.vel_command_b[env_ids] = torch.tensor([0.5, 0.0, 0.1])

    def _update_command(self):
        self.parent_update_called = True


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

    quiet_reference_module = ModuleType(
        "holomotion.src.env.isaaclab_components.quiet_reference"
    )

    class _FakeReferenceState:
        def __init__(self, path, num_envs, device):
            self.path = path
            self.num_envs = num_envs
            self.device = device
            self.assigned = None

        @classmethod
        def from_yaml(cls, path, num_envs, device):
            return cls(path=path, num_envs=num_envs, device=device)

        def assign_by_command(self, env_ids, commands_b):
            self.assigned = (env_ids.clone(), commands_b.clone())

    quiet_reference_module.QuietReferenceState = _FakeReferenceState

    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, velocity_module.__name__, velocity_module)
    monkeypatch.setitem(
        sys.modules,
        quiet_reference_module.__name__,
        quiet_reference_module,
    )

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


def test_command_attaches_reference_state_from_metadata(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)
    cfg = SimpleNamespace(
        reference_metadata_path="data/quiet_references/g1_quiet_metadata.yaml",
        imitation_prob_initial=1.0,
        quiet_mode_prob=1.0,
    )
    env = SimpleNamespace(num_envs=2, device="cpu")

    command = module.QuietGoalVelocityCommand(cfg, env)

    assert env.quiet_reference_state.path == (
        "data/quiet_references/g1_quiet_metadata.yaml"
    )
    assert command.quiet_reference_state is env.quiet_reference_state


def test_command_assigns_reference_after_command_update(monkeypatch):
    module = _load_quiet_command_module(monkeypatch)
    cfg = SimpleNamespace(
        reference_metadata_path="data/quiet_references/g1_quiet_metadata.yaml",
        imitation_prob_initial=1.0,
        quiet_mode_prob=1.0,
    )
    env = SimpleNamespace(num_envs=2, device="cpu")
    command = module.QuietGoalVelocityCommand(cfg, env)

    command._resample_command([0, 1])
    command._update_command()

    assigned_env_ids, assigned_commands = (
        command.quiet_reference_state.assigned
    )
    assert command.parent_update_called
    assert torch.equal(assigned_env_ids, torch.tensor([0, 1]))
    assert torch.equal(
        assigned_commands,
        torch.tensor([[0.5, 0.0, 0.1], [0.5, 0.0, 0.1]]),
    )

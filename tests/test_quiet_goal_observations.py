import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_PATH = (
    ROOT
    / "holomotion"
    / "src"
    / "env"
    / "isaaclab_components"
    / "isaaclab_observation.py"
)


class _DummyConfig:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _load_observation_module(monkeypatch):
    isaaclab = ModuleType("isaaclab")
    isaaclab_mdp = ModuleType("isaaclab.envs.mdp")
    isaaclab_math = ModuleType("isaaclab.utils.math")
    isaaclab_math.__getattr__ = lambda _name: (lambda *args, **kwargs: None)
    isaaclab_noise = ModuleType("isaaclab.utils.noise")
    isaaclab_noise.__getattr__ = lambda _name: _DummyConfig

    isaaclab_envs = ModuleType("isaaclab.envs")
    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_envs.ManagerBasedRLEnvCfg = _DummyConfig
    isaaclab_envs.ViewerCfg = _DummyConfig
    isaaclab_sim = ModuleType("isaaclab.sim")
    isaaclab_sim.__getattr__ = lambda _name: _DummyConfig
    isaaclab_actuators = ModuleType("isaaclab.actuators")
    isaaclab_actuators.ImplicitActuatorCfg = _DummyConfig
    isaaclab_assets = ModuleType("isaaclab.assets")
    isaaclab_assets.Articulation = object
    isaaclab_assets.ArticulationCfg = _DummyConfig
    isaaclab_assets.AssetBaseCfg = _DummyConfig
    isaaclab_managers = ModuleType("isaaclab.managers")
    isaaclab_managers.__getattr__ = lambda _name: _DummyConfig
    isaaclab_markers = ModuleType("isaaclab.markers")
    isaaclab_markers.VisualizationMarkers = _DummyConfig
    isaaclab_markers.VisualizationMarkersCfg = _DummyConfig
    isaaclab_markers_config = ModuleType("isaaclab.markers.config")
    isaaclab_markers_config.FRAME_MARKER_CFG = _DummyConfig
    isaaclab_scene = ModuleType("isaaclab.scene")
    isaaclab_scene.InteractiveSceneCfg = _DummyConfig
    isaaclab_sensors = ModuleType("isaaclab.sensors")
    isaaclab_sensors.ContactSensorCfg = _DummyConfig
    isaaclab_sensors.RayCasterCfg = _DummyConfig
    isaaclab_sensors.patterns = _DummyConfig
    isaaclab_terrains = ModuleType("isaaclab.terrains")
    isaaclab_terrains.TerrainImporterCfg = _DummyConfig
    isaaclab_utils = ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    omegaconf = ModuleType("omegaconf")
    omegaconf.DictConfig = dict
    omegaconf.ListConfig = list
    omegaconf.OmegaConf = SimpleNamespace(
        to_container=lambda value, resolve=True: value
    )

    fake_utils_module = ModuleType(
        "holomotion.src.env.isaaclab_components.isaaclab_utils"
    )
    fake_utils_module.resolve_holo_config = lambda value: value
    fake_frame_utils = ModuleType("holomotion.src.utils.frame_utils")
    fake_frame_utils.positions_world_to_env_frame = lambda pos, _env: pos
    fake_frame_utils.root_relative_positions_from_env_frame = (
        lambda pos, root_pos, root_quat: pos - root_pos[:, None, :]
    )

    isaaclab.envs = isaaclab_envs
    isaaclab.sim = isaaclab_sim
    isaaclab.actuators = isaaclab_actuators
    isaaclab.assets = isaaclab_assets
    isaaclab.managers = isaaclab_managers
    isaaclab.markers = isaaclab_markers
    isaaclab.scene = isaaclab_scene
    isaaclab.sensors = isaaclab_sensors
    isaaclab.terrains = isaaclab_terrains
    isaaclab.utils = isaaclab_utils
    isaaclab_envs.mdp = isaaclab_mdp
    isaaclab_utils.math = isaaclab_math
    isaaclab_utils.noise = isaaclab_noise

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.envs.mdp": isaaclab_mdp,
        "isaaclab.utils.math": isaaclab_math,
        "isaaclab.utils.noise": isaaclab_noise,
        "isaaclab.envs": isaaclab_envs,
        "isaaclab.sim": isaaclab_sim,
        "isaaclab.actuators": isaaclab_actuators,
        "isaaclab.assets": isaaclab_assets,
        "isaaclab.managers": isaaclab_managers,
        "isaaclab.markers": isaaclab_markers,
        "isaaclab.markers.config": isaaclab_markers_config,
        "isaaclab.scene": isaaclab_scene,
        "isaaclab.sensors": isaaclab_sensors,
        "isaaclab.terrains": isaaclab_terrains,
        "isaaclab.utils": isaaclab_utils,
        "omegaconf": omegaconf,
        "holomotion.src.env.isaaclab_components.isaaclab_utils": (
            fake_utils_module
        ),
        "holomotion.src.utils.frame_utils": fake_frame_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "quiet_observation_under_test",
        OBSERVATION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeCommand:
    quiet_mode = torch.tensor([1.0, 0.0])
    task_ids = torch.tensor([0, 1])


class _FakeCommandManager:
    def get_term(self, name):
        assert name == "base_velocity"
        return _FakeCommand()


class _FakeEnv:
    num_envs = 2
    device = "cpu"
    command_manager = _FakeCommandManager()


def test_mirror_velocity_command_supports_quiet_command(monkeypatch):
    observation = _load_observation_module(monkeypatch)

    command = torch.tensor([[1.0, 0.3, 0.2, 0.4, 1.0]])

    mirrored = observation.MirrorFunctions.mirror_velocity_command(command)

    assert torch.equal(mirrored, torch.tensor([[1.0, 0.3, -0.2, -0.4, 1.0]]))


def test_quiet_goal_velocity_command_observation(monkeypatch):
    observation = _load_observation_module(monkeypatch)
    velocity = torch.tensor([[0.2, 0.0, 0.0], [0.0, 0.0, 0.0]])
    observation.isaaclab_mdp.generated_commands = (
        lambda _env, command_name: velocity
    )

    obs = observation.ObservationFunctions._get_obs_quiet_goal_velocity_command(
        _FakeEnv()
    )

    assert torch.equal(
        obs,
        torch.tensor(
            [
                [1.0, 0.2, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_quiet_task_id_observation_is_column_vector(monkeypatch):
    observation = _load_observation_module(monkeypatch)

    obs = observation.ObservationFunctions._get_obs_quiet_task_id(_FakeEnv())

    assert obs.shape == (2, 1)
    assert torch.equal(obs[:, 0], torch.tensor([0.0, 1.0]))


def test_quiet_actor_observation_schema_excludes_references():
    cfg_path = (
        ROOT
        / "holomotion/config/env/observations/velocity_tracking/"
        "obs_quiet_goal_velocity.yaml"
    )
    cfg = yaml.safe_load(cfg_path.read_text())
    terms = cfg["obs"]["obs_groups"]["unified"]["atomic_obs_list"]
    actor_terms = {
        next(iter(term.values()))["func"]
        for term in terms
        if next(iter(term)).startswith("actor_")
    }

    assert "quiet_goal_velocity_command" in actor_terms
    assert not any("ref" in func for func in actor_terms)

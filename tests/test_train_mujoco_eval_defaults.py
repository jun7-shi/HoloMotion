from omegaconf import OmegaConf

from holomotion.src.training.train import _get_enabled_mujoco_eval_config


def test_missing_mujoco_eval_config_is_disabled():
    cfg = OmegaConf.create({})

    assert _get_enabled_mujoco_eval_config(cfg) is None


def test_disabled_mujoco_eval_config_is_disabled():
    cfg = OmegaConf.create({"mujoco_eval": {"enabled": False}})

    assert _get_enabled_mujoco_eval_config(cfg) is None


def test_enabled_mujoco_eval_config_is_returned():
    cfg = OmegaConf.create({"mujoco_eval": {"enabled": True}})

    assert _get_enabled_mujoco_eval_config(cfg) is cfg.mujoco_eval

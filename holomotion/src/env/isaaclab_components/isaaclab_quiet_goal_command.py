# Project HoloMotion
#
# Copyright (c) 2024-2026 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

from __future__ import annotations
from dataclasses import MISSING, dataclass
from importlib import import_module
from typing import Sequence

import torch
from isaaclab.utils import configclass

from holomotion.src.env.isaaclab_components.quiet_reference import (
    QuietReferenceState,
)

_velocity_command = import_module(
    "holomotion.src.env.isaaclab_components."
    "isaaclab_velocity_tracking_command"
)
HoloMotionUniformVelocityCommand = (
    _velocity_command.HoloMotionUniformVelocityCommand
)
HoloMotionUniformVelocityCommandCfg = (
    _velocity_command.HoloMotionUniformVelocityCommandCfg
)
_convert_ranges_dict_to_object = (
    _velocity_command._convert_ranges_dict_to_object
)


@dataclass
class QuietGoalTaskState:
    """Per-environment task and quiet-mode buffers."""

    num_envs: int
    device: str | torch.device

    def __post_init__(self):
        self.task_name_to_id = {
            "quiet_style_imitation": 0,
            "goal_velocity_generalization": 1,
        }
        self.task_ids = torch.ones(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self.quiet_mode = torch.ones(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )

    def assign(
        self,
        env_ids: torch.Tensor,
        imitation_mask: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.task_ids.device, dtype=torch.long)
        imitation_mask = imitation_mask.to(
            device=self.task_ids.device,
            dtype=torch.bool,
        )
        self.task_ids[env_ids] = torch.where(
            imitation_mask,
            torch.zeros_like(env_ids),
            torch.ones_like(env_ids),
        )

    def assign_quiet_mode(
        self,
        env_ids: torch.Tensor,
        quiet_mask: torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.quiet_mode.device, dtype=torch.long)
        quiet_mask = quiet_mask.to(
            device=self.quiet_mode.device,
            dtype=torch.bool,
        )
        self.quiet_mode[env_ids] = quiet_mask.to(dtype=self.quiet_mode.dtype)


class QuietGoalVelocityCommand(HoloMotionUniformVelocityCommand):
    """Velocity command with quiet mode and task sampling buffers."""

    cfg: "QuietGoalVelocityCommandCfg"

    def __init__(self, cfg: "QuietGoalVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.task_state = QuietGoalTaskState(self.num_envs, self.device)
        env.holo_task_ids = self.task_state.task_ids
        env.holo_task_name_to_id = self.task_state.task_name_to_id
        self._pending_reference_env_ids: torch.Tensor | None = None
        self.quiet_reference_state = None
        if str(self.cfg.reference_metadata_path).strip():
            self.quiet_reference_state = QuietReferenceState.from_yaml(
                self.cfg.reference_metadata_path,
                num_envs=self.num_envs,
                device=self.device,
            )
            env.quiet_reference_state = self.quiet_reference_state

    @property
    def quiet_mode(self) -> torch.Tensor:
        return self.task_state.quiet_mode

    @property
    def task_ids(self) -> torch.Tensor:
        return self.task_state.task_ids

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(
            env_ids,
            device=self.device,
            dtype=torch.long,
        )
        imitation_mask = (
            torch.rand(env_ids_t.numel(), device=self.device)
            < float(self.cfg.imitation_prob_initial)
        )
        quiet_mask = (
            torch.rand(env_ids_t.numel(), device=self.device)
            < float(self.cfg.quiet_mode_prob)
        )
        self.task_state.assign(env_ids_t, imitation_mask)
        self.task_state.assign_quiet_mode(env_ids_t, quiet_mask)
        self._pending_reference_env_ids = env_ids_t

    def _update_command(self):
        super()._update_command()
        if (
            self.quiet_reference_state is None
            or self._pending_reference_env_ids is None
        ):
            return
        env_ids = self._pending_reference_env_ids
        self.quiet_reference_state.assign_by_command(
            env_ids=env_ids,
            commands_b=self.vel_command_b[env_ids],
        )
        self._pending_reference_env_ids = None


@configclass
class QuietGoalVelocityCommandCfg(HoloMotionUniformVelocityCommandCfg):
    """Configuration for quiet goal-velocity command sampling."""

    class_type: type = QuietGoalVelocityCommand

    asset_name: str = MISSING
    quiet_mode_prob: float = 1.0
    imitation_prob_initial: float = 1.0
    imitation_prob_final: float = 0.5
    reference_metadata_path: str = ""


@configclass
class QuietGoalVelocityCommandsCfg:
    pass


def build_quiet_goal_velocity_commands_config(
    command_config_dict: dict,
) -> QuietGoalVelocityCommandsCfg:
    commands_cfg = QuietGoalVelocityCommandsCfg()

    for name, cfg in command_config_dict.items():
        command_type = cfg.get("type", "QuietGoalVelocityCommandCfg")
        params = cfg.get("params", {}).copy()

        if "ranges" in params and isinstance(params["ranges"], dict):
            params["ranges"] = _convert_ranges_dict_to_object(
                params["ranges"]
            )

        if "limit_ranges" in params and isinstance(
            params["limit_ranges"],
            dict,
        ):
            params["limit_ranges"] = _convert_ranges_dict_to_object(
                params["limit_ranges"]
            )

        if command_type != "QuietGoalVelocityCommandCfg":
            raise ValueError(
                f"Unknown quiet goal velocity command type: {command_type}"
            )

        setattr(commands_cfg, name, QuietGoalVelocityCommandCfg(**params))

    return commands_cfg

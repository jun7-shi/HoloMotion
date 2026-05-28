from __future__ import annotations
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class QuietReferenceClip:
    path: str
    robot: str
    mean_vx: float
    mean_vy: float
    mean_yaw_rate: float
    quiet_score: float


@dataclass(frozen=True)
class QuietReferenceIndex:
    clips: tuple[QuietReferenceClip, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QuietReferenceIndex":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        clips = tuple(
            QuietReferenceClip(**clip_payload)
            for clip_payload in payload["clips"]
        )
        return cls(clips=clips)

    def match(
        self,
        vx: float,
        vy: float,
        yaw_rate: float,
        max_lin_error: float,
        max_yaw_error: float,
    ) -> list[QuietReferenceClip]:
        matches = []
        for clip in self.clips:
            lin_error = (
                (clip.mean_vx - vx) ** 2 + (clip.mean_vy - vy) ** 2
            ) ** 0.5
            yaw_error = abs(clip.mean_yaw_rate - yaw_rate)
            if lin_error <= max_lin_error and yaw_error <= max_yaw_error:
                matches.append(clip)
        return sorted(matches, key=lambda clip: clip.quiet_score, reverse=True)


@dataclass(frozen=True)
class QuietReferenceTrajectory:
    clip: QuietReferenceClip
    joint_names: tuple[str, ...]
    joint_positions: torch.Tensor
    base_velocity: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_positions.shape[0])


class QuietReferenceState:
    """Runtime reference assignment state for quiet style rewards."""

    def __init__(
        self,
        index: QuietReferenceIndex,
        trajectories: tuple[QuietReferenceTrajectory, ...],
        num_envs: int,
        device: str | torch.device,
    ) -> None:
        self.index = index
        self.trajectories = trajectories
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.assigned_clip_ids = torch.full(
            (num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        num_envs: int,
        device: str | torch.device,
    ) -> "QuietReferenceState":
        metadata_path = Path(path)
        index = QuietReferenceIndex.from_yaml(metadata_path)
        trajectories = tuple(
            _load_clip_trajectory(clip, metadata_path.parent, device)
            for clip in index.clips
        )
        return cls(
            index=index,
            trajectories=trajectories,
            num_envs=num_envs,
            device=device,
        )

    def assign_by_command(
        self,
        env_ids: torch.Tensor,
        commands_b: torch.Tensor,
        max_lin_error: float = 0.25,
        max_yaw_error: float = 0.25,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        commands_b = commands_b.to(device=self.device, dtype=torch.float32)
        for row, env_id in zip(commands_b, env_ids, strict=True):
            matches = self.index.match(
                vx=float(row[0].item()),
                vy=float(row[1].item()),
                yaw_rate=float(row[2].item()),
                max_lin_error=max_lin_error,
                max_yaw_error=max_yaw_error,
            )
            if len(matches) == 0:
                self.assigned_clip_ids[env_id] = -1
            else:
                self.assigned_clip_ids[env_id] = self._clip_id(matches[0])

    def joint_pose_error(
        self,
        env,
        lower_body_joint_pattern: str,
        asset_cfg,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        joint_ids, reference_joint_ids = self._matched_joint_ids(
            asset.joint_names,
            lower_body_joint_pattern,
        )
        if len(joint_ids) == 0:
            return torch.zeros(env.num_envs, device=env.device)

        current = asset.data.joint_pos[:, joint_ids]
        target = torch.zeros_like(current)
        valid = self.assigned_clip_ids >= 0
        if not torch.any(valid):
            return torch.zeros(env.num_envs, device=env.device)

        for clip_id in torch.unique(self.assigned_clip_ids[valid]).tolist():
            clip_env_ids = (self.assigned_clip_ids == clip_id).nonzero(
                as_tuple=False
            ).flatten()
            trajectory = self.trajectories[int(clip_id)]
            frame_ids = self._frame_ids(env, clip_env_ids, trajectory)
            target[clip_env_ids] = trajectory.joint_positions[
                frame_ids
            ][:, reference_joint_ids]

        error = torch.mean(torch.square(current - target), dim=1)
        return torch.where(valid, error, torch.zeros_like(error))

    def base_velocity_error(self, env) -> torch.Tensor:
        asset = env.scene["robot"]
        target = torch.zeros(env.num_envs, 3, device=env.device)
        valid = self.assigned_clip_ids >= 0
        if not torch.any(valid):
            return torch.zeros(env.num_envs, device=env.device)

        for clip_id in torch.unique(self.assigned_clip_ids[valid]).tolist():
            clip_env_ids = (self.assigned_clip_ids == clip_id).nonzero(
                as_tuple=False
            ).flatten()
            target[clip_env_ids] = self.trajectories[
                int(clip_id)
            ].base_velocity

        measured = torch.zeros_like(target)
        measured[:, :2] = asset.data.root_lin_vel_b[:, :2]
        measured[:, 2] = asset.data.root_ang_vel_b[:, 2]
        error = torch.sum(torch.square(measured - target), dim=1)
        return torch.where(valid, error, torch.zeros_like(error))

    def _clip_id(self, clip: QuietReferenceClip) -> int:
        for index, candidate in enumerate(self.index.clips):
            if candidate.path == clip.path:
                return index
        return -1

    def _matched_joint_ids(
        self,
        asset_joint_names: list[str],
        lower_body_joint_pattern: str,
    ) -> tuple[list[int], list[int]]:
        pattern = re.compile(lower_body_joint_pattern)
        if len(self.trajectories) == 0:
            return [], []
        reference_joint_names = self.trajectories[0].joint_names
        reference_name_to_id = {
            name: index for index, name in enumerate(reference_joint_names)
        }

        joint_ids = []
        reference_joint_ids = []
        for joint_id, joint_name in enumerate(asset_joint_names):
            if not pattern.fullmatch(joint_name):
                continue
            reference_joint_id = reference_name_to_id.get(joint_name)
            if reference_joint_id is None:
                continue
            joint_ids.append(joint_id)
            reference_joint_ids.append(reference_joint_id)
        return joint_ids, reference_joint_ids

    def _frame_ids(
        self,
        env,
        env_ids: torch.Tensor,
        trajectory: QuietReferenceTrajectory,
    ) -> torch.Tensor:
        if hasattr(env, "episode_length_buf"):
            step_ids = env.episode_length_buf[env_ids].to(
                device=self.device,
                dtype=torch.long,
            )
        else:
            step_ids = torch.zeros_like(env_ids, dtype=torch.long)
        return step_ids % trajectory.num_frames


def _load_clip_trajectory(
    clip: QuietReferenceClip,
    metadata_dir: Path,
    device: str | torch.device,
) -> QuietReferenceTrajectory:
    path = _resolve_clip_path(clip.path, metadata_dir)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported quiet reference format: {path}")
    joint_names, joint_positions = _load_bones_seed_csv(path)
    base_velocity = torch.tensor(
        [clip.mean_vx, clip.mean_vy, clip.mean_yaw_rate],
        dtype=torch.float32,
        device=device,
    )
    return QuietReferenceTrajectory(
        clip=clip,
        joint_names=joint_names,
        joint_positions=joint_positions.to(device=device),
        base_velocity=base_velocity,
    )


def _resolve_clip_path(path: str, metadata_dir: Path) -> Path:
    clip_path = Path(path)
    if clip_path.is_absolute():
        return clip_path
    if clip_path.exists():
        return clip_path
    return metadata_dir / clip_path


def _load_bones_seed_csv(path: Path) -> tuple[tuple[str, ...], torch.Tensor]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        joint_columns = [
            name
            for name in (reader.fieldnames or [])
            if name.endswith("_joint_dof")
        ]
        joint_names = tuple(
            name.removesuffix("_dof") for name in joint_columns
        )
        rows = []
        for row in reader:
            rows.append(
                [
                    math.radians(float(row[column]))
                    for column in joint_columns
                ]
            )

    if len(rows) == 0:
        raise ValueError(f"Quiet reference CSV has no frames: {path}")
    joint_positions = torch.tensor(rows, dtype=torch.float32)
    return joint_names, joint_positions

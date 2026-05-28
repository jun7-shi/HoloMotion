from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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

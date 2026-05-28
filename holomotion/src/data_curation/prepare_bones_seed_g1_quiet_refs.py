from __future__ import annotations
import argparse
import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import yaml

DEFAULT_CSV_ROOT = Path("/data/jun7.shi/datasets/bones-seed/g1/csv")
DEFAULT_METADATA_ROOT = Path("/data/jun7.shi/datasets/bones-seed/metadata")
DEFAULT_OUTPUT_DIR = Path("data/quiet_references/g1/bones_seed_csv")
DEFAULT_METADATA_OUTPUT = Path("data/quiet_references/g1_quiet_metadata.yaml")
DEFAULT_SPEED_TARGETS = (0.5, 0.6, 0.8, 1.0)

BLOCKED_TEXT = (
    "arc",
    "circle",
    "crawl",
    "crouch",
    "grab",
    "hands on back",
    "inj_",
    "injured",
    "injury",
    "jog",
    "jump",
    "kick",
    "pull",
    "push",
    "run",
    "stair",
    "stoop",
    "turn",
    "wall",
)

FORWARD_NAME_PATTERNS = (
    "neutral_walk_180",
    "walk_180",
    "walk_ff_loop_180",
    "walk_ff_start_180",
    "walk_ff_stop_180",
    "walk_forward",
    "neutral_walk_forward",
)


@dataclass(frozen=True)
class MotionStats:
    mean_vx: float
    mean_vy: float
    mean_yaw_rate: float
    quiet_score: float


@dataclass(frozen=True)
class SelectedClip:
    filename: str
    source_path: Path
    output_path: Path
    manifest_path: str
    stats: MotionStats


def _is_zeroish(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return text in {"", "0", "0.0", "false", "none"}


def _is_oneish(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "1.0", "true"}


def _row_text(row: dict[str, str]) -> str:
    keys = (
        "filename",
        "move_name",
        "category",
        "content_name",
        "content_short_description",
        "content_short_description_2",
        "content_technical_description",
        "content_type_of_movement",
        "content_uniform_style",
    )
    return " ".join(str(row.get(key, "")) for key in keys).lower()


def _has_forward_name(row: dict[str, str]) -> bool:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("filename", "move_g1_path", "content_short_description_2")
    ).lower()
    return any(pattern in text for pattern in FORWARD_NAME_PATTERNS)


def is_candidate_row(row: dict[str, str]) -> bool:
    if not row.get("move_g1_path"):
        return False
    if row.get("category") != "Basic Locomotion Neutral":
        return False
    if (
        str(row.get("content_type_of_movement", "")).strip().lower()
        != "walking"
    ):
        return False
    if str(row.get("content_uniform_style", "")).strip().lower() != "neutral":
        return False
    if str(row.get("content_body_position", "")).strip().lower() != "standing":
        return False
    if not _is_oneish(row.get("content_horizontal_move")):
        return False
    if not _is_zeroish(row.get("content_vertical_move")):
        return False
    if not _is_zeroish(row.get("content_props")):
        return False
    if not _is_zeroish(row.get("content_complex_action")):
        return False

    text = _row_text(row)
    if "walk" not in text:
        return False
    if not _has_forward_name(row):
        return False
    return not any(blocked in text for blocked in BLOCKED_TEXT)


def resolve_g1_csv_path(csv_root: Path, move_g1_path: str) -> Path:
    rel_path = Path(move_g1_path)
    parts = rel_path.parts
    if len(parts) >= 3 and parts[0] == "g1" and parts[1] == "csv":
        return csv_root.joinpath(*parts[2:])
    return csv_root / rel_path


def _unwrap_degrees(values: Sequence[float]) -> list[float]:
    unwrapped: list[float] = []
    offset = 0.0
    previous: float | None = None
    for value in values:
        if previous is not None:
            delta = value - previous
            if delta > 180.0:
                offset -= 360.0
            elif delta < -180.0:
                offset += 360.0
        unwrapped.append(value + offset)
        previous = value
    return unwrapped


def _rms(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def compute_motion_stats(
    csv_path: Path,
    *,
    fps: float,
    position_scale: float,
    trim_fraction: float,
) -> MotionStats:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    yaws: list[float] = []

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            xs.append(float(row["root_translateX"]) * position_scale)
            ys.append(float(row["root_translateY"]) * position_scale)
            zs.append(float(row["root_translateZ"]) * position_scale)
            yaws.append(float(row["root_rotateZ"]))

    if len(xs) < 2:
        raise ValueError(f"{csv_path} has fewer than two frames")

    start = max(0, min(len(xs) - 2, int(len(xs) * trim_fraction)))
    end = min(
        len(xs) - 1, max(start + 1, int(len(xs) * (1.0 - trim_fraction)))
    )
    duration = (end - start) / fps
    if duration <= 0.0:
        raise ValueError(f"{csv_path} has invalid duration after trimming")

    unwrapped_yaw = _unwrap_degrees(yaws)
    mean_yaw = math.radians(
        sum(unwrapped_yaw[start : end + 1]) / (end - start + 1)
    )
    global_vx = (xs[end] - xs[start]) / duration
    global_vy = (ys[end] - ys[start]) / duration
    mean_vx = global_vx * math.cos(mean_yaw) + global_vy * math.sin(mean_yaw)
    mean_vy = -global_vx * math.sin(mean_yaw) + global_vy * math.cos(mean_yaw)
    mean_yaw_rate = (
        math.radians(unwrapped_yaw[end] - unwrapped_yaw[start]) / duration
    )

    frame_range = range(start, end)
    vertical_speed_rms = _rms(
        (zs[index + 1] - zs[index]) * fps for index in frame_range
    )
    xy_speeds = [
        (
            (xs[index + 1] - xs[index]) * fps,
            (ys[index + 1] - ys[index]) * fps,
        )
        for index in frame_range
    ]
    xy_accel_rms = _rms(
        math.hypot(
            xy_speeds[index + 1][0] - xy_speeds[index][0],
            xy_speeds[index + 1][1] - xy_speeds[index][1],
        )
        * fps
        for index in range(len(xy_speeds) - 1)
    )
    quiet_penalty = (
        2.0 * vertical_speed_rms
        + 0.10 * xy_accel_rms
        + 2.0 * abs(mean_vy)
        + abs(mean_yaw_rate)
    )
    quiet_score = 1.0 / (1.0 + quiet_penalty)

    return MotionStats(
        mean_vx=round(mean_vx, 6),
        mean_vy=round(mean_vy, 6),
        mean_yaw_rate=round(mean_yaw_rate, 6),
        quiet_score=round(quiet_score, 6),
    )


def load_candidate_clips(
    *,
    csv_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    repo_root: Path,
    fps: float,
    position_scale: float,
    trim_fraction: float,
) -> list[SelectedClip]:
    candidates: list[SelectedClip] = []
    with metadata_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if not is_candidate_row(row):
                continue
            source_path = resolve_g1_csv_path(csv_root, row["move_g1_path"])
            if not source_path.is_file():
                continue
            try:
                stats = compute_motion_stats(
                    source_path,
                    fps=fps,
                    position_scale=position_scale,
                    trim_fraction=trim_fraction,
                )
            except (KeyError, ValueError):
                continue
            if stats.mean_vx <= 0.05:
                continue

            relative_source = source_path.relative_to(csv_root)
            output_path = output_dir / relative_source
            manifest_path = output_path.relative_to(repo_root).as_posix()
            candidates.append(
                SelectedClip(
                    filename=row["filename"],
                    source_path=source_path,
                    output_path=output_path,
                    manifest_path=manifest_path,
                    stats=stats,
                )
            )
    return candidates


def _motion_key(filename: str) -> str:
    return filename.removesuffix("_M")


def select_by_speed_targets(
    candidates: Sequence[SelectedClip],
    *,
    speed_targets: Sequence[float],
    clips_per_target: int,
) -> list[SelectedClip]:
    selected: list[SelectedClip] = []
    selected_keys: set[str] = set()
    for target in speed_targets:
        ranked = sorted(
            candidates,
            key=lambda clip: (
                abs(clip.stats.mean_vx - target),
                abs(clip.stats.mean_vy),
                abs(clip.stats.mean_yaw_rate),
                -clip.stats.quiet_score,
                clip.filename,
            ),
        )
        added_for_target = 0
        for clip in ranked:
            motion_key = _motion_key(clip.filename)
            if motion_key in selected_keys:
                continue
            selected.append(clip)
            selected_keys.add(motion_key)
            added_for_target += 1
            if added_for_target >= clips_per_target:
                break
    return selected


def write_manifest(
    selected: Sequence[SelectedClip],
    *,
    metadata_output: Path,
    metadata_csv: Path,
    fps: float,
    position_scale: float,
    trim_fraction: float,
    speed_targets: Sequence[float],
):
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": {
            "dataset": "bones-seed",
            "metadata_csv": str(metadata_csv),
            "fps": fps,
            "position_scale": position_scale,
            "trim_fraction": trim_fraction,
            "speed_targets": [float(target) for target in speed_targets],
            "quiet_score": (
                "kinematic proxy from root vertical speed, XY acceleration, "
                "lateral drift, and yaw rate"
            ),
        },
        "clips": [
            {
                "path": clip.manifest_path,
                "robot": "g1_29dof",
                "mean_vx": clip.stats.mean_vx,
                "mean_vy": clip.stats.mean_vy,
                "mean_yaw_rate": clip.stats.mean_yaw_rate,
                "quiet_score": clip.stats.quiet_score,
            }
            for clip in selected
        ],
    }
    metadata_output.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def prepare_references(
    *,
    csv_root: Path,
    metadata_csv: Path,
    output_dir: Path,
    metadata_output: Path,
    repo_root: Path,
    speed_targets: Sequence[float],
    clips_per_target: int,
    fps: float,
    position_scale: float,
    trim_fraction: float,
    dry_run: bool = False,
) -> list[SelectedClip]:
    repo_root = repo_root.resolve()
    output_dir = (
        (repo_root / output_dir).resolve()
        if not output_dir.is_absolute()
        else output_dir
    )
    metadata_output = (
        (repo_root / metadata_output).resolve()
        if not metadata_output.is_absolute()
        else metadata_output
    )
    candidates = load_candidate_clips(
        csv_root=csv_root,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        repo_root=repo_root,
        fps=fps,
        position_scale=position_scale,
        trim_fraction=trim_fraction,
    )
    selected = select_by_speed_targets(
        candidates,
        speed_targets=speed_targets,
        clips_per_target=clips_per_target,
    )
    if dry_run:
        return selected

    for clip in selected:
        clip.output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(clip.source_path, clip.output_path)
    write_manifest(
        selected,
        metadata_output=metadata_output,
        metadata_csv=metadata_csv,
        fps=fps,
        position_scale=position_scale,
        trim_fraction=trim_fraction,
        speed_targets=speed_targets,
    )
    return selected


def discover_metadata_csv(metadata_root: Path) -> Path:
    preferred = metadata_root / "seed_metadata_v004.csv"
    if preferred.is_file():
        return preferred
    matches = sorted(metadata_root.glob("seed_metadata_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No seed_metadata_*.csv found in {metadata_root}"
        )
    return matches[-1]


def _parse_speed_targets(value: str) -> tuple[float, ...]:
    targets = tuple(
        float(part.strip()) for part in value.split(",") if part.strip()
    )
    if not targets:
        raise argparse.ArgumentTypeError("expected at least one speed target")
    return targets


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy selected Bones Seed G1 walking CSV references and write "
            "the quiet WBC manifest."
        ),
    )
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument(
        "--metadata-root", type=Path, default=DEFAULT_METADATA_ROOT
    )
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--speed-targets",
        type=_parse_speed_targets,
        default=DEFAULT_SPEED_TARGETS,
        help="Comma-separated target forward speeds in m/s.",
    )
    parser.add_argument("--clips-per-target", type=int, default=3)
    parser.add_argument("--fps", type=float, default=120.0)
    parser.add_argument("--position-scale", type=float, default=0.01)
    parser.add_argument("--trim-fraction", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_csv = args.metadata_csv or discover_metadata_csv(
        args.metadata_root
    )
    selected = prepare_references(
        csv_root=args.csv_root,
        metadata_csv=metadata_csv,
        output_dir=args.output_dir,
        metadata_output=args.metadata_output,
        repo_root=args.repo_root,
        speed_targets=args.speed_targets,
        clips_per_target=args.clips_per_target,
        fps=args.fps,
        position_scale=args.position_scale,
        trim_fraction=args.trim_fraction,
        dry_run=args.dry_run,
    )
    for clip in selected:
        print(
            f"{clip.filename}: vx={clip.stats.mean_vx:.3f} "
            f"vy={clip.stats.mean_vy:.3f} yaw={clip.stats.mean_yaw_rate:.3f} "
            f"quiet={clip.stats.quiet_score:.3f} -> {clip.manifest_path}"
        )
    print(f"selected {len(selected)} clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

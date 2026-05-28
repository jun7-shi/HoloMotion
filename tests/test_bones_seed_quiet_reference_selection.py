import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "holomotion"
    / "src"
    / "data_curation"
    / "prepare_bones_seed_g1_quiet_refs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_bones_seed_g1_quiet_refs_under_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_g1_csv(path: Path, x_values: list[float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Frame,root_translateX,root_translateY,root_translateZ,root_rotateZ"
    ]
    lines.extend(
        f"{frame},{x_value},0.0,0.8,0.0"
        for frame, x_value in enumerate(x_values)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_references_filters_walks_copies_files_and_writes_manifest(
    tmp_path: Path,
):
    bones_root = tmp_path / "bones"
    csv_root = bones_root / "g1" / "csv"
    metadata_csv = bones_root / "metadata" / "seed_metadata_v004.csv"
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)

    good_path = csv_root / "230101" / "walk_180_R_001__A001.csv"
    jog_path = csv_root / "230101" / "jog_180_R_001__A001.csv"
    injured_path = (
        csv_root / "230101" / "inj_right_leg_walk_180_R_001__A001.csv"
    )
    _write_g1_csv(good_path, [0.0, 1.0, 2.0, 3.0])
    _write_g1_csv(jog_path, [0.0, 1.0, 2.0, 3.0])
    _write_g1_csv(injured_path, [0.0, 1.0, 2.0, 3.0])

    metadata_csv.write_text(
        "\n".join(
            [
                "filename,move_g1_path,category,content_type_of_movement,"
                "content_horizontal_move,content_vertical_move,"
                "content_uniform_style,content_body_position,"
                "content_props,content_complex_action,"
                "content_short_description,content_short_description_2,"
                "content_technical_description,move_duration_frames",
                "walk_180_R_001__A001,"
                "g1/csv/230101/walk_180_R_001__A001.csv,"
                "Basic Locomotion Neutral,walking,1,0,neutral,standing,0,0,"
                "walking facing forward,walk 180,"
                "walk forward from start to finish,4",
                "jog_180_R_001__A001,"
                "g1/csv/230101/jog_180_R_001__A001.csv,"
                "Basic Locomotion Neutral,walking,1,0,neutral,standing,0,0,"
                "jogging facing forward,jog 180,"
                "jog forward from start to finish,4",
                "inj_right_leg_walk_180_R_001__A001,"
                "g1/csv/230101/inj_right_leg_walk_180_R_001__A001.csv,"
                "Basic Locomotion Styles,walking,1,0,injured leg,standing,0,0,"
                "injured walk,injured leg walk 180,"
                "person walks with injured leg,4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repo_root = tmp_path / "repo"
    output_dir = (
        repo_root / "data" / "quiet_references" / "g1" / "bones_seed_csv"
    )
    metadata_output = (
        repo_root / "data" / "quiet_references" / "g1_quiet_metadata.yaml"
    )

    module = _load_module()
    selected = module.prepare_references(
        csv_root=csv_root,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        metadata_output=metadata_output,
        repo_root=repo_root,
        speed_targets=(1.0,),
        clips_per_target=1,
        fps=1.0,
        position_scale=1.0,
        trim_fraction=0.0,
    )

    assert [clip.filename for clip in selected] == ["walk_180_R_001__A001"]
    assert (output_dir / "230101" / good_path.name).is_file()
    assert not (output_dir / "230101" / jog_path.name).exists()
    assert not (output_dir / "230101" / injured_path.name).exists()

    manifest = yaml.safe_load(metadata_output.read_text(encoding="utf-8"))
    assert manifest["clips"] == [
        {
            "path": (
                "data/quiet_references/g1/bones_seed_csv/230101/"
                "walk_180_R_001__A001.csv"
            ),
            "robot": "g1_29dof",
            "mean_vx": pytest.approx(1.0),
            "mean_vy": pytest.approx(0.0),
            "mean_yaw_rate": pytest.approx(0.0),
            "quiet_score": pytest.approx(1.0),
        }
    ]


def test_candidate_filter_rejects_non_forward_angle_walks():
    module = _load_module()

    assert not module.is_candidate_row(
        {
            "filename": "walk_ff_stop_225_R_003__A255",
            "move_g1_path": "g1/csv/230308/walk_ff_stop_225_R_003__A255.csv",
            "category": "Basic Locomotion Neutral",
            "content_type_of_movement": "walking",
            "content_horizontal_move": "1",
            "content_vertical_move": "0",
            "content_uniform_style": "neutral",
            "content_body_position": "standing",
            "content_props": "0",
            "content_complex_action": "0",
            "content_short_description": "walking facing forward",
            "content_short_description_2": "walk ff stop 225",
            "content_technical_description": "walk forward and stop",
        }
    )

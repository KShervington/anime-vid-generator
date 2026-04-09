import json
from typer.testing import CliRunner
from anime_vid_generator.cli import app

runner = CliRunner()


def test_stage1_dry_run_outputs_valid_json(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage1", str(video), "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    class_types = {n["class_type"] for n in parsed.values()}
    assert "VHS_LoadVideo" in class_types
    assert "DWPose_Estimator" in class_types


def test_stage1_dry_run_sets_video_path(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage1", str(video), "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    load_nodes = [n for n in parsed.values() if n["class_type"] == "VHS_LoadVideo"]
    assert load_nodes[0]["inputs"]["video"] == str(video)


def test_stage1_nonexistent_video_exits_nonzero():
    result = runner.invoke(app, ["stage1", "/nonexistent/video.mp4"])
    assert result.exit_code != 0


def test_stage1_help_exits_zero():
    result = runner.invoke(app, ["stage1", "--help"])
    assert result.exit_code == 0

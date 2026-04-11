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


def test_stage2_dry_run_outputs_valid_json(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    class_types = {n["class_type"] for n in parsed.values()}
    assert "VHS_LoadVideo" in class_types
    assert "KSampler" in class_types


def test_stage2_dry_run_sets_video_path(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    load_nodes = [n for n in parsed.values() if n["class_type"] == "VHS_LoadVideo"]
    assert load_nodes[0]["inputs"]["video"] == str(video)


def test_stage2_dry_run_custom_prompt(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--prompt", "my anime style", "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    clip_nodes = [n for n in parsed.values() if n["class_type"] == "CLIPTextEncode"]
    texts = [n["inputs"]["text"] for n in clip_nodes]
    assert "my anime style" in texts


def test_stage2_dry_run_custom_negative_prompt(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--negative-prompt", "bad quality", "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    clip_nodes = [n for n in parsed.values() if n["class_type"] == "CLIPTextEncode"]
    texts = [n["inputs"]["text"] for n in clip_nodes]
    assert "bad quality" in texts


def test_stage2_dry_run_vae_encode_flag(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--vae-encode", "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    class_types = {n["class_type"] for n in parsed.values()}
    assert "VAEEncode" in class_types
    assert "EmptyLatentVideo" not in class_types


def test_stage2_dry_run_denoise_option(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    result = runner.invoke(app, ["stage2", str(video), "--denoise", "0.75", "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    ks_nodes = [n for n in parsed.values() if n["class_type"] == "KSampler"]
    assert ks_nodes[0]["inputs"]["denoise"] == 0.75


def test_stage2_dry_run_reference_image_option(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"fake")
    face = tmp_path / "face.png"
    face.write_bytes(b"fakeimage")
    result = runner.invoke(app, ["stage2", str(video), "--reference-image", str(face), "--dry-run"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    load_image_nodes = [n for n in parsed.values() if n["class_type"] == "LoadImage"]
    assert load_image_nodes[0]["inputs"]["image"] == str(face)


def test_stage2_nonexistent_video_exits_nonzero():
    result = runner.invoke(app, ["stage2", "/nonexistent/video.mp4"])
    assert result.exit_code != 0


def test_stage2_help_exits_zero():
    result = runner.invoke(app, ["stage2", "--help"])
    assert result.exit_code == 0

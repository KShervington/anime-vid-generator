from anime_vid_generator.config import HardwareConfig, Stage1Config, PipelineConfig


def test_hardware_vram_budget_fits_in_envelope():
    config = HardwareConfig()
    allocated = (
        config.base_model_vram_gb
        + config.controlnet_vram_gb
        + config.context_window_vram_gb
        + config.buffer_vram_gb
    )
    assert allocated <= config.vram_gb


def test_hardware_default_comfyui_url():
    assert HardwareConfig().comfyui_url == "http://127.0.0.1:8188"


def test_stage1_overlap_less_than_window():
    config = Stage1Config()
    assert config.context_overlap_frames < config.context_window_frames


def test_stage1_defaults_match_tdd():
    config = Stage1Config()
    assert config.context_window_frames == 32
    assert config.context_overlap_frames == 8
    assert config.fps == 24


def test_pipeline_config_composes_sub_configs():
    config = PipelineConfig()
    assert isinstance(config.hardware, HardwareConfig)
    assert isinstance(config.stage1, Stage1Config)

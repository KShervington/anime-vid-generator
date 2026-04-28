from anime_vid_generator.config import (
    HardwareConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    PipelineConfig,
)


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


def test_stage2_config_default_model_path():
    assert Stage2Config().model_path == "wan_2_6_nvfp4.gguf"


def test_stage2_config_default_ip_adapter_weight():
    assert Stage2Config().ip_adapter_weight == 0.8


def test_stage2_config_default_style_transfer_cfg():
    assert Stage2Config().style_transfer_cfg == 1.5


def test_stage2_config_default_positive_prompt():
    config = Stage2Config()
    assert config.positive_prompt == "cinematic anime, ufotable style, high quality"


def test_stage2_config_default_negative_prompt():
    config = Stage2Config()
    assert config.negative_prompt == "blurry, low quality, photorealistic"


def test_stage2_config_default_resolution():
    config = Stage2Config()
    assert config.width == 1280
    assert config.height == 720


def test_stage2_config_default_context_window():
    config = Stage2Config()
    assert config.context_window_frames == 32
    assert config.context_overlap_frames == 8


def test_stage2_config_default_sampler_settings():
    config = Stage2Config()
    assert config.sampler_steps == 20
    assert config.sampler_cfg == 7.0
    assert config.sampler_name == "euler"
    assert config.sampler_scheduler == "karras"


def test_stage2_config_default_latent_mode():
    assert Stage2Config().latent_mode == "empty"


def test_stage2_config_default_denoise():
    assert Stage2Config().denoise == 1.0


def test_stage2_config_default_seed():
    assert Stage2Config().seed == 0


def test_stage2_config_latent_mode_accepts_vae_encode():
    config = Stage2Config(latent_mode="vae_encode")
    assert config.latent_mode == "vae_encode"


def test_stage2_config_custom_prompt():
    config = Stage2Config(positive_prompt="my custom prompt")
    assert config.positive_prompt == "my custom prompt"


def test_pipeline_config_composes_stage2():
    config = PipelineConfig()
    assert isinstance(config.stage2, Stage2Config)


def test_stage3_config_defaults():
    config = Stage3Config()
    assert config.emitter_prompt == "sword tip"
    assert config.mask_dilation == 8
    assert config.vfx_lora_path == "Ufotable_Fire_Trails.safetensors"
    assert config.vfx_lora_strength == 0.85
    assert config.vfx_cfg == 8.5
    assert config.vfx_positive_prompt == "fire trails, vfx, ufotable style, elemental effects"
    assert config.vfx_negative_prompt == "blurry, low quality, photorealistic"
    assert config.sampler_steps == 20
    assert config.sampler_name == "euler"
    assert config.sampler_scheduler == "karras"
    assert config.denoise == 0.8
    assert config.seed == 0


def test_stage3_config_is_pydantic_model():
    from pydantic import BaseModel
    assert issubclass(Stage3Config, BaseModel)


def test_stage3_config_custom_values():
    config = Stage3Config(emitter_prompt="rear tire", vfx_cfg=9.0)
    assert config.emitter_prompt == "rear tire"
    assert config.vfx_cfg == 9.0


def test_stage3_config_vfx_cfg_is_float():
    config = Stage3Config()
    assert isinstance(config.vfx_cfg, float)


def test_stage3_config_mask_dilation_is_int():
    config = Stage3Config()
    assert isinstance(config.mask_dilation, int)


def test_pipeline_config_has_stage3():
    config = PipelineConfig()
    assert isinstance(config.stage3, Stage3Config)

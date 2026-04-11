from typing import Literal
from pydantic import BaseModel, Field


class HardwareConfig(BaseModel):
    vram_gb: float = 16.0
    base_model_vram_gb: float = 4.2
    controlnet_vram_gb: float = 2.0
    context_window_vram_gb: float = 5.5
    buffer_vram_gb: float = 4.3
    comfyui_url: str = "http://127.0.0.1:8188"


class Stage1Config(BaseModel):
    context_window_frames: int = 32
    context_overlap_frames: int = 8
    fps: int = 24
    optical_flow_raft_iters: int = 12
    zoe_depth_model: Literal["ZoeD_N", "ZoeD_K", "ZoeD_NK"] = "ZoeD_NK"
    canny_low_threshold: int = 100
    canny_high_threshold: int = 200


class Stage2Config(BaseModel):
    model_path: str = "wan_2_6_nvfp4.gguf"
    reference_image_path: str = ""
    ip_adapter_weight: float = 0.8
    style_transfer_cfg: float = 1.5
    positive_prompt: str = "cinematic anime, ufotable style, high quality"
    negative_prompt: str = "blurry, low quality, photorealistic"
    width: int = 1280
    height: int = 720
    context_window_frames: int = 32
    context_overlap_frames: int = 8
    sampler_steps: int = 20
    sampler_cfg: float = 7.0
    sampler_name: str = "euler"
    sampler_scheduler: str = "karras"
    latent_mode: Literal["empty", "vae_encode"] = "empty"
    denoise: float = 1.0
    seed: int = 0


class PipelineConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    stage2: Stage2Config = Field(default_factory=Stage2Config)
    target_fps: int = 24
    output_resolution: tuple[int, int] = (3840, 2160)

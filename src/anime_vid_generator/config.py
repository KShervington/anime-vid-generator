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


class PipelineConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    stage1: Stage1Config = Field(default_factory=Stage1Config)
    target_fps: int = 24
    output_resolution: tuple[int, int] = (3840, 2160)

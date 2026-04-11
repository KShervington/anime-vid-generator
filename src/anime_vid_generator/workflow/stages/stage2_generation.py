from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    load_video_node,
    layered_model_unload_node,
    gguf_loader_node,
    clip_text_encode_node,
    load_image_node,
    ip_adapter_faceid_plus_node,
    reference_only_node,
    style_transfer_block_node,
    empty_latent_video_node,
    vae_encode_node,
    flow_guided_noise_injection_node,
    free_long_node,
    ksampler_node,
)
from ...config import Stage1Config, Stage2Config
from .stage1_motion import (
    build_organic_bus,
    build_rigid_bus,
    build_temporal_unification,
)


@dataclass
class ModelLoadingBusResult:
    model_ref: NodeRef   # GGUF_Loader slot 0: model
    clip_ref: NodeRef    # GGUF_Loader slot 1: clip
    vae_ref: NodeRef     # GGUF_Loader slot 2: vae


@dataclass
class IdentityBusResult:
    conditioned_model_ref: NodeRef  # Style_Transfer_Block slot 0


@dataclass
class ConditioningBusResult:
    positive_ref: NodeRef  # CLIPTextEncode positive slot 0
    negative_ref: NodeRef  # CLIPTextEncode negative slot 0


@dataclass
class LatentBusResult:
    latent_ref: NodeRef  # Flow_Guided_Noise_Injection slot 0


@dataclass
class GenerationBusResult:
    latent_output: NodeRef  # KSampler slot 0


def build_model_loading_bus(
    builder: WorkflowBuilder,
    config: Stage2Config,
) -> ModelLoadingBusResult:
    """Add Layered_Model_Unload (standalone) and GGUF_Loader nodes."""
    unload = layered_model_unload_node()
    builder.add(unload)

    loader = gguf_loader_node()
    loader.inputs["model_path"] = config.model_path
    gguf_id = builder.add(loader)

    return ModelLoadingBusResult(
        model_ref=(gguf_id, 0),
        clip_ref=(gguf_id, 1),
        vae_ref=(gguf_id, 2),
    )

from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import NodeRef, layered_model_unload_node, gguf_loader_node
from ...config import Stage2Config


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

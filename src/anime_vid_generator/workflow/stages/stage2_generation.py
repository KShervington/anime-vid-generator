from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    layered_model_unload_node,
    gguf_loader_node,
    clip_text_encode_node,
    ip_adapter_faceid_plus_node,
    reference_only_node,
    style_transfer_block_node,
    empty_latent_video_node,
    vae_encode_node,
    flow_guided_noise_injection_node,
)
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


def build_conditioning_bus(
    builder: WorkflowBuilder,
    clip_ref: NodeRef,
    config: Stage2Config,
) -> ConditioningBusResult:
    """Add positive and negative CLIPTextEncode nodes."""
    pos = clip_text_encode_node()
    pos.inputs["text"] = config.positive_prompt
    pos.inputs["clip"] = clip_ref
    pos_id = builder.add(pos)

    neg = clip_text_encode_node()
    neg.inputs["text"] = config.negative_prompt
    neg.inputs["clip"] = clip_ref
    neg_id = builder.add(neg)

    return ConditioningBusResult(
        positive_ref=(pos_id, 0),
        negative_ref=(neg_id, 0),
    )


def build_identity_bus(
    builder: WorkflowBuilder,
    model_ref: NodeRef,
    face_ref: NodeRef,
    config: Stage2Config,
) -> IdentityBusResult:
    """Add IP_Adapter_FaceID_Plus → Reference_Only → Style_Transfer_Block chain."""
    ip = ip_adapter_faceid_plus_node()
    ip.inputs["model"] = model_ref
    ip.inputs["image"] = face_ref
    ip.inputs["weight"] = config.ip_adapter_weight
    ip_id = builder.add(ip)

    ref = reference_only_node()
    ref.inputs["model"] = (ip_id, 0)
    ref.inputs["reference"] = face_ref
    ref_id = builder.add(ref)

    style = style_transfer_block_node()
    style.inputs["model"] = (ref_id, 0)
    style.inputs["cfg"] = config.style_transfer_cfg
    style_id = builder.add(style)

    return IdentityBusResult(conditioned_model_ref=(style_id, 0))


def build_latent_bus(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    flow_map: NodeRef,
    vae_ref: NodeRef,
    config: Stage2Config,
) -> LatentBusResult:
    """Add EmptyLatentVideo or VAEEncode, then Flow_Guided_Noise_Injection."""
    if config.latent_mode == "empty":
        latent = empty_latent_video_node()
        latent.inputs["width"] = config.width
        latent.inputs["height"] = config.height
        latent.inputs["length"] = config.context_window_frames
        latent.inputs["batch_size"] = 1
        latent_id = builder.add(latent)
        latent_source_ref: NodeRef = (latent_id, 0)
    else:  # vae_encode
        encode = vae_encode_node()
        encode.inputs["pixels"] = image_ref
        encode.inputs["vae"] = vae_ref
        latent_id = builder.add(encode)
        latent_source_ref = (latent_id, 0)

    noise = flow_guided_noise_injection_node()
    noise.inputs["latents"] = latent_source_ref
    noise.inputs["flow_map"] = flow_map
    noise_id = builder.add(noise)

    return LatentBusResult(latent_ref=(noise_id, 0))

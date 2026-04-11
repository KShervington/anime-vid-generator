from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    load_video_node,
    load_image_node,
    layered_model_unload_node,
    gguf_loader_node,
    clip_text_encode_node,
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
from .stage1_motion import build_organic_bus, build_rigid_bus, build_temporal_unification


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


def build_generation_bus(
    builder: WorkflowBuilder,
    model_ref: NodeRef,
    positive_ref: NodeRef,
    negative_ref: NodeRef,
    latent_ref: NodeRef,
    config: Stage2Config,
) -> GenerationBusResult:
    """Add FreeLong spectral blending node then KSampler."""
    fl = free_long_node()
    fl.inputs["model"] = model_ref
    fl.inputs["context_frames"] = config.context_window_frames
    fl.inputs["overlap_frames"] = config.context_overlap_frames
    fl_id = builder.add(fl)

    ks = ksampler_node()
    ks.inputs["model"] = (fl_id, 0)
    ks.inputs["positive"] = positive_ref
    ks.inputs["negative"] = negative_ref
    ks.inputs["latent_image"] = latent_ref
    ks.inputs["steps"] = config.sampler_steps
    ks.inputs["cfg"] = config.sampler_cfg
    ks.inputs["sampler_name"] = config.sampler_name
    ks.inputs["scheduler"] = config.sampler_scheduler
    ks.inputs["denoise"] = config.denoise
    ks.inputs["tiled_sampling"] = True
    ks_id = builder.add(ks)

    return GenerationBusResult(latent_output=(ks_id, 0))


def build_stage2_workflow(
    video_path: str,
    config: Stage2Config | None = None,
) -> dict:
    """Build the complete Stage 2 workflow and return ComfyUI API JSON.

    Creates a single ComfyUI prompt containing all Stage 1 preprocessing
    nodes (using Stage1Config defaults) followed by all Stage 2 generation
    nodes. The Stage 1 TemporalUnificationResult.flow_map is wired directly
    into the Stage 2 latent bus.

    Args:
        video_path: Absolute path to the source video file.
        config: Stage2Config override; uses defaults if None.

    Returns:
        dict in ComfyUI /prompt API format, ready to POST.
    """
    if config is None:
        config = Stage2Config()

    stage1_config = Stage1Config()
    builder = WorkflowBuilder()

    # Stage 1 — preprocessing
    video = load_video_node()
    video.inputs["video"] = video_path
    video.inputs["frame_load_cap"] = 0
    video.inputs["skip_first_frames"] = 0
    vid_id = builder.add(video)
    image_ref: NodeRef = (vid_id, 0)

    build_organic_bus(builder, image_ref, stage1_config)
    build_rigid_bus(builder, image_ref, stage1_config)
    temporal_result = build_temporal_unification(builder, image_ref, stage1_config)

    # Stage 2 — generation
    model_result = build_model_loading_bus(builder, config)

    face = load_image_node()
    face.inputs["image"] = config.reference_image_path
    face_id = builder.add(face)
    face_ref: NodeRef = (face_id, 0)

    identity_result = build_identity_bus(builder, model_result.model_ref, face_ref, config)
    cond_result = build_conditioning_bus(builder, model_result.clip_ref, config)
    latent_result = build_latent_bus(
        builder, image_ref, temporal_result.flow_map, model_result.vae_ref, config
    )
    build_generation_bus(
        builder,
        identity_result.conditioned_model_ref,
        cond_result.positive_ref,
        cond_result.negative_ref,
        latent_result.latent_ref,
        config,
    )

    return builder.build()

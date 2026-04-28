from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    load_video_node,
    load_image_node,
    clip_text_encode_node,
    sam3_video_segmenter_node,
    mask_dilate_node,
    vae_encode_for_inpaint_node,
    lora_loader_node,
    ksampler_node,
)
from ...config import Stage1Config, Stage2Config, Stage3Config
from .stage1_motion import build_organic_bus, build_rigid_bus, build_temporal_unification
from .stage2_generation import (
    build_model_loading_bus,
    build_conditioning_bus,
    build_identity_bus,
    build_latent_bus,
    build_generation_bus,
)


@dataclass
class TrackingBusResult:
    dilated_mask_ref: NodeRef  # Mask_Dilate slot 0


def build_tracking_bus(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    flow_map: NodeRef,
    config: Stage3Config,
) -> TrackingBusResult:
    """Add SAM3_VideoSegmenter and Mask_Dilate nodes for emitter tracking."""
    sam = sam3_video_segmenter_node()
    sam.inputs["image"] = image_ref
    sam.inputs["emitter_prompt"] = config.emitter_prompt
    sam_id = builder.add(sam)

    dilate = mask_dilate_node()
    dilate.inputs["mask"] = (sam_id, 0)
    dilate.inputs["flow_map"] = flow_map
    dilate.inputs["dilation"] = config.mask_dilation
    dilate_id = builder.add(dilate)

    return TrackingBusResult(dilated_mask_ref=(dilate_id, 0))


@dataclass
class VFXInpaintingBusResult:
    latent_output: NodeRef  # KSampler slot 0


def build_vfx_inpainting_bus(
    builder: WorkflowBuilder,
    model_ref: NodeRef,
    vae_ref: NodeRef,
    image_ref: NodeRef,
    dilated_mask_ref: NodeRef,
    positive_ref: NodeRef,
    negative_ref: NodeRef,
    config: Stage3Config,
) -> VFXInpaintingBusResult:
    """Add LoRA_Loader, VAEEncodeForInpaint, and KSampler for the VFX inpainting pass."""
    lora = lora_loader_node()
    lora.inputs["model"] = model_ref
    lora.inputs["lora_name"] = config.vfx_lora_path
    lora.inputs["strength_model"] = config.vfx_lora_strength
    lora.inputs["strength_clip"] = config.vfx_lora_strength
    lora_id = builder.add(lora)

    encode = vae_encode_for_inpaint_node()
    encode.inputs["pixels"] = image_ref
    encode.inputs["vae"] = vae_ref
    encode.inputs["mask"] = dilated_mask_ref
    encode_id = builder.add(encode)

    ks = ksampler_node()
    ks.inputs["model"] = (lora_id, 0)
    ks.inputs["positive"] = positive_ref
    ks.inputs["negative"] = negative_ref
    ks.inputs["latent_image"] = (encode_id, 0)
    ks.inputs["steps"] = config.sampler_steps
    ks.inputs["cfg"] = config.vfx_cfg
    ks.inputs["sampler_name"] = config.sampler_name
    ks.inputs["scheduler"] = config.sampler_scheduler
    ks.inputs["denoise"] = config.denoise
    ks.inputs["seed"] = config.seed
    ks.inputs["tiled_sampling"] = True
    ks_id = builder.add(ks)

    return VFXInpaintingBusResult(latent_output=(ks_id, 0))


def build_stage3_workflow(
    video_path: str,
    config: Stage3Config | None = None,
) -> dict:
    """Build the complete Stage 3 workflow and return ComfyUI API JSON.

    Creates a single ComfyUI prompt containing Stage 1 preprocessing nodes,
    Stage 2 generation nodes, and Stage 3 VFX inpainting nodes. The tracking
    bus uses the Stage 1 optical flow map. The inpainting bus uses the Stage 2
    model, VAE, and newly created VFX CLIP conditioning.

    Args:
        video_path: Absolute path to the source video file.
        config: Stage3Config override; uses defaults if None.

    Returns:
        dict in ComfyUI /prompt API format, ready to POST.
    """
    if config is None:
        config = Stage3Config()

    stage1_config = Stage1Config()
    stage2_config = Stage2Config()
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
    model_result = build_model_loading_bus(builder, stage2_config)
    cond_result = build_conditioning_bus(builder, model_result.clip_ref, stage2_config)

    face = load_image_node()
    face.inputs["image"] = stage2_config.reference_image_path
    face_id = builder.add(face)
    face_ref: NodeRef = (face_id, 0)

    identity_result = build_identity_bus(builder, model_result.model_ref, face_ref, stage2_config)
    latent_result = build_latent_bus(
        builder, image_ref, temporal_result.flow_map, model_result.vae_ref, stage2_config
    )
    build_generation_bus(
        builder,
        identity_result.conditioned_model_ref,
        cond_result.positive_ref,
        cond_result.negative_ref,
        latent_result.latent_ref,
        stage2_config,
    )

    # Stage 3 — VFX inpainting
    tracking_result = build_tracking_bus(builder, image_ref, temporal_result.flow_map, config)

    vfx_pos = clip_text_encode_node()
    vfx_pos.inputs["text"] = config.vfx_positive_prompt
    vfx_pos.inputs["clip"] = model_result.clip_ref
    vfx_pos_id = builder.add(vfx_pos)
    vfx_pos_ref: NodeRef = (vfx_pos_id, 0)

    vfx_neg = clip_text_encode_node()
    vfx_neg.inputs["text"] = config.vfx_negative_prompt
    vfx_neg.inputs["clip"] = model_result.clip_ref
    vfx_neg_id = builder.add(vfx_neg)
    vfx_neg_ref: NodeRef = (vfx_neg_id, 0)

    build_vfx_inpainting_bus(
        builder,
        identity_result.conditioned_model_ref,
        model_result.vae_ref,
        image_ref,
        tracking_result.dilated_mask_ref,
        vfx_pos_ref,
        vfx_neg_ref,
        config,
    )

    return builder.build()

from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    sam3_video_segmenter_node,
    mask_dilate_node,
    vae_encode_for_inpaint_node,
    lora_loader_node,
    ksampler_node,
)
from ...config import Stage3Config


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

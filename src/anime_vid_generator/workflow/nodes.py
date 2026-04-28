from dataclasses import dataclass, field

# Points to another node's output: ("node_id_string", output_slot_index)
NodeRef = tuple[str, int]

# Any valid ComfyUI input: a node link or a scalar value
NodeInput = NodeRef | str | int | float | bool | list | None


@dataclass
class WorkflowNode:
    class_type: str
    inputs: dict[str, NodeInput] = field(default_factory=dict)
    meta_title: str | None = None

    def to_api_dict(self) -> dict:
        serialized_inputs = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in self.inputs.items()
        }
        result: dict = {"class_type": self.class_type, "inputs": serialized_inputs}
        if self.meta_title is not None:
            result["_meta"] = {"title": self.meta_title}
        return result


def load_video_node() -> WorkflowNode:
    return WorkflowNode(class_type="VHS_LoadVideo", meta_title="Load Input Video")


def dwpose_estimator_node() -> WorkflowNode:
    return WorkflowNode(class_type="DWPose_Estimator", meta_title="DWPose Estimator")


def densepose_node() -> WorkflowNode:
    return WorkflowNode(class_type="DensePose", meta_title="DensePose")


def lineart_anime_node() -> WorkflowNode:
    return WorkflowNode(class_type="ControlNet_LineArt_Anime", meta_title="LineArt Anime")


def canny_edge_node() -> WorkflowNode:
    return WorkflowNode(class_type="Canny_Edge", meta_title="Canny Edge")


def zoe_depth_node() -> WorkflowNode:
    return WorkflowNode(class_type="ZoeDepth", meta_title="ZoeDepth")


def unimatch_optical_flow_node() -> WorkflowNode:
    return WorkflowNode(class_type="Unimatch_Optical_Flow", meta_title="Unimatch Optical Flow")


def layered_model_unload_node() -> WorkflowNode:
    return WorkflowNode(class_type="Layered_Model_Unload", meta_title="Unload Stage 1 Models")


def gguf_loader_node() -> WorkflowNode:
    return WorkflowNode(class_type="GGUF_Loader", meta_title="Load Wan 2.6 NVFP4")


def clip_text_encode_node() -> WorkflowNode:
    return WorkflowNode(class_type="CLIPTextEncode", meta_title="CLIP Text Encode")


def load_image_node() -> WorkflowNode:
    return WorkflowNode(class_type="LoadImage", meta_title="Load Reference Image")


def ip_adapter_faceid_plus_node() -> WorkflowNode:
    return WorkflowNode(class_type="IP_Adapter_FaceID_Plus", meta_title="IP-Adapter FaceID Plus")


def reference_only_node() -> WorkflowNode:
    return WorkflowNode(class_type="Reference_Only", meta_title="Reference Only")


def style_transfer_block_node() -> WorkflowNode:
    return WorkflowNode(class_type="Style_Transfer_Block", meta_title="Style Transfer Block")


def empty_latent_video_node() -> WorkflowNode:
    return WorkflowNode(class_type="EmptyLatentVideo", meta_title="Empty Latent Video")


def vae_encode_node() -> WorkflowNode:
    return WorkflowNode(class_type="VAEEncode", meta_title="VAE Encode")


def flow_guided_noise_injection_node() -> WorkflowNode:
    return WorkflowNode(class_type="Flow_Guided_Noise_Injection", meta_title="Flow Guided Noise Injection")


def free_long_node() -> WorkflowNode:
    return WorkflowNode(class_type="FreeLong", meta_title="FreeLong Spectral Blending")


def ksampler_node() -> WorkflowNode:
    return WorkflowNode(class_type="KSampler", meta_title="KSampler")


def sam3_video_segmenter_node() -> WorkflowNode:
    return WorkflowNode(class_type="SAM3_VideoSegmenter", meta_title="SAM 3 Video Segmenter")


def mask_dilate_node() -> WorkflowNode:
    return WorkflowNode(class_type="Mask_Dilate", meta_title="Mask Dilate")


def vae_encode_for_inpaint_node() -> WorkflowNode:
    return WorkflowNode(class_type="VAEEncodeForInpaint", meta_title="VAE Encode For Inpaint")


def lora_loader_node() -> WorkflowNode:
    return WorkflowNode(class_type="LoRA_Loader", meta_title="VFX LoRA Loader")

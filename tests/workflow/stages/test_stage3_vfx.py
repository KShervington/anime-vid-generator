import pytest
from anime_vid_generator.config import Stage3Config
from anime_vid_generator.workflow.builder import WorkflowBuilder
from anime_vid_generator.workflow.nodes import (
    load_video_node,
    unimatch_optical_flow_node,
    gguf_loader_node,
    clip_text_encode_node,
)
from anime_vid_generator.workflow.stages.stage3_vfx import (
    TrackingBusResult,
    VFXInpaintingBusResult,
    build_tracking_bus,
    build_vfx_inpainting_bus,
    build_stage3_workflow,
)


def _find_node_id(workflow: dict, class_type: str) -> str:
    for node_id, node in workflow.items():
        if node["class_type"] == class_type:
            return node_id
    raise KeyError(f"No node with class_type={class_type!r} found in workflow")


@pytest.fixture
def builder_with_video_and_flow():
    builder = WorkflowBuilder()
    video = load_video_node()
    video.inputs["video"] = "/tmp/test.mp4"
    vid_id = builder.add(video)
    image_ref = (vid_id, 0)
    flow = unimatch_optical_flow_node()
    flow_id = builder.add(flow)
    flow_map = (flow_id, 0)
    return builder, image_ref, flow_map


def test_tracking_bus_returns_result_dataclass(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    result = build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    assert isinstance(result, TrackingBusResult)


def test_tracking_bus_adds_sam3_node(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    class_types = {n["class_type"] for n in workflow.values()}
    assert "SAM3_VideoSegmenter" in class_types


def test_tracking_bus_adds_mask_dilate_node(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    class_types = {n["class_type"] for n in workflow.values()}
    assert "Mask_Dilate" in class_types


def test_tracking_bus_sam3_links_image_ref(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    sam_id = _find_node_id(workflow, "SAM3_VideoSegmenter")
    assert workflow[sam_id]["inputs"]["image"] == list(image_ref)


def test_tracking_bus_sam3_sets_emitter_prompt(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    config = Stage3Config(emitter_prompt="rear tire")
    build_tracking_bus(builder, image_ref, flow_map, config)
    workflow = builder.build()
    sam_id = _find_node_id(workflow, "SAM3_VideoSegmenter")
    assert workflow[sam_id]["inputs"]["emitter_prompt"] == "rear tire"


def test_tracking_bus_mask_dilate_links_to_sam3(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    dilate_id = _find_node_id(workflow, "Mask_Dilate")
    sam_id = _find_node_id(workflow, "SAM3_VideoSegmenter")
    assert workflow[dilate_id]["inputs"]["mask"] == [sam_id, 0]


def test_tracking_bus_mask_dilate_links_flow_map(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    dilate_id = _find_node_id(workflow, "Mask_Dilate")
    assert workflow[dilate_id]["inputs"]["flow_map"] == list(flow_map)


def test_tracking_bus_mask_dilate_sets_dilation(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    config = Stage3Config(mask_dilation=16)
    build_tracking_bus(builder, image_ref, flow_map, config)
    workflow = builder.build()
    dilate_id = _find_node_id(workflow, "Mask_Dilate")
    assert workflow[dilate_id]["inputs"]["dilation"] == 16


def test_tracking_bus_result_ref_points_to_mask_dilate(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    result = build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    workflow = builder.build()
    dilate_id = _find_node_id(workflow, "Mask_Dilate")
    assert result.dilated_mask_ref == (dilate_id, 0)


def test_tracking_bus_result_ref_is_slot_0(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    result = build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    assert result.dilated_mask_ref[1] == 0


@pytest.fixture
def vfx_bus_refs(builder_with_video_and_flow):
    builder, image_ref, flow_map = builder_with_video_and_flow
    tracking_result = build_tracking_bus(builder, image_ref, flow_map, Stage3Config())
    model_node = gguf_loader_node()
    model_id = builder.add(model_node)
    model_ref = (model_id, 0)
    vae_ref = (model_id, 2)
    pos_node = clip_text_encode_node()
    pos_id = builder.add(pos_node)
    pos_ref = (pos_id, 0)
    neg_node = clip_text_encode_node()
    neg_id = builder.add(neg_node)
    neg_ref = (neg_id, 0)
    return builder, model_ref, vae_ref, image_ref, tracking_result.dilated_mask_ref, pos_ref, neg_ref


def test_vfx_inpainting_bus_returns_result_dataclass(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    result = build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    assert isinstance(result, VFXInpaintingBusResult)


def test_vfx_inpainting_bus_adds_lora_loader(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    class_types = {n["class_type"] for n in workflow.values()}
    assert "LoRA_Loader" in class_types


def test_vfx_inpainting_bus_adds_vae_encode_for_inpaint(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    class_types = {n["class_type"] for n in workflow.values()}
    assert "VAEEncodeForInpaint" in class_types


def test_vfx_inpainting_bus_lora_links_to_model_ref(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    lora_id = _find_node_id(workflow, "LoRA_Loader")
    assert workflow[lora_id]["inputs"]["model"] == list(model_ref)


def test_vfx_inpainting_bus_lora_sets_lora_name(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    config = Stage3Config(vfx_lora_path="custom_vfx.safetensors")
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, config
    )
    workflow = builder.build()
    lora_id = _find_node_id(workflow, "LoRA_Loader")
    assert workflow[lora_id]["inputs"]["lora_name"] == "custom_vfx.safetensors"


def test_vfx_inpainting_bus_vae_encode_links_mask(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    encode_id = _find_node_id(workflow, "VAEEncodeForInpaint")
    assert workflow[encode_id]["inputs"]["mask"] == list(dilated_mask_ref)


def test_vfx_inpainting_bus_ksampler_uses_high_cfg(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    config = Stage3Config(vfx_cfg=9.0)
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, config
    )
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["cfg"] == 9.0


def test_vfx_inpainting_bus_result_points_to_ksampler(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    result = build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert result.latent_output == (ks_id, 0)


def test_vfx_inpainting_bus_result_latent_output_is_slot_0(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    result = build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    assert result.latent_output[1] == 0


def test_vfx_inpainting_bus_vae_encode_links_pixels(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    encode_id = _find_node_id(workflow, "VAEEncodeForInpaint")
    assert workflow[encode_id]["inputs"]["pixels"] == list(image_ref)


def test_vfx_inpainting_bus_vae_encode_links_vae(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, Stage3Config()
    )
    workflow = builder.build()
    encode_id = _find_node_id(workflow, "VAEEncodeForInpaint")
    assert workflow[encode_id]["inputs"]["vae"] == list(vae_ref)


def test_vfx_inpainting_bus_lora_sets_strength_model(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    config = Stage3Config(vfx_lora_strength=0.6)
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, config
    )
    workflow = builder.build()
    lora_id = _find_node_id(workflow, "LoRA_Loader")
    assert workflow[lora_id]["inputs"]["strength_model"] == 0.6


def test_vfx_inpainting_bus_lora_sets_strength_clip(vfx_bus_refs):
    builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref = vfx_bus_refs
    config = Stage3Config(vfx_lora_strength=0.6)
    build_vfx_inpainting_bus(
        builder, model_ref, vae_ref, image_ref, dilated_mask_ref, pos_ref, neg_ref, config
    )
    workflow = builder.build()
    lora_id = _find_node_id(workflow, "LoRA_Loader")
    assert workflow[lora_id]["inputs"]["strength_clip"] == 0.6


def test_build_stage3_workflow_returns_dict():
    result = build_stage3_workflow("/tmp/test.mp4")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_stage3_workflow_contains_stage1_nodes():
    result = build_stage3_workflow("/tmp/test.mp4")
    class_types = {n["class_type"] for n in result.values()}
    assert "DWPose_Estimator" in class_types
    assert "Unimatch_Optical_Flow" in class_types


def test_build_stage3_workflow_contains_stage2_nodes():
    result = build_stage3_workflow("/tmp/test.mp4")
    class_types = {n["class_type"] for n in result.values()}
    assert "GGUF_Loader" in class_types
    assert "IP_Adapter_FaceID_Plus" in class_types


def test_build_stage3_workflow_contains_stage3_nodes():
    result = build_stage3_workflow("/tmp/test.mp4")
    class_types = {n["class_type"] for n in result.values()}
    assert "SAM3_VideoSegmenter" in class_types
    assert "Mask_Dilate" in class_types
    assert "VAEEncodeForInpaint" in class_types
    assert "LoRA_Loader" in class_types


def test_build_stage3_workflow_sets_video_path():
    result = build_stage3_workflow("/tmp/my_video.mp4")
    load_nodes = [n for n in result.values() if n["class_type"] == "VHS_LoadVideo"]
    assert load_nodes[0]["inputs"]["video"] == "/tmp/my_video.mp4"


def test_build_stage3_workflow_has_two_ksamplers():
    result = build_stage3_workflow("/tmp/test.mp4")
    ks_nodes = [n for n in result.values() if n["class_type"] == "KSampler"]
    assert len(ks_nodes) == 2


def test_build_stage3_workflow_vfx_ksampler_uses_config_cfg():
    config = Stage3Config(vfx_cfg=9.5)
    result = build_stage3_workflow("/tmp/test.mp4", config=config)
    ks_nodes = [n for n in result.values() if n["class_type"] == "KSampler"]
    cfgs = [n["inputs"]["cfg"] for n in ks_nodes]
    assert 9.5 in cfgs


def test_build_stage3_workflow_all_node_links_are_valid():
    """Every NodeRef link must reference an existing node ID in the workflow."""
    result = build_stage3_workflow("/tmp/test.mp4")
    node_ids = set(result.keys())
    for node in result.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in node_ids, f"Dead link: {v} references non-existent node"

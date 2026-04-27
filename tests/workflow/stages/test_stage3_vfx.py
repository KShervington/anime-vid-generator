import pytest
from anime_vid_generator.config import Stage3Config
from anime_vid_generator.workflow.builder import WorkflowBuilder
from anime_vid_generator.workflow.nodes import (
    load_video_node,
    unimatch_optical_flow_node,
)
from anime_vid_generator.workflow.stages.stage3_vfx import (
    TrackingBusResult,
    build_tracking_bus,
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

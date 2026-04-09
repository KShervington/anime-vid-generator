import pytest
from anime_vid_generator.config import Stage1Config
from anime_vid_generator.workflow.builder import WorkflowBuilder
from anime_vid_generator.workflow.nodes import load_video_node, NodeRef
from anime_vid_generator.workflow.stages.stage1_motion import (
    OrganicBusResult,
    RigidBusResult,
    TemporalUnificationResult,
    build_organic_bus,
    build_rigid_bus,
    build_temporal_unification,
    build_stage1_workflow,
)


@pytest.fixture
def builder_with_video() -> tuple[WorkflowBuilder, NodeRef]:
    builder = WorkflowBuilder()
    video = load_video_node()
    video.inputs["video"] = "/tmp/test.mp4"
    vid_id = builder.add(video)
    return builder, (vid_id, 0)


# ── Organic Bus ──────────────────────────────────────────────────────────────

def test_organic_bus_returns_result_dataclass(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_organic_bus(builder, image_ref, Stage1Config())
    assert isinstance(result, OrganicBusResult)


def test_organic_bus_adds_dwpose_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_organic_bus(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "DWPose_Estimator" in class_types


def test_organic_bus_adds_densepose_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_organic_bus(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "DensePose" in class_types


def test_organic_bus_dwpose_links_to_image_ref(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_organic_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    assert workflow[result.dwpose_output[0]]["inputs"]["image"] == list(image_ref)


def test_organic_bus_densepose_links_to_image_ref(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_organic_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    assert workflow[result.densepose_output[0]]["inputs"]["image"] == list(image_ref)


def test_organic_bus_dwpose_enables_hand_and_face_detection(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_organic_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    inputs = workflow[result.dwpose_output[0]]["inputs"]
    assert inputs["detect_hand"] is True
    assert inputs["detect_face"] is True
    assert inputs["detect_body"] is True


def test_organic_bus_output_refs_point_to_valid_nodes(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_organic_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    assert result.dwpose_output[0] in workflow
    assert result.densepose_output[0] in workflow


# ── Rigid Bus ─────────────────────────────────────────────────────────────────

def test_rigid_bus_returns_result_dataclass(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_rigid_bus(builder, image_ref, Stage1Config())
    assert isinstance(result, RigidBusResult)


def test_rigid_bus_adds_lineart_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_rigid_bus(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "ControlNet_LineArt_Anime" in class_types


def test_rigid_bus_adds_canny_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_rigid_bus(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Canny_Edge" in class_types


def test_rigid_bus_adds_zoe_depth_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_rigid_bus(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "ZoeDepth" in class_types


def test_rigid_bus_all_nodes_link_to_image_ref(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_rigid_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    for ref in (result.lineart_output, result.canny_output, result.zoedepth_mask):
        assert workflow[ref[0]]["inputs"]["image"] == list(image_ref)


def test_rigid_bus_canny_thresholds_from_config(builder_with_video):
    builder, image_ref = builder_with_video
    config = Stage1Config(canny_low_threshold=80, canny_high_threshold=180)
    result = build_rigid_bus(builder, image_ref, config)
    workflow = builder.build()
    canny_inputs = workflow[result.canny_output[0]]["inputs"]
    assert canny_inputs["low_threshold"] == 80
    assert canny_inputs["high_threshold"] == 180


def test_rigid_bus_zoedepth_model_from_config(builder_with_video):
    builder, image_ref = builder_with_video
    config = Stage1Config(zoe_depth_model="ZoeD_K")
    result = build_rigid_bus(builder, image_ref, config)
    workflow = builder.build()
    assert workflow[result.zoedepth_mask[0]]["inputs"]["model"] == "ZoeD_K"


def test_rigid_bus_output_refs_point_to_valid_nodes(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_rigid_bus(builder, image_ref, Stage1Config())
    workflow = builder.build()
    assert result.lineart_output[0] in workflow
    assert result.canny_output[0] in workflow
    assert result.zoedepth_mask[0] in workflow


# ── Temporal Unification ──────────────────────────────────────────────────────

def test_temporal_unification_returns_result_dataclass(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_temporal_unification(builder, image_ref, Stage1Config())
    assert isinstance(result, TemporalUnificationResult)


def test_temporal_unification_adds_unimatch_node(builder_with_video):
    builder, image_ref = builder_with_video
    build_temporal_unification(builder, image_ref, Stage1Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Unimatch_Optical_Flow" in class_types


def test_temporal_unification_links_to_image_ref(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_temporal_unification(builder, image_ref, Stage1Config())
    workflow = builder.build()
    assert workflow[result.flow_map[0]]["inputs"]["image"] == list(image_ref)


def test_temporal_unification_context_params_from_config(builder_with_video):
    builder, image_ref = builder_with_video
    config = Stage1Config(context_window_frames=16, context_overlap_frames=4)
    result = build_temporal_unification(builder, image_ref, config)
    workflow = builder.build()
    inputs = workflow[result.flow_map[0]]["inputs"]
    assert inputs["context_frames"] == 16
    assert inputs["overlap_frames"] == 4


def test_temporal_unification_output_ref_is_valid(builder_with_video):
    builder, image_ref = builder_with_video
    result = build_temporal_unification(builder, image_ref, Stage1Config())
    assert result.flow_map[0] in builder.build()


# ── Assembler ─────────────────────────────────────────────────────────────────

def test_build_stage1_workflow_returns_dict():
    result = build_stage1_workflow("/tmp/test.mp4")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_stage1_workflow_contains_all_node_types():
    result = build_stage1_workflow("/tmp/test.mp4")
    class_types = {n["class_type"] for n in result.values()}
    assert {
        "VHS_LoadVideo",
        "DWPose_Estimator",
        "DensePose",
        "ControlNet_LineArt_Anime",
        "Canny_Edge",
        "ZoeDepth",
        "Unimatch_Optical_Flow",
    }.issubset(class_types)


def test_build_stage1_workflow_video_path_in_load_node():
    video_path = "/absolute/path/to/video.mp4"
    result = build_stage1_workflow(video_path)
    load_nodes = [n for n in result.values() if n["class_type"] == "VHS_LoadVideo"]
    assert len(load_nodes) == 1
    assert load_nodes[0]["inputs"]["video"] == video_path


def test_build_stage1_workflow_all_node_links_are_valid():
    """Every NodeRef link must reference an existing node ID in the workflow."""
    result = build_stage1_workflow("/tmp/test.mp4")
    node_ids = set(result.keys())
    for node in result.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in node_ids, f"Dead link: {v} references non-existent node"


def test_build_stage1_workflow_accepts_custom_config():
    config = Stage1Config(context_window_frames=16, context_overlap_frames=4)
    result = build_stage1_workflow("/tmp/test.mp4", config=config)
    flow_nodes = [n for n in result.values() if n["class_type"] == "Unimatch_Optical_Flow"]
    assert flow_nodes[0]["inputs"]["context_frames"] == 16
    assert flow_nodes[0]["inputs"]["overlap_frames"] == 4

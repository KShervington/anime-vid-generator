import pytest
from anime_vid_generator.config import Stage1Config, Stage2Config
from anime_vid_generator.workflow.builder import WorkflowBuilder
from anime_vid_generator.workflow.nodes import load_video_node, NodeRef


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def builder_with_video() -> tuple[WorkflowBuilder, NodeRef]:
    builder = WorkflowBuilder()
    video = load_video_node()
    video.inputs["video"] = "/tmp/test.mp4"
    vid_id = builder.add(video)
    return builder, (vid_id, 0)


@pytest.fixture
def flow_map_ref(builder_with_video) -> tuple[WorkflowBuilder, NodeRef, NodeRef]:
    """builder, image_ref, flow_map_ref — as produced by Stage 1 temporal unification."""
    from anime_vid_generator.workflow.nodes import unimatch_optical_flow_node
    builder, image_ref = builder_with_video
    flow = unimatch_optical_flow_node()
    flow.inputs["image"] = image_ref
    flow_id = builder.add(flow)
    return builder, image_ref, (flow_id, 0)


# ── Model Loading Bus ─────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import (
    ModelLoadingBusResult,
    build_model_loading_bus,
)


def test_model_loading_bus_returns_result_dataclass(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    assert isinstance(result, ModelLoadingBusResult)


def test_model_loading_bus_adds_layered_model_unload(builder_with_video):
    builder, _ = builder_with_video
    build_model_loading_bus(builder, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Layered_Model_Unload" in class_types


def test_model_loading_bus_adds_gguf_loader(builder_with_video):
    builder, _ = builder_with_video
    build_model_loading_bus(builder, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "GGUF_Loader" in class_types


def test_model_loading_bus_gguf_uses_model_path_from_config(builder_with_video):
    builder, _ = builder_with_video
    config = Stage2Config(model_path="custom_model.gguf")
    result = build_model_loading_bus(builder, config)
    workflow = builder.build()
    assert workflow[result.model_ref[0]]["inputs"]["model_path"] == "custom_model.gguf"


def test_model_loading_bus_model_ref_is_slot_0(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    assert result.model_ref[1] == 0


def test_model_loading_bus_clip_ref_is_slot_1(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    assert result.clip_ref[1] == 1


def test_model_loading_bus_vae_ref_is_slot_2(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    assert result.vae_ref[1] == 2


def test_model_loading_bus_all_refs_point_to_same_gguf_node(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    assert result.model_ref[0] == result.clip_ref[0] == result.vae_ref[0]


def test_model_loading_bus_refs_point_to_valid_nodes(builder_with_video):
    builder, _ = builder_with_video
    result = build_model_loading_bus(builder, Stage2Config())
    workflow = builder.build()
    assert result.model_ref[0] in workflow
    assert result.vae_ref[0] in workflow

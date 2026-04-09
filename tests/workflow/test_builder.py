import pytest
from anime_vid_generator.workflow.builder import WorkflowBuilder
from anime_vid_generator.workflow.nodes import load_video_node, dwpose_estimator_node


def test_add_returns_string_id(builder):
    node_id = builder.add(load_video_node())
    assert isinstance(node_id, str)


def test_ids_start_at_one(builder):
    assert builder.add(load_video_node()) == "1"


def test_ids_increment_sequentially(builder):
    id1 = builder.add(load_video_node())
    id2 = builder.add(dwpose_estimator_node())
    assert id1 == "1"
    assert id2 == "2"


def test_build_keys_are_strings(builder):
    builder.add(load_video_node())
    assert all(isinstance(k, str) for k in builder.build().keys())


def test_build_contains_all_added_nodes(builder):
    builder.add(load_video_node())
    builder.add(dwpose_estimator_node())
    assert len(builder.build()) == 2


def test_build_serializes_literal_input(builder):
    node = load_video_node()
    node.inputs["video"] = "/tmp/test.mp4"
    builder.add(node)
    result = builder.build()
    assert result["1"]["class_type"] == "VHS_LoadVideo"
    assert result["1"]["inputs"]["video"] == "/tmp/test.mp4"


def test_build_serializes_noderef_link(builder):
    vid_id = builder.add(load_video_node())
    pose = dwpose_estimator_node()
    pose.inputs["image"] = (vid_id, 0)
    pose_id = builder.add(pose)
    result = builder.build()
    assert result[pose_id]["inputs"]["image"] == [vid_id, 0]

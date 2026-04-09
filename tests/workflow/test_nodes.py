from anime_vid_generator.workflow.nodes import (
    WorkflowNode,
    NodeRef,
    load_video_node,
    dwpose_estimator_node,
    densepose_node,
    lineart_anime_node,
    canny_edge_node,
    zoe_depth_node,
    unimatch_optical_flow_node,
)


def test_literal_input_serializes_as_bare_value():
    node = WorkflowNode(class_type="TestNode")
    node.inputs["resolution"] = 512
    assert node.to_api_dict()["inputs"]["resolution"] == 512


def test_noderef_input_serializes_as_list():
    node = WorkflowNode(class_type="TestNode")
    ref: NodeRef = ("3", 1)
    node.inputs["image"] = ref
    assert node.to_api_dict()["inputs"]["image"] == ["3", 1]


def test_meta_title_included_when_set():
    node = WorkflowNode(class_type="TestNode", meta_title="My Node")
    assert node.to_api_dict()["_meta"] == {"title": "My Node"}


def test_no_meta_key_when_title_is_none():
    node = WorkflowNode(class_type="TestNode")
    assert "_meta" not in node.to_api_dict()


def test_load_video_node_class_type():
    assert load_video_node().class_type == "VHS_LoadVideo"


def test_dwpose_estimator_node_class_type():
    assert dwpose_estimator_node().class_type == "DWPose_Estimator"


def test_densepose_node_class_type():
    assert densepose_node().class_type == "DensePose"


def test_lineart_anime_node_class_type():
    assert lineart_anime_node().class_type == "ControlNet_LineArt_Anime"


def test_canny_edge_node_class_type():
    assert canny_edge_node().class_type == "Canny_Edge"


def test_zoe_depth_node_class_type():
    assert zoe_depth_node().class_type == "ZoeDepth"


def test_unimatch_optical_flow_node_class_type():
    assert unimatch_optical_flow_node().class_type == "Unimatch_Optical_Flow"

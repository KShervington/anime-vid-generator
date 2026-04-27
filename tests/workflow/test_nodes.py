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


from anime_vid_generator.workflow.nodes import (
    layered_model_unload_node,
    gguf_loader_node,
    clip_text_encode_node,
    load_image_node,
    ip_adapter_faceid_plus_node,
    reference_only_node,
    style_transfer_block_node,
    empty_latent_video_node,
    vae_encode_node,
    flow_guided_noise_injection_node,
    free_long_node,
    ksampler_node,
    sam3_video_segmenter_node,
    mask_dilate_node,
    vae_encode_for_inpaint_node,
    lora_loader_node,
)


def test_layered_model_unload_node_class_type():
    assert layered_model_unload_node().class_type == "Layered_Model_Unload"


def test_gguf_loader_node_class_type():
    assert gguf_loader_node().class_type == "GGUF_Loader"


def test_clip_text_encode_node_class_type():
    assert clip_text_encode_node().class_type == "CLIPTextEncode"


def test_load_image_node_class_type():
    assert load_image_node().class_type == "LoadImage"


def test_ip_adapter_faceid_plus_node_class_type():
    assert ip_adapter_faceid_plus_node().class_type == "IP_Adapter_FaceID_Plus"


def test_reference_only_node_class_type():
    assert reference_only_node().class_type == "Reference_Only"


def test_style_transfer_block_node_class_type():
    assert style_transfer_block_node().class_type == "Style_Transfer_Block"


def test_empty_latent_video_node_class_type():
    assert empty_latent_video_node().class_type == "EmptyLatentVideo"


def test_vae_encode_node_class_type():
    assert vae_encode_node().class_type == "VAEEncode"


def test_flow_guided_noise_injection_node_class_type():
    assert flow_guided_noise_injection_node().class_type == "Flow_Guided_Noise_Injection"


def test_free_long_node_class_type():
    assert free_long_node().class_type == "FreeLong"


def test_ksampler_node_class_type():
    assert ksampler_node().class_type == "KSampler"


def test_all_stage2_factories_return_workflow_node():
    from anime_vid_generator.workflow.nodes import WorkflowNode
    factories = [
        layered_model_unload_node,
        gguf_loader_node,
        clip_text_encode_node,
        load_image_node,
        ip_adapter_faceid_plus_node,
        reference_only_node,
        style_transfer_block_node,
        empty_latent_video_node,
        vae_encode_node,
        flow_guided_noise_injection_node,
        free_long_node,
        ksampler_node,
    ]
    for factory in factories:
        assert isinstance(factory(), WorkflowNode), f"{factory.__name__} did not return WorkflowNode"


def test_all_stage2_factories_have_empty_inputs_by_default():
    factories = [
        layered_model_unload_node,
        gguf_loader_node,
        clip_text_encode_node,
        load_image_node,
        ip_adapter_faceid_plus_node,
        reference_only_node,
        style_transfer_block_node,
        empty_latent_video_node,
        vae_encode_node,
        flow_guided_noise_injection_node,
        free_long_node,
        ksampler_node,
    ]
    for factory in factories:
        node = factory()
        assert node.inputs == {}, f"{factory.__name__} should return node with empty inputs"


def test_sam3_video_segmenter_node_class_type():
    node = sam3_video_segmenter_node()
    assert node.class_type == "SAM3_VideoSegmenter"


def test_mask_dilate_node_class_type():
    node = mask_dilate_node()
    assert node.class_type == "Mask_Dilate"


def test_vae_encode_for_inpaint_node_class_type():
    node = vae_encode_for_inpaint_node()
    assert node.class_type == "VAEEncodeForInpaint"


def test_lora_loader_node_class_type():
    node = lora_loader_node()
    assert node.class_type == "LoRA_Loader"

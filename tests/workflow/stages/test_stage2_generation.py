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


# ── Conditioning Bus ──────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import (
    ConditioningBusResult,
    build_conditioning_bus,
)


def test_conditioning_bus_returns_result_dataclass(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    assert isinstance(result, ConditioningBusResult)


def test_conditioning_bus_adds_two_clip_text_encode_nodes(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert class_types.count("CLIPTextEncode") == 2


def test_conditioning_bus_positive_text_from_config(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    config = Stage2Config(positive_prompt="my positive prompt")
    result = build_conditioning_bus(builder, model_result.clip_ref, config)
    workflow = builder.build()
    assert workflow[result.positive_ref[0]]["inputs"]["text"] == "my positive prompt"


def test_conditioning_bus_negative_text_from_config(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    config = Stage2Config(negative_prompt="my negative prompt")
    result = build_conditioning_bus(builder, model_result.clip_ref, config)
    workflow = builder.build()
    assert workflow[result.negative_ref[0]]["inputs"]["text"] == "my negative prompt"


def test_conditioning_bus_positive_links_to_clip(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    workflow = builder.build()
    assert workflow[result.positive_ref[0]]["inputs"]["clip"] == list(model_result.clip_ref)


def test_conditioning_bus_negative_links_to_clip(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    workflow = builder.build()
    assert workflow[result.negative_ref[0]]["inputs"]["clip"] == list(model_result.clip_ref)


def test_conditioning_bus_positive_ref_is_slot_0(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    assert result.positive_ref[1] == 0


def test_conditioning_bus_negative_ref_is_slot_0(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    assert result.negative_ref[1] == 0


def test_conditioning_bus_positive_and_negative_are_different_nodes(builder_with_video):
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    assert result.positive_ref[0] != result.negative_ref[0]


# ── Identity Bus ──────────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import (
    IdentityBusResult,
    build_identity_bus,
)
from anime_vid_generator.workflow.nodes import load_image_node


@pytest.fixture
def model_and_face_refs(builder_with_video):
    """Returns (builder, model_ref, face_ref) with face node already added."""
    builder, _ = builder_with_video
    model_result = build_model_loading_bus(builder, Stage2Config())
    face = load_image_node()
    face.inputs["image"] = "/tmp/face.png"
    face_id = builder.add(face)
    return builder, model_result.model_ref, (face_id, 0)


def test_identity_bus_returns_result_dataclass(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    assert isinstance(result, IdentityBusResult)


def test_identity_bus_adds_ip_adapter_node(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "IP_Adapter_FaceID_Plus" in class_types


def test_identity_bus_adds_reference_only_node(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Reference_Only" in class_types


def test_identity_bus_adds_style_transfer_block_node(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Style_Transfer_Block" in class_types


def test_identity_bus_ip_adapter_links_to_model_ref(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    ip_node_id = _find_node_id(workflow, "IP_Adapter_FaceID_Plus")
    assert workflow[ip_node_id]["inputs"]["model"] == list(model_ref)


def test_identity_bus_ip_adapter_links_to_face_ref(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    ip_node_id = _find_node_id(workflow, "IP_Adapter_FaceID_Plus")
    assert workflow[ip_node_id]["inputs"]["image"] == list(face_ref)


def test_identity_bus_ip_adapter_weight_from_config(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    config = Stage2Config(ip_adapter_weight=0.6)
    result = build_identity_bus(builder, model_ref, face_ref, config)
    workflow = builder.build()
    ip_node_id = _find_node_id(workflow, "IP_Adapter_FaceID_Plus")
    assert workflow[ip_node_id]["inputs"]["weight"] == 0.6


def test_identity_bus_reference_only_links_to_ip_adapter_output(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    ref_node_id = _find_node_id(workflow, "Reference_Only")
    ip_node_id = _find_node_id(workflow, "IP_Adapter_FaceID_Plus")
    assert workflow[ref_node_id]["inputs"]["model"] == [ip_node_id, 0]


def test_identity_bus_reference_only_links_face_ref_as_reference(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    ref_node_id = _find_node_id(workflow, "Reference_Only")
    assert workflow[ref_node_id]["inputs"]["reference"] == list(face_ref)


def test_identity_bus_style_transfer_links_to_reference_only_output(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    style_node_id = _find_node_id(workflow, "Style_Transfer_Block")
    ref_node_id = _find_node_id(workflow, "Reference_Only")
    assert workflow[style_node_id]["inputs"]["model"] == [ref_node_id, 0]


def test_identity_bus_style_transfer_cfg_from_config(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    config = Stage2Config(style_transfer_cfg=2.0)
    result = build_identity_bus(builder, model_ref, face_ref, config)
    workflow = builder.build()
    style_node_id = _find_node_id(workflow, "Style_Transfer_Block")
    assert workflow[style_node_id]["inputs"]["cfg"] == 2.0


def test_identity_bus_conditioned_model_ref_is_slot_0(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    assert result.conditioned_model_ref[1] == 0


def test_identity_bus_conditioned_model_ref_points_to_style_transfer(model_and_face_refs):
    builder, model_ref, face_ref = model_and_face_refs
    result = build_identity_bus(builder, model_ref, face_ref, Stage2Config())
    workflow = builder.build()
    style_node_id = _find_node_id(workflow, "Style_Transfer_Block")
    assert result.conditioned_model_ref[0] == style_node_id


# helper used in identity bus and latent bus tests
def _find_node_id(workflow: dict, class_type: str) -> str:
    for node_id, node in workflow.items():
        if node["class_type"] == class_type:
            return node_id
    raise KeyError(f"No node with class_type={class_type!r} found in workflow")


# ── Latent Bus ────────────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import (
    LatentBusResult,
    build_latent_bus,
)
from anime_vid_generator.workflow.nodes import NodeRef, unimatch_optical_flow_node


@pytest.fixture
def latent_bus_prereqs(builder_with_video):
    """Returns (builder, image_ref, flow_map_ref, vae_ref) ready for build_latent_bus."""
    builder, image_ref = builder_with_video
    # simulate stage1 optical flow
    flow = unimatch_optical_flow_node()
    flow.inputs["image"] = image_ref
    flow_id = builder.add(flow)
    flow_map: NodeRef = (flow_id, 0)
    # simulate gguf vae output
    model_result = build_model_loading_bus(builder, Stage2Config())
    return builder, image_ref, flow_map, model_result.vae_ref


def test_latent_bus_returns_result_dataclass(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    result = build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config())
    assert isinstance(result, LatentBusResult)


def test_latent_bus_empty_mode_adds_empty_latent_video(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="empty"))
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "EmptyLatentVideo" in class_types


def test_latent_bus_empty_mode_does_not_add_vae_encode(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="empty"))
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "VAEEncode" not in class_types


def test_latent_bus_vae_encode_mode_adds_vae_encode(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="vae_encode"))
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "VAEEncode" in class_types


def test_latent_bus_vae_encode_mode_does_not_add_empty_latent_video(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="vae_encode"))
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "EmptyLatentVideo" not in class_types


def test_latent_bus_empty_latent_uses_width_height_from_config(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    config = Stage2Config(width=832, height=480, latent_mode="empty")
    build_latent_bus(builder, image_ref, flow_map, vae_ref, config)
    workflow = builder.build()
    latent_node_id = _find_node_id(workflow, "EmptyLatentVideo")
    inputs = workflow[latent_node_id]["inputs"]
    assert inputs["width"] == 832
    assert inputs["height"] == 480


def test_latent_bus_empty_latent_uses_context_window_as_length(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    config = Stage2Config(context_window_frames=16, latent_mode="empty")
    build_latent_bus(builder, image_ref, flow_map, vae_ref, config)
    workflow = builder.build()
    latent_node_id = _find_node_id(workflow, "EmptyLatentVideo")
    assert workflow[latent_node_id]["inputs"]["length"] == 16


def test_latent_bus_empty_latent_batch_size_is_1(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="empty"))
    workflow = builder.build()
    latent_node_id = _find_node_id(workflow, "EmptyLatentVideo")
    assert workflow[latent_node_id]["inputs"]["batch_size"] == 1


def test_latent_bus_vae_encode_links_to_image_ref(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="vae_encode"))
    workflow = builder.build()
    vae_node_id = _find_node_id(workflow, "VAEEncode")
    assert workflow[vae_node_id]["inputs"]["pixels"] == list(image_ref)


def test_latent_bus_vae_encode_links_to_vae_ref(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config(latent_mode="vae_encode"))
    workflow = builder.build()
    vae_node_id = _find_node_id(workflow, "VAEEncode")
    assert workflow[vae_node_id]["inputs"]["vae"] == list(vae_ref)


def test_latent_bus_adds_flow_guided_noise_injection(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "Flow_Guided_Noise_Injection" in class_types


def test_latent_bus_flow_noise_links_to_flow_map(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config())
    workflow = builder.build()
    fgni_id = _find_node_id(workflow, "Flow_Guided_Noise_Injection")
    assert workflow[fgni_id]["inputs"]["flow_map"] == list(flow_map)


def test_latent_bus_latent_ref_is_slot_0(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    result = build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config())
    assert result.latent_ref[1] == 0


def test_latent_bus_latent_ref_points_to_flow_noise_node(latent_bus_prereqs):
    builder, image_ref, flow_map, vae_ref = latent_bus_prereqs
    result = build_latent_bus(builder, image_ref, flow_map, vae_ref, Stage2Config())
    workflow = builder.build()
    fgni_id = _find_node_id(workflow, "Flow_Guided_Noise_Injection")
    assert result.latent_ref[0] == fgni_id


# ── Generation Bus ────────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import (
    GenerationBusResult,
    build_generation_bus,
)
from anime_vid_generator.workflow.nodes import unimatch_optical_flow_node


@pytest.fixture
def generation_bus_prereqs(builder_with_video):
    """Returns (builder, conditioned_model_ref, positive_ref, negative_ref, latent_ref)."""
    builder, image_ref = builder_with_video
    # model loading
    model_result = build_model_loading_bus(builder, Stage2Config())
    # face image
    face = load_image_node()
    face.inputs["image"] = "/tmp/face.png"
    face_id = builder.add(face)
    face_ref: NodeRef = (face_id, 0)
    # identity
    identity_result = build_identity_bus(builder, model_result.model_ref, face_ref, Stage2Config())
    # conditioning
    cond_result = build_conditioning_bus(builder, model_result.clip_ref, Stage2Config())
    # flow map
    flow = unimatch_optical_flow_node()
    flow.inputs["image"] = image_ref
    flow_id = builder.add(flow)
    flow_map: NodeRef = (flow_id, 0)
    # latent
    latent_result = build_latent_bus(builder, image_ref, flow_map, model_result.vae_ref, Stage2Config())
    return (
        builder,
        identity_result.conditioned_model_ref,
        cond_result.positive_ref,
        cond_result.negative_ref,
        latent_result.latent_ref,
    )


def test_generation_bus_returns_result_dataclass(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    result = build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    assert isinstance(result, GenerationBusResult)


def test_generation_bus_adds_free_long_node(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "FreeLong" in class_types


def test_generation_bus_adds_ksampler_node(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    class_types = [n["class_type"] for n in builder.build().values()]
    assert "KSampler" in class_types


def test_generation_bus_freelong_links_to_conditioned_model(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    fl_id = _find_node_id(workflow, "FreeLong")
    assert workflow[fl_id]["inputs"]["model"] == list(model_ref)


def test_generation_bus_freelong_context_params_from_config(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    config = Stage2Config(context_window_frames=16, context_overlap_frames=4)
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, config)
    workflow = builder.build()
    fl_id = _find_node_id(workflow, "FreeLong")
    inputs = workflow[fl_id]["inputs"]
    assert inputs["context_frames"] == 16
    assert inputs["overlap_frames"] == 4


def test_generation_bus_ksampler_links_to_freelong_output(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    fl_id = _find_node_id(workflow, "FreeLong")
    assert workflow[ks_id]["inputs"]["model"] == [fl_id, 0]


def test_generation_bus_ksampler_links_to_positive(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["positive"] == list(pos_ref)


def test_generation_bus_ksampler_links_to_negative(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["negative"] == list(neg_ref)


def test_generation_bus_ksampler_links_to_latent(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["latent_image"] == list(latent_ref)


def test_generation_bus_ksampler_sampler_settings_from_config(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    config = Stage2Config(sampler_steps=30, sampler_cfg=8.0, sampler_name="dpm_2", sampler_scheduler="sgm_uniform")
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, config)
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    inputs = workflow[ks_id]["inputs"]
    assert inputs["steps"] == 30
    assert inputs["cfg"] == 8.0
    assert inputs["sampler_name"] == "dpm_2"
    assert inputs["scheduler"] == "sgm_uniform"


def test_generation_bus_ksampler_denoise_from_config(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    config = Stage2Config(denoise=0.75)
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, config)
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["denoise"] == 0.75


def test_generation_bus_ksampler_seed_from_config(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    config = Stage2Config(seed=42)
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, config)
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["seed"] == 42


def test_generation_bus_ksampler_tiled_sampling_is_true(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert workflow[ks_id]["inputs"]["tiled_sampling"] is True


def test_generation_bus_latent_output_is_slot_0(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    result = build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    assert result.latent_output[1] == 0


def test_generation_bus_latent_output_points_to_ksampler(generation_bus_prereqs):
    builder, model_ref, pos_ref, neg_ref, latent_ref = generation_bus_prereqs
    result = build_generation_bus(builder, model_ref, pos_ref, neg_ref, latent_ref, Stage2Config())
    workflow = builder.build()
    ks_id = _find_node_id(workflow, "KSampler")
    assert result.latent_output[0] == ks_id


# ── Assembler ─────────────────────────────────────────────────────────────────

from anime_vid_generator.workflow.stages.stage2_generation import build_stage2_workflow


def test_build_stage2_workflow_returns_dict():
    result = build_stage2_workflow("/tmp/test.mp4")
    assert isinstance(result, dict)
    assert len(result) > 0


def test_build_stage2_workflow_contains_all_stage1_node_types():
    result = build_stage2_workflow("/tmp/test.mp4")
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


def test_build_stage2_workflow_contains_all_stage2_node_types():
    result = build_stage2_workflow("/tmp/test.mp4")
    class_types = {n["class_type"] for n in result.values()}
    assert {
        "Layered_Model_Unload",
        "GGUF_Loader",
        "CLIPTextEncode",
        "IP_Adapter_FaceID_Plus",
        "Reference_Only",
        "Style_Transfer_Block",
        "EmptyLatentVideo",
        "Flow_Guided_Noise_Injection",
        "FreeLong",
        "KSampler",
    }.issubset(class_types)


def test_build_stage2_workflow_with_vae_encode_mode_has_vae_encode_not_empty_latent():
    config = Stage2Config(latent_mode="vae_encode")
    result = build_stage2_workflow("/tmp/test.mp4", config=config)
    class_types = {n["class_type"] for n in result.values()}
    assert "VAEEncode" in class_types
    assert "EmptyLatentVideo" not in class_types


def test_build_stage2_workflow_video_path_in_load_node():
    video_path = "/absolute/path/to/video.mp4"
    result = build_stage2_workflow(video_path)
    load_nodes = [n for n in result.values() if n["class_type"] == "VHS_LoadVideo"]
    assert len(load_nodes) == 1
    assert load_nodes[0]["inputs"]["video"] == video_path


def test_build_stage2_workflow_model_path_from_config():
    config = Stage2Config(model_path="my_custom.gguf")
    result = build_stage2_workflow("/tmp/test.mp4", config=config)
    gguf_nodes = [n for n in result.values() if n["class_type"] == "GGUF_Loader"]
    assert len(gguf_nodes) == 1
    assert gguf_nodes[0]["inputs"]["model_path"] == "my_custom.gguf"


def test_build_stage2_workflow_all_node_links_are_valid():
    """Every NodeRef link must reference an existing node ID in the workflow."""
    result = build_stage2_workflow("/tmp/test.mp4")
    node_ids = set(result.keys())
    for node in result.values():
        for v in node["inputs"].values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in node_ids, f"Dead link: {v} references non-existent node"


def test_build_stage2_workflow_uses_stage1_config_defaults_internally():
    """Stage 1 nodes should use Stage1Config defaults (context_window=32, overlap=8)."""
    result = build_stage2_workflow("/tmp/test.mp4")
    flow_nodes = [n for n in result.values() if n["class_type"] == "Unimatch_Optical_Flow"]
    assert flow_nodes[0]["inputs"]["context_frames"] == 32
    assert flow_nodes[0]["inputs"]["overlap_frames"] == 8


def test_build_stage2_workflow_none_config_uses_defaults():
    result = build_stage2_workflow("/tmp/test.mp4", config=None)
    ks_nodes = [n for n in result.values() if n["class_type"] == "KSampler"]
    assert len(ks_nodes) == 1
    assert ks_nodes[0]["inputs"]["steps"] == 20


def test_build_stage2_workflow_ksampler_tiled_sampling_always_true():
    result = build_stage2_workflow("/tmp/test.mp4")
    ks_nodes = [n for n in result.values() if n["class_type"] == "KSampler"]
    assert ks_nodes[0]["inputs"]["tiled_sampling"] is True


def test_build_stage2_workflow_has_exactly_one_layered_model_unload():
    result = build_stage2_workflow("/tmp/test.mp4")
    unload_nodes = [n for n in result.values() if n["class_type"] == "Layered_Model_Unload"]
    assert len(unload_nodes) == 1


def test_build_stage2_workflow_has_exactly_two_clip_text_encode_nodes():
    result = build_stage2_workflow("/tmp/test.mp4")
    clip_nodes = [n for n in result.values() if n["class_type"] == "CLIPTextEncode"]
    assert len(clip_nodes) == 2

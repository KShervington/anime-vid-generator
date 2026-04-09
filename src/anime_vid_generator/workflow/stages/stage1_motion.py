from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    dwpose_estimator_node,
    densepose_node,
    lineart_anime_node,
    canny_edge_node,
    zoe_depth_node,
    unimatch_optical_flow_node,
    load_video_node,
)
from ...config import Stage1Config


@dataclass
class OrganicBusResult:
    dwpose_output: NodeRef    # slot 0: skeletal + facial pose map
    densepose_output: NodeRef  # slot 0: volumetric depth map


@dataclass
class RigidBusResult:
    lineart_output: NodeRef  # slot 0: line-art preprocessed image
    canny_output: NodeRef    # slot 0: edge map
    zoedepth_mask: NodeRef   # slot 0: metric depth map


@dataclass
class TemporalUnificationResult:
    flow_map: NodeRef  # slot 0: optical flow map


def build_organic_bus(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    config: Stage1Config,
) -> OrganicBusResult:
    """Add DWPose + DensePose nodes, both reading video frames from image_ref."""
    pose = dwpose_estimator_node()
    pose.inputs["image"] = image_ref
    pose.inputs["detect_hand"] = True
    pose.inputs["detect_face"] = True
    pose.inputs["detect_body"] = True
    pose_id = builder.add(pose)

    dense = densepose_node()
    dense.inputs["image"] = image_ref
    dense.inputs["model"] = "DensePose_R50_FPN_s1x"
    dense_id = builder.add(dense)

    return OrganicBusResult(
        dwpose_output=(pose_id, 0),
        densepose_output=(dense_id, 0),
    )


def build_rigid_bus(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    config: Stage1Config,
) -> RigidBusResult:
    """Add LineArt, Canny, and ZoeDepth nodes, all reading from image_ref."""
    lineart = lineart_anime_node()
    lineart.inputs["image"] = image_ref
    lineart.inputs["coarse"] = False
    lineart_id = builder.add(lineart)

    canny = canny_edge_node()
    canny.inputs["image"] = image_ref
    canny.inputs["low_threshold"] = config.canny_low_threshold
    canny.inputs["high_threshold"] = config.canny_high_threshold
    canny_id = builder.add(canny)

    depth = zoe_depth_node()
    depth.inputs["image"] = image_ref
    depth.inputs["model"] = config.zoe_depth_model
    depth_id = builder.add(depth)

    return RigidBusResult(
        lineart_output=(lineart_id, 0),
        canny_output=(canny_id, 0),
        zoedepth_mask=(depth_id, 0),
    )


def build_temporal_unification(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    config: Stage1Config,
) -> TemporalUnificationResult:
    """Add Unimatch Optical Flow node, reading video frames from image_ref."""
    flow = unimatch_optical_flow_node()
    flow.inputs["image"] = image_ref
    flow.inputs["raft_iters"] = config.optical_flow_raft_iters
    flow.inputs["context_frames"] = config.context_window_frames
    flow.inputs["overlap_frames"] = config.context_overlap_frames
    flow_id = builder.add(flow)

    return TemporalUnificationResult(flow_map=(flow_id, 0))


def build_stage1_workflow(
    video_path: str,
    config: Stage1Config | None = None,
) -> dict:
    """Build the complete Stage 1 workflow and return ComfyUI API JSON.

    Args:
        video_path: Absolute path to the source video file.
        config: Stage1Config override; uses defaults if None.

    Returns:
        dict in ComfyUI /prompt API format, ready to POST.
    """
    if config is None:
        config = Stage1Config()

    builder = WorkflowBuilder()

    video = load_video_node()
    video.inputs["video"] = video_path
    video.inputs["frame_load_cap"] = 0
    video.inputs["skip_first_frames"] = 0
    vid_id = builder.add(video)
    image_ref: NodeRef = (vid_id, 0)

    build_organic_bus(builder, image_ref, config)
    build_rigid_bus(builder, image_ref, config)
    build_temporal_unification(builder, image_ref, config)

    return builder.build()

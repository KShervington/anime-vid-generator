from dataclasses import dataclass

from ..builder import WorkflowBuilder
from ..nodes import (
    NodeRef,
    sam3_video_segmenter_node,
    mask_dilate_node,
)
from ...config import Stage3Config


@dataclass
class TrackingBusResult:
    dilated_mask_ref: NodeRef  # Mask_Dilate slot 0


def build_tracking_bus(
    builder: WorkflowBuilder,
    image_ref: NodeRef,
    flow_map: NodeRef,
    config: Stage3Config,
) -> TrackingBusResult:
    """Add SAM3_VideoSegmenter and Mask_Dilate nodes for emitter tracking."""
    sam = sam3_video_segmenter_node()
    sam.inputs["image"] = image_ref
    sam.inputs["emitter_prompt"] = config.emitter_prompt
    sam_id = builder.add(sam)

    dilate = mask_dilate_node()
    dilate.inputs["mask"] = (sam_id, 0)
    dilate.inputs["flow_map"] = flow_map
    dilate.inputs["dilation"] = config.mask_dilation
    dilate_id = builder.add(dilate)

    return TrackingBusResult(dilated_mask_ref=(dilate_id, 0))

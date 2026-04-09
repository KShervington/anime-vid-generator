from dataclasses import dataclass, field

# Points to another node's output: ("node_id_string", output_slot_index)
NodeRef = tuple[str, int]

# Any valid ComfyUI input: a node link or a scalar value
NodeInput = NodeRef | str | int | float | bool | list | None


@dataclass
class WorkflowNode:
    class_type: str
    inputs: dict[str, NodeInput] = field(default_factory=dict)
    meta_title: str | None = None

    def to_api_dict(self) -> dict:
        serialized_inputs = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in self.inputs.items()
        }
        result: dict = {"class_type": self.class_type, "inputs": serialized_inputs}
        if self.meta_title is not None:
            result["_meta"] = {"title": self.meta_title}
        return result


def load_video_node() -> WorkflowNode:
    return WorkflowNode(class_type="VHS_LoadVideo", meta_title="Load Input Video")


def dwpose_estimator_node() -> WorkflowNode:
    return WorkflowNode(class_type="DWPose_Estimator", meta_title="DWPose Estimator")


def densepose_node() -> WorkflowNode:
    return WorkflowNode(class_type="DensePose", meta_title="DensePose")


def lineart_anime_node() -> WorkflowNode:
    return WorkflowNode(class_type="ControlNet_LineArt_Anime", meta_title="LineArt Anime")


def canny_edge_node() -> WorkflowNode:
    return WorkflowNode(class_type="Canny_Edge", meta_title="Canny Edge")


def zoe_depth_node() -> WorkflowNode:
    return WorkflowNode(class_type="ZoeDepth", meta_title="ZoeDepth")


def unimatch_optical_flow_node() -> WorkflowNode:
    return WorkflowNode(class_type="Unimatch_Optical_Flow", meta_title="Unimatch Optical Flow")

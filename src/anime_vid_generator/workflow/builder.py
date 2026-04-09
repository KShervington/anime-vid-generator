from .nodes import WorkflowNode


class WorkflowBuilder:
    def __init__(self) -> None:
        self._nodes: dict[str, WorkflowNode] = {}
        self._counter: int = 1

    def add(self, node: WorkflowNode) -> str:
        """Register a node and return its auto-assigned string ID."""
        node_id = str(self._counter)
        self._nodes[node_id] = node
        self._counter += 1
        return node_id

    def build(self) -> dict:
        """Serialize all registered nodes to ComfyUI /prompt API format."""
        return {node_id: node.to_api_dict() for node_id, node in self._nodes.items()}

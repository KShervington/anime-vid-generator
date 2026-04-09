import json
import uuid
from typing import Any, AsyncIterator

import httpx
import websockets


class ComfyUIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188") -> None:
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    async def submit_workflow(self, workflow: dict[str, Any]) -> str:
        """POST workflow to ComfyUI and return the assigned prompt_id."""
        payload = {"prompt": workflow, "client_id": self.client_id}
        async with httpx.AsyncClient() as http:
            response = await http.post(f"{self.base_url}/prompt", json=payload)
            response.raise_for_status()
            return response.json()["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict[str, Any]:
        """Fetch completed generation history for a prompt_id."""
        async with httpx.AsyncClient() as http:
            response = await http.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()

    async def monitor_progress(self, prompt_id: str) -> AsyncIterator[dict[str, Any]]:
        """Yield WebSocket progress events until execution for prompt_id completes."""
        ws_url = f"{self.base_url.replace('http', 'ws')}/ws?clientId={self.client_id}"
        async with websockets.connect(ws_url) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                yield msg
                if (
                    msg.get("type") == "executing"
                    and msg.get("data", {}).get("node") is None
                    and msg.get("data", {}).get("prompt_id") == prompt_id
                ):
                    break

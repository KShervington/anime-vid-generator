import json
import pytest
import httpx
import respx
from anime_vid_generator.client import ComfyUIClient


@pytest.fixture
def client() -> ComfyUIClient:
    return ComfyUIClient("http://127.0.0.1:8188")


@respx.mock
async def test_submit_workflow_posts_to_prompt_endpoint(client):
    route = respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "abc-123"})
    )
    result = await client.submit_workflow({"1": {"class_type": "Test", "inputs": {}}})
    assert result == "abc-123"
    assert route.called


@respx.mock
async def test_submit_workflow_includes_client_id(client):
    respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "abc"})
    )
    await client.submit_workflow({"1": {"class_type": "X", "inputs": {}}})
    payload = json.loads(respx.calls.last.request.content)
    assert payload["client_id"] == client.client_id


@respx.mock
async def test_submit_workflow_raises_on_http_error(client):
    respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=httpx.Response(400, json={"error": "bad prompt"})
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.submit_workflow({})


@respx.mock
async def test_get_history_returns_parsed_response(client):
    prompt_id = "abc-123"
    expected = {prompt_id: {"outputs": {}, "status": {"completed": True}}}
    respx.get(f"http://127.0.0.1:8188/history/{prompt_id}").mock(
        return_value=httpx.Response(200, json=expected)
    )
    result = await client.get_history(prompt_id)
    assert result[prompt_id]["status"]["completed"] is True


@respx.mock
async def test_get_history_raises_on_http_error(client):
    respx.get("http://127.0.0.1:8188/history/bad-id").mock(
        return_value=httpx.Response(404, text="not found")
    )
    with pytest.raises(httpx.HTTPStatusError):
        await client.get_history("bad-id")


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


async def test_monitor_progress_yields_events_and_stops_on_completion(client):
    """monitor_progress should yield events and stop when the executing/None event arrives for our prompt_id."""
    prompt_id = "test-prompt-id"

    # Simulate WebSocket messages
    messages = [
        json.dumps({"type": "progress", "data": {"node": "1", "value": 5, "max": 10, "prompt_id": prompt_id}}),
        json.dumps({"type": "executing", "data": {"node": "2", "prompt_id": prompt_id}}),
        json.dumps({"type": "executing", "data": {"node": None, "prompt_id": prompt_id}}),  # completion signal
        json.dumps({"type": "progress", "data": {"node": "3", "prompt_id": prompt_id}}),  # should NOT be yielded
    ]

    async def fake_ws_iter():
        for msg in messages:
            yield msg

    mock_ws = MagicMock()
    mock_ws.__aiter__ = lambda self: fake_ws_iter()
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)

    with patch("anime_vid_generator.client.websockets.connect", return_value=mock_ws):
        collected = []
        async for event in client.monitor_progress(prompt_id):
            collected.append(event)

    # Should yield the first 3 messages, stop at the "node=None" completion signal
    assert len(collected) == 3
    assert collected[0]["type"] == "progress"
    assert collected[1]["type"] == "executing"
    assert collected[2]["data"]["node"] is None

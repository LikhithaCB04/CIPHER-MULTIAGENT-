import asyncio
import json
import httpx
import websockets
import pytest
import uuid

@pytest.mark.asyncio
async def test_streaming_pipeline_sequence():
    task_id = f"test-{uuid.uuid4().hex[:6]}"
    payload = {
        "task_id": task_id,
        "description": "test task for fullstack streaming",
        "task_type": "fullstack",
        "context": "React frontend component test",
        "priority": "low"
    }

    ws_url = "ws://localhost:8000/ws"
    http_url = "http://localhost:8000/run"

    async with websockets.connect(ws_url) as ws:
        async def run_http():
            async with httpx.AsyncClient() as client:
                await client.post(http_url, json=payload, timeout=30.0)

        task = asyncio.create_task(run_http())

        events_received = []
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                event = json.loads(msg)
                
                if event.get("task_id") == task_id:
                    events_received.append(event.get("event"))
                    if event.get("event") == "pipeline_complete":
                        break
        except asyncio.TimeoutError:
            pass
        
        await task

        assert "task_received" in events_received, "Missing task_received"
        assert "agent_thought" in events_received, "Missing agent_thought"
        assert "agent_finished" in events_received, "Missing agent_finished"
        assert "pipeline_complete" in events_received, "Missing pipeline_complete"

        first_thought_idx = events_received.index("agent_thought")
        finished_idx = events_received.index("agent_finished")
        assert first_thought_idx < finished_idx, "Thoughts should arrive before agent_finished"
        
        assert events_received[-1] == "pipeline_complete", "pipeline_complete should be final event"

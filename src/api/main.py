from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json
from pydantic import BaseModel
from src.agents.orchestrator import create_supervisor_graph
from typing import Optional
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ORCA AI API", description="Marine Ecosystem Reasoning with Collaborative Agents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LangGraph Orchestrator
graph = create_supervisor_graph()

# ── Session-based conversation history for multi-turn context ──────────────
# Maps session_id -> list of {role, content} dicts
_session_histories: dict[str, list[dict]] = {}
MAX_HISTORY_TURNS = 10  # Keep last 10 turns (5 exchanges) per session

def _get_history(session_id: str) -> list[dict]:
    """Get conversation history for a session."""
    return _session_histories.get(session_id, [])

def _append_history(session_id: str, role: str, content: str):
    """Append a turn to conversation history, trimming to MAX_HISTORY_TURNS."""
    if session_id not in _session_histories:
        _session_histories[session_id] = []
    _session_histories[session_id].append({"role": role, "content": content})
    # Trim to keep only the most recent turns
    if len(_session_histories[session_id]) > MAX_HISTORY_TURNS:
        _session_histories[session_id] = _session_histories[session_id][-MAX_HISTORY_TURNS:]

class ChatRequest(BaseModel):
    query: str
    target_language: Optional[str] = "English"
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set. Please set it in a .env file or your terminal.")
        
    try:
        session_id = request.session_id or "default"
        history = _get_history(session_id)

        # Pass the human query + conversation history into the state graph
        result = await graph.ainvoke({
            "messages": [HumanMessage(content=request.query)],
            "target_language": request.target_language,
            "conversation_history": history
        })
        
        # The graph executes, and we want the final meaningful output
        final_message = ""
        for msg in reversed(result["messages"]):
            if getattr(msg, "type", "") == "ai" or getattr(msg, "type", "") == "AIMessageChunk":
                content = msg.content
                if isinstance(content, list):
                    text_blocks = [block.get("text", "") for block in content if isinstance(block, dict) and "text" in block]
                    content = "".join(text_blocks)
                
                content = str(content).strip()
                if content and content != "[]":
                    final_message = content
                    break
                    
        if not final_message:
            final_message = "No clear response provided by the agents."

        # Update conversation history with this exchange
        _append_history(session_id, "user", request.query)
        _append_history(session_id, "assistant", final_message)
            
        return ChatResponse(response=final_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent graph: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY environment variable not set.")

    import asyncio

    session_id = request.session_id or "default"
    history = _get_history(session_id)

    async def event_generator():
        MAX_RETRIES = 2
        collected_text = ""  # Collect the full response for history

        for attempt in range(MAX_RETRIES + 1):
            try:
                async for event in graph.astream_events({
                    "messages": [HumanMessage(content=request.query)],
                    "target_language": request.target_language,
                    "conversation_history": history
                }, version="v1"):
                    kind = event["event"]
                    name = event["name"]

                    data_payload = {"event": kind, "name": name, "content": ""}

                    if kind == "on_chat_model_stream":
                        tags = event.get("tags", [])
                        if "visualization_llm" not in tags and "geospatial_llm" not in tags:
                            chunk = event["data"]["chunk"]
                            if hasattr(chunk, "content"):
                                if isinstance(chunk.content, str):
                                    data_payload["content"] = chunk.content
                                    collected_text += chunk.content
                                elif isinstance(chunk.content, list):
                                    text_blocks = [b.get("text", "") for b in chunk.content if isinstance(b, dict) and "text" in b]
                                    joined = "".join(text_blocks)
                                    data_payload["content"] = joined
                                    collected_text += joined
                            yield f"data: {json.dumps(data_payload)}\n\n"

                    if kind == "on_tool_end":
                        data_payload["output"] = event.get("data", {}).get("output", "")
                        yield f"data: {json.dumps(data_payload)}\n\n"

                    if kind == "on_chain_end" and name == "visualization":
                        output = event["data"].get("output", {})
                        if "layers" in output:
                            yield f"data: {json.dumps({'event': 'on_layers_ready', 'layers': output['layers']})}\n\n"

                # Stream completed successfully — update conversation history
                _append_history(session_id, "user", request.query)
                if collected_text.strip():
                    _append_history(session_id, "assistant", collected_text.strip())

                yield "data: [DONE]\n\n"
                return  # Exit retry loop on success

            except Exception as e:
                err_str = str(e)
                is_connection_err = any(kw in err_str.lower() for kw in [
                    "wsarecv", "connection", "forcibly closed", "reset", "eof", "stream reading"
                ])

                if is_connection_err and attempt < MAX_RETRIES:
                    # Notify frontend of retry, then wait and try again
                    retry_msg = f"Connection interrupted — retrying ({attempt + 1}/{MAX_RETRIES})..."
                    yield f"data: {json.dumps({'event': 'on_chat_model_stream', 'content': retry_msg})}\n\n"
                    await asyncio.sleep(1.5)
                    continue  # retry

                # Final failure — send clean error
                friendly = "The API connection was interrupted. Please try again." if is_connection_err else f"Error: {err_str}"
                yield f"data: {json.dumps({'error': friendly})}\n\n"
                yield "data: [DONE]\n\n"
                return

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/map-data/chlorophyll")
async def get_chlorophyll_grid():
    """
    Returns the full Copernicus chlorophyll grid as a list of points for the frontend heatmap.
    Each point: { lat, lon, chl, is_pfz }
    """
    try:
        from src.tools.copernicus_loader import _load, _chl_data, _lats, _lons
        import numpy as np
        if not _load():
            return {"grid": [], "error": "Copernicus data not loaded"}
        grid = []
        for i, lat in enumerate(_lats):
            for j, lon in enumerate(_lons):
                val = _chl_data[i, j]
                if not np.isnan(val):
                    chl = round(float(val), 4)
                    grid.append({ "lat": float(lat), "lon": float(lon), "chl": chl, "is_pfz": chl >= 1.0 })
        return { "grid": grid, "count": len(grid), "coverage": { "min_lat": float(_lats.min()), "max_lat": float(_lats.max()), "min_lon": float(_lons.min()), "max_lon": float(_lons.max()) } }
    except Exception as e:
        return {"grid": [], "error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "ORCA API"}

"""OpenAI-compatible /v1/chat/completions shim over AWS Bedrock Converse.

Purpose: lets Hindsight (which speaks OpenAI HTTP only) drive Bedrock-hosted
models for the LongMemEval sweep without modifying Hindsight itself.

Quirks handled (per memory/reference_aws_bedrock_gotchas):
- `us.` inference-profile prefix injected for Anthropic + Llama 4 + a few
  Nemotron IDs that require it.
- `additionalModelRequestFields.reasoning_effort` injected for gpt-oss
  models (Bedrock rejects "minimal"; "low" is the floor).
- `reasoningContent` blocks stripped from the user-visible content but their
  tokens are still counted in usage.

Non-streaming only. Hindsight v0.6.x does not request stream=true on the
retain extraction path (confirmed by grep over the open-source server).
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("bedrock_proxy")

REGION = os.environ.get("AWS_REGION", "us-east-1")

# Models that require the us. inference-profile prefix on Bedrock.
INFERENCE_PROFILE_PREFIX_MODELS = {
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-6-v1:0",
    "meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama4-maverick-17b-instruct-v1:0",
}

# gpt-oss family: needs reasoning_effort. Bedrock floor is "low".
GPT_OSS_MODELS = {
    "openai.gpt-oss-20b-1:0",
    "openai.gpt-oss-120b-1:0",
}

_bedrock = boto3.client(
    "bedrock-runtime",
    region_name=REGION,
    config=BotoConfig(read_timeout=600, connect_timeout=15, retries={"max_attempts": 2}),
)

app = FastAPI(title="bedrock-proxy", version="0.1.0")


class Message(BaseModel):
    role: str
    content: Any  # str or list of content parts


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Any] = None
    stream: Optional[bool] = False
    # OpenAI passes some we ignore (presence_penalty etc.); pydantic allows extras off by default.

    class Config:
        extra = "ignore"


def _resolve_model_id(model: str) -> str:
    if model in INFERENCE_PROFILE_PREFIX_MODELS or model.startswith("anthropic.claude-sonnet-4") \
            or model.startswith("meta.llama4-"):
        return f"us.{model}"
    return model


def _to_bedrock_messages(messages: List[Message]) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, str]]]]:
    """Convert OpenAI messages to Bedrock converse messages + system blocks."""
    system_blocks: List[Dict[str, str]] = []
    out: List[Dict[str, Any]] = []
    for m in messages:
        content_text = m.content if isinstance(m.content, str) else \
            "".join(p.get("text", "") for p in m.content if isinstance(p, dict))
        if m.role == "system":
            system_blocks.append({"text": content_text})
            continue
        role = "assistant" if m.role == "assistant" else "user"
        out.append({"role": role, "content": [{"text": content_text}]})
    return out, (system_blocks or None)


def _extract_text(content_blocks: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for b in content_blocks or []:
        # Skip reasoningContent — token-counted but not user-visible content.
        if "text" in b:
            parts.append(b["text"])
    return "".join(parts)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
def list_models() -> Dict[str, Any]:
    # Hindsight may probe this; return a stub.
    return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> Dict[str, Any]:
    if req.stream:
        # Hindsight doesn't need it; refuse loudly rather than fake it.
        raise HTTPException(status_code=400, detail="stream=true not implemented")

    model_id = _resolve_model_id(req.model)
    messages, system = _to_bedrock_messages(req.messages)

    inference_config: Dict[str, Any] = {}
    if req.temperature is not None:
        inference_config["temperature"] = req.temperature
    if req.top_p is not None:
        inference_config["topP"] = req.top_p
    max_tok = req.max_completion_tokens or req.max_tokens
    if max_tok is not None:
        inference_config["maxTokens"] = max_tok
    if req.stop:
        stops = req.stop if isinstance(req.stop, list) else [req.stop]
        inference_config["stopSequences"] = stops

    additional: Dict[str, Any] = {}
    if req.model in GPT_OSS_MODELS:
        additional["reasoning_effort"] = "low"

    kwargs: Dict[str, Any] = {
        "modelId": model_id,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if inference_config:
        kwargs["inferenceConfig"] = inference_config
    if additional:
        kwargs["additionalModelRequestFields"] = additional

    log.info("converse model=%s msgs=%d max=%s", model_id, len(messages), max_tok)
    try:
        resp = _bedrock.converse(**kwargs)
    except _bedrock.exceptions.ValidationException as e:
        raise HTTPException(status_code=400, detail=f"bedrock validation: {e}")
    except Exception as e:
        log.exception("bedrock converse failed")
        raise HTTPException(status_code=502, detail=f"bedrock: {e}")

    out_msg = resp.get("output", {}).get("message", {})
    text = _extract_text(out_msg.get("content", []))
    usage = resp.get("usage", {}) or {}
    stop_reason = resp.get("stopReason", "stop")
    finish_map = {"end_turn": "stop", "max_tokens": "length", "stop_sequence": "stop",
                  "tool_use": "tool_calls", "content_filtered": "content_filter"}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_map.get(stop_reason, "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens",
                                     usage.get("inputTokens", 0) + usage.get("outputTokens", 0)),
        },
    }

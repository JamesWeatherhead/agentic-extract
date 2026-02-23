"""VLM (Vision Language Model) client abstraction layer.

Provides a unified interface for sending vision requests to Claude and
Codex/GPT-4o. Handles image encoding, API calls, error handling, and
exponential backoff for rate limits.
"""
from __future__ import annotations

import asyncio
import base64
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VLMResponse:
    """Response from a VLM vision request."""

    content: Any
    confidence: float
    model: str
    usage_tokens: int
    duration_ms: int


class VLMClient(ABC):
    """Abstract base class for VLM clients."""

    @abstractmethod
    async def send_vision_request(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> VLMResponse:
        """Send an image + prompt to the VLM and return structured response."""
        ...


def _encode_image_base64(image_path: Path) -> str:
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _detect_media_type(image_path: Path) -> str:
    """Detect the media type from the file extension."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(suffix, "image/png")


class ClaudeClient(VLMClient):
    """Claude API client for vision requests.

    Uses the Anthropic SDK to send images with prompts. Handles
    exponential backoff on rate limits.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-20250514",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        import anthropic
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = anthropic.Anthropic(api_key=api_key)

    async def send_vision_request(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> VLMResponse:
        image_b64 = _encode_image_base64(image_path)
        media_type = _detect_media_type(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=messages,
                )
                duration_ms = int((time.monotonic() - start) * 1000)

                raw_text = response.content[0].text
                try:
                    content = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    content = {"raw_text": raw_text}

                total_tokens = (
                    response.usage.input_tokens + response.usage.output_tokens
                )
                return VLMResponse(
                    content=content,
                    confidence=0.9,
                    model=self.model,
                    usage_tokens=total_tokens,
                    duration_ms=duration_ms,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"VLM request failed after {self.max_retries} attempts: {last_error}"
        )


class CodexClient(VLMClient):
    """OpenAI/Codex API client for vision requests.

    Uses the OpenAI SDK with optional Structured Outputs
    (response_format) for schema enforcement.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        import openai
        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self._client = openai.OpenAI(api_key=api_key)

    async def send_vision_request(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> VLMResponse:
        image_b64 = _encode_image_base64(image_path)
        media_type = _detect_media_type(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction_output",
                    "schema": schema,
                },
            }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                start = time.monotonic()
                response = self._client.chat.completions.create(**kwargs)
                duration_ms = int((time.monotonic() - start) * 1000)

                raw_text = response.choices[0].message.content
                try:
                    content = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    content = {"raw_text": raw_text}

                total_tokens = (
                    response.usage.prompt_tokens
                    + response.usage.completion_tokens
                )
                return VLMResponse(
                    content=content,
                    confidence=0.9,
                    model=self.model,
                    usage_tokens=total_tokens,
                    duration_ms=duration_ms,
                )
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"VLM request failed after {self.max_retries} attempts: {last_error}"
        )

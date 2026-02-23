"""Tests for VLM client abstraction (Claude and Codex)."""
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic_extract.clients.vlm import (
    ClaudeClient,
    CodexClient,
    VLMResponse,
)


def test_vlm_response_dataclass():
    resp = VLMResponse(
        content={"text": "hello"},
        confidence=0.95,
        model="claude-opus-4-20250514",
        usage_tokens=150,
        duration_ms=1200,
    )
    assert resp.content == {"text": "hello"}
    assert resp.confidence == 0.95
    assert resp.model == "claude-opus-4-20250514"


def test_claude_client_init():
    client = ClaudeClient(api_key="test-key", model="claude-opus-4-20250514")
    assert client.model == "claude-opus-4-20250514"


def test_codex_client_init():
    client = CodexClient(api_key="test-key", model="gpt-4o")
    assert client.model == "gpt-4o"


@pytest.mark.asyncio
async def test_claude_client_send_vision_request(tmp_path: Path):
    # Create a tiny test image
    from PIL import Image
    img = Image.new("RGB", (10, 10), "red")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text='{"result": "extracted text"}')]
    mock_message.usage.input_tokens = 100
    mock_message.usage.output_tokens = 50

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        MockAnthropic.return_value = mock_client

        client = ClaudeClient(api_key="test-key", model="claude-opus-4-20250514")
        client._client = mock_client

        resp = await client.send_vision_request(
            image_path=img_path,
            prompt="Extract the text from this image.",
        )
        assert isinstance(resp, VLMResponse)
        assert resp.content is not None
        assert resp.usage_tokens == 150
        assert resp.duration_ms >= 0


@pytest.mark.asyncio
async def test_codex_client_send_vision_request(tmp_path: Path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "blue")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": "codex output"}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 80
    mock_response.usage.completion_tokens = 40

    with patch("openai.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        client = CodexClient(api_key="test-key", model="gpt-4o")
        client._client = mock_client

        resp = await client.send_vision_request(
            image_path=img_path,
            prompt="Extract the text from this image.",
        )
        assert isinstance(resp, VLMResponse)
        assert resp.usage_tokens == 120
        assert resp.duration_ms >= 0


@pytest.mark.asyncio
async def test_codex_client_structured_output(tmp_path: Path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "green")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    mock_choice = MagicMock()
    mock_choice.message.content = '{"headers": ["A"], "rows": [{"A": 1}]}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50

    with patch("openai.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        client = CodexClient(api_key="test-key", model="gpt-4o")
        client._client = mock_client

        schema = {
            "type": "object",
            "properties": {
                "headers": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array"},
            },
        }
        resp = await client.send_vision_request(
            image_path=img_path,
            prompt="Extract the table.",
            schema=schema,
        )
        assert resp.content is not None


@pytest.mark.asyncio
async def test_claude_client_handles_api_error(tmp_path: Path):
    from PIL import Image
    img = Image.new("RGB", (10, 10), "white")
    img_path = tmp_path / "test.png"
    img.save(img_path)

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API rate limit")
        MockAnthropic.return_value = mock_client

        client = ClaudeClient(
            api_key="test-key", model="claude-opus-4-20250514", max_retries=1,
        )
        client._client = mock_client

        with pytest.raises(RuntimeError, match="VLM request failed"):
            await client.send_vision_request(
                image_path=img_path,
                prompt="Extract text.",
            )

# tests/tools/test_docker_runner.py
"""Tests for Docker tool runner."""
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agentic_extract.tools.docker_runner import DockerTool, ToolOutput


def test_tool_output_dataclass():
    out = ToolOutput(stdout="hello", stderr="", exit_code=0, duration_ms=150)
    assert out.stdout == "hello"
    assert out.exit_code == 0
    assert out.duration_ms == 150


def test_docker_tool_init():
    tool = DockerTool(image_name="paddleocr:latest", default_timeout=60)
    assert tool.image_name == "paddleocr:latest"
    assert tool.default_timeout == 60
    assert tool.volumes == {}


def test_docker_tool_init_with_volumes():
    vols = {"/host/data": "/container/data"}
    tool = DockerTool(image_name="test:latest", default_timeout=30, volumes=vols)
    assert tool.volumes == {"/host/data": "/container/data"}


def test_docker_tool_build_command():
    tool = DockerTool(
        image_name="myimage:latest",
        default_timeout=30,
        volumes={"/data": "/data"},
    )
    cmd = tool._build_command(["--input", "/data/test.png"])
    assert cmd[0] == "docker"
    assert "run" in cmd
    assert "--rm" in cmd
    assert "myimage:latest" in cmd
    assert "-v" in cmd
    assert "/data:/data" in cmd
    assert "--input" in cmd
    assert "/data/test.png" in cmd


@patch("subprocess.run")
def test_docker_tool_run_success(mock_run: MagicMock):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="output data", stderr="",
    )
    tool = DockerTool(image_name="test:latest", default_timeout=30)
    result = tool.run(["--help"])
    assert result.exit_code == 0
    assert result.stdout == "output data"
    assert result.stderr == ""
    assert result.duration_ms >= 0
    mock_run.assert_called_once()


@patch("subprocess.run")
def test_docker_tool_run_captures_stderr(mock_run: MagicMock):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="Error: file not found",
    )
    tool = DockerTool(image_name="test:latest", default_timeout=30)
    result = tool.run(["--bad-arg"])
    assert result.exit_code == 1
    assert "file not found" in result.stderr


@patch("subprocess.run")
def test_docker_tool_run_timeout(mock_run: MagicMock):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=5)
    tool = DockerTool(image_name="test:latest", default_timeout=5)
    result = tool.run(["--slow-op"])
    assert result.exit_code == -1
    assert "timeout" in result.stderr.lower()


@patch("subprocess.run")
def test_docker_tool_run_image_not_found(mock_run: MagicMock):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=125,
        stdout="", stderr="Unable to find image 'fake:latest' locally",
    )
    tool = DockerTool(image_name="fake:latest", default_timeout=30)
    result = tool.run([])
    assert result.exit_code == 125
    assert "Unable to find image" in result.stderr


@patch("subprocess.run")
def test_docker_tool_pull(mock_run: MagicMock):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Status: Image is up to date", stderr="",
    )
    tool = DockerTool(image_name="test:latest", default_timeout=30)
    success = tool.pull()
    assert success is True
    call_args = mock_run.call_args[0][0]
    assert call_args == ["docker", "pull", "test:latest"]


@patch("subprocess.run")
def test_docker_tool_pull_failure(mock_run: MagicMock):
    mock_run.return_value = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="pull access denied",
    )
    tool = DockerTool(image_name="private:latest", default_timeout=30)
    success = tool.pull()
    assert success is False

"""Base Docker tool runner for executing open-source tools in containers.

Every open-source tool runs in its own Docker container. This module
provides the base class for building tool-specific wrappers. Uses
subprocess (not the Docker SDK) for simplicity and fewer dependencies.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class ToolOutput:
    """Result from running a Docker container."""

    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int


class DockerTool:
    """Base class for running open-source tools inside Docker containers.

    Args:
        image_name: Docker image name and tag (e.g. "paddleocr:latest").
        default_timeout: Maximum seconds before killing the container.
        volumes: Host-to-container volume mappings (e.g. {"/data": "/data"}).
    """

    def __init__(
        self,
        image_name: str,
        default_timeout: int = 120,
        volumes: dict[str, str] | None = None,
    ) -> None:
        self.image_name = image_name
        self.default_timeout = default_timeout
        self.volumes: dict[str, str] = volumes or {}

    def _build_command(self, args: list[str]) -> list[str]:
        """Build the full docker run command."""
        cmd = ["docker", "run", "--rm"]
        for host_path, container_path in self.volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
        cmd.append(self.image_name)
        cmd.extend(args)
        return cmd

    def run(
        self,
        args: list[str],
        timeout: int | None = None,
    ) -> ToolOutput:
        """Run the Docker container with the given arguments.

        Args:
            args: Command-line arguments to pass to the container entrypoint.
            timeout: Override the default timeout (seconds).

        Returns:
            ToolOutput with stdout, stderr, exit code, and duration.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        cmd = self._build_command(args)

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolOutput(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolOutput(
                stdout="",
                stderr=f"Timeout after {effective_timeout}s",
                exit_code=-1,
                duration_ms=duration_ms,
            )

    def pull(self) -> bool:
        """Pull the Docker image. Returns True on success."""
        result = subprocess.run(
            ["docker", "pull", self.image_name],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result.returncode == 0

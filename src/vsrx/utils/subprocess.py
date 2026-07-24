from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from vsrx.domain.errors import ExternalToolError


@dataclass(frozen=True, slots=True)
class CommandResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def run_command(
    args: Sequence[str | Path],
    *,
    timeout: float | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> CommandResult:
    command = tuple(str(item) for item in args)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExternalToolError(
            f"failed to execute command: {command[0]}", details={"args": command}
        ) from exc
    result = CommandResult(command, completed.returncode, completed.stdout, completed.stderr)
    if check and completed.returncode != 0:
        raise ExternalToolError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}",
            details={"stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:]},
        )
    return result

"""
This is the master script file for the project.

Wang Wenzhi
Time: 2026-08-27
"""

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Block 1. Define variables for common paths
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIR_CODES = PROJECT_ROOT / "codes"
DIR_OUTPUTS = PROJECT_ROOT / "outputs"
DIR_DATA = PROJECT_ROOT / "data"
DIR_RAWDATA = DIR_DATA / "a_raw_data"
DIR_TEMPDATA = DIR_DATA / "b_temp_data"
DIR_FINALDATA = DIR_DATA / "c_final_data"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Block 2. Define helper functions to deal with paths
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def relative_path(path: str | PathLike[str]) -> str:
    """
    Format a path relative to the project root when possible.

    Arguments:
        path: File or directory path to format.

    Returns:
        A POSIX-style path relative to ``PROJECT_ROOT`` when ``path`` is inside the project;
        otherwise, the POSIX-style form of ``path``.
    """
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_directory(path_directory: str | PathLike[str]) -> None:
    """
    Create a directory and any missing parent directories.

    Arguments:
        path_directory: Directory to create. An existing directory is left unchanged.
    """
    Path(path_directory).mkdir(parents=True, exist_ok=True)


def ensure_parent(path_file: str | PathLike[str]) -> None:
    """
    Create the parent directory of a file path if it does not exist.

    Arguments:
        path_file: File path whose parent directory should be created.
    """
    Path(path_file).parent.mkdir(parents=True, exist_ok=True)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Block 3. Define helper functions for console reporting
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


_CONSOLE = Console()
_STATUS_STYLES = {
    "info": ("INFO", "blue"),
    "success": ("SUCCESS", "green"),
    "warning": ("WARNING", "yellow"),
    "error": ("ERROR", "red"),
}


@dataclass(frozen=True)
class _RunRecord:
    """
    A class to store timing information returned by ``start_run``.
    """

    label: str
    started_at: datetime
    _started_monotonic: float


def _format_timestamp(timestamp: datetime) -> str:
    """
    Format a UTC timestamp for console output.

    Arguments:
        timestamp: UTC timestamp to format.

    Returns:
        The timestamp formatted as ``YYYY-MM-DD HH:MM:SS UTC``.
    """
    return timestamp.strftime(r"%Y-%m-%d %H:%M:%S UTC")


def _format_duration(elapsed_seconds: float) -> str:
    """
    Format an elapsed duration for console output.

    Arguments:
        elapsed_seconds: Elapsed duration in seconds.

    Returns:
        A compact human-readable duration rounded to the nearest tenth of a second below
        one minute and to the nearest second otherwise.
    """
    if elapsed_seconds < 59.95:
        return f"{elapsed_seconds:.1f} seconds"

    total_seconds = int(elapsed_seconds + 0.5)
    hours, remainder = divmod(total_seconds, 3_600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        hour_label = "hour" if hours == 1 else "hours"
        return f"{hours} {hour_label} {minutes:02d} minutes {seconds:02d} seconds"

    minute_label = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {minute_label} {seconds:02d} seconds"


def _validate_status_level(level: str) -> str:
    """
    Validate a console status level.

    Arguments:
        level: Status level to validate against the levels in ``_STATUS_STYLES``.

    Returns:
        The validated status level.
    """
    if not isinstance(level, str) or level not in _STATUS_STYLES:
        supported = ", ".join(_STATUS_STYLES)
        raise ValueError(f"Unknown message level {level!r}. Expected one of: {supported}.")
    return level


def _status_text(message: object, level: str) -> Text:
    """
    Build a consistently styled console status line.

    Arguments:
        message: Message to display without interpreting it as Rich markup.
        level: Status level determining the label and color.

    Returns:
        A Rich ``Text`` object containing the styled status line.
    """
    level = _validate_status_level(level)
    label, color = _STATUS_STYLES[level]
    line = Text()
    line.append(f"{label}: ", style=f"bold {color}")
    line.append(str(message))
    return line


def _normalize_finish_message(message: str | tuple[str, str]) -> tuple[str, str]:
    """
    Normalize a final message for ``finish_run``.

    Arguments:
        message: Message string or a two-item ``(level, message)`` tuple.

    Returns:
        A validated ``(level, message)`` tuple.
    """
    if isinstance(message, str):
        return "info", message

    if not isinstance(message, tuple) or len(message) != 2:
        raise TypeError(
            "Each finish message must be a string or a two-item (level, message) tuple."
        )

    level, text = message
    if not isinstance(text, str):
        raise TypeError("The message in a (level, message) tuple must be a string.")
    _validate_status_level(level)
    return level, text


def start_run(
    script: str | PathLike[str] = Path(__file__),
    *,
    label: str | None = None,
) -> _RunRecord:
    """
    Record and report the start of a script.

    Arguments:
        script: Path of the script being run, commonly ``__file__``.
        label: Optional display label. By default, the project-relative script path is used.

    Returns:
        A ``_RunRecord`` containing the display label and timing information.
    """
    started_at = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    script_label = str(label) if label is not None else relative_path(script)
    run = _RunRecord(
        label=script_label,
        started_at=started_at,
        _started_monotonic=started_monotonic,
    )

    heading = Text("START: ", style="bold blue")
    heading.append(run.label)
    heading.append(f" | {_format_timestamp(run.started_at)}")
    _CONSOLE.print(heading)
    return run


def report_status(message: object, *, level: str = "info") -> None:
    """
    Print an important script checkpoint using a consistent status style.

    Arguments:
        message: Message to display without interpreting it as Rich markup.
        level: Status level determining the label and color. Defaults to ``"info"``.
    """
    _CONSOLE.print(_status_text(message, level))


def report_progress(
    completed: int,
    *,
    description: str,
    total: int | None = None,
    unit: str = "items",
) -> None:
    """
    Print a stable, one-line progress snapshot.

    Arguments:
        completed: Number of items completed.
        description: Description of the work being measured.
        total: Optional total number of items. When omitted, no percentage is displayed.
        unit: Label for the counted items. Defaults to ``"items"``.
    """
    if total is None:
        progress = f"{completed:,} {unit}"
    else:
        percentage = 0.0 if total == 0 else completed / total * 100
        progress = f"{completed:,} / {total:,} {unit} ({percentage:.1f}%)"

    line = Text("PROGRESS: ", style="bold blue")
    line.append(str(description))
    line.append(f": {progress}")
    _CONSOLE.print(line)


def show_docstring(function: Callable[..., object]) -> None:
    """
    Display a function's fully qualified name and cleaned docstring.

    Arguments:
        function: Function whose name and docstring should be displayed.
    """
    module_name = getattr(function, "__module__", type(function).__module__)
    function_name = getattr(function, "__qualname__", type(function).__qualname__)
    qualified_name = f"{module_name}.{function_name}"
    docstring = inspect.getdoc(function) or "No docstring is available."
    _CONSOLE.print(
        Panel(
            Text(docstring),
            title=Text(qualified_name, style="bold blue"),
            title_align="left",
            border_style="blue",
            padding=(1, 2),
        )
    )


def finish_run(
    run: _RunRecord,
    *messages: str | tuple[str, str],
    outcome: str = "success",
) -> None:
    """
    Report a run's finish time, elapsed duration, and optional final messages.

    Arguments:
        run: Record returned by ``start_run``.
        *messages: Optional message strings or two-item ``(level, message)`` tuples.
        outcome: Run outcome, either ``"success"`` or ``"failed"``. Defaults to
            ``"success"``.
    """
    finished_at = datetime.now(timezone.utc)
    finished_monotonic = time.monotonic()

    if not isinstance(run, _RunRecord):
        raise TypeError("run must be the record returned by start_run().")
    if outcome not in ("success", "failed"):
        raise ValueError("outcome must be either 'success' or 'failed'.")

    normalized_messages = [_normalize_finish_message(message) for message in messages]
    elapsed_seconds = max(0.0, finished_monotonic - run._started_monotonic)
    color = "green" if outcome == "success" else "red"
    outcome_label = "FINISHED" if outcome == "success" else "FAILED"

    heading = Text(f"{outcome_label}: ", style=f"bold {color}")
    heading.append(run.label)
    heading.append(f" | {_format_timestamp(finished_at)} | {_format_duration(elapsed_seconds)}")
    _CONSOLE.print(heading)

    for level, message in normalized_messages:
        _CONSOLE.print(_status_text(message, level))

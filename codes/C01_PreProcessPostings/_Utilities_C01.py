"""
Task:
(1) Define common helper functions or global parameters to pre-process job postings.
(2) The key task in pre-processing is to identify the informative/meaningful job postings.
(3) Informative job postings will be sent to the next step to construct selectivity measures.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-21
"""

import os
import sqlite3
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from inspect import getdoc
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

CODES_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODES_ROOT))
import main

relative_path = main.relative_path


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Path-related global parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


FOLDER_DATA = main.DIR_TEMPDATA / "C01_PreProcessPostings"
main.ensure_directory(FOLDER_DATA)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Two functions for handling temporary datasets
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def prepare_output(outpath: Path, overwrite: bool = True) -> Path:
    """
    Return a path to store a temporary dataset before directly modifying the final output data.
    """
    main.ensure_parent(outpath)
    if outpath.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {outpath}")
    temp_output = outpath.with_suffix(outpath.suffix + ".tmp")
    if temp_output.exists():
        temp_output.unlink()
    return temp_output


def publish_output(temp_output: Path, final_output: Path) -> None:
    """
    Replace the final data with a temporary dataset.
    """
    os.replace(temp_output, final_output)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. A function to initialize a sqlite3 connection
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def create_sqlite_table(
    tabpath: Path, sql_command: str, *more_sql_commands: str, overwrite: bool = True
) -> sqlite3.Connection:
    """
    Connect to a SQLite database, execute one or more SQL commands, and return the connection.
    """
    if tabpath.exists() and not overwrite:
        tabpath.unlink()
    connection = sqlite3.connect(tabpath)
    try:
        for command in (sql_command, *more_sql_commands):
            connection.execute(command)
    except Exception:
        connection.close()
        raise
    return connection


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step z. Other utility functions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def record_time() -> datetime:
    return datetime.now(timezone.utc)


def report_status(text: str) -> None:
    print("=" * 80)
    print(f"{text}")
    print("=" * 80)
    print("\n")


def richprint_docstring(function: Callable[..., object]) -> None:
    """Print a function's docstring in a titled Rich panel."""
    function_name = f"{function.__module__}.{function.__name__}"
    function_docstring = getdoc(function) or "No docstring is available."
    Console().print(
        Panel(
            Text(function_docstring),
            title=Text(function_name, style="bold blue"),
            title_align="left",
            border_style="blue",
            padding=(1, 2),
        )
    )

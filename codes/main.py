"""
This is the master script file for the project.

Wang Wenzhi
Time: 2026-08-27
"""

from os import PathLike
from pathlib import Path

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define variables for common paths
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIR_CODES = PROJECT_ROOT / "codes"
DIR_OUTPUTS = PROJECT_ROOT / "outputs"
DIR_DATA = PROJECT_ROOT / "data"
DIR_RAWDATA = DIR_DATA / "a_raw_data"
DIR_TEMPDATA = DIR_DATA / "b_temp_data"
DIR_FINALDATA = DIR_DATA / "c_final_data"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Define helper functions to deal with paths
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

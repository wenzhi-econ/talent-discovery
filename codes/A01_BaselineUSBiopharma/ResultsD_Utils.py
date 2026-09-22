"""
Task:
    Provide shared plotting and validation helpers for the Stage D result scripts.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Descriptions of outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    matplotlib, pandas, and Pillow.

To be imported by:
    ResultsD01 through ResultsD11 in codes/A01_BaselineUSBiopharma.

Notes:
(1) This utility module defines helpers and paths without running a data pipeline on import.
(2) Every figure is written to a temporary PNG, verified, and then moved into place.


Wang Wenzhi
Time: 2026-09-22
"""

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image

from codes import main as project


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Block 1. Define shared paths and visual settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


STAGE_D_DIRECTORY = project.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
OUTPUT_DIRECTORY = project.DIR_OUTPUTS / "A01_BaselineUSBiopharma"
FIGURE_WIDTH_INCHES = 12.8
FIGURE_HEIGHT_INCHES = 7.2
FIGURE_DPI = 200
COLOR_MATCHED = "#003399"
COLOR_UNMATCHED = "#C9CED6"
COLOR_POSTINGS = "#CC5500"
COLOR_ACCENT = "#2A7F62"
COLOR_GRID = "#D9D9D9"
COLOR_TEXT_LIGHT = "#4D4D4D"
MISSING_TEXT = ("", "empty", "null", "none", "nan", "na", "n/a")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 1. Validate required DataFrame columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def require_columns(frame: pd.DataFrame, columns: Collection[str], *, name: str) -> None:
    """
    Require a nonempty DataFrame with unique labels and named columns.

    Parameters
    ----------
    frame : pd.DataFrame
        DataFrame to validate without modification.
    columns : Collection[str]
        Column labels required by the calling result script.
    name : str
        Human-readable dataset name used in error messages.

    Raises
    ------
    ValueError
        Raised when the frame is empty, has duplicate labels, or lacks required columns.
    """
    missing_columns = sorted(set(columns) - set(frame.columns))
    if frame.empty:
        raise ValueError(f"{name} must be nonempty.")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} contains duplicate column labels.")
    if missing_columns:
        raise ValueError(f"{name} is missing columns: {missing_columns}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 2. Apply the common scientific-figure axis style
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def style_axis(axis: Axes, *, grid_axis: str = "y") -> None:
    """
    Apply the common Stage D axis and grid style.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis modified in place.
    grid_axis : str
        Axis receiving grid lines, either ``"x"``, ``"y"``, or ``"both"``.
    """
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=COLOR_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=11)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 3. Save and validate a PNG figure
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def save_figure(figure: Figure, output_file: Path) -> None:
    """
    Save a white-background PNG after validating the temporary image.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Completed Matplotlib figure to save.
    output_file : pathlib.Path
        Final PNG destination. An existing file is replaced.
    """
    project.ensure_parent(output_file)
    temporary_output = output_file.with_suffix(f"{output_file.suffix}.incomplete")
    figure.savefig(
        temporary_output,
        format="png",
        dpi=FIGURE_DPI,
        facecolor="white",
        bbox_inches="tight",
    )
    with Image.open(temporary_output) as saved_image:
        saved_image.verify()
    temporary_output.replace(output_file)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 4. Construct an empirical cumulative distribution
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def empirical_cdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct sorted values and empirical cumulative probabilities.

    Parameters
    ----------
    values : pd.Series
        Positive, nonmissing numeric observations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted values and cumulative probabilities in the interval zero to one.

    Raises
    ------
    ValueError
        Raised when the input is empty, missing, nonnumeric, or nonpositive.
    """
    numeric_values = pd.to_numeric(values, errors="coerce")
    if numeric_values.empty:
        raise ValueError("The empirical distribution requires at least one observation.")
    if numeric_values.isna().any() or numeric_values.le(0).any():
        raise ValueError("The empirical distribution requires positive nonmissing values.")
    sorted_values = np.sort(numeric_values.to_numpy(dtype="float64"))
    cumulative_probability = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    return sorted_values, cumulative_probability

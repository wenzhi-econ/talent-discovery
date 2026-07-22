"""
Task:
    Summarize detailed roles among focal BioPharm employees active in December 2022.

Inputs:
    data/b_temp_data/A02_Industry_BioPharm/BioPharm_UserPositions_FocalSpells.parquet

Outputs:
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500Composition.csv
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500Summary.csv
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500Diagnostics.csv
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500DateDiagnostics.csv
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500Overall.png
    outputs/A02_Industry_BioPharm/C05_FocalEmployeeRoleK1500ByCategory.png

Notes:
(1) Focal employees have job category Scientist or Engineer and are active on 2022-12-01.
(2) The unit is an employee-firm pair, matching the employee definition used in C03.
(3) If multiple focal spells are active, the most recently started spell defines the role.
(4) The CSV reports every role; figures display only the most common roles.

Time: 2026-07-21
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import pyarrow.parquet as pq


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify paths and parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TOPICS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOPICS_ROOT))
import main  # noqa: E402

DATA_DIRECTORY = main.DIR_TEMPDATA / "A02_Industry_BioPharm"
OUTPUT_DIRECTORY = main.DIR_OUTPUTS / "A02_Industry_BioPharm"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

PATH_POSITIONS = DATA_DIRECTORY / "BioPharm_UserPositions_FocalSpells.parquet"
PATH_COMPOSITION = OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500Composition.csv"
PATH_SUMMARY = OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500Summary.csv"
PATH_DIAGNOSTICS = OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500Diagnostics.csv"
PATH_DATE_DIAGNOSTICS = (
    OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500DateDiagnostics.csv"
)
PATH_FIGURE_OVERALL = OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500Overall.png"
PATH_FIGURE_CATEGORY = OUTPUT_DIRECTORY / "C05_FocalEmployeeRoleK1500ByCategory.png"

SNAPSHOT_DATE = pd.Timestamp("2022-12-01")
FOCAL_JOB_CATEGORIES = ("Scientist", "Engineer")
ROLE_COLUMN = "role_k1500"
MISSING_ROLE = "Missing"
BATCH_SIZE = 500_000
FIGURE_DPI = 300
NUMBER_OVERALL_ROLES_TO_PLOT = 30
NUMBER_CATEGORY_ROLES_TO_PLOT = 20


def iterate_parquet_batches(path: Path, columns: list[str]):
    """Yield selected Parquet columns as pandas DataFrames of manageable size."""
    parquet_file = pq.ParquetFile(path)
    for record_batch in parquet_file.iter_batches(columns=columns, batch_size=BATCH_SIZE):
        yield record_batch.to_pandas()


def convert_date_with_diagnostics(
    values: pd.Series,
    variable: str,
    dataset: str,
) -> tuple[pd.Series, dict]:
    """Convert a date variable and describe unsuccessful conversions."""
    text_values = values.astype("string").str.strip()
    missing_or_blank = text_values.isna() | text_values.eq("")
    converted = pd.to_datetime(text_values.mask(missing_or_blank), errors="coerce")
    failed_conversion = ~missing_or_blank & converted.isna()
    nonmissing_string_count = int((~missing_or_blank).sum())

    diagnostic = {
        "dataset": dataset,
        "variable": variable,
        "row_count": int(len(values)),
        "missing_or_blank_count": int(missing_or_blank.sum()),
        "nonmissing_string_count": nonmissing_string_count,
        "failed_conversion_count": int(failed_conversion.sum()),
    }
    return converted, diagnostic


def summarize_role_composition(
    employee_firms: pd.DataFrame,
    sample_name: str,
) -> tuple[pd.DataFrame, dict]:
    """Calculate a complete role distribution and concentration statistics."""
    role_counts = (
        employee_firms.groupby(ROLE_COLUMN, as_index=False, observed=True)
        .size()
        .rename(columns={"size": "employee_firm_count"})
        .sort_values(["employee_firm_count", ROLE_COLUMN], ascending=[False, True])
        .reset_index(drop=True)
    )
    denominator = int(role_counts["employee_firm_count"].sum())
    role_counts["sample"] = sample_name
    role_counts["denominator_employee_firms"] = denominator
    role_counts["employee_firm_share"] = role_counts["employee_firm_count"] / denominator
    role_counts["rank"] = range(1, len(role_counts) + 1)
    role_counts["cumulative_share"] = role_counts["employee_firm_share"].cumsum()

    shares = role_counts["employee_firm_share"]
    concentration_summary = {
        "sample": sample_name,
        "employee_firm_count": denominator,
        "distinct_role_count": int(len(role_counts)),
        "top_5_share": float(shares.head(5).sum()),
        "top_10_share": float(shares.head(10).sum()),
        "top_20_share": float(shares.head(20).sum()),
        "herfindahl_index": float(shares.pow(2).sum()),
    }
    concentration_summary["effective_role_count"] = (
        1 / concentration_summary["herfindahl_index"]
    )
    return role_counts, concentration_summary


def save_overall_role_figure(composition: pd.DataFrame, path: Path) -> None:
    """Plot the most common detailed roles in the combined focal-employee sample."""
    plot_data = composition.loc[
        composition["sample"].eq("All focal employees")
    ].head(NUMBER_OVERALL_ROLES_TO_PLOT)
    plot_data = plot_data.sort_values("employee_firm_share", ascending=True)

    figure, axis = plt.subplots(figsize=(12, 10))
    axis.barh(plot_data[ROLE_COLUMN], plot_data["employee_firm_share"], color="#4472C4")
    axis.set_xlabel("Percentage of focal employee-firm observations")
    axis.set_ylabel("Detailed job role")
    axis.set_title(
        "Most common detailed roles among active BioPharm scientists and engineers\n"
        "December 2022",
        loc="left",
    )
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)

    figure.text(
        0.125,
        0.015,
        "Notes:\n"
        f"(1) The figure displays the {NUMBER_OVERALL_ROLES_TO_PLOT} most common "
        f"{ROLE_COLUMN} values; the CSV reports every role.\n"
        "(2) Each employee-firm pair is counted once using the most recently started "
        "active focal spell.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.14)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_category_role_figure(composition: pd.DataFrame, path: Path) -> None:
    """Plot detailed roles separately within Scientist and Engineer categories."""
    category_data = composition.loc[
        composition["sample"].isin(FOCAL_JOB_CATEGORIES)
    ].copy()
    category_data = category_data.loc[
        category_data["rank"].le(NUMBER_CATEGORY_ROLES_TO_PLOT)
    ]
    maximum_share = category_data["employee_firm_share"].max()
    common_axis_maximum = (int(maximum_share * 20) + 1) / 20

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10), sharex=True)
    for axis, category in zip(axes, FOCAL_JOB_CATEGORIES):
        plot_data = category_data.loc[category_data["sample"].eq(category)]
        plot_data = plot_data.sort_values("employee_firm_share", ascending=True)
        axis.barh(
            plot_data[ROLE_COLUMN],
            plot_data["employee_firm_share"],
            color="#4472C4",
        )
        axis.set_title(category)
        axis.set_xlabel(f"Percentage within {category.lower()} employee-firm observations")
        axis.set_xlim(0, common_axis_maximum)
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)

    axes[0].set_ylabel("Detailed job role")
    figure.suptitle(
        "Detailed-role composition within broad focal occupations, December 2022",
        x=0.125,
        ha="left",
    )
    figure.text(
        0.125,
        0.015,
        "Notes:\n"
        f"(1) Each panel displays its {NUMBER_CATEGORY_ROLES_TO_PLOT} most common "
        f"{ROLE_COLUMN} values; the CSV reports every role.\n"
        "(2) The panels use the same percentage scale for comparison. Each employee-firm "
        "pair is counted once.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    figure.subplots_adjust(bottom=0.14, top=0.90, wspace=0.55)
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Identify focal employment spells active in December 2022
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) Only variables required for the snapshot and role summary are loaded.
(2) The inclusive snapshot rule is identical to the rule used in C03.
(3) Rows passing the restrictions are small enough to combine after batch filtering.
"""
position_columns = [
    "rcid",
    "user_id",
    "startdate",
    "enddate",
    "job_category",
    ROLE_COLUMN,
]
active_position_batches = []
date_diagnostic_records = []

for positions_batch in iterate_parquet_batches(PATH_POSITIONS, position_columns):
    enddate_nonmissing = positions_batch["enddate"].notna() & (
        positions_batch["enddate"].astype("string").str.strip().ne("")
    )
    positions_batch["startdate"], start_diagnostic = convert_date_with_diagnostics(
        positions_batch["startdate"],
        variable="startdate",
        dataset="BioPharm_UserPositions_FocalSpells",
    )
    positions_batch["enddate"], end_diagnostic = convert_date_with_diagnostics(
        positions_batch["enddate"],
        variable="enddate",
        dataset="BioPharm_UserPositions_FocalSpells",
    )
    date_diagnostic_records.extend([start_diagnostic, end_diagnostic])

    focal_category = positions_batch["job_category"].isin(FOCAL_JOB_CATEGORIES)
    active_spell = (
        positions_batch["startdate"].le(SNAPSHOT_DATE)
        & (~enddate_nonmissing | positions_batch["enddate"].ge(SNAPSHOT_DATE))
    )
    active_position_batches.append(
        positions_batch.loc[focal_category & active_spell, position_columns].copy()
    )

active_positions = pd.concat(active_position_batches, ignore_index=True)
active_positions[ROLE_COLUMN] = active_positions[ROLE_COLUMN].astype("string").str.strip()
active_positions[ROLE_COLUMN] = active_positions[ROLE_COLUMN].mask(
    active_positions[ROLE_COLUMN].eq("")
)
active_positions[ROLE_COLUMN] = active_positions[ROLE_COLUMN].fillna(MISSING_ROLE)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Select one primary detailed role per employee-firm pair
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


"""
Notes:
(1) The employee-firm pair is the unit used when C03 counts focal employees by firm.
(2) Multiple active spells can otherwise cause the same employee to enter several roles.
(3) The latest start date selects the current role when active spells overlap.
(4) Alphabetical tie-breaking makes selection reproducible when start dates are equal.
"""
employee_firm_columns = ["user_id", "rcid"]
role_counts_per_employee_firm = active_positions.groupby(
    employee_firm_columns,
    observed=True,
)[ROLE_COLUMN].nunique(dropna=False)
category_counts_per_employee_firm = active_positions.groupby(
    employee_firm_columns,
    observed=True,
)["job_category"].nunique(dropna=False)

active_positions = active_positions.sort_values(
    employee_firm_columns + ["startdate", "job_category", ROLE_COLUMN],
    ascending=[True, True, False, True, True],
    na_position="last",
)
primary_roles = active_positions.drop_duplicates(employee_firm_columns, keep="first").copy()


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Save diagnostics for dates, coverage, and overlapping roles
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


date_diagnostics = pd.DataFrame(date_diagnostic_records)
date_diagnostics = date_diagnostics.groupby(
    ["dataset", "variable"],
    as_index=False,
    observed=True,
).agg(
    row_count=("row_count", "sum"),
    missing_or_blank_count=("missing_or_blank_count", "sum"),
    nonmissing_string_count=("nonmissing_string_count", "sum"),
    failed_conversion_count=("failed_conversion_count", "sum"),
)
date_diagnostics["failed_conversion_share"] = (
    date_diagnostics["failed_conversion_count"]
    / date_diagnostics["nonmissing_string_count"].replace(0, pd.NA)
).fillna(0.0)
date_diagnostics.to_csv(PATH_DATE_DIAGNOSTICS, index=False)

diagnostics = pd.DataFrame(
    [
        {"metric": "active_focal_position_rows", "value": len(active_positions)},
        {"metric": "active_focal_employee_firm_pairs", "value": len(primary_roles)},
        {
            "metric": "employee_firm_pairs_with_multiple_roles",
            "value": int(role_counts_per_employee_firm.gt(1).sum()),
        },
        {
            "metric": "employee_firm_pairs_with_multiple_categories",
            "value": int(category_counts_per_employee_firm.gt(1).sum()),
        },
        {
            "metric": "employee_firm_pairs_with_missing_primary_role",
            "value": int(primary_roles[ROLE_COLUMN].eq(MISSING_ROLE).sum()),
        },
    ]
)
diagnostics["share_of_employee_firm_pairs"] = (
    diagnostics["value"] / len(primary_roles)
)
diagnostics.loc[
    diagnostics["metric"].eq("active_focal_position_rows"),
    "share_of_employee_firm_pairs",
] = pd.NA
diagnostics.to_csv(PATH_DIAGNOSTICS, index=False)

print("Date conversion diagnostics:")
print(date_diagnostics.to_string(index=False))
print("\nRole-selection diagnostics:")
print(diagnostics.to_string(index=False))


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Calculate complete role distributions and concentration statistics
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


composition_frames = []
concentration_records = []

overall_composition, overall_concentration = summarize_role_composition(
    primary_roles,
    sample_name="All focal employees",
)
composition_frames.append(overall_composition)
concentration_records.append(overall_concentration)

for job_category in FOCAL_JOB_CATEGORIES:
    category_employee_firms = primary_roles.loc[
        primary_roles["job_category"].eq(job_category)
    ]
    category_composition, category_concentration = summarize_role_composition(
        category_employee_firms,
        sample_name=job_category,
    )
    composition_frames.append(category_composition)
    concentration_records.append(category_concentration)

composition = pd.concat(composition_frames, ignore_index=True)
composition = composition[
    [
        "sample",
        ROLE_COLUMN,
        "employee_firm_count",
        "denominator_employee_firms",
        "employee_firm_share",
        "rank",
        "cumulative_share",
    ]
]
composition.to_csv(PATH_COMPOSITION, index=False)

concentration_summary = pd.DataFrame(concentration_records)
concentration_summary.to_csv(PATH_SUMMARY, index=False)

print("\nRole concentration summary:")
print(concentration_summary.to_string(index=False))
print("\nMost common detailed roles among all focal employees:")
overall_top_roles = composition.loc[
    composition["sample"].eq("All focal employees")
].head(30)
print(overall_top_roles.to_string(index=False))


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Plot the overall and broad-category role compositions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


save_overall_role_figure(composition, PATH_FIGURE_OVERALL)
save_category_role_figure(composition, PATH_FIGURE_CATEGORY)

print(f"\nSaved table: {main.relative_path(PATH_COMPOSITION)}")
print(f"Saved table: {main.relative_path(PATH_SUMMARY)}")
print(f"Saved table: {main.relative_path(PATH_DIAGNOSTICS)}")
print(f"Saved table: {main.relative_path(PATH_DATE_DIAGNOSTICS)}")
print(f"Saved figure: {main.relative_path(PATH_FIGURE_OVERALL)}")
print(f"Saved figure: {main.relative_path(PATH_FIGURE_CATEGORY)}")

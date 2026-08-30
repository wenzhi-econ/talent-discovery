"""Summarize employment histories of Revelio users linked to USPTO inventors.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/Inventors_USPTO_UserPositions/*.parquet

Outputs:
(a) outputs/B01_ConstructAnalysisSample/D03_USPTOInventors/summary_statistics.csv
(b) outputs/B01_ConstructAnalysisSample/D03_USPTOInventors/data_quality.csv
(c) outputs/B01_ConstructAnalysisSample/D03_USPTOInventors/distributions/*.csv
(d) outputs/B01_ConstructAnalysisSample/D03_USPTOInventors/figures/*.pdf
(e) outputs/B01_ConstructAnalysisSample/D03_USPTOInventors/README.md

Description:
(1) All Parquet parts are scanned as one logical dataset; no physical append is required.
(2) Counts and distributions in this script are employment-spell weighted. Missing categories
    are retained in distribution denominators and reported explicitly.
(3) Complete distributions are saved as CSV files. High-cardinality occupation and industry
    figures show the leading nonmissing categories and aggregate the remainder into ``<Other>``.
(4) "Inventors" means distinct inventor-linked Revelio ``user_id`` values. PatentsView inventor
    IDs are not present in the position files and are therefore not counted here.
(5) Inventor-balanced or event-anchored summaries are intentionally deferred until the
    inventor-level transformation is finalized.

Run from the project root with the Talent environment:

    conda run -s -n Talent python -m codes.B01_ConstructAnalysisSample.D03_USPTOInventors

Wang Wenzhi, with the help of Codex
Time: 2026-08-28
"""

from __future__ import annotations

import argparse
import re
import tempfile
import textwrap
from pathlib import Path

import duckdb
import matplotlib
import pandas as pd

from codes import main

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter

MISSING_LABEL = "<Missing>"
OTHER_LABEL = "<Other>"
DEFAULT_TOP_N = 30
MISSING_TEXT_VALUES = ("", "empty", "null", "none", "nan", "na", "n/a")
EXPECTED_ROLE_COLUMNS = (
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1500",
)
EXPECTED_RICS_COLUMNS = ("rics_k50", "rics_k200", "rics_k400")
REQUIRED_COLUMNS = (
    "user_id",
    "position_id",
    "rcid",
    "country",
    "seniority",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
    "role_k1500",
    "rics_k400",
)


def hierarchy_number(column_name: str) -> int:
    """Return the numeric resolution in a Revelio hierarchy column name."""

    match = re.search(r"_k(\d+)$", column_name)
    return int(match.group(1)) if match else -1


def quote_identifier(value: str) -> str:
    """Quote a trusted schema identifier for DuckDB."""

    return '"' + value.replace('"', '""') + '"'


def quote_literal(value: str | Path) -> str:
    """Quote text or a path as a DuckDB SQL literal."""

    return "'" + str(value).replace("'", "''") + "'"


def source_columns(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
) -> tuple[str, ...]:
    """Read the Parquet schema without materializing the dataset."""

    description = connection.execute(
        f"DESCRIBE SELECT * FROM read_parquet({quote_literal(source_glob)})"
    ).fetchdf()
    return tuple(description["column_name"].astype(str))


def normalized_text(column: str) -> str:
    """Map null, blank, and delivered missing-value strings to the missing label."""

    identifier = quote_identifier(column)
    missing_values = ", ".join(quote_literal(value) for value in MISSING_TEXT_VALUES)
    return (
        f"CASE WHEN {identifier} IS NULL OR "
        f"LOWER(TRIM(CAST({identifier} AS VARCHAR))) IN ({missing_values}) "
        f"THEN {quote_literal(MISSING_LABEL)} "
        f"ELSE TRIM(CAST({identifier} AS VARCHAR)) END AS {identifier}"
    )


def prepare_staging_table(
    connection: duckdb.DuckDBPyConnection,
    source_glob: str,
    role_columns: tuple[str, ...],
    rics_columns: tuple[str, ...],
    row_limit: int | None,
) -> None:
    """Stage only columns used by the summaries, scanning all Parquet parts once."""

    text_columns = (
        "country",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        *role_columns,
        *rics_columns,
    )
    select_columns = [
        "TRY_CAST(user_id AS BIGINT) AS user_id",
        "TRY_CAST(position_id AS BIGINT) AS position_id",
        "TRY_CAST(rcid AS BIGINT) AS rcid",
        "TRY_CAST(seniority AS BIGINT) AS seniority",
        *(normalized_text(column) for column in text_columns),
    ]
    limit_clause = f"LIMIT {row_limit}" if row_limit is not None else ""
    connection.execute(
        "CREATE TABLE positions AS SELECT "
        + ", ".join(select_columns)
        + f" FROM read_parquet({quote_literal(source_glob)}) {limit_clause}"
    )


def get_basic_statistics(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct requested counts and compact source-data diagnostics."""

    missing = quote_literal(MISSING_LABEL)
    raw = connection.execute(
        f"""
        SELECT COUNT(*)::BIGINT AS employment_spell_rows,
               COUNT(DISTINCT user_id)::BIGINT AS inventor_linked_users,
               COUNT(DISTINCT position_id)::BIGINT AS distinct_position_ids,
               COUNT(*) FILTER (WHERE position_id IS NULL)::BIGINT
                   AS missing_position_ids,
               COUNT(DISTINCT rcid)::BIGINT AS companies,
               COUNT(DISTINCT NULLIF(country, {missing}))::BIGINT AS countries,
               COUNT(DISTINCT NULLIF(onet_code, {missing}))::BIGINT
                   AS onet_occupations,
               COUNT(DISTINCT NULLIF(role_k1500, {missing}))::BIGINT
                   AS role_k1500_occupations,
               COUNT(DISTINCT NULLIF(rics_k400, {missing}))::BIGINT
                   AS rics_k400_industries
        FROM positions
        """
    ).fetchdf().iloc[0]

    summary_rows = (
        (
            "Inventor-linked Revelio users",
            raw["inventor_linked_users"],
            "Distinct nonmissing user_id values; a Revelio user is the inventor unit here.",
        ),
        (
            "Employment-spell rows",
            raw["employment_spell_rows"],
            "Rows across all Parquet parts; each source row is treated as one spell.",
        ),
        (
            "Companies",
            raw["companies"],
            "Distinct nonmissing Revelio company IDs (rcid).",
        ),
        (
            "Countries",
            raw["countries"],
            "Distinct nonmissing country values.",
        ),
        (
            "ONET occupations",
            raw["onet_occupations"],
            "Distinct nonmissing onet_code values.",
        ),
        (
            "role_k1500 occupations",
            raw["role_k1500_occupations"],
            "Distinct nonmissing role_k1500 values.",
        ),
        (
            "rics_k400 industries",
            raw["rics_k400_industries"],
            "Distinct nonmissing rics_k400 values.",
        ),
    )
    summary = pd.DataFrame(summary_rows, columns=["metric", "value", "definition"])
    summary["value"] = summary["value"].astype("int64")

    nonmissing_position_rows = (
        int(raw["employment_spell_rows"]) - int(raw["missing_position_ids"])
    )
    duplicate_excess = nonmissing_position_rows - int(raw["distinct_position_ids"])
    quality_rows = (
        ("Employment-spell rows", raw["employment_spell_rows"]),
        ("Distinct position_id values", raw["distinct_position_ids"]),
        ("Rows with missing position_id", raw["missing_position_ids"]),
        ("Excess rows among nonmissing position_id values", duplicate_excess),
    )
    quality = pd.DataFrame(quality_rows, columns=["diagnostic", "value"])
    quality["value"] = quality["value"].astype("int64")
    return summary, quality


def simple_distribution(
    connection: duckdb.DuckDBPyConnection,
    variable: str,
) -> pd.DataFrame:
    """Count one categorical variable, including its explicit missing category."""

    identifier = quote_identifier(variable)
    distribution = connection.execute(
        f"""
        SELECT {quote_literal(variable)} AS variable,
               CAST({identifier} AS VARCHAR) AS value,
               CAST(NULL AS VARCHAR) AS title,
               CAST({identifier} AS VARCHAR) AS label,
               COUNT(*)::BIGINT AS count
        FROM positions
        GROUP BY {identifier}
        ORDER BY count DESC, value
        """
    ).fetchdf()
    return complete_distribution(distribution)


def titled_distribution(
    connection: duckdb.DuckDBPyConnection,
    variable: str,
    title_column: str,
) -> pd.DataFrame:
    """Count a code and attach its most frequent nonmissing title deterministically."""

    value = quote_identifier(variable)
    title = quote_identifier(title_column)
    missing = quote_literal(MISSING_LABEL)
    distribution = connection.execute(
        f"""
        WITH value_counts AS (
            SELECT {value} AS value, COUNT(*)::BIGINT AS count
            FROM positions
            GROUP BY {value}
        ), title_counts AS (
            SELECT {value} AS value, {title} AS title, COUNT(*)::BIGINT AS title_count
            FROM positions
            WHERE {value} <> {missing} AND {title} <> {missing}
            GROUP BY {value}, {title}
        ), preferred_titles AS (
            SELECT value, title
            FROM (
                SELECT value, title,
                       ROW_NUMBER() OVER (
                           PARTITION BY value ORDER BY title_count DESC, title
                       ) AS title_rank
                FROM title_counts
            )
            WHERE title_rank = 1
        )
        SELECT {quote_literal(variable)} AS variable,
               CAST(value_counts.value AS VARCHAR) AS value,
               preferred_titles.title,
               CASE
                   WHEN value_counts.value = {missing} THEN {missing}
                   WHEN preferred_titles.title IS NULL THEN value_counts.value
                   ELSE value_counts.value || ' - ' || preferred_titles.title
               END AS label,
               value_counts.count
        FROM value_counts
        LEFT JOIN preferred_titles USING (value)
        ORDER BY count DESC, value_counts.value
        """
    ).fetchdf()
    return complete_distribution(distribution)


def complete_distribution(distribution: pd.DataFrame) -> pd.DataFrame:
    """Add shares and deterministic within-variable ranks to aggregate counts."""

    distribution = distribution.copy()
    total = int(distribution["count"].sum())
    distribution["share"] = distribution["count"] / total if total else float("nan")
    distribution["rank"] = range(1, len(distribution) + 1)
    return distribution[["variable", "value", "title", "label", "count", "share", "rank"]]


def seniority_distribution(
    connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """Count seniority values and order numeric levels before missing values."""

    distribution = connection.execute(
        f"""
        SELECT 'seniority' AS variable,
               COALESCE(CAST(seniority AS VARCHAR), {quote_literal(MISSING_LABEL)}) AS value,
               CAST(NULL AS VARCHAR) AS title,
               COALESCE(CAST(seniority AS VARCHAR), {quote_literal(MISSING_LABEL)}) AS label,
               COUNT(*)::BIGINT AS count,
               seniority AS seniority_order
        FROM positions
        GROUP BY seniority
        ORDER BY seniority_order NULLS LAST
        """
    ).fetchdf()
    total = int(distribution["count"].sum())
    distribution["share"] = distribution["count"] / total if total else float("nan")
    distribution["rank"] = range(1, len(distribution) + 1)
    return distribution[
        ["variable", "value", "title", "label", "count", "share", "rank"]
    ]


def classification_coverage(distributions: pd.DataFrame) -> pd.DataFrame:
    """Summarize nonmissing coverage and distinct categories for every classification."""

    rows = []
    for variable, group in distributions.groupby("variable", sort=False):
        total = int(group["count"].sum())
        missing_count = int(group.loc[group["value"].eq(MISSING_LABEL), "count"].sum())
        rows.append(
            {
                "variable": variable,
                "employment_spell_rows": total,
                "nonmissing_rows": total - missing_count,
                "nonmissing_share": (total - missing_count) / total if total else float("nan"),
                "distinct_nonmissing_categories": int(group["value"].ne(MISSING_LABEL).sum()),
            }
        )
    return pd.DataFrame(rows)


def abbreviate_label(value: object, width: int = 72) -> str:
    """Truncate a plot label while leaving complete text in the CSV output."""

    text = str(value)
    return textwrap.shorten(text, width=width, placeholder="...")


def top_categories(distribution: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """Keep leading nonmissing categories and aggregate the remainder for plotting."""

    plot_data = distribution.copy()
    if top_n is None:
        return plot_data.sort_values(["count", "label"], ascending=[False, True])

    missing = plot_data.loc[plot_data["value"].eq(MISSING_LABEL)]
    nonmissing = plot_data.loc[plot_data["value"].ne(MISSING_LABEL)]
    shown = nonmissing.head(top_n).copy()
    remainder = nonmissing.iloc[top_n:]
    pieces = [shown]
    if not remainder.empty:
        pieces.append(
            pd.DataFrame(
                {
                    "variable": [plot_data["variable"].iloc[0]],
                    "value": [OTHER_LABEL],
                    "title": [pd.NA],
                    "label": [OTHER_LABEL],
                    "count": [int(remainder["count"].sum())],
                    "share": [float(remainder["share"].sum())],
                    "rank": [pd.NA],
                }
            )
        )
    if not missing.empty:
        pieces.append(missing)
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["count", "label"],
        ascending=[False, True],
    )


def style_axes(axis: plt.Axes) -> None:
    """Apply the shared minimal figure style."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def plot_horizontal_distribution(
    distribution: pd.DataFrame,
    title: str,
    output_path: Path | None = None,
    pdf: PdfPages | None = None,
    top_n: int | None = None,
) -> None:
    """Plot a descending horizontal share distribution to a PDF path or page."""

    plot_data = top_categories(distribution, top_n).reset_index(drop=True)
    plot_data["plot_label"] = plot_data["label"].map(abbreviate_label)
    category_count = len(plot_data)
    figure_height = max(4.5, 1.7 + 0.27 * category_count)
    figure, axis = plt.subplots(figsize=(11, figure_height))
    colors = [
        "#9CA3AF" if value in (MISSING_LABEL, OTHER_LABEL) else "#2563EB"
        for value in plot_data["value"]
    ]
    axis.barh(plot_data["plot_label"], plot_data["share"], color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Share of employment spells")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title(title, loc="left", fontsize=13, weight="bold", pad=12)
    style_axes(axis)
    if top_n is not None:
        figure.text(
            0.01,
            0.005,
            f"Top {top_n} nonmissing categories; remaining categories are grouped as "
            f"{OTHER_LABEL}. Missing values remain explicit. Denominator: all spells.",
            fontsize=8,
            color="#4B5563",
        )
    else:
        figure.text(
            0.01,
            0.005,
            "All observed categories are shown. Missing values remain explicit. "
            "Denominator: all spells.",
            fontsize=8,
            color="#4B5563",
        )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    if pdf is not None:
        pdf.savefig(figure, bbox_inches="tight")
    elif output_path is not None:
        figure.savefig(output_path, bbox_inches="tight")
    else:
        plt.close(figure)
        raise ValueError("Either output_path or pdf must be provided.")
    plt.close(figure)


def plot_seniority(distribution: pd.DataFrame, output_path: Path) -> None:
    """Plot the seniority distribution in numeric level order."""

    figure, axis = plt.subplots(figsize=(9, 5.5))
    colors = [
        "#9CA3AF" if value == MISSING_LABEL else "#2563EB"
        for value in distribution["value"]
    ]
    axis.bar(distribution["label"], distribution["share"], color=colors)
    axis.set_xlabel("Seniority level")
    axis.set_ylabel("Share of employment spells")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axis.set_title(
        "Seniority distribution across inventor employment spells",
        loc="left",
        fontsize=13,
        weight="bold",
        pad=12,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)
    figure.text(
        0.01,
        0.005,
        "Missing values remain explicit. Denominator: all spells.",
        fontsize=8,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.025, 1, 1))
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def write_classification_pdf(
    distributions: pd.DataFrame,
    variables: tuple[str, ...],
    kind: str,
    output_path: Path,
    top_n: int,
) -> None:
    """Write one horizontal bar-chart page per requested classification variable."""

    with PdfPages(output_path) as pdf:
        for variable in variables:
            plot_horizontal_distribution(
                distributions.loc[distributions["variable"].eq(variable)],
                title=(
                    f"{kind} distribution across inventor employment spells: {variable}"
                ),
                pdf=pdf,
                top_n=top_n,
            )


def markdown_table(summary: pd.DataFrame) -> str:
    """Render the requested integer summary as a small Markdown table."""

    lines = ["| Measure | Value | Definition |", "|---|---:|---|"]
    for row in summary.itertuples(index=False):
        definition = str(row.definition).replace("|", "\\|")
        lines.append(f"| {row.metric} | {int(row.value):,} | {definition} |")
    return "\n".join(lines)


def write_readme(
    output_dir: Path,
    summary: pd.DataFrame,
    role_columns: tuple[str, ...],
    rics_columns: tuple[str, ...],
    top_n: int,
    row_limit: int | None,
) -> None:
    """Write a compact guide to definitions and generated files."""

    development_note = (
        f"\n**Development run:** only the first {row_limit:,} source rows were used.\n"
        if row_limit is not None
        else ""
    )
    content = f"""# USPTO inventor employment-history summary
{development_note}
These statistics are weighted by employment spells. Each row in the multi-part Parquet
dataset contributes equally, so inventors with more recorded positions receive more weight.

## Requested counts

{markdown_table(summary)}

## Outputs

- `summary_statistics.csv`: requested counts and their definitions.
- `data_quality.csv`: position-ID diagnostics.
- `distributions/seniority.csv`: complete seniority distribution.
- `distributions/country.csv`: complete country distribution.
- `distributions/occupations.csv`: complete ONET and role distributions.
- `distributions/industries.csv`: complete NAICS and RICS distributions.
- `distributions/classification_coverage.csv`: nonmissing coverage by classification.
- `figures/seniority_distribution.pdf`: seniority distribution.
- `figures/country_distribution.pdf`: every country, ordered by descending spell share.
- `figures/occupation_distributions.pdf`: one page for each occupation variable.
- `figures/industry_distributions.pdf`: one page for each industry variable.

Occupation variables: {", ".join(("onet_code", *role_columns))}.

Industry variables: {", ".join(("naics_code", *rics_columns))}.

The occupation and industry PDFs display the top {top_n} nonmissing categories on each
page and group the remaining categories as `{OTHER_LABEL}`. Their CSV files retain every
category. All distribution denominators include missing classifications, which are displayed
as `{MISSING_LABEL}`.

"Inventor" in these outputs is an inventor-linked Revelio `user_id`. The position extracts do
not contain `pv_inventor_id`, so they cannot distinguish the small number of users linked to
multiple PatentsView inventor IDs.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def build_summaries(
    input_dir: Path,
    output_dir: Path,
    top_n: int = DEFAULT_TOP_N,
    row_limit: int | None = None,
) -> pd.DataFrame:
    """Build the complete spell-weighted summary package and return requested counts."""

    if top_n < 1:
        raise ValueError("top_n must be a positive integer.")
    if row_limit is not None and row_limit < 1:
        raise ValueError("row_limit must be a positive integer when provided.")

    parquet_files = tuple(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    distributions_dir = output_dir / "distributions"
    figures_dir = output_dir / "figures"
    distributions_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_glob = (input_dir / "*.parquet").as_posix()

    with tempfile.TemporaryDirectory(prefix="d03_uspto_", dir=output_dir) as work_dir:
        connection = duckdb.connect(str(Path(work_dir) / "work.duckdb"))
        try:
            available_columns = source_columns(connection, source_glob)
            missing_required = sorted(set(REQUIRED_COLUMNS) - set(available_columns))
            if missing_required:
                raise ValueError(f"Input is missing required fields: {missing_required}")

            role_columns = tuple(
                sorted(
                    (
                        column
                        for column in available_columns
                        if column.startswith("role_k")
                    ),
                    key=hierarchy_number,
                )
            )
            rics_columns = tuple(
                sorted(
                    (
                        column
                        for column in available_columns
                        if column.startswith("rics_k")
                    ),
                    key=hierarchy_number,
                )
            )
            missing_expected_roles = sorted(set(EXPECTED_ROLE_COLUMNS) - set(role_columns))
            missing_expected_rics = sorted(set(EXPECTED_RICS_COLUMNS) - set(rics_columns))
            if missing_expected_roles or missing_expected_rics:
                raise ValueError(
                    "Input hierarchy differs from the expected Revelio schema. "
                    f"Missing role columns: {missing_expected_roles}; "
                    f"missing RICS columns: {missing_expected_rics}."
                )

            prepare_staging_table(
                connection,
                source_glob,
                role_columns,
                rics_columns,
                row_limit,
            )
            summary, quality = get_basic_statistics(connection)
            seniority = seniority_distribution(connection)
            country = simple_distribution(connection, "country")
            occupation_tables = [
                titled_distribution(connection, "onet_code", "onet_title"),
                *(simple_distribution(connection, variable) for variable in role_columns),
            ]
            industry_tables = [
                titled_distribution(connection, "naics_code", "naics_description"),
                *(simple_distribution(connection, variable) for variable in rics_columns),
            ]
        finally:
            connection.close()

    occupations = pd.concat(occupation_tables, ignore_index=True)
    industries = pd.concat(industry_tables, ignore_index=True)
    coverage = classification_coverage(pd.concat([occupations, industries], ignore_index=True))

    summary.to_csv(output_dir / "summary_statistics.csv", index=False)
    quality.to_csv(output_dir / "data_quality.csv", index=False)
    seniority.to_csv(distributions_dir / "seniority.csv", index=False)
    country.to_csv(distributions_dir / "country.csv", index=False)
    occupations.to_csv(distributions_dir / "occupations.csv", index=False)
    industries.to_csv(distributions_dir / "industries.csv", index=False)
    coverage.to_csv(distributions_dir / "classification_coverage.csv", index=False)

    plot_seniority(seniority, figures_dir / "seniority_distribution.pdf")
    plot_horizontal_distribution(
        country,
        "Country distribution across inventor employment spells",
        output_path=figures_dir / "country_distribution.pdf",
        top_n=None,
    )
    write_classification_pdf(
        occupations,
        ("onet_code", *role_columns),
        "Occupation",
        figures_dir / "occupation_distributions.pdf",
        top_n,
    )
    write_classification_pdf(
        industries,
        ("naics_code", *rics_columns),
        "Industry",
        figures_dir / "industry_distributions.pdf",
        top_n,
    )
    write_readme(output_dir, summary, role_columns, rics_columns, top_n, row_limit)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line paths and development options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=(
            main.DIR_TEMPDATA
            / "B01_ConstructAnalysisSample"
            / "Inventors_USPTO_UserPositions"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            main.DIR_OUTPUTS
            / "B01_ConstructAnalysisSample"
            / "D03_USPTOInventors"
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of nonmissing categories shown per occupation/industry figure page.",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=None,
        help="Optional development-only row limit; omit for the complete sample.",
    )
    return parser.parse_args()


def run() -> None:
    """Run the summary workflow and report its requested counts."""

    arguments = parse_args()
    run_record = main.start_run(__file__)
    try:
        summary = build_summaries(
            arguments.input_dir,
            arguments.output_dir,
            top_n=arguments.top_n,
            row_limit=arguments.row_limit,
        )
    except Exception as error:
        main.finish_run(run_record, ("error", str(error)), outcome="failed")
        raise

    for row in summary.itertuples(index=False):
        main.report_status(f"{row.metric}: {int(row.value):,}")
    if arguments.row_limit is not None:
        main.report_status(
            f"Development run used only {arguments.row_limit:,} rows.",
            level="warning",
        )
    main.finish_run(
        run_record,
        f"Outputs saved to {main.relative_path(arguments.output_dir)}.",
    )


if __name__ == "__main__":
    run()

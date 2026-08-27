# ruff: noqa: PLR1711

"""Aggregate-table summary statistics for the universe of candidate focal new hires."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    layout_file="layouts/B02_FNH_SummaryStats.slides.json",
    auto_download=["html"],
)


@app.cell
def _():
    import math
    import re

    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import pycountry

    return alt, math, mo, pd, px, pycountry, re


@app.cell
def _(mo):
    mo.md(r"""
    # Universe of candidate focal new hires: Summary statistics

    Notes:
    - This notebook describes the broadest universe of candidate focal new hires.
    - The observation unit is at **user-company** level: one user can appear more than once when they are observed as a new hire at different companies.
    - All distributions below are simple averages over these user-company observations.

    The purpose of this notebook:
    - It is **not** as a description of the final research sample.
    - It starts with the broadest sample of new hires we can study, and all summary statistics I will present today should be used to guide the sample construction for our final analysis sample.
    - Charts in this report the occupation, industry, geography, and seniority distribution among this broadest universe.
    - **They will serve as a baseline when comparing with the inventor-matched sample of candidate new hires.**
    """)
    return


@app.cell
def _(alt, math, pd, pycountry, re):
    MISSING_LABEL = "<Missing>"
    MIN_ALL_COUNTRY_INDUSTRY_HIRES = 1_000
    US_LABEL = "United States"
    ALL_COUNTRIES_SCOPE = "__all__"
    US_SCOPE = "__us__"

    def hierarchy_number(column_name):
        match = re.search(r"_k(\d+)$", column_name)
        return int(match.group(1)) if match else -1

    def distribution_table(data, value_column, title_column=None):
        columns = [value_column] + ([title_column] if title_column else [])
        working = data.loc[:, columns].copy()
        for column in columns:
            working[column] = working[column].fillna(MISSING_LABEL).astype("string")
        summary = (
            working.groupby(columns, dropna=False, observed=True)
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["count", value_column], ascending=[False, True])
            .reset_index(drop=True)
        )
        summary["share"] = summary["count"] / len(data)
        summary["rank"] = range(1, len(summary) + 1)
        summary["value"] = summary[value_column].astype("string")
        if title_column:
            summary["display_label"] = (
                summary[value_column].astype("string")
                + " — "
                + summary[title_column].astype("string")
            )
        else:
            summary["display_label"] = summary["value"]
        return summary

    def joint_distribution_table(
        data,
        industry_column,
        occupation_column,
        title_columns,
    ):
        industry_title = title_columns.get(industry_column)
        occupation_title = title_columns.get(occupation_column)
        columns = [industry_column, occupation_column]
        for title_column in (industry_title, occupation_title):
            if title_column:
                columns.append(title_column)
        working = data.loc[:, columns].copy()
        for column in columns:
            working[column] = working[column].fillna(MISSING_LABEL).astype("string")
        working["industry_label"] = working[industry_column]
        if industry_title:
            working["industry_label"] += " — " + working[industry_title]
        working["occupation_label"] = working[occupation_column]
        if occupation_title:
            working["occupation_label"] += " — " + working[occupation_title]
        working["industry_value"] = working[industry_column]
        working["occupation_value"] = working[occupation_column]
        summary = (
            working.groupby(
                [
                    "industry_value",
                    "industry_label",
                    "occupation_value",
                    "occupation_label",
                ],
                dropna=False,
                observed=True,
            )
            .size()
            .rename("count")
            .reset_index()
            .sort_values(
                ["count", "industry_label", "occupation_label"],
                ascending=[False, True, True],
            )
            .reset_index(drop=True)
        )
        summary["share"] = summary["count"] / len(data)
        summary["rank"] = range(1, len(summary) + 1)
        summary["display_label"] = summary["industry_label"] + " × " + summary["occupation_label"]
        summary["value"] = summary["display_label"]
        return summary

    def category_selector_options(data, value_column, title_column=None):
        summary = distribution_table(data, value_column, title_column)
        summary = summary.loc[summary["value"] != MISSING_LABEL].drop_duplicates("value")
        return dict(zip(summary["display_label"], summary["value"], strict=True))

    def restrict_to_eligible_industries(
        summary,
        eligible_values,
        value_column="value",
    ):
        result = summary.loc[summary[value_column].astype("string").isin(eligible_values)].copy()
        result = result.reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    def available_country_options():
        return {
            "All countries": ALL_COUNTRIES_SCOPE,
            US_LABEL: US_SCOPE,
        }

    def resolve_country_scope(selections):
        selected = tuple(selections or ())
        if not selected or ALL_COUNTRIES_SCOPE in selected:
            return "all", "All countries"
        if US_SCOPE in selected:
            return "us", US_LABEL
        return "all", "All countries"

    def distribution_from_counts(
        counts,
        value_column,
        title_column=None,
        count_column="count",
    ):
        columns = ["value"] + (["title"] if title_column else [])
        summary = (
            counts.groupby(columns, dropna=False, observed=True)[count_column]
            .sum()
            .rename("count")
            .reset_index()
            .sort_values(["count", "value"], ascending=[False, True])
            .reset_index(drop=True)
        )
        summary["share"] = summary["count"] / summary["count"].sum()
        summary["rank"] = range(1, len(summary) + 1)
        summary["value"] = summary["value"].fillna(MISSING_LABEL).astype("string")
        summary[value_column] = summary["value"]
        if title_column:
            summary["title"] = summary["title"].fillna(MISSING_LABEL).astype("string")
            summary[title_column] = summary["title"]
            summary["display_label"] = summary["value"] + " — " + summary["title"]
        else:
            summary["display_label"] = summary["value"]
        return summary

    def crosswalk_from_counts(counts):
        pairs = counts.copy()
        pairs["left_value"] = pairs["left_value"].fillna(MISSING_LABEL).astype("string")
        pairs["right_value"] = pairs["right_value"].fillna(MISSING_LABEL).astype("string")
        pairs["left_title"] = pairs["left_title"].fillna("").astype("string")
        pairs["right_title"] = pairs["right_title"].fillna("").astype("string")
        pairs["left_label"] = pairs["left_value"]
        pairs.loc[pairs["left_title"] != "", "left_label"] += (
            " — " + pairs.loc[pairs["left_title"] != "", "left_title"]
        )
        pairs["right_label"] = pairs["right_value"]
        pairs.loc[pairs["right_title"] != "", "right_label"] += (
            " — " + pairs.loc[pairs["right_title"] != "", "right_title"]
        )
        pairs["share_within_left"] = pairs["count"] / pairs.groupby("left_label", observed=True)[
            "count"
        ].transform("sum")
        pairs = pairs.sort_values(
            ["left_label", "count", "right_label"], ascending=[True, False, True]
        ).reset_index(drop=True)
        pairs["rank"] = pairs.groupby("left_label", observed=True).cumcount() + 1
        return pairs

    def joint_from_counts(counts, denominator=None):
        summary = counts.copy()
        summary["industry_value"] = summary["industry_value"].astype("string")
        summary["occupation_value"] = summary["occupation_value"].astype("string")
        summary["industry_title"] = summary["industry_title"].fillna("").astype("string")
        summary["occupation_title"] = summary["occupation_title"].fillna("").astype("string")
        summary["industry_label"] = summary["industry_value"]
        summary.loc[summary["industry_title"] != "", "industry_label"] += (
            " — " + summary.loc[summary["industry_title"] != "", "industry_title"]
        )
        summary["occupation_label"] = summary["occupation_value"]
        summary.loc[summary["occupation_title"] != "", "occupation_label"] += (
            " — " + summary.loc[summary["occupation_title"] != "", "occupation_title"]
        )
        summary = summary.sort_values(
            ["count", "industry_label", "occupation_label"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
        total = summary["count"].sum() if denominator is None else denominator
        summary["share"] = summary["count"] / total
        summary["rank"] = range(1, len(summary) + 1)
        summary["display_label"] = summary["industry_label"] + " × " + summary["occupation_label"]
        summary["value"] = summary["display_label"]
        return summary

    def crosswalk_table(data, left_columns, right_column, right_title_column=None):
        columns = [*left_columns, right_column]
        if right_title_column:
            columns.append(right_title_column)
        working = data.loc[:, columns].copy()
        for column in columns:
            working[column] = working[column].fillna(MISSING_LABEL).astype("string")
        pairs = (
            working.groupby(columns, dropna=False, observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        pairs["left_label"] = pairs[left_columns[0]].astype("string")
        if len(left_columns) > 1:
            pairs["left_label"] = (
                pairs[left_columns[0]].astype("string")
                + " — "
                + pairs[left_columns[1]].astype("string")
            )
        pairs["right_label"] = pairs[right_column].astype("string")
        if right_title_column:
            pairs["right_label"] = (
                pairs[right_column].astype("string")
                + " — "
                + pairs[right_title_column].astype("string")
            )
        pairs["share_within_left"] = pairs["count"] / pairs.groupby("left_label", observed=True)[
            "count"
        ].transform("sum")
        pairs = pairs.sort_values(
            ["left_label", "count", "right_label"], ascending=[True, False, True]
        ).reset_index(drop=True)
        pairs["rank"] = pairs.groupby("left_label", observed=True).cumcount() + 1
        return pairs

    def make_share_chart(
        summary,
        title,
        top_n=None,
        baseline=None,
        x_title="Share within selected country scope",
    ):
        top = summary.head(top_n).copy() if top_n else summary.copy()
        if baseline is not None:
            base_shares = baseline[["display_label", "share"]].rename(
                columns={"share": "baseline_share"}
            )
            top = top.merge(base_shares, on="display_label", how="left")
            top["baseline_share"] = top["baseline_share"].fillna(0.0)
        else:
            top["baseline_share"] = math.nan
        if top.empty:
            return alt.Chart(pd.DataFrame({"display_label": []})).mark_bar()
        order = top["display_label"].tolist()
        maximum = float(top[["share", "baseline_share"]].max().max())
        domain = [0.0, maximum * 1.16 if maximum else 1.0]
        tooltip = [
            alt.Tooltip("display_label:N", title="Category"),
            alt.Tooltip("count:Q", title="Candidate new hires", format=","),
            alt.Tooltip("share:Q", title="Share", format=".2%"),
            alt.Tooltip("rank:Q", title="Rank", format="d"),
        ]
        if baseline is not None:
            tooltip.append(
                alt.Tooltip("baseline_share:Q", title="Universe baseline share", format=".2%")
            )
        base = alt.Chart(top).encode(
            y=alt.Y(
                "display_label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=440, labelPadding=6),
            ),
            tooltip=tooltip,
        )
        bars = base.mark_bar(color="#2563EB", opacity=0.85).encode(
            x=alt.X(
                "share:Q",
                title=x_title,
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=domain),
            )
        )
        labels = base.mark_text(align="left", baseline="middle", dx=4).encode(
            x=alt.X("share:Q"), text=alt.Text("share:Q", format=".1%")
        )
        layers = [bars, labels]
        if baseline is not None:
            layers.append(
                base.mark_point(shape="diamond", filled=True, color="#B91C1C", size=90).encode(
                    x=alt.X("baseline_share:Q")
                )
            )
        return (
            alt.layer(*layers)
            .properties(
                width="container",
                height=max(280, len(top) * 20),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )

    def make_crosswalk_chart(pairs, selected_left, title, baseline_pairs=None):
        selected = pairs.loc[pairs["left_label"] == selected_left].copy()
        if baseline_pairs is not None:
            base = baseline_pairs.loc[
                baseline_pairs["left_label"] == selected_left,
                ["right_label", "share_within_left"],
            ].rename(columns={"share_within_left": "baseline_share"})
            selected = selected.merge(base, on="right_label", how="left")
            selected["baseline_share"] = selected["baseline_share"].fillna(0.0)
        else:
            selected["baseline_share"] = math.nan
        if selected.empty:
            return alt.Chart(pd.DataFrame({"right_label": []})).mark_bar()
        order = selected["right_label"].tolist()
        maximum = float(selected[["share_within_left", "baseline_share"]].max().max())
        domain = [0.0, maximum * 1.16 if maximum else 1.0]
        tooltip = [
            alt.Tooltip("left_label:N", title="Selected category"),
            alt.Tooltip("right_label:N", title="Comparison category"),
            alt.Tooltip("count:Q", title="Candidate new hires", format=","),
            alt.Tooltip("share_within_left:Q", title="Share", format=".2%"),
            alt.Tooltip("rank:Q", title="Rank", format="d"),
        ]
        if baseline_pairs is not None:
            tooltip.append(
                alt.Tooltip("baseline_share:Q", title="Universe baseline share", format=".2%")
            )
        base = alt.Chart(selected).encode(
            y=alt.Y(
                "right_label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=440, labelPadding=6),
            ),
            tooltip=tooltip,
        )
        bars = base.mark_bar(color="#0F766E", opacity=0.85).encode(
            x=alt.X(
                "share_within_left:Q",
                title="Share within selected category",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=domain),
            )
        )
        labels = base.mark_text(align="left", baseline="middle", dx=4).encode(
            x=alt.X("share_within_left:Q"),
            text=alt.Text("share_within_left:Q", format=".1%"),
        )
        layers = [bars, labels]
        if baseline_pairs is not None:
            layers.append(
                base.mark_point(shape="diamond", filled=True, color="#B91C1C", size=90).encode(
                    x=alt.X("baseline_share:Q")
                )
            )
        return (
            alt.layer(*layers)
            .properties(
                width="container",
                height=max(300, len(selected) * 20),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )

    def country_iso3(country_name):
        aliases = {
            "Bolivia": "BOL",
            "Brunei": "BRN",
            "Cape Verde": "CPV",
            "Czech Republic": "CZE",
            "Democratic Republic of the Congo": "COD",
            "East Timor": "TLS",
            "Hong Kong": "HKG",
            "Iran": "IRN",
            "Ivory Coast": "CIV",
            "Kosovo": "XKX",
            "Laos": "LAO",
            "Macau": "MAC",
            "Moldova": "MDA",
            "North Korea": "PRK",
            "Palestine": "PSE",
            "Republic of the Congo": "COG",
            "Russia": "RUS",
            "South Korea": "KOR",
            "Syria": "SYR",
            "Taiwan": "TWN",
            "Tanzania": "TZA",
            "Turkey": "TUR",
            "United Kingdom": "GBR",
            "United States": "USA",
            "Venezuela": "VEN",
            "Vietnam": "VNM",
        }
        if country_name in aliases:
            return aliases[country_name]
        try:
            return pycountry.countries.lookup(str(country_name)).alpha_3
        except LookupError:
            return None

    def us_state_code(state_name):
        codes = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "District of Columbia": "DC",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "Washington, D.C.": "DC",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
        }
        return codes.get(state_name)

    return (
        MISSING_LABEL,
        MIN_ALL_COUNTRY_INDUSTRY_HIRES,
        available_country_options,
        category_selector_options,
        country_iso3,
        crosswalk_from_counts,
        crosswalk_table,
        distribution_from_counts,
        distribution_table,
        hierarchy_number,
        joint_from_counts,
        joint_distribution_table,
        make_crosswalk_chart,
        make_share_chart,
        resolve_country_scope,
        restrict_to_eligible_industries,
        us_state_code,
    )


@app.cell
def _(hierarchy_number, mo, pd):
    AGGREGATE_DIR = (
        mo.notebook_location().parents[1]
        / "data"
        / "b_temp_data"
        / "B01_ConstructAnalysisSample"
        / "C01_UniverseCandidateFNH"
    )
    REQUIRED_AGGREGATES = (
        "metadata.parquet",
        "schema_report.parquet",
        "role_onet_crosswalk.parquet",
        "country_counts.parquet",
        "us_state_counts.parquet",
        "seniority_counts.parquet",
        "time_counts.parquet",
    )
    missing_aggregates = [
        name for name in REQUIRED_AGGREGATES if not (AGGREGATE_DIR / name).exists()
    ]
    if missing_aggregates:
        raise FileNotFoundError(
            "Run C01_DataPrep_UniverseCandidateFNH_AggTables.py first. "
            f"Missing aggregate files: {missing_aggregates}"
        )
    metadata = pd.read_parquet(AGGREGATE_DIR / "metadata.parquet").iloc[0]

    def _columns(name):
        value = str(metadata[name])
        return tuple(item for item in value.split("|") if item)

    AVAILABLE_ROLE_COLUMNS = tuple(sorted(_columns("available_role_columns"), key=hierarchy_number))
    AVAILABLE_RICS_COLUMNS = tuple(sorted(_columns("available_rics_columns"), key=hierarchy_number))
    EXPECTED_ROLE_COLUMNS = (
        "role_k50",
        "role_k150",
        "role_k300",
        "role_k500",
        "role_k1000",
        "role_k1500",
    )
    EXPECTED_RICS_COLUMNS = ("rics_k50", "rics_k200", "rics_k400")

    def load_classification_counts(variable, scope_key="all"):
        table = pd.read_parquet(AGGREGATE_DIR / "classification" / f"{variable}.parquet")
        return table.loc[table["scope_key"] == scope_key].copy()

    def load_crosswalk_counts(left_variable, right_variable, scope_key="all"):
        table = pd.read_parquet(
            AGGREGATE_DIR / "crosswalk" / f"{left_variable}__{right_variable}.parquet"
        )
        return table.loc[table["scope_key"] == scope_key].copy()

    def load_joint_counts(industry_variable, occupation_variable, scope_key="all"):
        table = pd.read_parquet(
            AGGREGATE_DIR / "joint" / f"{industry_variable}__{occupation_variable}.parquet"
        )
        return table.loc[table["scope_key"] == scope_key].copy()

    return (
        AGGREGATE_DIR,
        AVAILABLE_RICS_COLUMNS,
        AVAILABLE_ROLE_COLUMNS,
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        load_classification_counts,
        load_crosswalk_counts,
        load_joint_counts,
        metadata,
    )


@app.cell
def _(
    AGGREGATE_DIR,
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    MISSING_LABEL,
    available_country_options,
    crosswalk_from_counts,
    distribution_from_counts,
    hierarchy_number,
    load_classification_counts,
    load_crosswalk_counts,
    metadata,
    pd,
):
    CLASSIFICATION_LABELS = {
        "onet_code": "ONET code and title",
        "naics_code": "NAICS code and description",
        **{
            column: f"Revelio role K{hierarchy_number(column):,}"
            for column in AVAILABLE_ROLE_COLUMNS
        },
        **{
            column: f"Revelio industry K{hierarchy_number(column):,}"
            for column in AVAILABLE_RICS_COLUMNS
        },
    }
    title_columns = {"onet_code": "onet_title", "naics_code": "naics_description"}
    country_selector_options = available_country_options()
    industry_selector_options = {
        CLASSIFICATION_LABELS[column]: column for column in ("naics_code", *AVAILABLE_RICS_COLUMNS)
    }
    occupation_selector_options = {
        CLASSIFICATION_LABELS[column]: column for column in ("onet_code", *AVAILABLE_ROLE_COLUMNS)
    }
    default_industry_column = (
        "rics_k400"
        if "rics_k400" in AVAILABLE_RICS_COLUMNS
        else AVAILABLE_RICS_COLUMNS[-1]
        if AVAILABLE_RICS_COLUMNS
        else "naics_code"
    )
    classification_columns = (
        "onet_code",
        *AVAILABLE_ROLE_COLUMNS,
        "naics_code",
        *AVAILABLE_RICS_COLUMNS,
    )
    distribution_tables = {
        column: distribution_from_counts(
            load_classification_counts(column),
            column,
            title_columns.get(column),
        )
        for column in classification_columns
    }
    eligible_industries_by_column = {
        column: frozenset(
            summary.loc[
                (summary["value"] != MISSING_LABEL)
                & (summary["count"] >= MIN_ALL_COUNTRY_INDUSTRY_HIRES),
                "value",
            ].astype(str)
        )
        for column, summary in distribution_tables.items()
        if column == "naics_code" or column in AVAILABLE_RICS_COLUMNS
    }

    def classification_options(variable):
        summary = distribution_tables[variable]
        summary = summary.loc[summary["value"] != MISSING_LABEL]
        return dict(zip(summary["display_label"], summary["value"], strict=True))

    role_crosswalk_rows = pd.read_parquet(AGGREGATE_DIR / "role_onet_crosswalk.parquet")
    if "role_k1500" in AVAILABLE_ROLE_COLUMNS:
        role_crosswalk_rows["ONET code and title"] = (
            role_crosswalk_rows["onet_code"].astype("string")
            + " — "
            + role_crosswalk_rows["onet_title"].astype("string")
        )
        role_onet_crosswalk = role_crosswalk_rows.loc[
            :, [*AVAILABLE_ROLE_COLUMNS, "ONET code and title"]
        ].reset_index(drop=True)
        role_counts = (
            role_crosswalk_rows.loc[role_crosswalk_rows["role_k1500"] != MISSING_LABEL]
            .groupby("role_k1500", observed=True)
            .size()
        )
        role_onet_cardinality_violations = int((role_counts > 1).sum())
    else:
        role_onet_crosswalk = pd.DataFrame(columns=[*AVAILABLE_ROLE_COLUMNS, "ONET code and title"])
        role_onet_cardinality_violations = 0

    finest_rics_column = AVAILABLE_RICS_COLUMNS[-1] if AVAILABLE_RICS_COLUMNS else None
    rics_naics_pairs = (
        crosswalk_from_counts(load_crosswalk_counts(finest_rics_column, "naics_code"))
        if finest_rics_column
        else pd.DataFrame()
    )
    naics_rics_pairs = (
        crosswalk_from_counts(load_crosswalk_counts("naics_code", finest_rics_column))
        if finest_rics_column
        else pd.DataFrame()
    )
    candidate_count = int(metadata["candidate_count"])
    classification_stats = pd.DataFrame(
        [
            {
                "Variable": column,
                "Classification": CLASSIFICATION_LABELS[column],
                "Nonmissing categories": int(
                    summary.loc[summary["value"] != MISSING_LABEL, "value"].nunique()
                ),
                "Missing rows": int(summary.loc[summary["value"] == MISSING_LABEL, "count"].sum()),
                "Missing share": float(
                    summary.loc[summary["value"] == MISSING_LABEL, "count"].sum() / candidate_count
                ),
            }
            for column, summary in distribution_tables.items()
        ]
    )
    schema_report = pd.read_parquet(AGGREGATE_DIR / "schema_report.parquet")
    title_diagnostics = pd.read_parquet(AGGREGATE_DIR / "title_diagnostics.parquet")
    onet_title_diagnostic = title_diagnostics.loc[
        title_diagnostics["classification"] == "ONET",
        ["code", "distinct_titles"],
    ].rename(columns={"code": "onet_code"})
    naics_title_diagnostic = title_diagnostics.loc[
        title_diagnostics["classification"] == "NAICS",
        ["code", "distinct_titles"],
    ].rename(
        columns={
            "code": "naics_code",
            "distinct_titles": "distinct_descriptions",
        }
    )
    basic_numbers = {
        "candidate_count": candidate_count,
        "distinct_users": int(metadata["distinct_users"]),
        "distinct_companies": int(metadata["distinct_companies"]),
        "distinct_countries": int(metadata["distinct_countries"]),
    }
    return (
        CLASSIFICATION_LABELS,
        basic_numbers,
        classification_options,
        classification_stats,
        country_selector_options,
        default_industry_column,
        eligible_industries_by_column,
        finest_rics_column,
        industry_selector_options,
        naics_rics_pairs,
        naics_title_diagnostic,
        occupation_selector_options,
        onet_title_diagnostic,
        rics_naics_pairs,
        role_onet_cardinality_violations,
        role_onet_crosswalk,
        schema_report,
        title_columns,
    )


@app.cell
def _(
    basic_numbers,
    classification_stats,
    mo,
    naics_title_diagnostic,
    onet_title_diagnostic,
    schema_report,
):
    onet_conflicts = int((onet_title_diagnostic["distinct_titles"] > 1).sum())
    naics_conflicts = int((naics_title_diagnostic["distinct_descriptions"] > 1).sum())
    _candidate_count = basic_numbers["candidate_count"]
    mo.vstack(
        [
            mo.md("## 1. Basic numbers"),
            mo.md(
                """
                The sample construction process is:

                1. Keep employment spells in the two-digit occupation groups **17: Architecture and Engineering occupations** and **19: Life, Physical, and Social Science occupations**.
                2. Retain spells starting from January 2021 through December 2023.
                3. Exclude spells with missing geography or job-title information.
                4. Exclude internship positions.
                5. Retain one employment spell within each user-company cell. This is the **universe sample of candidate focal new hires**.

                Notes:
                - This is the broadest sample of the new hires.
                - Further restrictions on industries and occupations will be guided by the summary statistics below.
                - The sample is at the user-company level, so one user can appear multiple times as a new hire at different companies.
                - All summary statistics are simple averages over these user-company observations.
                """
            ),
            mo.md(
                f"""
                | Number | Value |
                |---|---:|
                | Candidate focal new hires (user-company level) | {_candidate_count:,} |
                | Distinct users | {basic_numbers["distinct_users"]:,} |
                | Distinct companies | {basic_numbers["distinct_companies"]:,} |
                | Distinct countries | {basic_numbers["distinct_countries"]:,} |
                """
            ),
            mo.accordion(
                {
                    "Classification coverage": mo.ui.table(
                        classification_stats,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Requested-variable coverage": mo.ui.table(
                        schema_report, pagination=False, show_column_summaries=False
                    ),
                    "Label diagnostics": mo.md(
                        f"ONET codes with multiple titles: **{onet_conflicts:,}**; "
                        f"NAICS codes with multiple descriptions: **{naics_conflicts:,}**."
                    ),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The occupation distribution

    - How does Revelio assign an occupation to a user's employment spell?
        - Revelio first construct the user's occupation (with different levels of aggregation: `role_k50`, `role_k150`, `role_k300`, `role_k500`, `role_k1000`, `role_k1500`) based on their own occupation classification method (which is not public).
        - Then Revelio constructs a crosswalk mapping from their own occupation classification (`role_k1500`) to ONET occupations.
    - This implies:
        - Each occupation in `role_k1500` is mapped to 1 and only 1 ONET occupation; while one ONET occupation could be mapped to multiple roles in `role_k1500`.
        - For the series of Revelio's own occupation classifications, a role in a finer classification (e.g., in `role_k500`) is mapped to 1 and only 1 role in a broader classification (e.g., in `role_k50`); while one role in a broader classification is mapped to multiple roles in a finer classification.
    - **The main message from results in this section is that Revelio's occupation classifications are kind of messy.**
        - It is important to understand the limitations of the mapping from Revelio's own occupation classifications to ONET occupations.
        - It helps reveal whether a prominent category is meaningful or an unpleasant result from the mapping itself.
    """)
    return


@app.cell
def _(country_selector_options, mo):
    onet_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    return (onet_country_selector,)


@app.cell
def _(
    distribution_from_counts,
    load_classification_counts,
    make_share_chart,
    mo,
    onet_country_selector,
    resolve_country_scope,
):
    _scope_key, _scope_label = resolve_country_scope(onet_country_selector.value)
    onet_summary = distribution_from_counts(
        load_classification_counts("onet_code", _scope_key),
        "onet_code",
        "onet_title",
    )
    _chart = make_share_chart(
        onet_summary,
        f"ONET occupation distribution — {_scope_label}",
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 2.1. ONET occupation distribution

                - We have a very large share of "Astronomers" and "Historians". 
                - This is related to the specific procedures how Revelio assigns the ONET occupation codes. 
                - I will talk more on this point later. But in short:
                    - **Do not read the bars as evidence that a worker classified as a "historian" in ONET is literally a historian.**
                    - Unexpected occupation names should be traced through the crosswalks below and checked against reported job titles before they are used to define the focal occupation sample.
                    - From my own observations, unexpected ONET occupations often indicate measurement errors.
                """
            ),
            onet_country_selector,
            _chart,
            mo.accordion(
                {
                    "View all ONET occupation counts": mo.ui.table(
                        onet_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(AVAILABLE_ROLE_COLUMNS, country_selector_options, mo):
    role_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    role_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_ROLE_COLUMNS),
        value="role_k1500" if "role_k1500" in AVAILABLE_ROLE_COLUMNS else None,
        label="Revelio occupation variable",
    )
    role_top_n_selector = mo.ui.number(
        start=1, stop=1000, step=1, value=50, label="Number of top occupations"
    )
    return role_country_selector, role_top_n_selector, role_variable_selector


@app.cell
def _(
    distribution_from_counts,
    load_classification_counts,
    make_share_chart,
    mo,
    resolve_country_scope,
    role_country_selector,
    role_top_n_selector,
    role_variable_selector,
):
    role_variable = role_variable_selector.value
    _scope_key, _scope_label = resolve_country_scope(role_country_selector.value)
    role_summary = distribution_from_counts(
        load_classification_counts(role_variable, _scope_key), role_variable
    )
    role_top_n = max(1, int(role_top_n_selector.value))
    _chart = make_share_chart(
        role_summary,
        f"Top {role_top_n} occupations in {role_variable} — {_scope_label}",
        role_top_n,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 2.2. Revelio's own occupation distribution

                - From `role_k50` to `role_k1500`, an occupation's name become more ambiguous.
                - It is often hard to judge an occupation's exact nature simply from its name.
                - This is a limitation inherited in Revelio's standardization of a user's occupation variable.
                """
            ),
            mo.hstack(
                [role_country_selector, role_variable_selector, role_top_n_selector],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View all categories": mo.ui.table(
                        role_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(AVAILABLE_ROLE_COLUMNS, country_selector_options, mo):
    onet_role_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    onet_role_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_ROLE_COLUMNS),
        value="role_k1500" if "role_k1500" in AVAILABLE_ROLE_COLUMNS else None,
        label="Revelio occupation variable",
    )
    return onet_role_country_selector, onet_role_variable_selector


@app.cell
def _(
    MISSING_LABEL,
    crosswalk_from_counts,
    load_crosswalk_counts,
    mo,
    onet_role_country_selector,
    onet_role_variable_selector,
    resolve_country_scope,
):
    onet_role_variable = onet_role_variable_selector.value
    _scope_key, onet_role_scope_label = resolve_country_scope(onet_role_country_selector.value)
    onet_role_pairs = crosswalk_from_counts(
        load_crosswalk_counts("onet_code", onet_role_variable, _scope_key)
    )
    if onet_role_pairs.empty:
        onet_role_selector = None
    else:
        _choices = {
            f"{left} ({int(group['count'].sum()):,} hires)": left
            for left, group in onet_role_pairs.groupby("left_label", sort=False)
            if left != MISSING_LABEL
        }
        onet_role_selector = mo.ui.dropdown(
            options=_choices,
            value=next(iter(_choices)) if _choices else None,
            searchable=True,
            label="ONET occupation",
            full_width=True,
        )
    return (
        onet_role_pairs,
        onet_role_scope_label,
        onet_role_selector,
        onet_role_variable,
    )


@app.cell
def _(
    make_crosswalk_chart,
    mo,
    onet_role_country_selector,
    onet_role_pairs,
    onet_role_scope_label,
    onet_role_selector,
    onet_role_variable,
    onet_role_variable_selector,
):
    _control = (
        onet_role_selector
        if onet_role_selector is not None
        else mo.callout(mo.md(f"`{onet_role_variable}` is unavailable."), kind="warn")
    )
    _items = [
        mo.md("### 2.3. Crosswalk from ONET to Revelio's own occupation"),
        mo.md(R"""
            - This is the crosswalk from ONET to Revelio's own occupation classifications.
            - An expected composition often indicates a systematic crosswalk problem.
            """),
        mo.hstack(
            [onet_role_country_selector, onet_role_variable_selector, _control],
            justify="start",
            gap=2,
            widths="equal",
        ),
    ]
    if onet_role_selector is not None:
        _selected = onet_role_selector.value
        _chart = make_crosswalk_chart(
            onet_role_pairs,
            _selected,
            f"Revelio {onet_role_variable} composition of {_selected} — {onet_role_scope_label}",
        )
        _items.append(_chart)
    mo.vstack(_items, gap=1)
    return


@app.cell
def _(mo, role_onet_cardinality_violations, role_onet_crosswalk):
    if role_onet_crosswalk.empty:
        _content = mo.callout(mo.md("`role_k1500` is unavailable."), kind="warn")
    else:
        if role_onet_cardinality_violations:
            _diagnostic = mo.callout(
                mo.md(
                    f"**Warning:** {role_onet_cardinality_violations:,} nonmissing "
                    "`role_k1500` categories map to more than one combination of "
                    "Revelio hierarchy values and ONET occupation."
                ),
                kind="warn",
            )
        else:
            _diagnostic = mo.md(R"""
                - This is the crosswalk from Revelio's own occupation classifications to ONET.
                - Each occupation in `role_k1500` maps to 1 and only 1 ONET occupation; while 1 ONET occupation could be mapped to multiple roles in `role_k1500`.
                - The table is ordered by the `role_k1500` category's share of candidate focal new hires, from largest to smallest.
                - Main message: **Revelio's own occupation classifications are messy and often don't match to their literal meaning.**
                """)
        _content = mo.vstack(
            [
                _diagnostic,
                mo.ui.table(
                    role_onet_crosswalk,
                    pagination=True,
                    page_size=20,
                    show_column_summaries=False,
                ),
            ],
            gap=1,
        )
    mo.vstack(
        [
            mo.md("### 2.4. Crosswalk from Revelio's own occupation to ONET"),
            _content,
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    My suggestions on occupation classifications:

    - We shouldn't use a single Revelio's occupation variable for sample construction.
        - The ONET occupations are very unstable (recall the Astronomers and Historians).
        - We shouldn't either start with Revelio's own occupation classifications, because these variables often have ambiguous names (we cannot tell the exact nature of an occupation simply from its name itself).
    - The best workflow is:
        - We start from a selected industry (or a set of industries) because Revelio's industry classifications seem to be more consistent across NAICS and their own industry classification system.
        - **Next, using the results in Section 4 of this report, we select occupations that have relatively large share within the selected industry.**
        - Optionally, we can do further restrictions based on users' reported job titles for robustness checks.
            - For example, exclude those new hires whose self-reported job titles are clearly not what we want.
            - For example, we keep only the job titles within the ONET codes that frequently show up in the sample.
    - In summary, these are limitations inherited in Revelio's data, and we need to carefully deal with measurement errors of users' occupations.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The industry distribution

    - There are 2 main sets of variables indicating a user's industry.
        - `naics`: The NAICS industry classifications at 6 digits (used mainly in North America: US, Canada).
        - `rics_k50`, `rics_k200`, `rics_k400`: Revelio's own industry classifications (at different levels of aggregation).
    - Revelio doesn't document how it constructs these industry variables.
    - Unlike occupation classifications (where we know ONET codes are derived from Revelio's own occupation classifications), **there is no clear map between NAICS industries and Revelio's own industries**.
        - For example, "Pharmaceutical Preparation Manufacturing" (in NAICS) maps to multiple industries in `rics_k400`: Pharmaceuticals (54.5%); Pharmaceutical Manufacturing (32.7%); Biotechnology and Life Sciences (3.7%); Life Sciences and Diagnostics (2.5%) Biopharmaceuticals and Healthcare Services (1.8%).
        - In the other way around, "Pharmaceuticals" (in `rics_k400`) maps to multiple NAICS industries: Pharmaceutical Preparation Manufacturing (70.5%); Biological Product (except Diagnostics) Manufacturing (13.7%); Research and Development in Biotechnology (except Nanobiotechnology) (6.1%); Surgical and Medical Instrument Manufacturing (5.3%).
    - The good thing is that the Revelio's own industry classifications are clear and intuitive enough, so I suggest we start from them.
    """)
    return


@app.cell
def _(country_selector_options, mo):
    naics_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    naics_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of eligible NAICS industries",
    )
    return naics_country_selector, naics_top_n_selector


@app.cell
def _(
    distribution_from_counts,
    eligible_industries_by_column,
    load_classification_counts,
    make_share_chart,
    mo,
    naics_country_selector,
    naics_top_n_selector,
    resolve_country_scope,
    restrict_to_eligible_industries,
):
    _scope_key, _scope_label = resolve_country_scope(naics_country_selector.value)
    _summary = distribution_from_counts(
        load_classification_counts("naics_code", _scope_key),
        "naics_code",
        "naics_description",
    )
    _summary = restrict_to_eligible_industries(
        _summary, eligible_industries_by_column["naics_code"]
    )
    _chart = make_share_chart(
        _summary,
        f"Top NAICS industries — {_scope_label}",
        int(naics_top_n_selector.value),
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 3.1. NAICS distribution

                Industries must have at least 1,000 candidate new hires in the all-country
                sample. Top-N truncates the resulting count ranking and therefore reveals a
                stable prefix.
                """
            ),
            mo.hstack(
                [naics_country_selector, naics_top_n_selector],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View all eligible NAICS categories": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(AVAILABLE_RICS_COLUMNS, country_selector_options, mo):
    rics_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    rics_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_RICS_COLUMNS),
        value="rics_k400" if "rics_k400" in AVAILABLE_RICS_COLUMNS else None,
        label="Revelio industry variable",
    )
    rics_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of eligible Revelio industries",
    )
    return rics_country_selector, rics_top_n_selector, rics_variable_selector


@app.cell
def _(
    distribution_from_counts,
    eligible_industries_by_column,
    load_classification_counts,
    make_share_chart,
    mo,
    resolve_country_scope,
    restrict_to_eligible_industries,
    rics_country_selector,
    rics_top_n_selector,
    rics_variable_selector,
):
    _variable = rics_variable_selector.value
    _top_n = int(rics_top_n_selector.value)
    _scope_key, _scope_label = resolve_country_scope(rics_country_selector.value)
    _summary = distribution_from_counts(
        load_classification_counts(_variable, _scope_key), _variable
    )
    _summary = restrict_to_eligible_industries(_summary, eligible_industries_by_column[_variable])
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industries in {_variable} — {_scope_label}",
        _top_n,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 3.2. Revelio's own industry distribution

                - The `rics_k400` view is a practical starting point for defining industries.
                - NAICS can be used for robustness checks to see whether conclusions depend on the proprietary Revelio classification.
                - Industry eligibility requires at least 1,000 candidate new hires in the
                  all-country sample; Top-N only truncates the stable eligible ranking.
                """
            ),
            mo.hstack(
                [rics_country_selector, rics_variable_selector, rics_top_n_selector],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View all eligible Revelio industry categories": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(MISSING_LABEL, country_selector_options, mo, naics_rics_pairs):
    naics_rics_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    if naics_rics_pairs.empty:
        naics_rics_selector = None
    else:
        _choices = {
            f"{left} ({int(group['count'].sum()):,} hires)": left
            for left, group in naics_rics_pairs.groupby("left_label", sort=False)
            if left != MISSING_LABEL
        }
        naics_rics_selector = mo.ui.dropdown(
            options=_choices,
            value=next(iter(_choices)) if _choices else None,
            searchable=True,
            label="NAICS code and description",
            full_width=True,
        )
    return naics_rics_country_selector, naics_rics_selector


@app.cell
def _(
    crosswalk_from_counts,
    finest_rics_column,
    load_crosswalk_counts,
    make_crosswalk_chart,
    mo,
    naics_rics_country_selector,
    naics_rics_selector,
    resolve_country_scope,
):
    _control = (
        naics_rics_selector
        if naics_rics_selector is not None
        else mo.callout(mo.md("No RICS field is available."), kind="warn")
    )
    _items = [
        mo.md("### 3.3. Crosswalk from NAICS to Revelio's own industry"),
        mo.md(R"""
            - Select a NAICS category to see its finest Revelio industry composition.
            - One-to-many mappings show that a narrow NAICS label translates into multiple Revelio's industries.
            - The displayed conditional shares can quantify that measurement ambiguity.
            """),
        mo.hstack(
            [naics_rics_country_selector, _control],
            justify="start",
            gap=2,
            widths="equal",
        ),
    ]
    if naics_rics_selector is not None:
        _selected = naics_rics_selector.value
        _scope_key, _scope_label = resolve_country_scope(naics_rics_country_selector.value)
        _pairs = crosswalk_from_counts(
            load_crosswalk_counts("naics_code", finest_rics_column, _scope_key)
        )
        _chart = make_crosswalk_chart(
            _pairs,
            _selected,
            f"{finest_rics_column} composition of {_selected} — {_scope_label}",
        )
        _selected_pairs = _pairs.loc[_pairs["left_label"] == _selected].copy()
        _items.extend(
            [
                _chart,
                mo.accordion(
                    {
                        "View selected NAICS-to-Revelio crosswalk": mo.ui.table(
                            _selected_pairs,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ]
        )
    mo.vstack(_items, gap=1)
    return


@app.cell
def _(
    MISSING_LABEL,
    country_selector_options,
    finest_rics_column,
    mo,
    rics_naics_pairs,
):
    rics_naics_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    if rics_naics_pairs.empty:
        rics_naics_selector = None
    else:
        _choices = {
            f"{left} ({int(group['count'].sum()):,} hires)": left
            for left, group in rics_naics_pairs.groupby("left_label", sort=False)
            if left != MISSING_LABEL
        }
        rics_naics_selector = mo.ui.dropdown(
            options=_choices,
            value=next(iter(_choices)) if _choices else None,
            searchable=True,
            label=finest_rics_column,
            full_width=True,
        )
    return rics_naics_country_selector, rics_naics_selector


@app.cell
def _(
    crosswalk_from_counts,
    finest_rics_column,
    load_crosswalk_counts,
    make_crosswalk_chart,
    mo,
    resolve_country_scope,
    rics_naics_country_selector,
    rics_naics_selector,
):
    _control = (
        rics_naics_selector
        if rics_naics_selector is not None
        else mo.callout(mo.md("No RICS field is available."), kind="warn")
    )
    _items = [
        mo.md("### 3.4. Crosswalk from Revelio's own industry to NAICS"),
        mo.md(
            rf"""
            - Select a `{finest_rics_column}` category to see its NAICS composition.
            - A dispersed NAICS composition suggests that we should do NAICS-based restrictions as robustness checks.
            """
        ),
        mo.hstack(
            [rics_naics_country_selector, _control],
            justify="start",
            gap=2,
            widths="equal",
        ),
    ]
    if rics_naics_selector is not None:
        _selected = rics_naics_selector.value
        _scope_key, _scope_label = resolve_country_scope(rics_naics_country_selector.value)
        _pairs = crosswalk_from_counts(
            load_crosswalk_counts(finest_rics_column, "naics_code", _scope_key)
        )
        _chart = make_crosswalk_chart(
            _pairs,
            _selected,
            f"NAICS composition of {_selected} ({finest_rics_column}) — {_scope_label}",
        )
        _selected_pairs = _pairs.loc[_pairs["left_label"] == _selected].copy()
        _items.extend(
            [
                _chart,
                mo.accordion(
                    {
                        "View selected Revelio-to-NAICS crosswalk": mo.ui.table(
                            _selected_pairs,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ]
        )
    mo.vstack(_items, gap=1)
    return


@app.cell
def _(
    CLASSIFICATION_LABELS,
    country_selector_options,
    default_industry_column,
    industry_selector_options,
    mo,
    occupation_selector_options,
):
    industry_occupation_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    industry_occupation_industry_selector = mo.ui.dropdown(
        options=industry_selector_options,
        value=CLASSIFICATION_LABELS[default_industry_column],
        label="Industry variable",
        full_width=True,
    )
    industry_occupation_occupation_selector = mo.ui.dropdown(
        options=occupation_selector_options,
        value=CLASSIFICATION_LABELS["onet_code"],
        label="Occupation variable",
        full_width=True,
    )
    industry_occupation_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of top eligible industry-occupation combinations",
    )
    return (
        industry_occupation_country_selector,
        industry_occupation_industry_selector,
        industry_occupation_occupation_selector,
        industry_occupation_top_n_selector,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 4. Industry-occupation distribution

    - In this section, I document the joint and marginal distribution of industry-occupation combinations among the universe sample of candidate focal new hires.
    - Based on my current understanding of the data, an intuitive way to select the final analysis sample is:
        - We start with an industry that is patents-intensive (e.g., BioPharma based on Revelio's own industry classifications `rics_k400`) -- this part of evidence will show in the next part about the match rates.
        - Next, using summary statistics from this section, we select ONET occupations that constitute most of the scientists/engineers in the selected industry.
    - In what follows:
        - I will first document the joint distribution of industry-occupation combinations.
        - Next, I will document the occupation distribution within a selected industry (or a set of industries).
        - Finally, I will document the industry distribution within a selected occupation (or a set of occupations).
    """)
    return


@app.cell
def _(
    industry_occupation_country_selector,
    industry_occupation_industry_selector,
    industry_occupation_occupation_selector,
    industry_occupation_top_n_selector,
    joint_from_counts,
    load_classification_counts,
    load_joint_counts,
    make_share_chart,
    mo,
    resolve_country_scope,
):
    _industry_column = industry_occupation_industry_selector.value
    _occupation_column = industry_occupation_occupation_selector.value
    _top_n = max(1, int(industry_occupation_top_n_selector.value))
    _scope_key, _scope_label = resolve_country_scope(industry_occupation_country_selector.value)
    _summary = joint_from_counts(
        load_joint_counts(_industry_column, _occupation_column, _scope_key),
        denominator=load_classification_counts("onet_code", _scope_key)["count"].sum(),
    )
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industry-occupation combinations — {_scope_label}",
        _top_n,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 4.1. Joint industry-occupation distribution

                - Each bar reports the share of the universe sample in an industry-occupation combination within the selected country scope.
                - Joint cells reveal concentrations that can be hidden in the separate occupation and industry distributions.
                - Industries must have at least 1,000 candidate new hires in the all-country
                  sample. The Top-N control changes the number of eligible combinations displayed,
                  not the denominator used to calculate shares.
                """
            ),
            mo.hstack(
                [
                    industry_occupation_country_selector,
                    industry_occupation_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            mo.hstack(
                [
                    industry_occupation_industry_selector,
                    industry_occupation_occupation_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View eligible industry-occupation combinations": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    CLASSIFICATION_LABELS,
    country_selector_options,
    default_industry_column,
    industry_selector_options,
    mo,
    occupation_selector_options,
):
    occupation_within_industry_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    occupation_within_industry_industry_variable_selector = mo.ui.dropdown(
        options=industry_selector_options,
        value=CLASSIFICATION_LABELS[default_industry_column],
        label="Industry variable",
        full_width=True,
    )
    occupation_within_industry_occupation_variable_selector = mo.ui.dropdown(
        options=occupation_selector_options,
        value=CLASSIFICATION_LABELS["onet_code"],
        label="Occupation variable",
        full_width=True,
    )
    occupation_within_industry_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of top occupations",
    )
    return (
        occupation_within_industry_country_selector,
        occupation_within_industry_industry_variable_selector,
        occupation_within_industry_occupation_variable_selector,
        occupation_within_industry_top_n_selector,
    )


@app.cell
def _(
    classification_options,
    eligible_industries_by_column,
    mo,
    occupation_within_industry_industry_variable_selector,
):
    _industry_column = occupation_within_industry_industry_variable_selector.value
    _industry_options = classification_options(_industry_column)
    _eligible_values = eligible_industries_by_column[_industry_column]
    _industry_options = {
        label: value for label, value in _industry_options.items() if value in _eligible_values
    }
    _default_industries = (
        "Biotechnology and Life Sciences",
        "Pharmaceutical Manufacturing",
        "Pharmaceuticals",
    )
    _default_labels = [
        industry for industry in _default_industries if industry in _industry_options
    ]
    occupation_within_industry_industry_selector = mo.ui.multiselect(
        options=_industry_options,
        value=_default_labels,
        label="Industries",
        full_width=True,
    )
    return (occupation_within_industry_industry_selector,)


@app.cell
def _(
    distribution_from_counts,
    load_joint_counts,
    make_share_chart,
    mo,
    occupation_within_industry_country_selector,
    occupation_within_industry_industry_selector,
    occupation_within_industry_industry_variable_selector,
    occupation_within_industry_occupation_variable_selector,
    occupation_within_industry_top_n_selector,
    resolve_country_scope,
    title_columns,
):
    _industry_column = occupation_within_industry_industry_variable_selector.value
    _occupation_column = occupation_within_industry_occupation_variable_selector.value
    _selected_industries = tuple(occupation_within_industry_industry_selector.value or ())
    _top_n = max(1, int(occupation_within_industry_top_n_selector.value))
    _scope_key, _scope_label = resolve_country_scope(
        occupation_within_industry_country_selector.value
    )
    _pairs = load_joint_counts(_industry_column, _occupation_column, _scope_key)
    _pairs = _pairs.loc[_pairs["industry_value"].isin(_selected_industries)].copy()
    _counts = (
        _pairs.groupby(
            ["occupation_value", "occupation_title"],
            dropna=False,
            observed=True,
        )["count"]
        .sum()
        .reset_index()
        .rename(columns={"occupation_value": "value", "occupation_title": "title"})
    )
    _summary = distribution_from_counts(
        _counts,
        _occupation_column,
        title_columns.get(_occupation_column),
    )
    _figure = (
        make_share_chart(
            _summary,
            f"Top {_top_n} occupations within the selected industry set — {_scope_label}",
            _top_n,
            x_title="Share within selected industry set",
        )
        if _selected_industries
        else mo.callout(mo.md("Select at least one industry."), kind="warn")
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 4.2. Conditional occupation distribution within an industry

                - The selected industries are pooled before calculating the occupation shares.
                - Each bar reports an occupation's share of universe-sample hires in the selected country and industry set.
                - The industry selector includes only categories with at least 1,000 candidate
                  new hires in the all-country sample.
                """
            ),
            mo.hstack(
                [
                    occupation_within_industry_country_selector,
                    occupation_within_industry_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            mo.hstack(
                [
                    occupation_within_industry_industry_variable_selector,
                    occupation_within_industry_occupation_variable_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            occupation_within_industry_industry_selector,
            _figure,
            mo.accordion(
                {
                    "View occupations within selected industries": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    CLASSIFICATION_LABELS,
    country_selector_options,
    default_industry_column,
    industry_selector_options,
    mo,
    occupation_selector_options,
):
    industry_within_occupation_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    industry_within_occupation_industry_variable_selector = mo.ui.dropdown(
        options=industry_selector_options,
        value=CLASSIFICATION_LABELS[default_industry_column],
        label="Industry variable",
        full_width=True,
    )
    industry_within_occupation_occupation_variable_selector = mo.ui.dropdown(
        options=occupation_selector_options,
        value=CLASSIFICATION_LABELS["onet_code"],
        label="Occupation variable",
        full_width=True,
    )
    industry_within_occupation_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of eligible industries",
    )
    return (
        industry_within_occupation_country_selector,
        industry_within_occupation_industry_variable_selector,
        industry_within_occupation_occupation_variable_selector,
        industry_within_occupation_top_n_selector,
    )


@app.cell
def _(
    classification_options,
    industry_within_occupation_occupation_variable_selector,
    mo,
):
    _occupation_column = industry_within_occupation_occupation_variable_selector.value
    _occupation_options = classification_options(_occupation_column)
    _default_occupation_titles = (
        "Microbiologists",
        "Chemical Engineers",
        "Bioengineers and Biomedical Engineers",
        "Biochemists and Biophysicists",
        "Chemists",
        "Animal Scientists",
    )
    _default_labels = [
        option_label
        for title in _default_occupation_titles
        for option_label in _occupation_options
        if option_label == title or option_label.endswith(f" — {title}")
    ]
    industry_within_occupation_occupation_selector = mo.ui.multiselect(
        options=_occupation_options,
        value=_default_labels,
        label="Occupations",
        full_width=True,
    )
    return (industry_within_occupation_occupation_selector,)


@app.cell
def _(
    distribution_from_counts,
    eligible_industries_by_column,
    industry_within_occupation_country_selector,
    industry_within_occupation_industry_variable_selector,
    industry_within_occupation_occupation_selector,
    industry_within_occupation_occupation_variable_selector,
    industry_within_occupation_top_n_selector,
    load_joint_counts,
    make_share_chart,
    mo,
    resolve_country_scope,
    restrict_to_eligible_industries,
    title_columns,
):
    _industry_column = industry_within_occupation_industry_variable_selector.value
    _occupation_column = industry_within_occupation_occupation_variable_selector.value
    _selected_occupations = tuple(industry_within_occupation_occupation_selector.value or ())
    _top_n = max(1, int(industry_within_occupation_top_n_selector.value))
    _scope_key, _scope_label = resolve_country_scope(
        industry_within_occupation_country_selector.value
    )
    _pairs = load_joint_counts(_industry_column, _occupation_column, _scope_key)
    _pairs = _pairs.loc[_pairs["occupation_value"].isin(_selected_occupations)].copy()
    _counts = (
        _pairs.groupby(
            ["industry_value", "industry_title"],
            dropna=False,
            observed=True,
        )["count"]
        .sum()
        .reset_index()
        .rename(columns={"industry_value": "value", "industry_title": "title"})
    )
    _summary = distribution_from_counts(
        _counts,
        _industry_column,
        title_columns.get(_industry_column),
    )
    _summary = restrict_to_eligible_industries(
        _summary, eligible_industries_by_column[_industry_column]
    )
    _figure = (
        make_share_chart(
            _summary,
            f"Top {_top_n} industries within the selected occupation set — {_scope_label}",
            _top_n,
            x_title="Share within selected occupation set",
        )
        if _selected_occupations
        else mo.callout(mo.md("Select at least one occupation."), kind="warn")
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 4.3. Conditional industry distribution within an occupation

                - The selected occupations are pooled before calculating the industry shares.
                - Each bar reports an industry's share of universe-sample hires in the selected country and occupation set.
                - Only industries with at least 1,000 candidate new hires in the all-country
                  sample are eligible; Top-N truncates their stable count ranking.
                """
            ),
            mo.hstack(
                [
                    industry_within_occupation_country_selector,
                    industry_within_occupation_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            mo.hstack(
                [
                    industry_within_occupation_industry_variable_selector,
                    industry_within_occupation_occupation_variable_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            industry_within_occupation_occupation_selector,
            _figure,
            mo.accordion(
                {
                    "View industries within selected occupations": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(AGGREGATE_DIR, country_iso3, math, pd, us_state_code):
    country_summary = pd.read_parquet(AGGREGATE_DIR / "country_counts.parquet")
    country_summary = country_summary.rename(columns={"country": "value"})
    country_summary["country"] = country_summary["value"]
    country_summary["display_label"] = country_summary["value"]
    country_summary["share"] = country_summary["count"] / country_summary["count"].sum()
    country_summary["rank"] = range(1, len(country_summary) + 1)
    country_summary["iso3"] = country_summary["country"].map(country_iso3)
    country_summary["log10_count"] = country_summary["count"].map(
        lambda count: math.log10(count) if count > 0 else 0
    )
    mapped_country_summary = country_summary.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = country_summary.loc[country_summary["iso3"].isna()].copy()
    us_state_summary = pd.read_parquet(AGGREGATE_DIR / "us_state_counts.parquet")
    us_state_summary["share_within_country"] = (
        us_state_summary["count"] / us_state_summary["count"].sum()
    )
    us_state_summary["state_code"] = us_state_summary["state"].map(us_state_code)
    state_map_data = us_state_summary.dropna(subset=["state_code"]).copy()
    unmatched_state_data = us_state_summary.loc[us_state_summary["state_code"].isna()].copy()
    state_map_coverage = (
        state_map_data["count"].sum() / us_state_summary["count"].sum()
        if not us_state_summary.empty
        else 0.0
    )
    return (
        mapped_country_summary,
        state_map_coverage,
        state_map_data,
        unmapped_country_summary,
        unmatched_state_data,
    )


@app.cell
def _(mapped_country_summary, mo, px, unmapped_country_summary):
    _figure = px.choropleth(
        mapped_country_summary,
        locations="iso3",
        color="log10_count",
        hover_name="country",
        hover_data={
            "iso3": False,
            "log10_count": False,
            "count": ":,",
            "share": ":.2%",
        },
        labels={
            "count": "Candidate new hires",
            "share": "Global share",
            "log10_count": "Log10 candidate new hires",
        },
        color_continuous_scale="Blues",
        projection="natural earth",
        title="Candidate focal new hires by country",
    ).update_geos(showframe=False, showcoastlines=True)
    mo.vstack(
        [
            mo.md(
                """
                ## 5. Other results

                ### 5.1. Geography distribution

                - The country map reports absolute candidate-hire concentration and global shares.
                """
            ),
            _figure,
            mo.accordion(
                {
                    "View mapped country data": mo.ui.table(
                        mapped_country_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View country counts not mapped to ISO-3": mo.ui.table(
                        unmapped_country_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo, px, state_map_coverage, state_map_data, unmatched_state_data):
    if state_map_data.empty:
        _figure = mo.callout(
            mo.md("No U.S. state labels matched the state-code mapping."), kind="warn"
        )
    else:
        _figure = px.choropleth(
            state_map_data,
            locations="state_code",
            locationmode="USA-states",
            scope="usa",
            color="share_within_country",
            hover_name="state",
            hover_data={
                "state_code": False,
                "count": ":,",
                "share_within_country": ":.2%",
            },
            labels={
                "count": "Candidate new hires",
                "share_within_country": "U.S. share",
            },
            color_continuous_scale="Blues",
            title="Candidate focal new hires by U.S. state",
        ).update_geos(scope="usa", visible=False)
    mo.vstack(
        [
            mo.md(
                R"""
                - State shares use all U.S. candidate focal new hires as the denominator.
                """
            ),
            _figure,
            mo.md(f"The state-code mapping covers **{state_map_coverage:.2%}** of U.S. hires."),
            mo.accordion(
                {
                    "View mapped U.S. state data": mo.ui.table(
                        state_map_data,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View unmatched U.S. state labels": mo.ui.table(
                        unmatched_state_data,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(country_selector_options, mo):
    seniority_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    return (seniority_country_selector,)


@app.cell
def _(
    AGGREGATE_DIR,
    alt,
    distribution_from_counts,
    mo,
    pd,
    resolve_country_scope,
    seniority_country_selector,
):
    _scope_key, _scope_label = resolve_country_scope(seniority_country_selector.value)
    _counts = pd.read_parquet(AGGREGATE_DIR / "seniority_counts.parquet")
    _counts = _counts.loc[_counts["scope_key"] == _scope_key].copy()
    seniority_summary = distribution_from_counts(_counts, "seniority")
    seniority_summary["seniority_order"] = pd.to_numeric(
        seniority_summary["value"], errors="coerce"
    )
    seniority_summary = seniority_summary.sort_values(
        ["seniority_order", "display_label"], na_position="last"
    )
    order = seniority_summary["display_label"].tolist()
    _figure = (
        alt.Chart(seniority_summary)
        .mark_bar(color="#7C3AED", opacity=0.85)
        .encode(
            x=alt.X(
                "display_label:O",
                sort=order,
                title="Seniority level",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "share:Q",
                title="Share of candidate new hires",
                axis=alt.Axis(format=".1%"),
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Seniority"),
                alt.Tooltip("count:Q", title="Candidate new hires", format=","),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
            ],
        )
        .properties(
            width="container",
            height=360,
            title=alt.TitleParams(text=f"Seniority distribution — {_scope_label}", anchor="start"),
        )
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.2. Seniority distribution

                - Shares use candidate focal new hires in the selected country scope as the
                  denominator.
                """
            ),
            seniority_country_selector,
            _figure,
            mo.accordion(
                {
                    "View all seniority counts": mo.ui.table(
                        seniority_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(country_selector_options, mo):
    time_series_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    return (time_series_country_selector,)


@app.cell
def _(
    AGGREGATE_DIR,
    alt,
    mo,
    pd,
    resolve_country_scope,
    time_series_country_selector,
):
    _scope_key, _scope_label = resolve_country_scope(time_series_country_selector.value)
    time_series = pd.read_parquet(AGGREGATE_DIR / "time_counts.parquet")
    time_series = time_series.loc[
        time_series["scope_key"] == _scope_key, ["start_month", "count"]
    ].copy()
    _figure = (
        alt.Chart(time_series)
        .mark_line(point=True, color="#0369A1")
        .encode(
            x=alt.X("start_month:T", title="Employment start month"),
            y=alt.Y("count:Q", title="Candidate focal new hires"),
            tooltip=[
                alt.Tooltip("start_month:T", title="Start month", format="%b %Y"),
                alt.Tooltip("count:Q", title="Candidate focal new hires", format=","),
            ],
        )
        .properties(
            width="container",
            height=360,
            title=alt.TitleParams(
                text=f"Candidate focal new hires over time — {_scope_label}",
                anchor="start",
            ),
        )
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.3. Time-series

                - Monthly counts are shown for the selected country scope.
                """
            ),
            time_series_country_selector,
            _figure,
            mo.accordion(
                {
                    "View all monthly counts": mo.ui.table(
                        time_series,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()

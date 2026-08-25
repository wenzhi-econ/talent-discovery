# ruff: noqa: B018, PLR1711

"""Summary statistics for the inventor-matched candidate focal new hires."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def _():
    import math
    import re

    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import pyarrow.dataset as ds
    import pycountry

    return alt, ds, math, mo, pd, px, pycountry, re


@app.cell
def _(mo):
    mo.md(r"""
    # Inventor-matched candidate focal new hires: summary statistics

    This notebook describes the candidate focal new hires whose users have at least one
    inventor ID in the Revelio-to-PatentsView crosswalk. The observation unit remains a
    **user-company observation**. Multiple inventor IDs for one user do not duplicate that
    observation. All matched-sample distributions are simple averages over matched
    user-company observations; red diamonds and the additional comparison series show the
    corresponding universe baseline.
    """)
    return


@app.cell
def _(alt, math, pd, pycountry, re):
    MISSING_LABEL = "<Missing>"

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
        pairs["share_within_left"] = pairs["count"] / pairs.groupby(
            "left_label", observed=True
        )["count"].transform("sum")
        pairs = pairs.sort_values(
            ["left_label", "count", "right_label"], ascending=[True, False, True]
        ).reset_index(drop=True)
        pairs["rank"] = pairs.groupby("left_label", observed=True).cumcount() + 1
        return pairs

    def make_share_chart(summary, title, top_n=None, baseline=None):
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
            alt.Tooltip("count:Q", title="Inventor-matched hires", format=","),
            alt.Tooltip("share:Q", title="Matched share", format=".2%"),
            alt.Tooltip("rank:Q", title="Matched rank", format="d"),
            alt.Tooltip(
                "baseline_share:Q", title="Universe baseline share", format=".2%"
            ),
        ]
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
                title="Share of inventor-matched hires",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=domain),
            )
        )
        labels = base.mark_text(align="left", baseline="middle", dx=4).encode(
            x=alt.X("share:Q"), text=alt.Text("share:Q", format=".1%")
        )
        diamonds = base.mark_point(
            shape="diamond", filled=True, color="#B91C1C", size=90
        ).encode(x=alt.X("baseline_share:Q"))
        return (
            alt.layer(bars, labels, diamonds)
            .properties(
                width="container",
                height=max(280, len(top) * 20),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )

    def make_crosswalk_chart(pairs, selected_left, title, baseline_pairs):
        selected = pairs.loc[pairs["left_label"] == selected_left].copy()
        baseline = baseline_pairs.loc[
            baseline_pairs["left_label"] == selected_left,
            ["right_label", "share_within_left"],
        ].rename(columns={"share_within_left": "baseline_share"})
        selected = selected.merge(baseline, on="right_label", how="left")
        selected["baseline_share"] = selected["baseline_share"].fillna(0.0)
        if selected.empty:
            return alt.Chart(pd.DataFrame({"right_label": []})).mark_bar()
        order = selected["right_label"].tolist()
        maximum = float(selected[["share_within_left", "baseline_share"]].max().max())
        domain = [0.0, maximum * 1.16 if maximum else 1.0]
        base = alt.Chart(selected).encode(
            y=alt.Y(
                "right_label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=440, labelPadding=6),
            ),
            tooltip=[
                alt.Tooltip("left_label:N", title="Selected category"),
                alt.Tooltip("right_label:N", title="Comparison category"),
                alt.Tooltip("count:Q", title="Inventor-matched hires", format=","),
                alt.Tooltip("share_within_left:Q", title="Matched share", format=".2%"),
                alt.Tooltip(
                    "baseline_share:Q", title="Universe baseline share", format=".2%"
                ),
                alt.Tooltip("rank:Q", title="Matched rank", format="d"),
            ],
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
        diamonds = base.mark_point(
            shape="diamond", filled=True, color="#B91C1C", size=90
        ).encode(x=alt.X("baseline_share:Q"))
        return (
            alt.layer(bars, labels, diamonds)
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
        country_iso3,
        crosswalk_table,
        distribution_table,
        hierarchy_number,
        make_crosswalk_chart,
        make_share_chart,
        us_state_code,
    )


@app.cell
def _(ds, hierarchy_number, mo, pd):
    INPUT_DIR = (
        mo.notebook_location().parents[1]
        / "data"
        / "b_temp_data"
        / "B01_ConstructAnalysisSample"
        / "FocalNewHires_AllIndustries"
    )
    CROSSWALK_PATH = (
        mo.notebook_location().parents[1]
        / "data"
        / "a_raw_data"
        / "A_Revelio"
        / "revelio_user_id_patentsview_id.csv"
    )
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
        "rcid",
        "country",
        "state",
        "seniority",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
    )
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(f"Inventor crosswalk does not exist: {CROSSWALK_PATH}")
    parquet_files = tuple(sorted(INPUT_DIR.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in: {INPUT_DIR}")
    dataset = ds.dataset(INPUT_DIR, format="parquet")
    available_columns = tuple(dataset.schema.names)
    missing_required = sorted(set(REQUIRED_COLUMNS) - set(available_columns))
    if missing_required:
        raise ValueError(f"Input is missing required fields: {missing_required}")
    if "start_month" in available_columns:
        date_column = "start_month"
    elif "startdate" in available_columns:
        date_column = "startdate"
    else:
        raise ValueError("Input must contain either `start_month` or `startdate`.")
    available_role_columns = tuple(
        sorted(
            (column for column in available_columns if column.startswith("role_k")),
            key=hierarchy_number,
        )
    )
    available_rics_columns = tuple(
        sorted(
            (column for column in available_columns if column.startswith("rics_k")),
            key=hierarchy_number,
        )
    )
    analysis_columns = tuple(
        dict.fromkeys(
            [
                *REQUIRED_COLUMNS,
                date_column,
                *available_role_columns,
                *available_rics_columns,
            ]
        )
    )
    universe_fnh = pd.read_parquet(
        INPUT_DIR,
        columns=list(analysis_columns),
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    if date_column == "start_month":
        universe_fnh["start_month"] = pd.to_datetime(
            universe_fnh[date_column], errors="coerce"
        )
    else:
        universe_fnh["start_month"] = (
            pd.to_datetime(universe_fnh[date_column], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    links = pd.read_csv(
        CROSSWALK_PATH,
        usecols=["user_id", "pv_inventor_id"],
        dtype={"user_id": "Int64", "pv_inventor_id": "string"},
    )
    valid_links = links.dropna(subset=["user_id", "pv_inventor_id"]).copy()
    valid_links["pv_inventor_id"] = valid_links["pv_inventor_id"].str.strip()
    valid_links = valid_links.loc[valid_links["pv_inventor_id"].ne("")]
    linked_users = valid_links["user_id"].drop_duplicates()
    if linked_users.empty:
        raise ValueError("Inventor crosswalk contains no rows with both IDs.")
    fnh = universe_fnh.loc[universe_fnh["user_id"].isin(linked_users)].copy()
    if fnh.empty:
        raise ValueError("No focal-hire observations match the inventor crosswalk.")
    user_link_counts = valid_links.groupby("user_id").size()
    link_diagnostics = pd.DataFrame(
        [
            {
                "Crosswalk rows": len(links),
                "Rows with both IDs": len(valid_links),
                "Unique linked users": int(linked_users.nunique()),
                "Unique inventor IDs": int(valid_links["pv_inventor_id"].nunique()),
                "Users with multiple crosswalk rows": int((user_link_counts > 1).sum()),
                "Maximum rows per user": int(user_link_counts.max()),
                "Missing user IDs": int(links["user_id"].isna().sum()),
                "Missing inventor IDs": int(links["pv_inventor_id"].isna().sum()),
                "Matched focal-hire observations": len(fnh),
                "Matched users represented": int(fnh["user_id"].nunique(dropna=True)),
            }
        ]
    )
    return (
        CROSSWALK_PATH,
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        INPUT_DIR,
        available_rics_columns,
        available_role_columns,
        fnh,
        link_diagnostics,
        parquet_files,
        universe_fnh,
    )


@app.cell
def _(
    MISSING_LABEL,
    available_rics_columns,
    available_role_columns,
    crosswalk_table,
    distribution_table,
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    fnh,
    hierarchy_number,
    pd,
    universe_fnh,
):
    CLASSIFICATION_LABELS = {
        "onet_code": "O*NET code and title",
        "naics_code": "NAICS code and description",
        **{
            column: f"Revelio role K{hierarchy_number(column):,}"
            for column in available_role_columns
        },
        **{
            column: f"Revelio industry K{hierarchy_number(column):,}"
            for column in available_rics_columns
        },
    }
    title_columns = {"onet_code": "onet_title", "naics_code": "naics_description"}
    classification_columns = (
        "onet_code",
        *available_role_columns,
        "naics_code",
        *available_rics_columns,
    )
    distribution_tables = {
        column: distribution_table(fnh, column, title_columns.get(column))
        for column in classification_columns
    }
    baseline_distribution_tables = {
        column: distribution_table(universe_fnh, column, title_columns.get(column))
        for column in classification_columns
    }
    onet_role_pairs = (
        crosswalk_table(fnh, ["onet_code", "onet_title"], "role_k1500")
        if "role_k1500" in fnh.columns
        else pd.DataFrame()
    )
    baseline_onet_role_pairs = (
        crosswalk_table(universe_fnh, ["onet_code", "onet_title"], "role_k1500")
        if "role_k1500" in universe_fnh.columns
        else pd.DataFrame()
    )
    role_onet_pairs = (
        crosswalk_table(fnh, ["role_k1500"], "onet_code", "onet_title")
        if "role_k1500" in fnh.columns
        else pd.DataFrame()
    )
    baseline_role_onet_pairs = (
        crosswalk_table(universe_fnh, ["role_k1500"], "onet_code", "onet_title")
        if "role_k1500" in universe_fnh.columns
        else pd.DataFrame()
    )
    finest_rics_column = available_rics_columns[-1] if available_rics_columns else None
    rics_naics_pairs = (
        crosswalk_table(fnh, [finest_rics_column], "naics_code", "naics_description")
        if finest_rics_column
        else pd.DataFrame()
    )
    baseline_rics_naics_pairs = (
        crosswalk_table(
            universe_fnh, [finest_rics_column], "naics_code", "naics_description"
        )
        if finest_rics_column
        else pd.DataFrame()
    )
    naics_rics_pairs = (
        crosswalk_table(fnh, ["naics_code", "naics_description"], finest_rics_column)
        if finest_rics_column
        else pd.DataFrame()
    )
    baseline_naics_rics_pairs = (
        crosswalk_table(
            universe_fnh, ["naics_code", "naics_description"], finest_rics_column
        )
        if finest_rics_column
        else pd.DataFrame()
    )
    classification_stats = pd.DataFrame(
        [
            {
                "Variable": column,
                "Classification": CLASSIFICATION_LABELS[column],
                "Matched nonmissing categories": int(
                    summary.loc[summary["value"] != MISSING_LABEL, "value"].nunique()
                ),
                "Matched missing rows": int(
                    summary.loc[summary["value"] == MISSING_LABEL, "count"].sum()
                ),
                "Matched missing share": float(
                    summary.loc[summary["value"] == MISSING_LABEL, "count"].sum()
                    / len(fnh)
                ),
            }
            for column, summary in distribution_tables.items()
        ]
    )
    expected = [
        "onet_code",
        "onet_title",
        *EXPECTED_ROLE_COLUMNS,
        *EXPECTED_RICS_COLUMNS,
    ]
    expected.extend(
        ["naics_code", "naics_description", "country", "state", "seniority"]
    )
    schema_report = pd.DataFrame(
        [
            {
                "Variable": column,
                "Status": "Available"
                if column in fnh.columns
                else "Absent from input schema",
                "Missing rows": int(fnh[column].isna().sum())
                if column in fnh.columns
                else pd.NA,
                "Missing share": (
                    float(fnh[column].isna().mean()) if column in fnh.columns else pd.NA
                ),
            }
            for column in expected
        ]
    )
    onet_title_diagnostic = (
        fnh[["onet_code", "onet_title"]]
        .dropna()
        .groupby("onet_code", observed=True)["onet_title"]
        .nunique()
        .rename("distinct_titles")
        .reset_index()
    )
    naics_title_diagnostic = (
        fnh[["naics_code", "naics_description"]]
        .dropna()
        .groupby("naics_code", observed=True)["naics_description"]
        .nunique()
        .rename("distinct_descriptions")
        .reset_index()
    )
    basic_numbers = {
        "candidate_count": len(fnh),
        "distinct_users": int(fnh["user_id"].nunique(dropna=True)),
        "distinct_companies": int(fnh["rcid"].nunique(dropna=True)),
        "distinct_countries": int(fnh["country"].nunique(dropna=True)),
        "universe_count": len(universe_fnh),
    }
    return (
        CLASSIFICATION_LABELS,
        baseline_distribution_tables,
        baseline_naics_rics_pairs,
        baseline_onet_role_pairs,
        baseline_rics_naics_pairs,
        baseline_role_onet_pairs,
        basic_numbers,
        classification_stats,
        distribution_tables,
        finest_rics_column,
        naics_rics_pairs,
        naics_title_diagnostic,
        onet_role_pairs,
        onet_title_diagnostic,
        role_onet_pairs,
        rics_naics_pairs,
        schema_report,
    )


@app.cell
def _(
    basic_numbers,
    classification_stats,
    link_diagnostics,
    mo,
    naics_title_diagnostic,
    onet_title_diagnostic,
    schema_report,
):
    onet_conflicts = int((onet_title_diagnostic["distinct_titles"] > 1).sum())
    naics_conflicts = int((naics_title_diagnostic["distinct_descriptions"] > 1).sum())
    _candidate_count = basic_numbers["candidate_count"]
    match_share = basic_numbers["candidate_count"] / basic_numbers["universe_count"]
    mo.vstack(
        [
            mo.md("## 1. Basic numbers"),
            mo.md(
                """
                The universe sample is constructed by: (i) retaining employment spells in
                two-digit occupation groups 17 (Architecture and Engineering) and 19 (Life,
                Physical, and Social Science); (ii) retaining starts from January 2021 through
                December 2023; (iii) excluding missing geography or job-title information;
                (iv) excluding internships; and (v) retaining one spell per user-company cell.

                This matched notebook adds one restriction: retain a user-company observation
                when the user has at least one inventor ID in the Revelio-to-PatentsView
                crosswalk. Duplicate crosswalk rows or multiple inventor IDs do not duplicate
                the user-company observation. This remains a user-company sample, so one user
                may appear at multiple companies. All summary statistics are simple averages
                over matched user-company observations.
                """
            ),
            mo.md(
                f"""
                | Number | Value |
                |---|---:|
                | Inventor-matched candidate focal new hires | {_candidate_count:,} |
                | Distinct matched users | {basic_numbers["distinct_users"]:,} |
                | Distinct matched companies | {basic_numbers["distinct_companies"]:,} |
                | Distinct matched countries | {basic_numbers["distinct_countries"]:,} |
                | Universe candidate focal new hires | {basic_numbers["universe_count"]:,} |
                | Matched share of universe observations | {match_share:.2%} |
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
                    "Inventor crosswalk diagnostics": mo.ui.table(
                        link_diagnostics, pagination=False, show_column_summaries=False
                    ),
                    "Label diagnostics": mo.md(
                        f"O*NET codes with multiple titles: **{onet_conflicts:,}**; "
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

    Bars report the inventor-matched sample. Red diamonds report the universe baseline
    share for the same category.
    """)
    return


@app.cell
def _(baseline_distribution_tables, distribution_tables, make_share_chart, mo):
    _summary = distribution_tables["onet_code"]
    _chart = make_share_chart(
        _summary,
        "O*NET occupation distribution",
        baseline=baseline_distribution_tables["onet_code"],
    )
    mo.vstack(
        [
            mo.md("### 2.1. O*NET occupation distribution"),
            _chart,
            mo.accordion(
                {
                    "View matched O*NET counts": mo.ui.table(
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
def _(available_role_columns, mo):
    role_variable_selector = mo.ui.dropdown(
        options=list(available_role_columns),
        value="role_k1500" if "role_k1500" in available_role_columns else None,
        label="Revelio occupation variable",
    )
    role_top_n_selector = mo.ui.number(
        start=1, stop=1000, step=1, value=50, label="Number of top occupations"
    )
    mo.vstack(
        [
            mo.md("### 2.2. Revelio's own occupation distribution"),
            mo.hstack(
                [role_variable_selector, role_top_n_selector],
                justify="start",
                gap=2,
            ),
        ],
        gap=1,
    )
    return role_top_n_selector, role_variable_selector


@app.cell
def _(
    baseline_distribution_tables,
    distribution_tables,
    make_share_chart,
    mo,
    role_top_n_selector,
    role_variable_selector,
):
    _variable = role_variable_selector.value
    _top_n = max(1, int(role_top_n_selector.value))
    _summary = distribution_tables[_variable]
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} occupations in {_variable}",
        _top_n,
        baseline=baseline_distribution_tables[_variable],
    )
    mo.vstack(
        [
            _chart,
            mo.accordion(
                {
                    "View matched categories": mo.ui.table(
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
def _(MISSING_LABEL, mo, onet_role_pairs):
    if onet_role_pairs.empty:
        onet_role_selector = None
        _control = mo.callout(mo.md("`role_k1500` is unavailable."), kind="warn")
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
            label="O*NET occupation",
            full_width=True,
        )
        _control = onet_role_selector
    mo.vstack(
        [
            mo.md("### 2.3. Crosswalk from O*NET to Revelio's own occupation"),
            mo.md(
                "Matched bars and universe-baseline diamonds show conditional shares."
            ),
            _control,
        ],
        gap=1,
    )
    return (onet_role_selector,)


@app.cell
def _(
    baseline_onet_role_pairs,
    make_crosswalk_chart,
    mo,
    onet_role_pairs,
    onet_role_selector,
):
    _output = mo.md("")
    if onet_role_selector is not None:
        _selected = onet_role_selector.value
        _chart = make_crosswalk_chart(
            onet_role_pairs,
            _selected,
            f"Revelio role composition of {_selected}",
            baseline_onet_role_pairs,
        )
        _output = mo.vstack([_chart], gap=1)
    _output


@app.cell
def _(MISSING_LABEL, mo, role_onet_pairs):
    if role_onet_pairs.empty:
        role_onet_selector = None
        _control = mo.callout(mo.md("`role_k1500` is unavailable."), kind="warn")
    else:
        _choices = {
            f"{left} ({int(group['count'].sum()):,} hires)": left
            for left, group in role_onet_pairs.groupby("left_label", sort=False)
            if left != MISSING_LABEL
        }
        role_onet_selector = mo.ui.dropdown(
            options=_choices,
            value=next(iter(_choices)) if _choices else None,
            searchable=True,
            label="Revelio role K1,500",
            full_width=True,
        )
        _control = role_onet_selector
    mo.vstack(
        [
            mo.md("### 2.4. Crosswalk from Revelio's own occupation to O*NET"),
            mo.md(
                "Matched bars and universe-baseline diamonds show conditional shares."
            ),
            _control,
        ],
        gap=1,
    )
    return (role_onet_selector,)


@app.cell
def _(
    baseline_role_onet_pairs,
    make_crosswalk_chart,
    mo,
    role_onet_pairs,
    role_onet_selector,
):
    _output = mo.md("")
    if role_onet_selector is not None:
        _selected = role_onet_selector.value
        _chart = make_crosswalk_chart(
            role_onet_pairs,
            _selected,
            f"O*NET composition of {_selected}",
            baseline_role_onet_pairs,
        )
        _output = mo.vstack([_chart], gap=1)
    _output


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The industry distribution

    Bars report the inventor-matched sample. Red diamonds report the universe baseline
    share for the same category.
    """)
    return


@app.cell
def _(mo):
    naics_top_n_selector = mo.ui.number(
        start=1, stop=2000, step=1, value=50, label="Number of top NAICS industries"
    )
    mo.vstack([mo.md("### 3.1. NAICS distribution"), naics_top_n_selector], gap=1)
    return (naics_top_n_selector,)


@app.cell
def _(
    baseline_distribution_tables,
    distribution_tables,
    make_share_chart,
    mo,
    naics_top_n_selector,
):
    _chart = make_share_chart(
        distribution_tables["naics_code"],
        "Top NAICS industries",
        int(naics_top_n_selector.value),
        baseline=baseline_distribution_tables["naics_code"],
    )
    _chart


@app.cell
def _(available_rics_columns, mo):
    rics_variable_selector = mo.ui.dropdown(
        options=list(available_rics_columns),
        value="rics_k400" if "rics_k400" in available_rics_columns else None,
        label="Revelio industry variable",
    )
    rics_top_n_selector = mo.ui.number(
        start=1, stop=2000, step=1, value=50, label="Number of top Revelio industries"
    )
    mo.vstack(
        [
            mo.md("### 3.2. Revelio's own industry distribution"),
            mo.hstack(
                [rics_variable_selector, rics_top_n_selector], justify="start", gap=2
            ),
        ],
        gap=1,
    )
    return rics_top_n_selector, rics_variable_selector


@app.cell
def _(
    baseline_distribution_tables,
    distribution_tables,
    make_share_chart,
    mo,
    rics_top_n_selector,
    rics_variable_selector,
):
    _variable = rics_variable_selector.value
    _top_n = int(rics_top_n_selector.value)
    _chart = make_share_chart(
        distribution_tables[_variable],
        f"Top {_top_n} industries in {_variable}",
        _top_n,
        baseline=baseline_distribution_tables[_variable],
    )
    _chart


@app.cell
def _(MISSING_LABEL, mo, naics_rics_pairs):
    if naics_rics_pairs.empty:
        naics_rics_selector = None
        _control = mo.callout(mo.md("No RICS field is available."), kind="warn")
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
        _control = naics_rics_selector
    mo.vstack(
        [
            mo.md("### 3.3. Crosswalk from NAICS to Revelio's own industry"),
            mo.md(
                "Matched bars and universe-baseline diamonds show conditional shares."
            ),
            _control,
        ],
        gap=1,
    )
    return (naics_rics_selector,)


@app.cell
def _(
    baseline_naics_rics_pairs,
    finest_rics_column,
    make_crosswalk_chart,
    mo,
    naics_rics_pairs,
    naics_rics_selector,
):
    _output = mo.md("")
    if naics_rics_selector is not None:
        _selected = naics_rics_selector.value
        _chart = make_crosswalk_chart(
            naics_rics_pairs,
            _selected,
            f"{finest_rics_column} composition of {_selected}",
            baseline_naics_rics_pairs,
        )
        _output = mo.vstack([_chart], gap=1)
    _output


@app.cell
def _(MISSING_LABEL, finest_rics_column, mo, rics_naics_pairs):
    if rics_naics_pairs.empty:
        rics_naics_selector = None
        _control = mo.callout(mo.md("No RICS field is available."), kind="warn")
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
        _control = rics_naics_selector
    mo.vstack(
        [
            mo.md("### 3.4. Crosswalk from Revelio's own industry to NAICS"),
            mo.md(
                "Matched bars and universe-baseline diamonds show conditional shares."
            ),
            _control,
        ],
        gap=1,
    )
    return (rics_naics_selector,)


@app.cell
def _(
    baseline_rics_naics_pairs,
    finest_rics_column,
    make_crosswalk_chart,
    mo,
    rics_naics_pairs,
    rics_naics_selector,
):
    _output = mo.md("")
    if rics_naics_selector is not None:
        _selected = rics_naics_selector.value
        _chart = make_crosswalk_chart(
            rics_naics_pairs,
            _selected,
            f"NAICS composition of {_selected} ({finest_rics_column})",
            baseline_rics_naics_pairs,
        )
        _output = mo.vstack([_chart], gap=1)
    _output


@app.cell
def _(country_iso3, distribution_table, fnh, math, universe_fnh, us_state_code):
    country_summary = distribution_table(fnh, "country")
    baseline_country_summary = distribution_table(universe_fnh, "country")
    country_summary["iso3"] = country_summary["country"].map(country_iso3)
    country_summary["log10_count"] = country_summary["count"].map(
        lambda count: math.log10(count) if count > 0 else 0
    )
    country_baseline = baseline_country_summary[["display_label", "share"]].rename(
        columns={"share": "baseline_share"}
    )
    country_summary = country_summary.merge(
        country_baseline, on="display_label", how="left"
    )
    country_summary["baseline_share"] = country_summary["baseline_share"].fillna(0.0)
    mapped_country_summary = country_summary.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = country_summary.loc[
        country_summary["iso3"].isna()
    ].copy()
    state_working = fnh.loc[fnh["country"] == "United States", ["state"]].copy()
    state_baseline_working = universe_fnh.loc[
        universe_fnh["country"] == "United States", ["state"]
    ].copy()
    for working in [state_working, state_baseline_working]:
        working["state"] = working["state"].fillna("<Missing>")
    us_state_summary = (
        state_working.groupby("state", observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    baseline_state_summary = (
        state_baseline_working.groupby("state", observed=True)
        .size()
        .rename("baseline_count")
        .reset_index()
    )
    us_state_summary["share_within_country"] = (
        us_state_summary["count"] / us_state_summary["count"].sum()
    )
    baseline_state_summary["baseline_share_within_country"] = (
        baseline_state_summary["baseline_count"]
        / baseline_state_summary["baseline_count"].sum()
    )
    us_state_summary = us_state_summary.merge(
        baseline_state_summary, on="state", how="left"
    )
    us_state_summary["baseline_share_within_country"] = us_state_summary[
        "baseline_share_within_country"
    ].fillna(0.0)
    us_state_summary["state_code"] = us_state_summary["state"].map(us_state_code)
    state_map_data = us_state_summary.dropna(subset=["state_code"]).copy()
    unmatched_state_data = us_state_summary.loc[
        us_state_summary["state_code"].isna()
    ].copy()
    state_map_coverage = (
        state_map_data["count"].sum() / us_state_summary["count"].sum()
        if not us_state_summary.empty
        else 0.0
    )
    return (
        mapped_country_summary,
        state_map_coverage,
        state_map_data,
        unmatched_state_data,
        unmapped_country_summary,
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
            "baseline_share": ":.2%",
        },
        labels={
            "count": "Inventor-matched hires",
            "share": "Matched share",
            "baseline_share": "Universe baseline share",
            "log10_count": "Log10 inventor-matched hires",
        },
        color_continuous_scale="Blues",
        projection="natural earth",
        title="Inventor-matched candidate focal new hires by country",
    ).update_geos(showframe=False, showcoastlines=True)
    mo.vstack(
        [
            mo.md("## 4. Other results\n\n### 4.1. Geography distribution"),
            mo.md("Hover over a country to compare matched and universe shares."),
            _figure,
            mo.accordion(
                {
                    "View country counts not mapped to ISO-3": mo.ui.table(
                        unmapped_country_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )


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
                "baseline_share_within_country": ":.2%",
            },
            labels={
                "count": "Inventor-matched hires",
                "share_within_country": "Matched U.S. share",
                "baseline_share_within_country": "Universe baseline U.S. share",
            },
            color_continuous_scale="Blues",
            title="Inventor-matched candidate focal new hires by U.S. state",
        ).update_geos(scope="usa", visible=False)
    mo.vstack(
        [
            mo.md(
                "State-map hover values report the matched and universe shares within the "
                "United States."
            ),
            _figure,
            mo.md(
                f"The state-code mapping covers **{state_map_coverage:.2%}** of matched U.S. "
                "hires."
            ),
            mo.accordion(
                {
                    "View unmatched U.S. state labels": mo.ui.table(
                        unmatched_state_data,
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
def _(alt, distribution_table, fnh, mo, pd, universe_fnh):
    matched = distribution_table(fnh, "seniority")[
        ["display_label", "count", "share"]
    ].copy()
    baseline = distribution_table(universe_fnh, "seniority")[
        ["display_label", "share"]
    ].copy()
    baseline = baseline.rename(columns={"share": "baseline_share"})
    seniority = matched.merge(baseline, on="display_label", how="outer").fillna(0.0)
    seniority["sample"] = "Inventor-matched"
    baseline_plot = seniority[["display_label", "baseline_share"]].rename(
        columns={"baseline_share": "share"}
    )
    baseline_plot["sample"] = "Universe baseline"
    matched_plot = seniority[["display_label", "share"]].copy()
    matched_plot["sample"] = "Inventor-matched"
    plot_data = pd.concat([matched_plot, baseline_plot], ignore_index=True)
    plot_data["seniority_order"] = pd.to_numeric(
        plot_data["display_label"], errors="coerce"
    )
    order = (
        plot_data[["display_label", "seniority_order"]]
        .drop_duplicates()
        .sort_values(["seniority_order", "display_label"], na_position="last")[
            "display_label"
        ]
        .tolist()
    )
    _figure = (
        alt.Chart(plot_data)
        .mark_bar()
        .encode(
            x=alt.X(
                "display_label:O",
                sort=order,
                title="Seniority level",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("share:Q", title="Share", axis=alt.Axis(format=".1%")),
            xOffset=alt.XOffset("sample:N"),
            color=alt.Color(
                "sample:N",
                scale=alt.Scale(
                    domain=["Inventor-matched", "Universe baseline"],
                    range=["#7C3AED", "#CBD5E1"],
                ),
                title=None,
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Seniority"),
                alt.Tooltip("sample:N", title="Sample"),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
            ],
        )
        .properties(width="container", height=380)
        .configure_view(stroke=None)
    )
    mo.vstack([mo.md("### 4.2. Seniority distribution"), _figure], gap=1)


@app.cell
def _(alt, fnh, mo, pd, universe_fnh):
    matched_series = (
        fnh.dropna(subset=["start_month"])
        .assign(
            start_month=lambda data: (
                pd.to_datetime(data["start_month"]).dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby("start_month", observed=True)
        .size()
        .rename("matched_count")
    )
    baseline_series = (
        universe_fnh.dropna(subset=["start_month"])
        .assign(
            start_month=lambda data: (
                pd.to_datetime(data["start_month"]).dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby("start_month", observed=True)
        .size()
        .rename("universe_count")
    )
    time_series = (
        pd.concat([matched_series, baseline_series], axis=1).fillna(0).reset_index()
    )
    time_series["matched_share_of_universe"] = time_series[
        "matched_count"
    ] / time_series["universe_count"].replace(0, pd.NA)
    time_series["matched_month_share"] = (
        time_series["matched_count"] / time_series["matched_count"].sum()
    )
    time_series["universe_month_share"] = (
        time_series["universe_count"] / time_series["universe_count"].sum()
    )
    _matched_month_plot = time_series[
        [
            "start_month",
            "matched_count",
            "matched_month_share",
            "matched_share_of_universe",
        ]
    ].rename(columns={"matched_count": "count", "matched_month_share": "share"})
    _matched_month_plot["sample"] = "Inventor-matched"
    _baseline_month_plot = time_series[
        [
            "start_month",
            "universe_count",
            "universe_month_share",
            "matched_share_of_universe",
        ]
    ].rename(columns={"universe_count": "count", "universe_month_share": "share"})
    _baseline_month_plot["sample"] = "Universe baseline"
    _share_plot_data = pd.concat(
        [_matched_month_plot, _baseline_month_plot], ignore_index=True
    )
    _count_figure = (
        alt.Chart(time_series)
        .mark_line(point=True, color="#7C3AED")
        .encode(
            x=alt.X("start_month:T", title="Employment start month"),
            y=alt.Y("matched_count:Q", title="Inventor-matched candidate new hires"),
            tooltip=[
                alt.Tooltip("start_month:T", title="Start month", format="%b %Y"),
                alt.Tooltip(
                    "matched_count:Q", title="Inventor-matched hires", format=","
                ),
                alt.Tooltip(
                    "matched_share_of_universe:Q",
                    title="Matched share of universe",
                    format=".2%",
                ),
            ],
        )
        .properties(width="container", height=320)
        .configure_view(stroke=None)
    )
    _share_figure = (
        alt.Chart(_share_plot_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("start_month:T", title="Employment start month"),
            y=alt.Y(
                "share:Q", title="Share of each sample", axis=alt.Axis(format=".1%")
            ),
            color=alt.Color(
                "sample:N",
                scale=alt.Scale(
                    domain=["Inventor-matched", "Universe baseline"],
                    range=["#7C3AED", "#64748B"],
                ),
                title="Sample",
            ),
            tooltip=[
                alt.Tooltip("start_month:T", title="Start month", format="%b %Y"),
                alt.Tooltip("sample:N", title="Sample"),
                alt.Tooltip("count:Q", title="Candidate focal new hires", format=","),
                alt.Tooltip(
                    "matched_share_of_universe:Q",
                    title="Matched share of universe",
                    format=".2%",
                ),
            ],
        )
        .properties(width="container", height=360)
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md("### 4.3. Time-series"),
            mo.md("Monthly count of inventor-matched candidate focal new hires."),
            _count_figure,
            mo.md(
                "Monthly shares compare the timing composition of the matched and universe "
                "samples on a common scale."
            ),
            _share_figure,
        ],
        gap=1,
    )


if __name__ == "__main__":
    app.run()

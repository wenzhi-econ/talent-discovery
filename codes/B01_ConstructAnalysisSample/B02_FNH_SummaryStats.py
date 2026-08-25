# ruff: noqa: PLR1711

"""Summary statistics for the universe of candidate focal new hires."""

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
    import pyarrow.dataset as ds
    import pycountry

    return alt, ds, math, mo, pd, px, pycountry, re


@app.cell
def _(mo):
    mo.md(r"""
    # Universe of candidate focal new hires: summary statistics

    This notebook describes the broadest universe of candidate focal new hires. The
    observation unit is a **user-company observation**: one user can appear more than
    once when they are observed as a new hire at different companies. All distributions
    below are simple averages over these user-company observations.
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
            alt.Tooltip("count:Q", title="Candidate new hires", format=","),
            alt.Tooltip("share:Q", title="Share", format=".2%"),
            alt.Tooltip("rank:Q", title="Rank", format="d"),
        ]
        if baseline is not None:
            tooltip.append(
                alt.Tooltip(
                    "baseline_share:Q", title="Universe baseline share", format=".2%"
                )
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
                title="Share of candidate new hires",
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
                base.mark_point(
                    shape="diamond", filled=True, color="#B91C1C", size=90
                ).encode(x=alt.X("baseline_share:Q"))
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
                alt.Tooltip(
                    "baseline_share:Q", title="Universe baseline share", format=".2%"
                )
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
                base.mark_point(
                    shape="diamond", filled=True, color="#B91C1C", size=90
                ).encode(x=alt.X("baseline_share:Q"))
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
    PARQUET_FILES = tuple(sorted(INPUT_DIR.glob("*.parquet")))
    if not PARQUET_FILES:
        raise FileNotFoundError(f"No Parquet files found in: {INPUT_DIR}")
    dataset = ds.dataset(INPUT_DIR, format="parquet")
    AVAILABLE_COLUMNS = tuple(dataset.schema.names)
    missing_required = sorted(set(REQUIRED_COLUMNS) - set(AVAILABLE_COLUMNS))
    if missing_required:
        raise ValueError(f"Input is missing required fields: {missing_required}")
    if "start_month" in AVAILABLE_COLUMNS:
        DATE_COLUMN = "start_month"
    elif "startdate" in AVAILABLE_COLUMNS:
        DATE_COLUMN = "startdate"
    else:
        raise ValueError("Input must contain either `start_month` or `startdate`.")
    AVAILABLE_ROLE_COLUMNS = tuple(
        sorted(
            (column for column in AVAILABLE_COLUMNS if column.startswith("role_k")),
            key=hierarchy_number,
        )
    )
    AVAILABLE_RICS_COLUMNS = tuple(
        sorted(
            (column for column in AVAILABLE_COLUMNS if column.startswith("rics_k")),
            key=hierarchy_number,
        )
    )
    ANALYSIS_COLUMNS = tuple(
        dict.fromkeys(
            [
                *REQUIRED_COLUMNS,
                DATE_COLUMN,
                *AVAILABLE_ROLE_COLUMNS,
                *AVAILABLE_RICS_COLUMNS,
            ]
        )
    )
    fnh = pd.read_parquet(
        INPUT_DIR,
        columns=list(ANALYSIS_COLUMNS),
        engine="pyarrow",
        dtype_backend="pyarrow",
    )
    if DATE_COLUMN == "start_month":
        fnh["start_month"] = pd.to_datetime(fnh[DATE_COLUMN], errors="coerce")
    else:
        fnh["start_month"] = (
            pd.to_datetime(fnh[DATE_COLUMN], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp()
        )
    if fnh.empty:
        raise ValueError("The focal-new-hire input contains no observations.")
    return (
        AVAILABLE_RICS_COLUMNS,
        AVAILABLE_ROLE_COLUMNS,
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        fnh,
    )


@app.cell
def _(
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    MISSING_LABEL,
    crosswalk_table,
    distribution_table,
    fnh,
    hierarchy_number,
    pd,
):
    CLASSIFICATION_LABELS = {
        "onet_code": "O*NET code and title",
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
    classification_columns = (
        "onet_code",
        *AVAILABLE_ROLE_COLUMNS,
        "naics_code",
        *AVAILABLE_RICS_COLUMNS,
    )
    distribution_tables = {
        column: distribution_table(fnh, column, title_columns.get(column))
        for column in classification_columns
    }
    if "role_k1500" in fnh.columns:
        _role_crosswalk_columns = [
            *AVAILABLE_ROLE_COLUMNS,
            "onet_code",
            "onet_title",
        ]
        _role_crosswalk_rows = fnh.loc[:, _role_crosswalk_columns].copy()
        for _column in _role_crosswalk_columns:
            _role_crosswalk_rows[_column] = (
                _role_crosswalk_rows[_column]
                .fillna(MISSING_LABEL)
                .astype("string")
            )
        _role_crosswalk_rows = _role_crosswalk_rows.drop_duplicates()
        _role_shares = distribution_tables["role_k1500"][["value", "share"]]
        _role_crosswalk_ordered = _role_crosswalk_rows.merge(
            _role_shares,
            left_on="role_k1500",
            right_on="value",
            how="left",
            validate="many_to_one",
        )
        _role_crosswalk_ordered["O*NET code and title"] = (
            _role_crosswalk_ordered["onet_code"]
            + " — "
            + _role_crosswalk_ordered["onet_title"]
        )
        _role_crosswalk_ordered = _role_crosswalk_ordered.sort_values(
            ["share", "role_k1500", "O*NET code and title"],
            ascending=[False, True, True],
        )
        role_onet_crosswalk = (
            _role_crosswalk_ordered.loc[
                :, [*AVAILABLE_ROLE_COLUMNS, "O*NET code and title"]
            ]
            .reset_index(drop=True)
        )
        _role_mapping_counts = (
            _role_crosswalk_rows.loc[
                _role_crosswalk_rows["role_k1500"] != MISSING_LABEL
            ].groupby("role_k1500", observed=True).size()
        )
        role_onet_cardinality_violations = int((_role_mapping_counts > 1).sum())
    else:
        role_onet_crosswalk = pd.DataFrame(
            columns=[*AVAILABLE_ROLE_COLUMNS, "O*NET code and title"]
        )
        role_onet_cardinality_violations = 0
    finest_rics_column = AVAILABLE_RICS_COLUMNS[-1] if AVAILABLE_RICS_COLUMNS else None
    rics_naics_pairs = (
        crosswalk_table(fnh, [finest_rics_column], "naics_code", "naics_description")
        if finest_rics_column
        else pd.DataFrame()
    )
    naics_rics_pairs = (
        crosswalk_table(fnh, ["naics_code", "naics_description"], finest_rics_column)
        if finest_rics_column
        else pd.DataFrame()
    )
    classification_stats = pd.DataFrame(
        [
            {
                "Variable": column,
                "Classification": CLASSIFICATION_LABELS[column],
                "Nonmissing categories": int(
                    summary.loc[summary["value"] != MISSING_LABEL, "value"].nunique()
                ),
                "Missing rows": int(
                    summary.loc[summary["value"] == MISSING_LABEL, "count"].sum()
                ),
                "Missing share": float(
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
    }
    return (
        basic_numbers,
        classification_stats,
        distribution_tables,
        finest_rics_column,
        naics_rics_pairs,
        naics_title_diagnostic,
        onet_title_diagnostic,
        rics_naics_pairs,
        role_onet_cardinality_violations,
        role_onet_crosswalk,
        schema_report,
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

                1. Keep employment spells in the two-digit occupation groups **17:
                   Architecture and Engineering occupations** and **19: Life, Physical,
                   and Social Science occupations**.
                2. Retain spells starting from **January 2021 through December 2023**.
                3. Exclude spells with missing geography or job-title information.
                4. Exclude internship positions.
                5. Retain one employment spell within each user-company cell. This is the
                   **universe sample of candidate focal new hires**.

                This is the broadest sample of the new hires. Further restrictions on
                industries and occupations will be guided by the summary statistics below.
                The sample is at the user-company level, so one user can appear multiple
                times as a new hire at different companies. All summary statistics are
                simple averages over these user-company observations.
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

    Shares are calculated over all candidate focal new hires, including observations with
    missing classification values.
    """)
    return


@app.cell
def _(distribution_tables, make_share_chart, mo):
    onet_summary = distribution_tables["onet_code"]
    _chart = make_share_chart(onet_summary, "O*NET occupation distribution")
    mo.vstack(
        [
            mo.md("### 2.1. O*NET occupation distribution"),
            _chart,
            mo.accordion(
                {
                    "View all O*NET occupation counts": mo.ui.table(
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
def _(AVAILABLE_ROLE_COLUMNS, mo):
    role_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_ROLE_COLUMNS),
        value="role_k1500" if "role_k1500" in AVAILABLE_ROLE_COLUMNS else None,
        label="Revelio occupation variable",
    )
    role_top_n_selector = mo.ui.number(
        start=1, stop=1000, step=1, value=50, label="Number of top occupations"
    )
    return role_top_n_selector, role_variable_selector


@app.cell
def _(
    distribution_tables,
    make_share_chart,
    mo,
    role_top_n_selector,
    role_variable_selector,
):
    role_variable = role_variable_selector.value
    role_summary = distribution_tables[role_variable]
    role_top_n = max(1, int(role_top_n_selector.value))
    _chart = make_share_chart(
        role_summary, f"Top {role_top_n} occupations in {role_variable}", role_top_n
    )
    mo.vstack(
        [
            mo.md("### 2.2. Revelio's own occupation distribution"),
            mo.hstack(
                [role_variable_selector, role_top_n_selector],
                justify="start",
                gap=2,
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
def _(AVAILABLE_ROLE_COLUMNS, mo):
    onet_role_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_ROLE_COLUMNS),
        value="role_k1500" if "role_k1500" in AVAILABLE_ROLE_COLUMNS else None,
        label="Revelio occupation variable",
    )
    return (onet_role_variable_selector,)


@app.cell
def _(
    MISSING_LABEL,
    crosswalk_table,
    fnh,
    mo,
    onet_role_variable_selector,
    pd,
):
    onet_role_variable = onet_role_variable_selector.value
    onet_role_pairs = (
        crosswalk_table(
            fnh, ["onet_code", "onet_title"], onet_role_variable
        )
        if onet_role_variable in fnh.columns
        else pd.DataFrame()
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
            label="O*NET occupation",
            full_width=True,
        )
    return onet_role_pairs, onet_role_selector, onet_role_variable


@app.cell
def _(
    make_crosswalk_chart,
    mo,
    onet_role_pairs,
    onet_role_selector,
    onet_role_variable,
    onet_role_variable_selector,
):
    _control = (
        onet_role_selector
        if onet_role_selector is not None
        else mo.callout(
            mo.md(f"`{onet_role_variable}` is unavailable."), kind="warn"
        )
    )
    _items = [
        mo.md("### 2.3. Crosswalk from O*NET to Revelio's own occupation"),
        mo.md(
            "Select a Revelio occupation variable and an O*NET occupation to see "
            "the corresponding role composition."
        ),
        mo.hstack(
            [onet_role_variable_selector, _control],
            justify="start",
            gap=2,
        ),
    ]
    if onet_role_selector is not None:
        _selected = onet_role_selector.value
        _chart = make_crosswalk_chart(
            onet_role_pairs,
            _selected,
            f"Revelio {onet_role_variable} composition of {_selected}",
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
                    "Revelio hierarchy values and O*NET occupation."
                ),
                kind="warn",
            )
        else:
            _diagnostic = mo.md(
                "Each nonmissing `role_k1500` category maps to exactly one combination "
                "of Revelio occupation hierarchy values and O*NET occupation in the "
                "current sample. Rows are ordered by the `role_k1500` category's share "
                "of candidate focal new hires, from largest to smallest."
            )
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
            mo.md("### 2.4. Crosswalk from Revelio's own occupation to O*NET"),
            _content,
        ],
        gap=1,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The industry distribution

    NAICS and Revelio's own industry classifications are shown separately because they
    need not produce the same categories or composition.
    """)
    return


@app.cell
def _(mo):
    naics_top_n_selector = mo.ui.number(
        start=1, stop=2000, step=1, value=50, label="Number of top NAICS industries"
    )
    return (naics_top_n_selector,)


@app.cell
def _(distribution_tables, make_share_chart, mo, naics_top_n_selector):
    _summary = distribution_tables["naics_code"]
    _chart = make_share_chart(
        _summary,
        "Top NAICS industries",
        int(naics_top_n_selector.value),
    )
    mo.vstack(
        [
            mo.md("### 3.1. NAICS distribution"),
            naics_top_n_selector,
            _chart,
            mo.accordion(
                {
                    "View all NAICS categories": mo.ui.table(
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
def _(AVAILABLE_RICS_COLUMNS, mo):
    rics_variable_selector = mo.ui.dropdown(
        options=list(AVAILABLE_RICS_COLUMNS),
        value="rics_k400" if "rics_k400" in AVAILABLE_RICS_COLUMNS else None,
        label="Revelio industry variable",
    )
    rics_top_n_selector = mo.ui.number(
        start=1, stop=2000, step=1, value=50, label="Number of top Revelio industries"
    )
    return rics_top_n_selector, rics_variable_selector


@app.cell
def _(
    distribution_tables,
    make_share_chart,
    mo,
    rics_top_n_selector,
    rics_variable_selector,
):
    _variable = rics_variable_selector.value
    _top_n = int(rics_top_n_selector.value)
    _summary = distribution_tables[_variable]
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industries in {_variable}",
        _top_n,
    )
    mo.vstack(
        [
            mo.md("### 3.2. Revelio's own industry distribution"),
            mo.hstack(
                [rics_variable_selector, rics_top_n_selector],
                justify="start",
                gap=2,
            ),
            _chart,
            mo.accordion(
                {
                    "View all Revelio industry categories": mo.ui.table(
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
def _(MISSING_LABEL, mo, naics_rics_pairs):
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
    return (naics_rics_selector,)


@app.cell
def _(
    finest_rics_column,
    make_crosswalk_chart,
    mo,
    naics_rics_pairs,
    naics_rics_selector,
):
    _control = (
        naics_rics_selector
        if naics_rics_selector is not None
        else mo.callout(mo.md("No RICS field is available."), kind="warn")
    )
    _items = [
        mo.md("### 3.3. Crosswalk from NAICS to Revelio's own industry"),
        mo.md(
            "Select a NAICS category to see its finest Revelio industry composition."
        ),
        _control,
    ]
    if naics_rics_selector is not None:
        _selected = naics_rics_selector.value
        _chart = make_crosswalk_chart(
            naics_rics_pairs,
            _selected,
            f"{finest_rics_column} composition of {_selected}",
        )
        _selected_pairs = naics_rics_pairs.loc[
            naics_rics_pairs["left_label"] == _selected
        ].copy()
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
def _(MISSING_LABEL, finest_rics_column, mo, rics_naics_pairs):
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
    return (rics_naics_selector,)


@app.cell
def _(
    finest_rics_column,
    make_crosswalk_chart,
    mo,
    rics_naics_pairs,
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
            f"Select a `{finest_rics_column}` category to see its NAICS composition."
        ),
        _control,
    ]
    if rics_naics_selector is not None:
        _selected = rics_naics_selector.value
        _chart = make_crosswalk_chart(
            rics_naics_pairs,
            _selected,
            f"NAICS composition of {_selected} ({finest_rics_column})",
        )
        _selected_pairs = rics_naics_pairs.loc[
            rics_naics_pairs["left_label"] == _selected
        ].copy()
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
def _(country_iso3, distribution_table, fnh, math, us_state_code):
    country_summary = distribution_table(fnh, "country")
    country_summary["iso3"] = country_summary["country"].map(country_iso3)
    country_summary["log10_count"] = country_summary["count"].map(
        lambda count: math.log10(count) if count > 0 else 0
    )
    mapped_country_summary = country_summary.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = country_summary.loc[
        country_summary["iso3"].isna()
    ].copy()
    state_working = fnh.loc[fnh["country"] == "United States", ["state"]].copy()
    state_working["state"] = state_working["state"].fillna("<Missing>")
    us_state_summary = (
        state_working.groupby("state", observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    us_state_summary["share_within_country"] = (
        us_state_summary["count"] / us_state_summary["count"].sum()
    )
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
            mo.md("## 4. Other results\n\n### 4.1. Geography distribution"),
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
                    )
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
                "State shares use all U.S. candidate focal new hires as the denominator, "
                "including missing or unmapped state labels."
            ),
            _figure,
            mo.md(
                f"The state-code mapping covers **{state_map_coverage:.2%}** of U.S. hires."
            ),
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
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(alt, distribution_table, fnh, mo, pd):
    seniority_summary = distribution_table(fnh, "seniority")
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
        .properties(width="container", height=360)
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md("### 4.2. Seniority distribution"),
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
def _(alt, fnh, mo, pd):
    time_series = (
        fnh.dropna(subset=["start_month"])
        .assign(
            start_month=lambda data: (
                pd.to_datetime(data["start_month"]).dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby("start_month", observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
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
        .properties(width="container", height=360)
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md("### 4.3. Time-series"),
            mo.md("Monthly count by employment start month."),
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

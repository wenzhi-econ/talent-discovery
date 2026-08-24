# ruff: noqa: B018, PLR1711

"""
Task:
    Describe the universe of candidate focal new hires.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHires_AllIndustries/*.parquet

Outputs:
(a) Interactive marimo views of occupation, industry, and geographic distributions.

Notes:
(1) The unit is a focal-hire spell: one retained spell per user-company pair.
(2) Counts and shares describe the observed focal-hire spells directly.
(3) Only requested analysis columns are loaded, but all rows are read into pandas memory.
(4) The notebook detects absent role_k* and rics_k* fields and reports them explicitly.
(5) The U.S. state map uses Plotly's built-in USA-states geometry.

Run:
    $fnh_notebook = "codes/B01_ConstructAnalysisSample/B02_FNH_SummaryStats.py"
    conda run -s -n Talent marimo edit $fnh_notebook

Wang Wenzhi, with the help of Codex
Time: 2026-08-24
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import math
    import re

    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import pyarrow.dataset as ds
    import pycountry

    return (
        alt,
        ds,
        math,
        mo,
        pd,
        px,
        pycountry,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Universe of candidate focal new hires

    This notebook describes the broad all-industry sample used to choose narrower
    occupation and industry restrictions. The unit is a **focal-hire spell**: one
    retained spell per user-company pair. Counts and shares describe the observed
    candidate sample directly.

    Missing classifications are retained as explicit categories. Controls are provided
    where they support comparisons across classifications.
    """)
    return


@app.cell(hide_code=True)
def _(
    alt,
    pycountry,
    re,
):
    MISSING_LABEL = "<Missing>"

    def hierarchy_number(column_name):
        """Return the numeric K level used to sort Revelio hierarchy fields."""

        _match = re.search(r"_k(\d+)$", column_name)
        return int(_match.group(1)) if _match else -1

    def distribution_table(data, value_column, title_column=None):
        """Aggregate a categorical field while retaining missing observations."""

        _columns = [value_column]
        if title_column is not None:
            _columns.append(title_column)

        _working = data.loc[:, _columns].copy()
        for _column in _columns:
            _working[_column] = _working[_column].fillna(MISSING_LABEL)

        _summary = (
            _working.groupby(_columns, dropna=False, observed=True)
            .size()
            .rename("count")
            .reset_index()
        )
        _summary = _summary.sort_values(
            ["count", value_column],
            ascending=[False, True],
        ).reset_index(drop=True)
        _summary["share"] = _summary["count"] / len(data)
        _summary["rank"] = range(1, len(_summary) + 1)
        _summary["value"] = _summary[value_column].astype("string")

        if title_column is None:
            _summary["display_label"] = _summary["value"]
        else:
            _title = _summary[title_column].astype("string")
            _summary["display_label"] = _summary["value"] + " — " + _title

        return _summary

    def make_bar_chart(summary, title, top_n):
        """Create a ranked horizontal share chart from an aggregated table."""

        _top = summary.head(top_n).copy()
        _order = _top["display_label"].tolist()
        _maximum_share = float(_top["share"].max())
        _share_domain = [0.0, _maximum_share * 1.12]
        _base = alt.Chart(_top).encode(
            y=alt.Y(
                "display_label:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelLimit=420, labelPadding=6),
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Category"),
                alt.Tooltip("count:Q", title="Focal-hire spells", format=","),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
                alt.Tooltip("rank:Q", title="Rank", format="d"),
            ],
        )
        _bars = _base.mark_bar(color="#2563EB", opacity=0.85).encode(
            x=alt.X(
                "share:Q",
                title="Share of focal-hire spells",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=_share_domain),
            )
        )
        _labels = _base.mark_text(align="left", baseline="middle", dx=4).encode(
            x=alt.X("share:Q"),
            text=alt.Text("share:Q", format=".1%"),
        )
        return (
            (_bars + _labels)
            .properties(
                width="container",
                height=max(280, top_n * 20),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )

    def country_iso3(country_name):
        """Map common Revelio country labels to ISO alpha-3 codes."""

        _aliases = {
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
        if country_name in _aliases:
            return _aliases[country_name]
        try:
            return pycountry.countries.lookup(str(country_name)).alpha_3
        except LookupError:
            return None

    def us_state_code(state_name):
        """Map delivered U.S. state names to USPS abbreviations."""

        _codes = {
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
        return _codes.get(state_name)

    return (
        MISSING_LABEL,
        country_iso3,
        distribution_table,
        hierarchy_number,
        make_bar_chart,
        us_state_code,
    )


@app.cell(hide_code=True)
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
    EXPECTED_RICS_COLUMNS = (
        "rics_k50",
        "rics_k200",
        "rics_k400",
    )
    REQUIRED_COLUMNS = (
        "country",
        "state",
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

    _dataset = ds.dataset(INPUT_DIR, format="parquet")
    AVAILABLE_COLUMNS = tuple(_dataset.schema.names)
    _missing_required = sorted(set(REQUIRED_COLUMNS) - set(AVAILABLE_COLUMNS))
    if _missing_required:
        raise ValueError(f"Input is missing required fields: {_missing_required}")

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
    return (
        ANALYSIS_COLUMNS,
        AVAILABLE_COLUMNS,
        AVAILABLE_RICS_COLUMNS,
        AVAILABLE_ROLE_COLUMNS,
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        INPUT_DIR,
        PARQUET_FILES,
        fnh,
    )


@app.cell(hide_code=True)
def _(
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    MISSING_LABEL,
    distribution_table,
    fnh,
    hierarchy_number,
    pd,
):
    CLASSIFICATION_LABELS = {
        "onet_code": "O*NET code and title",
        "naics_code": "NAICS code and description",
        **{
            _column: f"Revelio role K{hierarchy_number(_column):,}"
            for _column in AVAILABLE_ROLE_COLUMNS
        },
        **{
            _column: f"Revelio industry K{hierarchy_number(_column):,}"
            for _column in AVAILABLE_RICS_COLUMNS
        },
    }
    _title_columns = {
        "onet_code": "onet_title",
        "naics_code": "naics_description",
    }
    _classification_columns = (
        "onet_code",
        *AVAILABLE_ROLE_COLUMNS,
        *AVAILABLE_RICS_COLUMNS,
        "naics_code",
    )
    distribution_tables = {
        _column: distribution_table(
            fnh,
            _column,
            _title_columns.get(_column),
        )
        for _column in _classification_columns
    }

    _stats_rows = []
    for _column, _summary in distribution_tables.items():
        _missing = _summary.loc[_summary["value"] == MISSING_LABEL, "count"].sum()
        _stats_rows.append(
            {
                "Variable": _column,
                "Classification": CLASSIFICATION_LABELS[_column],
                "Nonmissing categories": int(
                    _summary.loc[_summary["value"] != MISSING_LABEL, "value"].nunique()
                ),
                "Missing rows": int(_missing),
                "Missing share": _missing / len(fnh),
            }
        )
    classification_stats = pd.DataFrame(_stats_rows)

    _onet_pairs = fnh[["onet_code", "onet_title"]].dropna()
    onet_title_diagnostic = (
        _onet_pairs.groupby("onet_code", observed=True)["onet_title"]
        .nunique()
        .rename("distinct_titles")
        .reset_index()
        .sort_values(["distinct_titles", "onet_code"], ascending=[False, True])
    )
    _naics_pairs = fnh[["naics_code", "naics_description"]].dropna()
    naics_title_diagnostic = (
        _naics_pairs.groupby("naics_code", observed=True)["naics_description"]
        .nunique()
        .rename("distinct_descriptions")
        .reset_index()
        .sort_values(["distinct_descriptions", "naics_code"], ascending=[False, True])
    )
    return (
        CLASSIFICATION_LABELS,
        classification_stats,
        distribution_tables,
        naics_title_diagnostic,
        onet_title_diagnostic,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Occupation distribution

    The three views describe the complete O\*NET distribution and examine the mapping
    between O\*NET occupations and Revelio's standardized roles in both directions.
    """)
    return


@app.cell(hide_code=True)
def _(distribution_tables, make_bar_chart, mo):
    selected_occupation_table = distribution_tables["onet_code"]
    _occupation_categories = len(selected_occupation_table)
    occupation_chart = make_bar_chart(
        selected_occupation_table,
        "O*NET code and title",
        _occupation_categories,
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 1.1. O\*NET occupation distribution

                All delivered O\*NET code-title categories are shown. Shares use all
                focal-hire spells as the denominator, including missing classifications.
                """
            ),
            occupation_chart,
            mo.md(
                f"The chart shows all **{_occupation_categories:,}** code-title "
                "combinations, including missing values."
            ),
            mo.accordion(
                {
                    "View all O*NET occupation counts": mo.ui.table(
                        selected_occupation_table,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return occupation_chart, selected_occupation_table


@app.cell(hide_code=True)
def _(MISSING_LABEL, fnh, pd):
    if "role_k1500" not in fnh.columns:
        onet_role_pairs = pd.DataFrame(
            columns=[
                "onet_code",
                "onet_title",
                "onet_label",
                "role_value",
                "count",
                "share_within_onet",
                "role_rank",
                "display_label",
            ]
        )
    else:
        _crosswalk = fnh[["onet_code", "onet_title", "role_k1500"]].copy()
        for _column in ["onet_code", "onet_title", "role_k1500"]:
            _crosswalk[_column] = _crosswalk[_column].fillna(MISSING_LABEL)
        _crosswalk = (
            _crosswalk.groupby(
                ["onet_code", "onet_title", "role_k1500"],
                observed=True,
            )
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={"role_k1500": "role_value"})
        )
        _crosswalk["share_within_onet"] = _crosswalk["count"] / _crosswalk.groupby(
            ["onet_code", "onet_title"],
            observed=True,
        )["count"].transform("sum")
        _crosswalk["onet_label"] = (
            _crosswalk["onet_code"].astype("string")
            + " — "
            + _crosswalk["onet_title"].astype("string")
        )
        _crosswalk["display_label"] = _crosswalk["role_value"].astype("string")
        onet_role_pairs = _crosswalk.sort_values(
            ["onet_code", "onet_title", "count", "role_value"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        onet_role_pairs["role_rank"] = (
            onet_role_pairs.groupby(
                ["onet_code", "onet_title"],
                observed=True,
            ).cumcount()
            + 1
        )
    return onet_role_pairs,


@app.cell(hide_code=True)
def _(MISSING_LABEL, distribution_tables, fnh, mo):
    if "role_k1500" not in fnh.columns:
        onet_role_selector = None
        _onet_role_control = mo.callout(
            mo.md(
                "The local Parquet schema does not contain `role_k1500`. Add this field "
                "to the upstream Fabric extract and rerun the notebook to activate the "
                "O*NET–role cross-classification."
            ),
            kind="warn",
        )
    else:
        _onet_summary = distribution_tables["onet_code"]
        _onet_summary = _onet_summary.loc[
            (_onet_summary["onet_code"] != MISSING_LABEL)
            & (_onet_summary["onet_title"] != MISSING_LABEL)
        ]
        _onet_options = {
            f"{_row.display_label} ({_row.count:,} spells)": _row.display_label
            for _row in _onet_summary.itertuples(index=False)
        }
        onet_role_selector = mo.ui.dropdown(
            options=_onet_options,
            value=next(iter(_onet_options)),
            searchable=True,
            label="O*NET occupation",
            full_width=True,
        )
        _onet_role_control = onet_role_selector

    mo.vstack(
        [
            mo.md(
                r"""
                ### 1.2. O\*NET–Revelio role cross-classification

                For a selected O\*NET occupation, the chart reports every delivered
                `role_k1500` value in descending order of its share within that occupation.
                Missing role values are retained as a category.
                """
            ),
            _onet_role_control,
        ],
        gap=1,
    )
    return onet_role_selector,


@app.cell(hide_code=True)
def _(alt, mo, onet_role_pairs, onet_role_selector):
    if onet_role_selector is None:
        onet_role_chart = None
        selected_onet_roles = onet_role_pairs.copy()
        _onet_role_output = mo.md("")
    else:
        _selected_onet = onet_role_selector.value
        selected_onet_roles = onet_role_pairs.loc[
            onet_role_pairs["onet_label"] == _selected_onet
        ].copy()
        _role_order = selected_onet_roles["display_label"].tolist()
        onet_role_chart = (
            alt.Chart(selected_onet_roles)
            .mark_bar(color="#0F766E", opacity=0.85)
            .encode(
                x=alt.X(
                    "share_within_onet:Q",
                    title="Share within O*NET occupation",
                    axis=alt.Axis(format=".1%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_role_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=[
                    alt.Tooltip("onet_label:N", title="O*NET occupation"),
                    alt.Tooltip("display_label:N", title="Revelio role K1,500"),
                    alt.Tooltip("count:Q", title="Focal-hire spells", format=","),
                    alt.Tooltip(
                        "share_within_onet:Q",
                        title="Within-O*NET share",
                        format=".2%",
                    ),
                    alt.Tooltip("role_rank:Q", title="Rank", format="d"),
                ],
            )
            .properties(
                width="container",
                height=max(320, len(selected_onet_roles) * 20),
                title=alt.TitleParams(
                    text=f"Revelio role composition of {_selected_onet}",
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        _dominant_share = selected_onet_roles.iloc[0]["share_within_onet"]
        _onet_role_output = mo.vstack(
            [
                onet_role_chart,
                mo.md(
                    f"All **{len(selected_onet_roles):,}** role values are shown; the "
                    f"largest accounts for **{_dominant_share:.1%}** of this occupation."
                ),
                mo.accordion(
                    {
                        "View all role values": mo.ui.table(
                            selected_onet_roles,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ],
            gap=1,
        )
    _onet_role_output
    return onet_role_chart, selected_onet_roles


@app.cell(hide_code=True)
def _(onet_role_pairs):
    role_onet_pairs = onet_role_pairs.copy()
    if not role_onet_pairs.empty:
        role_onet_pairs["share_within_role"] = (
            role_onet_pairs["count"]
            / role_onet_pairs.groupby("role_value", observed=True)["count"].transform(
                "sum"
            )
        )
        role_onet_pairs["display_label"] = role_onet_pairs["onet_label"]
        role_onet_pairs = role_onet_pairs.sort_values(
            ["role_value", "count", "onet_code", "onet_title"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)
        role_onet_pairs["onet_rank"] = (
            role_onet_pairs.groupby("role_value", observed=True).cumcount() + 1
        )
    return role_onet_pairs,


@app.cell(hide_code=True)
def _(MISSING_LABEL, distribution_tables, fnh, mo):
    if "role_k1500" not in fnh.columns:
        role_onet_selector = None
        _role_onet_control = mo.callout(
            mo.md("The Role–O*NET diagnostic needs the `role_k1500` field."),
            kind="warn",
        )
    else:
        _role_summary = distribution_tables["role_k1500"]
        _role_summary = _role_summary.loc[_role_summary["value"] != MISSING_LABEL]
        _role_options = {
            f"{_row.value} ({_row.count:,} spells)": _row.value
            for _row in _role_summary.itertuples(index=False)
        }
        role_onet_selector = mo.ui.dropdown(
            options=_role_options,
            value=next(iter(_role_options)),
            searchable=True,
            label="Revelio role K1,500",
            full_width=True,
        )
        _role_onet_control = role_onet_selector

    mo.vstack(
        [
            mo.md(
                r"""
                ### 1.3. Revelio role–O\*NET cross-classification

                For a selected `role_k1500` value, the chart reports every delivered
                O\*NET code-title category in descending order of its share within that
                Revelio role. Missing O\*NET classifications are retained.
                """
            ),
            _role_onet_control,
        ],
        gap=1,
    )
    return role_onet_selector,


@app.cell(hide_code=True)
def _(alt, mo, role_onet_pairs, role_onet_selector):
    if role_onet_selector is None:
        role_onet_chart = None
        selected_role_onet = role_onet_pairs.copy()
        _role_onet_output = mo.md("")
    else:
        _selected_role = role_onet_selector.value
        selected_role_onet = role_onet_pairs.loc[
            role_onet_pairs["role_value"] == _selected_role
        ].copy()
        _onet_order = selected_role_onet["display_label"].tolist()
        role_onet_chart = (
            alt.Chart(selected_role_onet)
            .mark_bar(color="#C2410C", opacity=0.85)
            .encode(
                x=alt.X(
                    "share_within_role:Q",
                    title="Share within Revelio role K1,500",
                    axis=alt.Axis(format=".1%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_onet_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=[
                    alt.Tooltip("role_value:N", title="Revelio role K1,500"),
                    alt.Tooltip("display_label:N", title="O*NET occupation"),
                    alt.Tooltip("count:Q", title="Focal-hire spells", format=","),
                    alt.Tooltip(
                        "share_within_role:Q",
                        title="Within-role share",
                        format=".2%",
                    ),
                    alt.Tooltip("onet_rank:Q", title="Rank", format="d"),
                ],
            )
            .properties(
                width="container",
                height=max(120, len(selected_role_onet) * 20),
                title=alt.TitleParams(
                    text=f"O*NET composition of {_selected_role}",
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        _dominant_onet_share = selected_role_onet.iloc[0]["share_within_role"]
        _role_onet_output = mo.vstack(
            [
                role_onet_chart,
                mo.md(
                    rf"All **{len(selected_role_onet):,}** O\*NET categories are shown; "
                    f"the largest accounts for **{_dominant_onet_share:.1%}** of this role."
                ),
                mo.accordion(
                    {
                        "View all O*NET categories": mo.ui.table(
                            selected_role_onet,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ],
            gap=1,
        )
    _role_onet_output
    return role_onet_chart, selected_role_onet


@app.cell(hide_code=True)
def _(AVAILABLE_RICS_COLUMNS, mo):
    _rics_note = (
        f"Available fields: {', '.join(f'`{column}`' for column in AVAILABLE_RICS_COLUMNS)}."
        if AVAILABLE_RICS_COLUMNS
        else "No `rics_k*` fields are available in the delivered extract."
    )
    mo.md(
        f"""
        ## 2. Industry distribution

        Choose between the available Revelio industry hierarchy fields and NAICS.
        Code-description pairs are kept together so label inconsistencies remain visible.

        {_rics_note}
        """
    )
    return


@app.cell(hide_code=True)
def _(AVAILABLE_RICS_COLUMNS, CLASSIFICATION_LABELS, mo):
    _industry_columns = [*AVAILABLE_RICS_COLUMNS, "naics_code"]
    _industry_options = {
        CLASSIFICATION_LABELS[_column]: _column for _column in _industry_columns
    }
    industry_selector = mo.ui.dropdown(
        options=_industry_options,
        value=next(iter(_industry_options)),
        searchable=True,
        label="Industry classification",
    )
    industry_top_n = mo.ui.slider(
        start=10,
        stop=30,
        step=5,
        value=20,
        show_value=True,
        label="Categories shown",
    )
    mo.hstack(
        [industry_selector, industry_top_n],
        justify="start",
        align="end",
        gap=2,
        wrap=True,
    )
    return industry_selector, industry_top_n


@app.cell(hide_code=True)
def _(
    CLASSIFICATION_LABELS,
    distribution_tables,
    industry_selector,
    industry_top_n,
    make_bar_chart,
    mo,
):
    selected_industry_column = industry_selector.value
    selected_industry_table = distribution_tables[selected_industry_column]
    _industry_coverage = selected_industry_table.head(industry_top_n.value)["share"].sum()
    _industry_categories = len(selected_industry_table)
    industry_chart = make_bar_chart(
        selected_industry_table,
        CLASSIFICATION_LABELS[selected_industry_column],
        industry_top_n.value,
    )
    mo.vstack(
        [
            industry_chart,
            mo.md(
                f"Displayed categories cover **{_industry_coverage:.1%}** of all "
                f"spells; the aggregated table contains {_industry_categories:,} "
                "code-label combinations including missing values."
            ),
            mo.accordion(
                {
                    "View industry counts": mo.ui.table(
                        selected_industry_table,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return industry_chart, selected_industry_column, selected_industry_table


@app.cell(hide_code=True)
def _(AVAILABLE_RICS_COLUMNS, MISSING_LABEL, fnh, pd):
    finest_rics_column = (
        AVAILABLE_RICS_COLUMNS[-1] if AVAILABLE_RICS_COLUMNS else None
    )
    if finest_rics_column is None:
        rics_naics_pairs = pd.DataFrame(
            columns=[
                "rics_value",
                "naics_code",
                "naics_description",
                "count",
                "share_within_rics",
            ]
        )
    else:
        _crosswalk = fnh[
            [finest_rics_column, "naics_code", "naics_description"]
        ].copy()
        for _column in [finest_rics_column, "naics_code", "naics_description"]:
            _crosswalk[_column] = _crosswalk[_column].fillna(MISSING_LABEL)
        _crosswalk = (
            _crosswalk.groupby(
                [finest_rics_column, "naics_code", "naics_description"],
                observed=True,
            )
            .size()
            .rename("count")
            .reset_index()
            .rename(columns={finest_rics_column: "rics_value"})
        )
        _crosswalk["share_within_rics"] = _crosswalk["count"] / _crosswalk.groupby(
            "rics_value",
            observed=True,
        )["count"].transform("sum")
        _crosswalk["display_label"] = (
            _crosswalk["naics_code"].astype("string")
            + " — "
            + _crosswalk["naics_description"].astype("string")
        )
        rics_naics_pairs = _crosswalk.sort_values(
            ["rics_value", "count", "naics_code"],
            ascending=[True, False, True],
        ).reset_index(drop=True)
    return finest_rics_column, rics_naics_pairs


@app.cell(hide_code=True)
def _(
    MISSING_LABEL,
    distribution_tables,
    finest_rics_column,
    mo,
):
    if finest_rics_column is None:
        rics_detail_selector = None
        _rics_control = mo.callout(
            mo.md("The RICS–NAICS diagnostic needs an available `rics_k*` field."),
            kind="warn",
        )
    else:
        _rics_summary = distribution_tables[finest_rics_column]
        _rics_summary = _rics_summary.loc[_rics_summary["value"] != MISSING_LABEL]
        _rics_options = {
            f"{_row.value} ({_row.count:,} spells)": _row.value
            for _row in _rics_summary.itertuples(index=False)
        }
        rics_detail_selector = mo.ui.dropdown(
            options=_rics_options,
            value=next(iter(_rics_options)),
            searchable=True,
            label=f"{finest_rics_column} category",
            full_width=True,
        )
        _rics_control = rics_detail_selector

    mo.vstack(
        [
            mo.md(
                """
                ### 2.1. RICS–NAICS cross-classification

                For a selected category in the finest available RICS hierarchy, the chart
                reports its leading NAICS code-description pairs. Concentration is a
                descriptive diagnostic, not a test that the taxonomies should coincide.
                """
            ),
            _rics_control,
        ],
        gap=1,
    )
    return rics_detail_selector,


@app.cell(hide_code=True)
def _(alt, finest_rics_column, mo, rics_detail_selector, rics_naics_pairs):
    if rics_detail_selector is None:
        rics_naics_chart = None
        selected_rics_naics = rics_naics_pairs.copy()
        _rics_output = mo.md("")
    else:
        selected_rics_value = rics_detail_selector.value
        selected_rics_naics = rics_naics_pairs.loc[
            rics_naics_pairs["rics_value"] == selected_rics_value
        ].copy()
        _top = selected_rics_naics.head(20)
        _order = _top["display_label"].tolist()
        rics_naics_chart = (
            alt.Chart(_top)
            .mark_bar(color="#7C3AED", opacity=0.85)
            .encode(
                x=alt.X(
                    "share_within_rics:Q",
                    title=f"Share within {finest_rics_column} category",
                    axis=alt.Axis(format=".1%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=[
                    alt.Tooltip("rics_value:N", title=finest_rics_column),
                    alt.Tooltip("display_label:N", title="NAICS"),
                    alt.Tooltip("count:Q", title="Focal-hire spells", format=","),
                    alt.Tooltip(
                        "share_within_rics:Q",
                        title="Within-RICS share",
                        format=".2%",
                    ),
                ],
            )
            .properties(
                width="container",
                height=420,
                title=alt.TitleParams(
                    text=f"NAICS composition of {selected_rics_value}",
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        _dominant_share = selected_rics_naics.iloc[0]["share_within_rics"]
        _top_coverage = selected_rics_naics.head(20)["share_within_rics"].sum()
        _rics_output = mo.vstack(
            [
                rics_naics_chart,
                mo.md(
                    f"The largest NAICS pair accounts for **{_dominant_share:.1%}** "
                    f"of this RICS category; the displayed pairs cover "
                    f"**{_top_coverage:.1%}**."
                ),
                mo.accordion(
                    {
                        "View all NAICS pairs": mo.ui.table(
                            selected_rics_naics,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ],
            gap=1,
        )
    _rics_output
    return rics_naics_chart, selected_rics_naics


@app.cell(hide_code=True)
def _(rics_naics_pairs):
    naics_rics_pairs = rics_naics_pairs.copy()
    if not naics_rics_pairs.empty:
        naics_rics_pairs["naics_label"] = naics_rics_pairs["display_label"]
        naics_rics_pairs["share_within_naics"] = (
            naics_rics_pairs["count"]
            / naics_rics_pairs.groupby(
                ["naics_code", "naics_description"],
                observed=True,
            )["count"].transform("sum")
        )
        naics_rics_pairs["display_label"] = naics_rics_pairs["rics_value"].astype(
            "string"
        )
        naics_rics_pairs = naics_rics_pairs.sort_values(
            ["naics_code", "naics_description", "count", "rics_value"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        naics_rics_pairs["rics_rank"] = (
            naics_rics_pairs.groupby(
                ["naics_code", "naics_description"],
                observed=True,
            ).cumcount()
            + 1
        )
    return naics_rics_pairs,


@app.cell(hide_code=True)
def _(MISSING_LABEL, distribution_tables, finest_rics_column, mo):
    if finest_rics_column is None:
        naics_rics_selector = None
        _naics_rics_control = mo.callout(
            mo.md("The NAICS–RICS diagnostic needs an available `rics_k*` field."),
            kind="warn",
        )
    else:
        _naics_summary = distribution_tables["naics_code"]
        _naics_summary = _naics_summary.loc[
            (_naics_summary["naics_code"] != MISSING_LABEL)
            & (_naics_summary["naics_description"] != MISSING_LABEL)
        ]
        _naics_options = {
            f"{_row.display_label} ({_row.count:,} spells)": _row.display_label
            for _row in _naics_summary.itertuples(index=False)
        }
        naics_rics_selector = mo.ui.dropdown(
            options=_naics_options,
            value=next(iter(_naics_options)),
            searchable=True,
            label="NAICS code and description",
            full_width=True,
        )
        _naics_rics_control = naics_rics_selector

    mo.vstack(
        [
            mo.md(
                """
                ### 2.2. NAICS–RICS cross-classification

                For a selected NAICS code-description category, the chart reports every
                delivered category in the finest available RICS hierarchy in descending
                order of its share within that NAICS category. Missing RICS values are
                retained.
                """
            ),
            _naics_rics_control,
        ],
        gap=1,
    )
    return naics_rics_selector,


@app.cell(hide_code=True)
def _(alt, finest_rics_column, mo, naics_rics_pairs, naics_rics_selector):
    if naics_rics_selector is None:
        naics_rics_chart = None
        selected_naics_rics = naics_rics_pairs.copy()
        _naics_rics_output = mo.md("")
    else:
        _selected_naics = naics_rics_selector.value
        selected_naics_rics = naics_rics_pairs.loc[
            naics_rics_pairs["naics_label"] == _selected_naics
        ].copy()
        _rics_order = selected_naics_rics["display_label"].tolist()
        naics_rics_chart = (
            alt.Chart(selected_naics_rics)
            .mark_bar(color="#B45309", opacity=0.85)
            .encode(
                x=alt.X(
                    "share_within_naics:Q",
                    title="Share within NAICS category",
                    axis=alt.Axis(format=".1%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_rics_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=[
                    alt.Tooltip("naics_label:N", title="NAICS"),
                    alt.Tooltip("display_label:N", title=finest_rics_column),
                    alt.Tooltip("count:Q", title="Focal-hire spells", format=","),
                    alt.Tooltip(
                        "share_within_naics:Q",
                        title="Within-NAICS share",
                        format=".2%",
                    ),
                    alt.Tooltip("rics_rank:Q", title="Rank", format="d"),
                ],
            )
            .properties(
                width="container",
                height=max(320, len(selected_naics_rics) * 20),
                title=alt.TitleParams(
                    text=f"{finest_rics_column} composition of {_selected_naics}",
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
        _dominant_rics_share = selected_naics_rics.iloc[0]["share_within_naics"]
        _naics_rics_output = mo.vstack(
            [
                naics_rics_chart,
                mo.md(
                    f"All **{len(selected_naics_rics):,}** RICS categories are shown; "
                    f"the largest accounts for **{_dominant_rics_share:.1%}** of this "
                    "NAICS category."
                ),
                mo.accordion(
                    {
                        "View all RICS categories": mo.ui.table(
                            selected_naics_rics,
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ],
            gap=1,
        )
    _naics_rics_output
    return naics_rics_chart, selected_naics_rics


@app.cell(hide_code=True)
def _(country_iso3, distribution_table, fnh, math):
    country_summary = distribution_table(fnh, "country")
    country_summary["iso3"] = country_summary["country"].map(country_iso3)
    country_summary["log10_count"] = country_summary["count"].map(math.log10)
    mapped_country_summary = country_summary.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = country_summary.loc[
        country_summary["iso3"].isna()
    ].copy()

    _state_working = fnh.loc[
        fnh["country"] == "United States",
        ["state"],
    ].copy()
    _state_working["state"] = _state_working["state"].fillna("<Missing>")
    us_state_summary = (
        _state_working.groupby("state", observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    us_state_summary["share_within_country"] = (
        us_state_summary["count"] / us_state_summary["count"].sum()
    )
    return (
        country_summary,
        mapped_country_summary,
        unmapped_country_summary,
        us_state_summary,
    )


@app.cell(hide_code=True)
def _(mapped_country_summary, mo, px, unmapped_country_summary):
    mo.md(
        """
        ## 3. Geographic distribution

        The world map uses ISO-3 country locations and a log10 color scale so countries
        outside the largest markets remain distinguishable. Hover labels report raw counts
        and shares.
        """
    )
    world_country_map = px.choropleth(
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
            "count": "Focal-hire spells",
            "share": "Global share",
            "log10_count": "log10 spells",
        },
        color_continuous_scale="Blues",
        projection="natural earth",
        title="Focal-hire spells by country",
    )
    world_country_map.update_geos(showframe=False, showcoastlines=True)
    world_country_map.update_layout(
        height=560,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
    )
    _unmapped_count = unmapped_country_summary["count"].sum()
    _mapped_count = mapped_country_summary["count"].sum()
    _iso_coverage = _mapped_count / (_mapped_count + _unmapped_count)
    mo.vstack(
        [
            world_country_map,
            mo.md(f"ISO mapping covers **{_iso_coverage:.2%}** of focal-hire spells."),
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
    return world_country_map,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.1. First-level administrative distribution

    This subsection is restricted to the United States. State shares use all U.S.
    focal-hire spells as the denominator, including unmatched or missing state labels.
    """)
    return


@app.cell(hide_code=True)
def _(px, us_state_code, us_state_summary):
    _us_states = us_state_summary.copy()
    _us_states["state_code"] = _us_states["state"].map(us_state_code)
    state_map_data = _us_states.dropna(subset=["state_code"]).copy()
    unmatched_state_data = _us_states.loc[_us_states["state_code"].isna()].sort_values(
        "count",
        ascending=False,
    )
    _us_total = _us_states["count"].sum()
    state_map_coverage = (
        state_map_data["count"].sum() / _us_total if _us_total else 0.0
    )

    if state_map_data.empty:
        state_map_figure = None
    else:
        state_map_figure = px.choropleth(
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
                "count": "Focal-hire spells",
                "share_within_country": "U.S. share",
            },
            color_continuous_scale="Blues",
            title="Focal-hire spells by U.S. state",
        )
        state_map_figure.update_geos(scope="usa", visible=False)
        state_map_figure.update_layout(
            height=600,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
        )
    return (
        state_map_coverage,
        state_map_data,
        state_map_figure,
        unmatched_state_data,
    )


@app.cell(hide_code=True)
def _(mo, state_map_coverage, state_map_figure, unmatched_state_data):
    if state_map_figure is None:
        _map_output = mo.callout(
            mo.md(
                "No delivered U.S. state labels matched the USPS state-code mapping."
            ),
            kind="warn",
        )
    else:
        _map_output = state_map_figure

    mo.vstack(
        [
            _map_output,
            mo.md(
                f"The USPS state-code mapping covers **{state_map_coverage:.2%}** of "
                "U.S. focal-hire spells."
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


@app.cell(hide_code=True)
def _(
    classification_stats,
    mo,
    naics_title_diagnostic,
    onet_title_diagnostic,
):
    _onet_conflicts = int((onet_title_diagnostic["distinct_titles"] > 1).sum())
    _naics_conflicts = int(
        (naics_title_diagnostic["distinct_descriptions"] > 1).sum()
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ## 4. Diagnostics

                - O\\*NET codes associated with multiple delivered titles:
                  **{_onet_conflicts:,}**
                - NAICS codes associated with multiple delivered descriptions:
                  **{_naics_conflicts:,}**
                """
            ),
            mo.accordion(
                {
                    "Classification coverage": mo.ui.table(
                        classification_stats,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "O*NET code-title diagnostic": mo.ui.table(
                        onet_title_diagnostic,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "NAICS code-description diagnostic": mo.ui.table(
                        naics_title_diagnostic,
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


@app.cell(hide_code=True)
def _(
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    MISSING_LABEL,
    PARQUET_FILES,
    fnh,
    mo,
    pd,
):
    _expected = [
        "onet_code",
        "onet_title",
        *EXPECTED_ROLE_COLUMNS,
        *EXPECTED_RICS_COLUMNS,
        "naics_code",
        "naics_description",
        "country",
        "state",
    ]
    _rows = []
    for _column in _expected:
        if _column in fnh.columns:
            _missing_count = int(fnh[_column].isna().sum())
            _rows.append(
                {
                    "Variable": _column,
                    "Status": "Available",
                    "Missing rows": _missing_count,
                    "Missing share": _missing_count / len(fnh),
                }
            )
        else:
            _rows.append(
                {
                    "Variable": _column,
                    "Status": "Absent from input schema",
                    "Missing rows": pd.NA,
                    "Missing share": pd.NA,
                }
            )
    schema_report = pd.DataFrame(_rows)

    _memory_mib = fnh.memory_usage(deep=True).sum() / 1024**2
    _missing_roles = sorted(set(EXPECTED_ROLE_COLUMNS) - set(AVAILABLE_ROLE_COLUMNS))
    _missing_rics = sorted(set(EXPECTED_RICS_COLUMNS) - set(AVAILABLE_RICS_COLUMNS))
    if _missing_roles or _missing_rics:
        _schema_message = (
            f"Missing requested role fields: {', '.join(_missing_roles) or 'none'}. "
            f"Missing requested RICS fields: {', '.join(_missing_rics) or 'none'}."
        )
        _schema_callout = mo.callout(mo.md(_schema_message), kind="warn")
    else:
        _schema_callout = mo.callout(
            mo.md("All requested role and industry hierarchy fields are available."),
            kind="success",
        )

    mo.vstack(
        [
            mo.md(
                f"""
                ## Sample and schema

                - **Focal-hire spells:** {len(fnh):,}
                - **Parquet parts:** {len(PARQUET_FILES):,}
                - **Countries:** {fnh['country'].nunique(dropna=True):,}
                - **Pandas memory for selected columns:** {_memory_mib:,.1f} MiB
                - **Missing marker in charts:** `{MISSING_LABEL}`
                """
            ),
            _schema_callout,
            mo.accordion(
                {
                    "View requested-variable coverage": mo.ui.table(
                        schema_report,
                        pagination=False,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return schema_report,


if __name__ == "__main__":
    app.run()

# ruff: noqa: B018, PLR1711

"""
Task:
    Characterize inventor-linkage rates in the candidate focal-new-hire sample.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHires_AllIndustries/*.parquet
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) Interactive marimo views of occupation, industry, industry-occupation, country,
    and U.S.-state inventor match rates.

Notes:
(1) The unit is a focal-hire spell: one retained spell per user-company pair.
(2) The inventor crosswalk is deduplicated to unique nonmissing user IDs before matching.
(3) Only linkage and requested analysis columns are read; strings are held as categoricals.
(4) Display thresholds affect charts and maps, not the complete downloadable tables.

Run:
    $fnh_match_notebook = "codes/B01_ConstructAnalysisSample/B03_FNH_InventorMatchRates.py"
    conda run -s -n Talent marimo edit $fnh_match_notebook

Wang Wenzhi, with the help of Codex
Time: 2026-08-24
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell(hide_code=True)
def imports():
    import math
    import re

    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import polars as pl
    import pyarrow.dataset as ds
    import pycountry

    return alt, ds, math, mo, pd, pl, px, pycountry, re


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # Inventor match rates among candidate focal new hires

    This notebook measures how often focal-hire spells can be linked to a USPTO
    inventor identifier. It supports the full sample, the United States, the non-U.S.
    sample, and user-selected countries or country sets.

    The main estimand is the **unweighted focal-hire-spell match rate**. The inventor
    crosswalk is first collapsed to unique, nonmissing `user_id` values, so a user linked
    to multiple inventor IDs cannot duplicate focal-hire spells. A matched user's status
    is attached to each of their retained user-company focal-hire spells.
    """)
    return


@app.cell(hide_code=True)
def helpers(alt, pd, pl, pycountry, re):
    MISSING_LABEL = "<Missing>"
    US_LABEL = "United States"

    def hierarchy_number(column_name):
        """Return the numeric K level used to sort Revelio hierarchy fields."""

        _match = re.search(r"_k(\d+)$", column_name)
        return int(_match.group(1)) if _match else -1

    def rate_aggregations():
        """Return the common spell- and user-level aggregation expressions."""

        return [
            pl.len().alias("candidate_spells"),
            pl.col("inventor_match").sum().cast(pl.Int64).alias("matched_spells"),
            pl.col("user_id").n_unique().alias("unique_users"),
            pl.col("user_id")
            .filter(pl.col("inventor_match"))
            .n_unique()
            .alias("matched_users"),
        ]

    def add_rate_statistics(summary):
        """Add match rates and 95% Wilson intervals to a grouped pandas table."""

        _result = summary.copy()
        if _result.empty:
            for _column in ["match_rate", "user_match_rate", "ci_low", "ci_high"]:
                _result[_column] = pd.Series(dtype="float64")
            return _result

        _n = _result["candidate_spells"].astype(float)
        _p = _result["matched_spells"] / _n
        _z = 1.96
        _denominator = 1.0 + _z**2 / _n
        _center = (_p + _z**2 / (2.0 * _n)) / _denominator
        _margin = (
            _z * (_p * (1.0 - _p) / _n + _z**2 / (4.0 * _n**2)) ** 0.5 / _denominator
        )
        _result["match_rate"] = _p
        _result["user_match_rate"] = _result["matched_users"] / _result["unique_users"]
        _result["ci_low"] = (_center - _margin).clip(lower=0.0)
        _result["ci_high"] = (_center + _margin).clip(upper=1.0)
        return _result

    def grouped_match_rates(data, group_columns):
        """Aggregate inventor coverage over one or more categorical columns."""

        _summary = (
            data.group_by(list(group_columns), maintain_order=False)
            .agg(rate_aggregations())
            .sort("candidate_spells", descending=True)
            .to_pandas()
        )
        return add_rate_statistics(_summary)

    def classification_match_rates(data, value_column, title_column=None):
        """Aggregate one classification and attach a stable display label."""

        _aggregations = rate_aggregations()
        if title_column is not None:
            _aggregations.append(pl.col(title_column).first().alias(title_column))

        _summary = (
            data.group_by(value_column, maintain_order=False)
            .agg(_aggregations)
            .sort("candidate_spells", descending=True)
            .to_pandas()
        )
        _summary = add_rate_statistics(_summary)
        _summary["group_value"] = _summary[value_column].astype(str)
        if title_column is None:
            _summary["display_label"] = _summary["group_value"]
        else:
            _summary["display_label"] = (
                _summary["group_value"] + " — " + _summary[title_column].astype(str)
            )
        return _summary

    def make_rate_chart(
        summary,
        title,
        overall_rate,
        top_n,
        min_count,
        color,
    ):
        """Plot rates and Wilson intervals for the largest eligible categories."""

        _eligible = summary.loc[summary["candidate_spells"] >= min_count].copy()
        _shown = (
            _eligible.nlargest(top_n, "candidate_spells")
            .sort_values(["match_rate", "candidate_spells"], ascending=[False, False])
            .reset_index(drop=True)
        )
        if _shown.empty:
            return None, _shown

        _x_max = max(
            float(_shown["ci_high"].max()) * 1.12,
            overall_rate * 1.05,
            0.01,
        )
        _order = _shown["display_label"].tolist()
        _base = alt.Chart(_shown).encode(
            y=alt.Y(
                "display_label:N",
                sort=_order,
                title=None,
                axis=alt.Axis(labelLimit=430, labelPadding=6),
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Category"),
                alt.Tooltip("candidate_spells:Q", title="Candidate spells", format=","),
                alt.Tooltip("matched_spells:Q", title="Matched spells", format=","),
                alt.Tooltip("match_rate:Q", title="Match rate", format=".2%"),
                alt.Tooltip("ci_low:Q", title="95% CI lower", format=".2%"),
                alt.Tooltip("ci_high:Q", title="95% CI upper", format=".2%"),
                alt.Tooltip("unique_users:Q", title="Unique users", format=","),
                alt.Tooltip(
                    "user_match_rate:Q",
                    title="Unique-user match rate",
                    format=".2%",
                ),
            ],
        )
        _intervals = _base.mark_rule(color=color, opacity=0.55).encode(
            x=alt.X(
                "ci_low:Q",
                title="Inventor match rate (95% Wilson interval)",
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=[0.0, _x_max]),
            ),
            x2="ci_high:Q",
        )
        _points = _base.mark_point(
            color=color,
            filled=True,
            size=75,
        ).encode(x=alt.X("match_rate:Q"))
        _labels = _base.mark_text(
            align="left",
            baseline="middle",
            dx=6,
            color="#111827",
        ).encode(
            x=alt.X("ci_high:Q"),
            text=alt.Text("match_rate:Q", format=".1%"),
        )
        _reference = (
            alt.Chart(pd.DataFrame({"overall_rate": [overall_rate]}))
            .mark_rule(color="#6B7280", strokeDash=[6, 4])
            .encode(x=alt.X("overall_rate:Q"))
        )
        _chart = (
            (_reference + _intervals + _points + _labels)
            .properties(
                width="container",
                height=max(300, len(_shown) * 22),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )
        return _chart, _shown

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
        """Map delivered U.S. state labels to USPS abbreviations."""

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
        US_LABEL,
        classification_match_rates,
        country_iso3,
        grouped_match_rates,
        hierarchy_number,
        make_rate_chart,
        us_state_code,
    )


@app.cell(hide_code=True)
def paths_and_schema(ds, hierarchy_number, mo):
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
    EXPECTED_RICS_COLUMNS = (
        "rics_k50",
        "rics_k200",
        "rics_k400",
    )
    REQUIRED_COLUMNS = (
        "user_id",
        "country",
        "state",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
    )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")
    if not CROSSWALK_PATH.exists():
        raise FileNotFoundError(f"Inventor crosswalk does not exist: {CROSSWALK_PATH}")

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
            (_column for _column in AVAILABLE_COLUMNS if _column.startswith("role_k")),
            key=hierarchy_number,
        )
    )
    AVAILABLE_RICS_COLUMNS = tuple(
        sorted(
            (_column for _column in AVAILABLE_COLUMNS if _column.startswith("rics_k")),
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
    return (
        ANALYSIS_COLUMNS,
        AVAILABLE_RICS_COLUMNS,
        AVAILABLE_ROLE_COLUMNS,
        CROSSWALK_PATH,
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        INPUT_DIR,
        PARQUET_FILES,
    )


@app.cell(hide_code=True)
def load_data(
    ANALYSIS_COLUMNS,
    CROSSWALK_PATH,
    INPUT_DIR,
    MISSING_LABEL,
    pd,
    pl,
):
    _patent_links = pl.read_csv(
        CROSSWALK_PATH,
        columns=["user_id", "pv_inventor_id"],
        schema_overrides={
            "user_id": pl.Int64,
            "pv_inventor_id": pl.String,
        },
    )
    _link_user_counts = _patent_links.group_by("user_id").len()
    _patent_users = _patent_links.select("user_id").drop_nulls().unique()
    link_diagnostics = pd.DataFrame(
        [
            {
                "Crosswalk rows": len(_patent_links),
                "Unique linked users": _patent_links["user_id"].n_unique(),
                "Unique inventor IDs": _patent_links["pv_inventor_id"].n_unique(),
                "Users with multiple rows": int(
                    _link_user_counts.select((pl.col("len") > 1).sum()).item()
                ),
                "Maximum rows per user": int(_link_user_counts["len"].max()),
                "Missing user IDs": int(_patent_links["user_id"].null_count()),
            }
        ]
    )

    _string_columns = tuple(
        _column for _column in ANALYSIS_COLUMNS if _column != "user_id"
    )
    _matched_users = _patent_users.lazy().with_columns(
        pl.lit(True).alias("inventor_match")
    )
    fnh = (
        pl.scan_parquet(str(INPUT_DIR / "*.parquet"))
        .select(list(ANALYSIS_COLUMNS))
        .join(_matched_users, on="user_id", how="left")
        .with_columns(
            pl.col("inventor_match").fill_null(False),
            *[
                pl.col(_column).fill_null(MISSING_LABEL).cast(pl.Categorical)
                for _column in _string_columns
            ],
        )
        .collect(engine="streaming")
    )
    return fnh, link_diagnostics


@app.cell(hide_code=True)
def classifications(
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    hierarchy_number,
):
    OCCUPATION_LABELS = {
        "onet_code": "O*NET code and title",
        **{
            _column: f"Revelio role K{hierarchy_number(_column):,}"
            for _column in AVAILABLE_ROLE_COLUMNS
        },
    }
    OCCUPATION_TITLES = {"onet_code": "onet_title"}
    INDUSTRY_LABELS = {
        "naics_code": "NAICS code and description",
        **{
            _column: f"Revelio industry K{hierarchy_number(_column):,}"
            for _column in AVAILABLE_RICS_COLUMNS
        },
    }
    INDUSTRY_TITLES = {"naics_code": "naics_description"}
    DEFAULT_INDUSTRY = (
        "rics_k400" if "rics_k400" in INDUSTRY_LABELS else next(iter(INDUSTRY_LABELS))
    )
    return (
        DEFAULT_INDUSTRY,
        INDUSTRY_LABELS,
        INDUSTRY_TITLES,
        OCCUPATION_LABELS,
        OCCUPATION_TITLES,
    )


@app.cell(hide_code=True)
def controls(
    DEFAULT_INDUSTRY,
    INDUSTRY_LABELS,
    MISSING_LABEL,
    OCCUPATION_LABELS,
    fnh,
    mo,
    pl,
):
    COUNTRY_OPTIONS = sorted(
        _country
        for _country in fnh["country"].cast(pl.String).unique().to_list()
        if _country != MISSING_LABEL
    )
    SCOPE_LABELS = {
        "full": "Full sample",
        "us": "United States only",
        "non_us": "Non-U.S. only",
        "custom": "Selected countries",
    }
    scope_selector = mo.ui.dropdown(
        options={_label: _value for _value, _label in SCOPE_LABELS.items()},
        value="Full sample",
        label="Sample scope",
        full_width=True,
    )
    custom_country_selector = mo.ui.multiselect(
        options=COUNTRY_OPTIONS,
        value=[],
        label="Countries used when scope is ‘Selected countries’",
        full_width=True,
    )
    occupation_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in OCCUPATION_LABELS.items()},
        value=OCCUPATION_LABELS["onet_code"],
        label="Occupation classification",
        full_width=True,
    )
    industry_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in INDUSTRY_LABELS.items()},
        value=INDUSTRY_LABELS[DEFAULT_INDUSTRY],
        label="Industry classification",
        full_width=True,
    )
    top_n_selector = mo.ui.slider(
        start=10,
        stop=100,
        step=5,
        value=40,
        show_value=True,
        label="Maximum categories in ranked charts",
        full_width=True,
    )
    min_group_size_selector = mo.ui.slider(
        steps=[1, 10, 25, 50, 100, 250, 500, 1_000, 2_500, 5_000],
        value=100,
        show_value=True,
        label="Minimum focal-hire spells per displayed group",
        full_width=True,
    )
    joint_axis_size_selector = mo.ui.slider(
        start=5,
        stop=30,
        step=5,
        value=15,
        show_value=True,
        label="Largest industries and occupations on heatmap axes",
        full_width=True,
    )
    joint_min_size_selector = mo.ui.slider(
        steps=[1, 5, 10, 25, 50, 100, 250, 500, 1_000],
        value=25,
        show_value=True,
        label="Minimum focal-hire spells per heatmap cell",
        full_width=True,
    )
    return (
        SCOPE_LABELS,
        custom_country_selector,
        industry_selector,
        joint_axis_size_selector,
        joint_min_size_selector,
        min_group_size_selector,
        occupation_selector,
        scope_selector,
        top_n_selector,
    )


@app.cell(hide_code=True)
def show_controls(
    custom_country_selector,
    industry_selector,
    joint_axis_size_selector,
    joint_min_size_selector,
    min_group_size_selector,
    mo,
    occupation_selector,
    scope_selector,
    top_n_selector,
):
    mo.vstack(
        [
            mo.md("## Analysis controls"),
            scope_selector,
            custom_country_selector,
            mo.md(
                "Country selections are applied only when **Selected countries** is "
                "the active scope. Missing country labels remain in the full sample but "
                "are excluded from the U.S., non-U.S., and custom scopes."
            ),
            occupation_selector,
            industry_selector,
            top_n_selector,
            min_group_size_selector,
            joint_axis_size_selector,
            joint_min_size_selector,
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def select_sample(
    MISSING_LABEL,
    SCOPE_LABELS,
    US_LABEL,
    custom_country_selector,
    fnh,
    math,
    pl,
    scope_selector,
):
    selected_countries = tuple(custom_country_selector.value)
    _scope_value = scope_selector.value
    if _scope_value == "full":
        selected_fnh = fnh
    elif _scope_value == "us":
        selected_fnh = fnh.filter(pl.col("country") == US_LABEL)
    elif _scope_value == "non_us":
        selected_fnh = fnh.filter(
            (pl.col("country") != US_LABEL) & (pl.col("country") != MISSING_LABEL)
        )
    else:
        selected_fnh = fnh.filter(pl.col("country").is_in(selected_countries))

    selected_scope_label = SCOPE_LABELS[_scope_value]
    _sample_row = selected_fnh.select(
        pl.len().alias("candidate_spells"),
        pl.col("inventor_match").sum().cast(pl.Int64).alias("matched_spells"),
        pl.col("user_id").n_unique().alias("unique_users"),
        pl.col("user_id")
        .filter(pl.col("inventor_match"))
        .n_unique()
        .alias("matched_users"),
        pl.col("country")
        .filter(pl.col("country") != MISSING_LABEL)
        .n_unique()
        .alias("countries"),
    ).to_dicts()[0]
    _sample_row["match_rate"] = (
        _sample_row["matched_spells"] / _sample_row["candidate_spells"]
        if _sample_row["candidate_spells"]
        else math.nan
    )
    _sample_row["user_match_rate"] = (
        _sample_row["matched_users"] / _sample_row["unique_users"]
        if _sample_row["unique_users"]
        else math.nan
    )
    sample_metrics = _sample_row
    return sample_metrics, selected_fnh, selected_scope_label


@app.cell(hide_code=True)
def sample_overview(mo, sample_metrics, selected_scope_label):
    if sample_metrics["candidate_spells"] == 0:
        _overview = mo.callout(
            mo.md(
                "The active sample is empty. Choose at least one country when using "
                "the **Selected countries** scope."
            ),
            kind="warn",
        )
    else:
        _overview = mo.md(
            f"""
            ## Active sample: {selected_scope_label}

            - **Focal-hire spells:** {sample_metrics["candidate_spells"]:,}
            - **Matched spells:** {sample_metrics["matched_spells"]:,}
            - **Spell match rate:** {sample_metrics["match_rate"]:.2%}
            - **Unique users:** {sample_metrics["unique_users"]:,}
            - **Matched unique users:** {sample_metrics["matched_users"]:,}
            - **Unique-user match rate:** {sample_metrics["user_match_rate"]:.2%}
            - **Nonmissing countries represented:** {sample_metrics["countries"]:,}

            The chart reference line marks the active sample's overall spell match rate.
            Confidence intervals are descriptive binomial intervals; they do not account
            for repeated users across focal-hire spells.
            """
        )
    _overview
    return


@app.cell(hide_code=True)
def occupation_rates(
    OCCUPATION_LABELS,
    OCCUPATION_TITLES,
    classification_match_rates,
    make_rate_chart,
    min_group_size_selector,
    occupation_selector,
    sample_metrics,
    selected_fnh,
    top_n_selector,
):
    occupation_column = occupation_selector.value
    occupation_title = OCCUPATION_LABELS[occupation_column]
    occupation_summary = classification_match_rates(
        selected_fnh,
        occupation_column,
        OCCUPATION_TITLES.get(occupation_column),
    )
    occupation_chart, _occupation_display = make_rate_chart(
        occupation_summary,
        f"Inventor match rates by {occupation_title}",
        sample_metrics["match_rate"],
        top_n_selector.value,
        min_group_size_selector.value,
        "#2563EB",
    )
    return (
        occupation_chart,
        occupation_column,
        occupation_summary,
        occupation_title,
    )


@app.cell(hide_code=True)
def occupation_output(mo, occupation_chart, occupation_summary):
    if occupation_chart is None:
        _occupation_figure = mo.callout(
            mo.md("No occupation meets the current sample and denominator settings."),
            kind="warn",
        )
    else:
        _occupation_figure = occupation_chart

    mo.vstack(
        [
            mo.md(
                "## 1. Occupation match rates\n\n"
                "The ranked chart selects the largest eligible categories by focal-hire "
                "spell count, then orders those categories by match rate. The table retains "
                "every category, including missing classifications and small denominators."
            ),
            _occupation_figure,
            mo.accordion(
                {
                    "View all occupation match-rate statistics": mo.ui.table(
                        occupation_summary,
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
def industry_rates(
    INDUSTRY_LABELS,
    INDUSTRY_TITLES,
    classification_match_rates,
    industry_selector,
    make_rate_chart,
    min_group_size_selector,
    sample_metrics,
    selected_fnh,
    top_n_selector,
):
    industry_column = industry_selector.value
    industry_title = INDUSTRY_LABELS[industry_column]
    industry_summary = classification_match_rates(
        selected_fnh,
        industry_column,
        INDUSTRY_TITLES.get(industry_column),
    )
    industry_chart, _industry_display = make_rate_chart(
        industry_summary,
        f"Inventor match rates by {industry_title}",
        sample_metrics["match_rate"],
        top_n_selector.value,
        min_group_size_selector.value,
        "#B45309",
    )
    return industry_chart, industry_column, industry_summary, industry_title


@app.cell(hide_code=True)
def industry_output(industry_chart, industry_summary, mo):
    if industry_chart is None:
        _industry_figure = mo.callout(
            mo.md("No industry meets the current sample and denominator settings."),
            kind="warn",
        )
    else:
        _industry_figure = industry_chart

    mo.vstack(
        [
            mo.md(
                "## 2. Industry match rates\n\n"
                "The same denominator and display rules are used for NAICS and Revelio's "
                "RICS hierarchies. Change the classification above to compare levels."
            ),
            _industry_figure,
            mo.accordion(
                {
                    "View all industry match-rate statistics": mo.ui.table(
                        industry_summary,
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
def joint_rates(
    MISSING_LABEL,
    alt,
    grouped_match_rates,
    industry_column,
    industry_summary,
    joint_axis_size_selector,
    joint_min_size_selector,
    math,
    occupation_column,
    occupation_summary,
    pl,
    sample_metrics,
    selected_fnh,
):
    joint_axis_size = joint_axis_size_selector.value
    _top_occupations = (
        occupation_summary.loc[occupation_summary["group_value"] != MISSING_LABEL]
        .nlargest(joint_axis_size, "candidate_spells")
        .copy()
    )
    _top_industries = (
        industry_summary.loc[industry_summary["group_value"] != MISSING_LABEL]
        .nlargest(joint_axis_size, "candidate_spells")
        .copy()
    )
    joint_axis_spells = selected_fnh.filter(
        pl.col(occupation_column)
        .cast(pl.String)
        .is_in(_top_occupations["group_value"].tolist())
        & pl.col(industry_column)
        .cast(pl.String)
        .is_in(_top_industries["group_value"].tolist())
    )
    joint_summary = grouped_match_rates(
        joint_axis_spells,
        [industry_column, occupation_column],
    )
    _occupation_labels = dict(
        zip(
            _top_occupations["group_value"],
            _top_occupations["display_label"],
            strict=True,
        )
    )
    _industry_labels = dict(
        zip(
            _top_industries["group_value"],
            _top_industries["display_label"],
            strict=True,
        )
    )
    joint_summary["occupation_label"] = (
        joint_summary[occupation_column].astype(str).map(_occupation_labels)
    )
    joint_summary["industry_label"] = (
        joint_summary[industry_column].astype(str).map(_industry_labels)
    )
    joint_display = joint_summary.loc[
        joint_summary["candidate_spells"] >= joint_min_size_selector.value
    ].copy()
    joint_axis_coverage = (
        len(joint_axis_spells) / sample_metrics["candidate_spells"]
        if sample_metrics["candidate_spells"]
        else math.nan
    )

    if joint_display.empty:
        joint_chart = None
    else:
        _occupation_order = _top_occupations["display_label"].tolist()
        _industry_order = _top_industries["display_label"].tolist()
        joint_chart = (
            alt.Chart(joint_display)
            .mark_rect(stroke="white", strokeWidth=0.4)
            .encode(
                x=alt.X(
                    "industry_label:N",
                    sort=_industry_order,
                    title=None,
                    axis=alt.Axis(labelAngle=-45, labelLimit=260),
                ),
                y=alt.Y(
                    "occupation_label:N",
                    sort=_occupation_order,
                    title=None,
                    axis=alt.Axis(labelLimit=330),
                ),
                color=alt.Color(
                    "match_rate:Q",
                    title="Match rate",
                    scale=alt.Scale(scheme="blues"),
                    legend=alt.Legend(format=".1%"),
                ),
                tooltip=[
                    alt.Tooltip("industry_label:N", title="Industry"),
                    alt.Tooltip("occupation_label:N", title="Occupation"),
                    alt.Tooltip(
                        "candidate_spells:Q",
                        title="Candidate spells",
                        format=",",
                    ),
                    alt.Tooltip("matched_spells:Q", title="Matched spells", format=","),
                    alt.Tooltip("match_rate:Q", title="Match rate", format=".2%"),
                    alt.Tooltip("ci_low:Q", title="95% CI lower", format=".2%"),
                    alt.Tooltip("ci_high:Q", title="95% CI upper", format=".2%"),
                ],
            )
            .properties(
                width="container",
                height=max(390, len(_top_occupations) * 28),
                title=alt.TitleParams(
                    text="Inventor match rates at industry–occupation level",
                    anchor="start",
                ),
            )
            .configure_view(stroke=None)
        )
    return joint_axis_coverage, joint_axis_size, joint_chart, joint_display


@app.cell(hide_code=True)
def joint_output(
    industry_title,
    joint_axis_coverage,
    joint_axis_size,
    joint_chart,
    joint_display,
    mo,
    occupation_title,
):
    if joint_chart is None:
        _joint_figure = mo.callout(
            mo.md("No industry–occupation cell meets the active denominator setting."),
            kind="warn",
        )
    else:
        _joint_figure = joint_chart

    mo.vstack(
        [
            mo.md(
                f"""
                ## 3. Industry–occupation match rates

                The heatmap crosses **{industry_title}** with **{occupation_title}**. For legibility,
                its axes contain the largest {joint_axis_size:,} nonmissing industries and
                occupations in the active sample. Those axis categories jointly cover
                **{joint_axis_coverage:.1%}** of active focal-hire spells. Blank cells fall
                below the current heatmap-cell denominator threshold.
                """
            ),
            _joint_figure,
            mo.accordion(
                {
                    "View displayed industry–occupation cells": mo.ui.table(
                        joint_display,
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
def geography_tables(
    MISSING_LABEL,
    US_LABEL,
    classification_match_rates,
    country_iso3,
    make_rate_chart,
    math,
    min_group_size_selector,
    pl,
    sample_metrics,
    selected_fnh,
    top_n_selector,
    us_state_code,
):
    country_summary = classification_match_rates(selected_fnh, "country")
    country_chart, _country_display = make_rate_chart(
        country_summary,
        "Inventor match rates by country",
        sample_metrics["match_rate"],
        top_n_selector.value,
        min_group_size_selector.value,
        "#0F766E",
    )

    _country_map_working = country_summary.loc[
        (country_summary["group_value"] != MISSING_LABEL)
        & (country_summary["candidate_spells"] >= min_group_size_selector.value)
    ].copy()
    _country_map_working["iso3"] = _country_map_working["group_value"].map(country_iso3)
    mapped_country_summary = _country_map_working.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = _country_map_working.loc[
        _country_map_working["iso3"].isna()
    ].copy()

    us_selected = selected_fnh.filter(pl.col("country") == US_LABEL)
    state_summary = classification_match_rates(us_selected, "state")
    state_chart, _state_display = make_rate_chart(
        state_summary,
        "Inventor match rates by U.S. state",
        (float(us_selected["inventor_match"].mean()) if len(us_selected) else math.nan),
        top_n_selector.value,
        min_group_size_selector.value,
        "#7C3AED",
    )
    _state_map_working = state_summary.loc[
        (state_summary["group_value"] != MISSING_LABEL)
        & (state_summary["candidate_spells"] >= min_group_size_selector.value)
    ].copy()
    _state_map_working["state_code"] = _state_map_working["group_value"].map(
        us_state_code
    )
    state_map_data = _state_map_working.dropna(subset=["state_code"]).copy()
    unmapped_state_summary = _state_map_working.loc[
        _state_map_working["state_code"].isna()
    ].copy()
    return (
        country_chart,
        country_summary,
        mapped_country_summary,
        state_chart,
        state_map_data,
        state_summary,
        unmapped_country_summary,
        unmapped_state_summary,
        us_selected,
    )


@app.cell(hide_code=True)
def maps(
    mapped_country_summary,
    px,
    sample_metrics,
    state_map_data,
    us_selected,
):
    if mapped_country_summary.empty:
        world_country_map = None
    else:
        _country_color_max = max(
            float(mapped_country_summary["match_rate"].max()),
            sample_metrics["match_rate"],
            0.01,
        )
        world_country_map = px.choropleth(
            mapped_country_summary,
            locations="iso3",
            color="match_rate",
            hover_name="group_value",
            hover_data={
                "iso3": False,
                "candidate_spells": ":,",
                "matched_spells": ":,",
                "match_rate": ":.2%",
                "ci_low": ":.2%",
                "ci_high": ":.2%",
            },
            labels={
                "candidate_spells": "Candidate spells",
                "matched_spells": "Matched spells",
                "match_rate": "Match rate",
                "ci_low": "95% CI lower",
                "ci_high": "95% CI upper",
            },
            color_continuous_scale="Blues",
            range_color=(0.0, _country_color_max),
            projection="natural earth",
            title="Inventor match rates by country",
        )
        world_country_map.update_geos(showframe=False, showcoastlines=True)
        world_country_map.update_layout(
            height=560,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )

    if state_map_data.empty:
        state_map_figure = None
    else:
        _state_color_max = max(
            float(state_map_data["match_rate"].max()),
            float(us_selected["inventor_match"].mean()),
            0.01,
        )
        state_map_figure = px.choropleth(
            state_map_data,
            locations="state_code",
            locationmode="USA-states",
            scope="usa",
            color="match_rate",
            hover_name="group_value",
            hover_data={
                "state_code": False,
                "candidate_spells": ":,",
                "matched_spells": ":,",
                "match_rate": ":.2%",
                "ci_low": ":.2%",
                "ci_high": ":.2%",
            },
            labels={
                "candidate_spells": "Candidate spells",
                "matched_spells": "Matched spells",
                "match_rate": "Match rate",
                "ci_low": "95% CI lower",
                "ci_high": "95% CI upper",
            },
            color_continuous_scale="Purples",
            range_color=(0.0, _state_color_max),
            title="Inventor match rates by U.S. state",
        )
        state_map_figure.update_geos(scope="usa", visible=False)
        state_map_figure.update_layout(
            height=600,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )
    return state_map_figure, world_country_map


@app.cell(hide_code=True)
def geography_output(
    country_chart,
    country_summary,
    mo,
    state_chart,
    state_map_figure,
    state_summary,
    unmapped_country_summary,
    unmapped_state_summary,
    world_country_map,
):
    _world_output = (
        world_country_map
        if world_country_map is not None
        else mo.callout(
            mo.md("No country can be mapped under the active sample and denominator."),
            kind="warn",
        )
    )
    _country_chart_output = (
        country_chart
        if country_chart is not None
        else mo.callout(
            mo.md("No country meets the active denominator setting."),
            kind="warn",
        )
    )
    _state_output = (
        state_map_figure
        if state_map_figure is not None
        else mo.callout(
            mo.md(
                "The active scope contains no mapped U.S. state meeting the current "
                "denominator setting. Include the United States or lower the threshold."
            ),
            kind="warn",
        )
    )
    _state_chart_output = state_chart if state_chart is not None else mo.md("")

    mo.vstack(
        [
            mo.md(
                "## 4. Geographic match rates\n\n"
                "Maps and ranked charts use the active denominator threshold. Full tables "
                "retain all country and U.S.-state categories. The state view is calculated "
                "only from U.S. spells that remain inside the active country scope."
            ),
            _country_chart_output,
            _world_output,
            mo.accordion(
                {
                    "View all country match-rate statistics": mo.ui.table(
                        country_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View eligible country labels not mapped to ISO-3": mo.ui.table(
                        unmapped_country_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                }
            ),
            mo.md("### 4.1. United States"),
            _state_chart_output,
            _state_output,
            mo.accordion(
                {
                    "View all U.S.-state match-rate statistics": mo.ui.table(
                        state_summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View eligible state labels not mapped to USPS codes": mo.ui.table(
                        unmapped_state_summary,
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
def diagnostics(
    AVAILABLE_RICS_COLUMNS,
    AVAILABLE_ROLE_COLUMNS,
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    MISSING_LABEL,
    PARQUET_FILES,
    fnh,
    link_diagnostics,
    mo,
    pd,
    pl,
):
    _classification_columns = (
        "onet_code",
        *AVAILABLE_ROLE_COLUMNS,
        "naics_code",
        *AVAILABLE_RICS_COLUMNS,
        "country",
        "state",
    )
    _missing_expected_roles = sorted(
        set(EXPECTED_ROLE_COLUMNS) - set(AVAILABLE_ROLE_COLUMNS)
    )
    _missing_expected_rics = sorted(
        set(EXPECTED_RICS_COLUMNS) - set(AVAILABLE_RICS_COLUMNS)
    )
    _coverage_rows = []
    for _column in _classification_columns:
        _missing_count = int(
            fnh.select((pl.col(_column) == MISSING_LABEL).sum()).item()
        )
        _coverage_rows.append(
            {
                "Variable": _column,
                "Nonmissing categories": int(
                    fnh.filter(pl.col(_column) != MISSING_LABEL)[_column].n_unique()
                ),
                "Missing rows": _missing_count,
                "Missing share": _missing_count / len(fnh),
            }
        )
    classification_coverage = pd.DataFrame(_coverage_rows)

    _onet_conflicts = (
        fnh.filter(pl.col("onet_code") != MISSING_LABEL)
        .group_by("onet_code")
        .agg(pl.col("onet_title").n_unique().alias("distinct_titles"))
        .filter(pl.col("distinct_titles") > 1)
        .sort("distinct_titles", descending=True)
        .to_pandas()
    )
    _naics_conflicts = (
        fnh.filter(pl.col("naics_code") != MISSING_LABEL)
        .group_by("naics_code")
        .agg(pl.col("naics_description").n_unique().alias("distinct_descriptions"))
        .filter(pl.col("distinct_descriptions") > 1)
        .sort("distinct_descriptions", descending=True)
        .to_pandas()
    )
    title_diagnostics = {
        "O*NET codes with multiple titles": len(_onet_conflicts),
        "NAICS codes with multiple descriptions": len(_naics_conflicts),
    }

    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. Data and linkage diagnostics

                - **Input Parquet parts:** {len(PARQUET_FILES):,}
                - **Full-sample focal-hire spells:** {len(fnh):,}
                - **In-memory analysis frame:** {fnh.estimated_size("mb"):,.1f} MiB
                - **O*NET codes with multiple delivered titles:**
                  {title_diagnostics["O*NET codes with multiple titles"]:,}
                - **NAICS codes with multiple delivered descriptions:**
                  {title_diagnostics["NAICS codes with multiple descriptions"]:,}

                Only the linkage and requested analysis columns are read. Classification
                strings are held as categoricals to reduce memory use.
                """
            ),
            mo.accordion(
                {
                    "Inventor-crosswalk diagnostics": mo.ui.table(
                        link_diagnostics,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Classification coverage": mo.ui.table(
                        classification_coverage,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "O*NET code-title conflicts": mo.ui.table(
                        _onet_conflicts,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "NAICS code-description conflicts": mo.ui.table(
                        _naics_conflicts,
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
def interpretation(mo):
    mo.callout(
        mo.md(r"""
        **Interpretation boundary.** This is a linkage/coverage measure, not a clean measure
        of inventive productivity. A nonmatch combines never patenting, patenting outside
        USPTO coverage, and failed user-to-inventor linkage. Cross-country and cross-field
        differences may therefore reflect PatentView scope and linkage quality as well as
        real inventive activity. Use the counts, intervals, and missing-classification rows
        when choosing restrictions; later robustness work should assess linkage selection.
        """),
        kind="warn",
    )
    return


if __name__ == "__main__":
    app.run()

# ruff: noqa: PLR1711

"""
Task:
    Summarize inventor-linkage rates in the candidate focal-new-hire sample.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHires_AllIndustries/*.parquet
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) Four reactive marimo sections covering basic counts, occupations, industries,
    and other geographic, seniority, and start-month results.

Notes:
(1) The focal-new-hire Parquet dataset has one retained row per user-company pair.
(2) The inventor crosswalk is reduced to users with a nonmissing inventor ID before matching.
(3) Every figure has its own local controls; there is no global analysis-control panel.
(4) User-company rates use focal-new-hire rows as the denominator. User rates use distinct
    users within each displayed group as the denominator.

Run:
    $fnh_match_notebook = "codes/B01_ConstructAnalysisSample/B03_FNH_InventorMatchRates.py"
    conda run -s -n Talent marimo edit $fnh_match_notebook

Wang Wenzhi, with the help of Codex
Time: 2026-08-24
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell()
def imports():
    import re

    import altair as alt
    import marimo as mo
    import pandas as pd
    import plotly.express as px
    import polars as pl
    import pyarrow.dataset as ds
    import pycountry

    return alt, ds, mo, pd, pl, px, pycountry, re


@app.cell()
def title(mo):
    mo.vstack(
        [
            mo.md(
                "# Inventor match rates among candidate focal new hires\n\n"
                "This notebook compares the candidate focal-new-hire universe constructed from "
                "`A_FocalNewHires_AllIndustries.ipynb` with the Revelio-to-PatentsView inventor "
                "crosswalk. The figures report linkage coverage, not a complete measure of "
                "inventive productivity. Controls are placed next to the result that they "
                "change."
            )
        ]
    )
    return


@app.cell()
def helpers(alt, pd, pl, pycountry, re):
    MISSING_LABEL = "<Missing>"
    US_LABEL = "United States"
    METRIC_OPTIONS = {
        "User-company level": "spell",
        "User level": "user",
    }
    SCOPE_OPTIONS = {
        "All countries": "all",
        "United States": "us",
        "Non-U.S.": "non_us",
        "Selected countries": "custom",
    }

    def hierarchy_number(column_name):
        """Return the numeric K level used to sort Revelio hierarchy fields."""

        _match = re.search(r"_k(\d+)$", column_name)
        return int(_match.group(1)) if _match else -1

    def metric_label(metric):
        return "user-company" if metric == "spell" else "user"

    def metric_columns(metric):
        if metric == "user":
            return "user_match_rate", "user_ci_low", "user_ci_high", "matched_users"
        return "spell_match_rate", "spell_ci_low", "spell_ci_high", "matched_spells"

    def rate_aggregations():
        """Return common aggregation expressions for both denominator definitions."""

        return [
            pl.len().alias("candidate_spells"),
            pl.col("inventor_match").cast(pl.Int64).sum().alias("matched_spells"),
            pl.col("user_id").n_unique().alias("unique_users"),
            pl.col("user_id")
            .filter(pl.col("inventor_match"))
            .n_unique()
            .alias("matched_users"),
        ]

    def add_rate_statistics(summary):
        """Attach spell- and user-level rates with descriptive Wilson intervals."""

        _result = summary.copy()
        _rate_columns = [
            "spell_match_rate",
            "user_match_rate",
            "spell_ci_low",
            "spell_ci_high",
            "user_ci_low",
            "user_ci_high",
        ]
        if _result.empty:
            for _column in _rate_columns:
                _result[_column] = pd.Series(dtype="float64")
            return _result

        def _wilson(successes, denominators):
            _n = denominators.astype(float)
            _p = (successes / _n).where(_n > 0)
            _z = 1.96
            _denominator = 1.0 + _z**2 / _n
            _center = (_p + _z**2 / (2.0 * _n)) / _denominator
            _margin = (
                _z
                * (_p * (1.0 - _p) / _n + _z**2 / (4.0 * _n**2)) ** 0.5
                / _denominator
            )
            return (_center - _margin).clip(lower=0.0), (_center + _margin).clip(
                upper=1.0
            )

        _spell_n = _result["candidate_spells"].astype(float)
        _user_n = _result["unique_users"].astype(float)
        _spell_low, _spell_high = _wilson(
            _result["matched_spells"],
            _spell_n,
        )
        _user_low, _user_high = _wilson(
            _result["matched_users"],
            _user_n,
        )
        _result["spell_match_rate"] = (_result["matched_spells"] / _spell_n).where(
            _spell_n > 0
        )
        _result["user_match_rate"] = (_result["matched_users"] / _user_n).where(
            _user_n > 0
        )
        _result["spell_ci_low"] = _spell_low
        _result["spell_ci_high"] = _spell_high
        _result["user_ci_low"] = _user_low
        _result["user_ci_high"] = _user_high
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

    def _display_value(value):
        if value is None or pd.isna(value):
            return MISSING_LABEL
        return str(value)

    def classification_match_rates(data, value_column, title_column=None):
        """Aggregate one classification and add readable value/display labels."""

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
        _summary["group_value"] = (
            _summary[value_column].map(_display_value).astype("string")
        )
        if title_column is None:
            _summary["display_label"] = _summary["group_value"]
        else:
            _summary["display_label"] = (
                _summary["group_value"]
                + " — "
                + _summary[title_column].map(_display_value).astype("string")
            )
        return _summary

    def sample_statistics(data):
        """Return counts needed for Section 1 and figure reference information."""

        _row = data.select(
            pl.len().alias("candidate_spells"),
            pl.col("inventor_match").cast(pl.Int64).sum().alias("matched_spells"),
            pl.col("user_id").n_unique().alias("unique_users"),
            pl.col("user_id")
            .filter(pl.col("inventor_match"))
            .n_unique()
            .alias("matched_users"),
        ).to_dicts()[0]
        _row["spell_match_rate"] = (
            _row["matched_spells"] / _row["candidate_spells"]
            if _row["candidate_spells"]
            else float("nan")
        )
        _row["user_match_rate"] = (
            _row["matched_users"] / _row["unique_users"]
            if _row["unique_users"]
            else float("nan")
        )
        return _row

    def available_countries(data):
        return sorted(
            _country
            for _country in data["country"].cast(pl.String).unique().to_list()
            if _country != MISSING_LABEL
        )

    def country_scope_frames(data, selected_countries):
        """Return the three requested country scopes, with custom replacing non-U.S."""

        _selected = tuple(selected_countries or ())
        _custom_label = (
            "Selected countries: " + ", ".join(_selected)
            if _selected
            else "Selected countries: none"
        )
        return (
            (
                "All countries",
                "all",
                data,
            ),
            (
                "United States",
                "us",
                data.filter(pl.col("country") == US_LABEL),
            ),
            (
                "Non-U.S." if not _selected else _custom_label,
                "non_us" if not _selected else "custom",
                data.filter(
                    pl.col("country").is_in(_selected)
                    if _selected
                    else (
                        (pl.col("country") != US_LABEL)
                        & (pl.col("country") != MISSING_LABEL)
                    )
                ),
            ),
        )

    def scoped_classification_match_rates(
        data,
        value_column,
        title_column=None,
        selected_countries=(),
    ):
        """Calculate one classification separately for all, U.S., and non-U.S./custom."""

        _summaries = []
        for _scope_label, _scope_key, _scope_data in country_scope_frames(
            data,
            selected_countries,
        ):
            _summary = classification_match_rates(
                _scope_data,
                value_column,
                title_column,
            )
            _summary["country_scope"] = _scope_label
            _summary["scope_key"] = _scope_key
            _summaries.append(_summary)
        return pd.concat(_summaries, ignore_index=True)

    def _chart_tooltips(metric, include_scope=False):
        _rate_column, _low_column, _high_column, _matched_column = metric_columns(
            metric
        )
        _tooltips = []
        if include_scope:
            _tooltips.append(alt.Tooltip("country_scope:N", title="Country scope"))
        _tooltips.extend(
            [
                alt.Tooltip("display_label:N", title="Category"),
                alt.Tooltip(
                    "candidate_spells:Q",
                    title="Candidate spells",
                    format=",",
                ),
                alt.Tooltip(
                    f"{_matched_column}:Q",
                    title="Matched observations",
                    format=",",
                ),
                alt.Tooltip(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    format=".2%",
                ),
                alt.Tooltip(f"{_low_column}:Q", title="95% CI lower", format=".2%"),
                alt.Tooltip(f"{_high_column}:Q", title="95% CI upper", format=".2%"),
                alt.Tooltip(
                    "unique_users:Q",
                    title="Unique users",
                    format=",",
                ),
                alt.Tooltip(
                    "matched_users:Q",
                    title="Matched users",
                    format=",",
                ),
            ]
        )
        return _tooltips

    def make_grouped_rate_chart(summary, title, metric, top_n, color_range):
        """Select large categories, then rank their bars by the all-country rate."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _all_country = summary.loc[
            (summary["scope_key"] == "all") & (summary["group_value"] != MISSING_LABEL)
        ].copy()
        _largest = _all_country.sort_values(
            ["candidate_spells", "display_label"],
            ascending=[False, True],
        )
        _top_values = _largest.head(int(top_n))["group_value"].tolist()
        _ranked = _all_country.loc[
            _all_country["group_value"].isin(_top_values)
        ].sort_values(
            [_rate_column, "candidate_spells"],
            ascending=[False, False],
            na_position="last",
        )
        _shown = summary.loc[summary["group_value"].isin(_top_values)].copy()
        if _shown.empty:
            return None, _shown

        _category_order = _ranked.loc[
            _ranked["group_value"].isin(_top_values), "display_label"
        ].tolist()
        _scope_order = [
            _scope for _scope in summary["country_scope"].drop_duplicates().tolist()
        ]
        _shown["country_scope"] = pd.Categorical(
            _shown["country_scope"],
            categories=_scope_order,
            ordered=True,
        )
        _rate_upper = min(
            1.0,
            max(float(_shown[_high_column].max()) * 1.08, 0.01),
        )
        _chart = (
            alt.Chart(_shown)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_category_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                yOffset=alt.YOffset("country_scope:N", sort=_scope_order),
                color=alt.Color(
                    "country_scope:N",
                    title="Country scope",
                    scale=alt.Scale(domain=_scope_order, range=color_range),
                ),
                tooltip=_chart_tooltips(metric, include_scope=True),
            )
            .properties(
                width="container",
                height=max(300, len(_category_order) * 28),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )
        return _chart, _shown

    def make_seniority_rate_chart(summary, title, metric, color_range):
        """Plot seniority on the x-axis in numeric order for each country scope."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _shown = summary.loc[summary["group_value"] != MISSING_LABEL].copy()
        if _shown.empty:
            return None, _shown
        _shown["seniority_order"] = pd.to_numeric(
            _shown["group_value"], errors="coerce"
        )
        _category_order = (
            _shown[["display_label", "seniority_order"]]
            .drop_duplicates()
            .sort_values(
                ["seniority_order", "display_label"],
                na_position="last",
            )["display_label"]
            .tolist()
        )
        _scope_order = _shown["country_scope"].drop_duplicates().tolist()
        _shown["country_scope"] = pd.Categorical(
            _shown["country_scope"],
            categories=_scope_order,
            ordered=True,
        )
        _rate_upper = min(
            1.0,
            max(float(_shown[_high_column].max()) * 1.08, 0.01),
        )
        _chart = (
            alt.Chart(_shown)
            .mark_bar()
            .encode(
                x=alt.X(
                    "display_label:O",
                    sort=_category_order,
                    title="Seniority level",
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=alt.Axis(format=".0%"),
                ),
                xOffset=alt.XOffset("country_scope:N", sort=_scope_order),
                color=alt.Color(
                    "country_scope:N",
                    title="Country scope",
                    scale=alt.Scale(domain=_scope_order, range=color_range),
                ),
                tooltip=_chart_tooltips(metric, include_scope=True),
            )
            .properties(
                width="container",
                height=400,
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )
        return _chart, _shown

    def make_single_rate_chart(summary, title, metric, top_n, color):
        """Select large categories, then rank their bars by the selected rate."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _eligible = summary.loc[summary["group_value"] != MISSING_LABEL].copy()
        _top_values = (
            _eligible.sort_values(
                ["candidate_spells", "display_label"],
                ascending=[False, True],
            )
            .head(int(top_n))["group_value"]
            .tolist()
        )
        _shown = _eligible.loc[_eligible["group_value"].isin(_top_values)].copy()
        _shown = _shown.sort_values(
            [_rate_column, "candidate_spells"],
            ascending=[False, False],
            na_position="last",
        )
        if _shown.empty:
            return None, _shown
        _order = _shown["display_label"].tolist()
        _rate_upper = min(
            1.0,
            max(float(_shown[_high_column].max()) * 1.08, 0.01),
        )
        _chart = (
            alt.Chart(_shown)
            .mark_bar(color=color)
            .encode(
                x=alt.X(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=alt.Axis(format=".0%"),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=_chart_tooltips(metric),
            )
            .properties(
                width="container",
                height=max(300, len(_shown) * 25),
                title=alt.TitleParams(text=title, anchor="start"),
            )
            .configure_view(stroke=None)
        )
        return _chart, _shown

    def make_time_chart(summary, title, metric, color_range):
        """Make a monthly line chart for the requested denominator definition."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _shown = summary.dropna(subset=["month"]).copy()
        if _shown.empty:
            return None, _shown
        _scope_order = _shown["country_scope"].drop_duplicates().tolist()
        _shown["country_scope"] = pd.Categorical(
            _shown["country_scope"],
            categories=_scope_order,
            ordered=True,
        )
        _rate_upper = min(
            1.0,
            max(float(_shown[_high_column].max()) * 1.08, 0.01),
        )
        _chart = (
            alt.Chart(_shown)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Focal-hire start month"),
                y=alt.Y(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=alt.Axis(format=".0%"),
                ),
                color=alt.Color(
                    "country_scope:N",
                    title="Country scope",
                    scale=alt.Scale(domain=_scope_order, range=color_range),
                ),
                tooltip=[
                    alt.Tooltip("country_scope:N", title="Country scope"),
                    alt.Tooltip("month:T", title="Start month", format="%Y-%m"),
                    alt.Tooltip(
                        "candidate_spells:Q",
                        title="Candidate spells",
                        format=",",
                    ),
                    alt.Tooltip(
                        f"{_rate_column}:Q",
                        title=f"{metric_label(metric).title()} match rate",
                        format=".2%",
                    ),
                    alt.Tooltip(
                        "unique_users:Q",
                        title="Unique users",
                        format=",",
                    ),
                ],
            )
            .properties(
                width="container",
                height=420,
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
        METRIC_OPTIONS,
        MISSING_LABEL,
        SCOPE_OPTIONS,
        US_LABEL,
        available_countries,
        classification_match_rates,
        country_iso3,
        grouped_match_rates,
        hierarchy_number,
        make_grouped_rate_chart,
        make_seniority_rate_chart,
        make_single_rate_chart,
        make_time_chart,
        metric_columns,
        metric_label,
        sample_statistics,
        scoped_classification_match_rates,
        us_state_code,
    )


@app.cell()
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
    REQUIRED_COLUMNS = (
        "user_id",
        "country",
        "state",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        "seniority",
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
    if "start_month" in AVAILABLE_COLUMNS:
        DATE_COLUMN = "start_month"
    elif "startdate" in AVAILABLE_COLUMNS:
        DATE_COLUMN = "startdate"
    else:
        raise ValueError("Input must contain either `start_month` or `startdate`.")

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
                DATE_COLUMN,
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
        DATE_COLUMN,
        INPUT_DIR,
        PARQUET_FILES,
    )


@app.cell()
def load_data(
    ANALYSIS_COLUMNS,
    CROSSWALK_PATH,
    DATE_COLUMN,
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
    ).with_columns(pl.col("pv_inventor_id").str.strip_chars())
    _valid_links = _patent_links.filter(
        pl.col("user_id").is_not_null()
        & pl.col("pv_inventor_id").is_not_null()
        & (pl.col("pv_inventor_id") != "")
    )
    _link_user_counts = _valid_links.group_by("user_id").len()
    _patent_users = _valid_links.select("user_id").unique()
    link_diagnostics = pd.DataFrame(
        [
            {
                "Crosswalk rows": len(_patent_links),
                "Rows with both IDs": len(_valid_links),
                "Unique linked users": len(_patent_users),
                "Unique inventor IDs": _valid_links["pv_inventor_id"].n_unique(),
                "Users with multiple rows": int(
                    _link_user_counts.select((pl.col("len") > 1).sum()).item()
                ),
                "Maximum rows per user": int(_link_user_counts["len"].max() or 0),
                "Missing user IDs": int(_patent_links["user_id"].null_count()),
                "Missing inventor IDs": int(
                    _patent_links["pv_inventor_id"].null_count()
                ),
                "Blank inventor IDs": int(
                    _patent_links.select((pl.col("pv_inventor_id") == "").sum()).item()
                ),
            }
        ]
    )

    _string_columns = tuple(
        _column
        for _column in ANALYSIS_COLUMNS
        if _column not in {"user_id", "seniority", DATE_COLUMN}
    )
    _matched_users = _patent_users.lazy().with_columns(
        pl.lit(True).alias("inventor_match")
    )
    _fnh_scan = (
        pl.scan_parquet(str(INPUT_DIR / "*.parquet"))
        .select(list(ANALYSIS_COLUMNS))
        .with_columns(
            pl.col(DATE_COLUMN)
            .cast(pl.String)
            .str.to_date(strict=False)
            .dt.truncate("1mo")
            .alias("start_month")
        )
        .join(_matched_users, on="user_id", how="left")
        .with_columns(
            pl.col("inventor_match").fill_null(False),
            *[
                pl.col(_column).fill_null(MISSING_LABEL).cast(pl.Categorical)
                for _column in _string_columns
            ],
        )
    )
    if DATE_COLUMN != "start_month":
        _fnh_scan = _fnh_scan.drop(DATE_COLUMN)
    fnh = _fnh_scan.collect(engine="streaming")
    return fnh, link_diagnostics


@app.cell()
def classifications(AVAILABLE_RICS_COLUMNS, AVAILABLE_ROLE_COLUMNS, hierarchy_number):
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


# -----------------------------------------------------------------------------
# Section 1. Basic numbers
# -----------------------------------------------------------------------------


@app.cell()
def basic_numbers(
    MISSING_LABEL,
    US_LABEL,
    available_countries,
    fnh,
    link_diagnostics,
    mo,
    pd,
    pl,
    sample_statistics,
):
    _scopes = (
        ("All countries", fnh),
        ("United States", fnh.filter(pl.col("country") == US_LABEL)),
        (
            "Non-U.S.",
            fnh.filter(
                (pl.col("country") != US_LABEL) & (pl.col("country") != MISSING_LABEL)
            ),
        ),
    )
    _rows = []
    for _scope_label, _scope_data in _scopes:
        _stats = sample_statistics(_scope_data)
        _rows.extend(
            [
                {
                    "Country scope": _scope_label,
                    "Observation level": "User-company",
                    "Candidate observations": _stats["candidate_spells"],
                    "Matched observations": _stats["matched_spells"],
                    "Match rate": _stats["spell_match_rate"],
                },
                {
                    "Country scope": _scope_label,
                    "Observation level": "User",
                    "Candidate observations": _stats["unique_users"],
                    "Matched observations": _stats["matched_users"],
                    "Match rate": _stats["user_match_rate"],
                },
            ]
        )
    basic_numbers_table = pd.DataFrame(_rows)
    mo.vstack(
        [
            mo.md(
                "## 1. Basic numbers\n\n"
                "The all-country row includes any retained records with a missing country. "
                "The U.S. and non-U.S. rows exclude missing-country records."
            ),
            mo.ui.table(
                basic_numbers_table,
                pagination=False,
                show_column_summaries=False,
                format_mapping={"Match rate": "0.00%"},
            ),
            mo.accordion(
                {
                    "Inventor-crosswalk diagnostics": mo.ui.table(
                        link_diagnostics,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Country coverage": mo.ui.table(
                        pd.DataFrame(
                            {"Nonmissing countries": [len(available_countries(fnh))]}
                        ),
                        pagination=False,
                        show_column_summaries=False,
                    ),
                }
            ),
        ],
        gap=1,
    )
    return basic_numbers_table


# -----------------------------------------------------------------------------
# Section 2. Match rates across different occupations
# -----------------------------------------------------------------------------


@app.cell()
def occupation_controls(
    METRIC_OPTIONS, OCCUPATION_LABELS, SCOPE_OPTIONS, available_countries, fnh, mo
):
    occupation_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition",
        full_width=True,
    )
    occupation_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in OCCUPATION_LABELS.items()},
        value=OCCUPATION_LABELS["onet_code"],
        label="Occupation classification",
        full_width=True,
    )
    occupation_country_selector = mo.ui.multiselect(
        options=available_countries(fnh),
        value=[],
        label="Countries replacing the non-U.S. series (optional)",
        full_width=True,
    )
    occupation_table_scope_selector = mo.ui.dropdown(
        options=SCOPE_OPTIONS,
        value="All countries",
        label="Country scope for the table",
        full_width=True,
    )
    occupation_show_table = mo.ui.checkbox(
        value=False,
        label="Display the occupation table",
    )
    return (
        occupation_country_selector,
        occupation_metric_selector,
        occupation_selector,
        occupation_show_table,
        occupation_table_scope_selector,
    )


@app.cell()
def occupation_top_n_control(MISSING_LABEL, fnh, mo, occupation_selector, pl):
    _occupation_column = occupation_selector.value
    _category_count = int(
        fnh.filter(pl.col(_occupation_column) != MISSING_LABEL)
        .select(pl.col(_occupation_column).n_unique())
        .item()
    )
    _max_categories = max(1, _category_count)
    _default_categories = (
        _max_categories
        if _occupation_column == "onet_code"
        else min(50, _max_categories)
    )
    occupation_top_n_selector = mo.ui.slider(
        start=1,
        stop=_max_categories,
        value=_default_categories,
        step=1,
        show_value=True,
        label="Number of occupation categories in the bar chart",
        full_width=True,
    )
    return occupation_top_n_selector


@app.cell()
def occupation_rates(
    OCCUPATION_LABELS,
    OCCUPATION_TITLES,
    fnh,
    make_grouped_rate_chart,
    metric_label,
    occupation_country_selector,
    occupation_metric_selector,
    occupation_selector,
    occupation_table_scope_selector,
    occupation_top_n_selector,
    scoped_classification_match_rates,
):
    occupation_column = occupation_selector.value
    occupation_title = OCCUPATION_LABELS[occupation_column]
    occupation_metric = occupation_metric_selector.value
    occupation_summary = scoped_classification_match_rates(
        fnh,
        occupation_column,
        OCCUPATION_TITLES.get(occupation_column),
        occupation_country_selector.value,
    )
    occupation_chart, _occupation_display = make_grouped_rate_chart(
        occupation_summary,
        (
            f"Inventor {metric_label(occupation_metric)}-level match rates by "
            f"{occupation_title}"
        ),
        occupation_metric,
        occupation_top_n_selector.value,
        ["#2563EB", "#0F766E", "#B45309"],
    )
    occupation_table = occupation_summary.loc[
        occupation_summary["scope_key"] == occupation_table_scope_selector.value
    ].copy()
    occupation_table = occupation_table.drop(columns=["scope_key"], errors="ignore")
    occupation_note = (
        "The top categories are selected by their all-country candidate counts, then the "
        "bars are ranked by the selected all-country match rate. The selected countries "
        "replace, rather than add to, the non-U.S. series."
    )
    return (
        occupation_chart,
        occupation_note,
        occupation_summary,
        occupation_table,
    )


@app.cell()
def occupation_output(
    mo,
    occupation_chart,
    occupation_country_selector,
    occupation_metric_selector,
    occupation_note,
    occupation_selector,
    occupation_show_table,
    occupation_table,
    occupation_table_scope_selector,
    occupation_top_n_selector,
):
    _figure = (
        occupation_chart
        if occupation_chart is not None
        else mo.callout(
            mo.md("No occupation has a nonmissing all-country denominator."),
            kind="warn",
        )
    )
    _table_output = mo.md("")
    if occupation_show_table.value:
        _table_output = mo.vstack(
            [
                occupation_table_scope_selector,
                mo.ui.table(
                    occupation_table,
                    pagination=True,
                    page_size=20,
                    selection="multi",
                    show_column_summaries=False,
                ),
            ],
            gap=1,
        )
    mo.vstack(
        [
            mo.md("## 2. Match rates across different occupations"),
            mo.vstack(
                [
                    occupation_metric_selector,
                    occupation_selector,
                    occupation_top_n_selector,
                    occupation_country_selector,
                    mo.md(occupation_note),
                ],
                gap=1,
            ),
            _figure,
            mo.accordion(
                {"Occupation table (select rows and country scope)": _table_output}
            ),
        ],
        gap=1,
    )
    return


# -----------------------------------------------------------------------------
# Section 3. Match rates across different industries
# -----------------------------------------------------------------------------


@app.cell()
def industry_controls(
    INDUSTRY_LABELS,
    METRIC_OPTIONS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
    DEFAULT_INDUSTRY,
):
    industry_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition",
        full_width=True,
    )
    industry_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in INDUSTRY_LABELS.items()},
        value=INDUSTRY_LABELS[DEFAULT_INDUSTRY],
        label="Industry classification",
        full_width=True,
    )
    industry_country_selector = mo.ui.multiselect(
        options=available_countries(fnh),
        value=[],
        label="Countries replacing the non-U.S. series (optional)",
        full_width=True,
    )
    industry_table_scope_selector = mo.ui.dropdown(
        options=SCOPE_OPTIONS,
        value="All countries",
        label="Country scope for the table",
        full_width=True,
    )
    industry_show_table = mo.ui.checkbox(
        value=False,
        label="Display the industry table",
    )
    return (
        industry_country_selector,
        industry_metric_selector,
        industry_selector,
        industry_show_table,
        industry_table_scope_selector,
    )


@app.cell()
def industry_top_n_control(MISSING_LABEL, fnh, industry_selector, mo, pl):
    _industry_column = industry_selector.value
    _category_count = int(
        fnh.filter(pl.col(_industry_column) != MISSING_LABEL)
        .select(pl.col(_industry_column).n_unique())
        .item()
    )
    _max_categories = max(1, _category_count)
    industry_top_n_selector = mo.ui.slider(
        start=1,
        stop=_max_categories,
        value=min(50, _max_categories),
        step=1,
        show_value=True,
        label="Number of industry categories in the bar chart",
        full_width=True,
    )
    return industry_top_n_selector


@app.cell()
def industry_rates(
    INDUSTRY_LABELS,
    INDUSTRY_TITLES,
    fnh,
    industry_country_selector,
    industry_metric_selector,
    industry_selector,
    industry_table_scope_selector,
    industry_top_n_selector,
    make_grouped_rate_chart,
    metric_label,
    scoped_classification_match_rates,
):
    industry_column = industry_selector.value
    industry_title = INDUSTRY_LABELS[industry_column]
    industry_metric = industry_metric_selector.value
    industry_summary = scoped_classification_match_rates(
        fnh,
        industry_column,
        INDUSTRY_TITLES.get(industry_column),
        industry_country_selector.value,
    )
    industry_chart, _industry_display = make_grouped_rate_chart(
        industry_summary,
        (
            f"Inventor {metric_label(industry_metric)}-level match rates by "
            f"{industry_title}"
        ),
        industry_metric,
        industry_top_n_selector.value,
        ["#B45309", "#2563EB", "#0F766E"],
    )
    industry_table = industry_summary.loc[
        industry_summary["scope_key"] == industry_table_scope_selector.value
    ].copy()
    industry_table = industry_table.drop(columns=["scope_key"], errors="ignore")
    industry_note = (
        "The top categories are selected by their all-country candidate counts, then the "
        "bars are ranked by the selected all-country match rate. The selected countries "
        "replace, rather than add to, the non-U.S. series."
    )
    return industry_chart, industry_note, industry_summary, industry_table


@app.cell()
def industry_output(
    industry_chart,
    industry_country_selector,
    industry_metric_selector,
    industry_note,
    industry_selector,
    industry_show_table,
    industry_table,
    industry_table_scope_selector,
    industry_top_n_selector,
    mo,
):
    _figure = (
        industry_chart
        if industry_chart is not None
        else mo.callout(
            mo.md("No industry has a nonmissing all-country denominator."),
            kind="warn",
        )
    )
    _table_output = mo.md("")
    if industry_show_table.value:
        _table_output = mo.vstack(
            [
                industry_table_scope_selector,
                mo.ui.table(
                    industry_table,
                    pagination=True,
                    page_size=20,
                    selection="multi",
                    show_column_summaries=False,
                ),
            ],
            gap=1,
        )
    mo.vstack(
        [
            mo.md("## 3. Match rates across different industries"),
            mo.vstack(
                [
                    industry_metric_selector,
                    industry_selector,
                    industry_top_n_selector,
                    industry_country_selector,
                    mo.md(industry_note),
                ],
                gap=1,
            ),
            _figure,
            mo.accordion(
                {"Industry table (select rows and country scope)": _table_output}
            ),
        ],
        gap=1,
    )
    return


# -----------------------------------------------------------------------------
# Section 4. Other results
# -----------------------------------------------------------------------------


@app.cell()
def country_controls(METRIC_OPTIONS, MISSING_LABEL, fnh, mo, pl):
    _country_count = int(
        fnh.filter(pl.col("country") != MISSING_LABEL)
        .select(pl.col("country").n_unique())
        .item()
    )
    country_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition for country results",
        full_width=True,
    )
    country_top_n_selector = mo.ui.slider(
        start=1,
        stop=max(1, _country_count),
        value=min(50, max(1, _country_count)),
        step=1,
        show_value=True,
        label="Number of countries in the bar chart",
        full_width=True,
    )
    country_show_table = mo.ui.checkbox(
        value=False,
        label="Display the country table",
    )
    return country_metric_selector, country_show_table, country_top_n_selector


@app.cell()
def country_rates(
    MISSING_LABEL,
    classification_match_rates,
    country_iso3,
    country_metric_selector,
    country_top_n_selector,
    fnh,
    make_single_rate_chart,
    metric_columns,
    metric_label,
    px,
):
    country_metric = country_metric_selector.value
    country_summary = classification_match_rates(fnh, "country")
    country_chart, _country_display = make_single_rate_chart(
        country_summary,
        (f"Inventor {metric_label(country_metric)}-level match rates across countries"),
        country_metric,
        country_top_n_selector.value,
        "#0F766E",
    )
    _country_map_working = country_summary.loc[
        country_summary["group_value"] != MISSING_LABEL
    ].copy()
    _country_map_working["iso3"] = _country_map_working["group_value"].map(country_iso3)
    mapped_country_summary = _country_map_working.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = _country_map_working.loc[
        _country_map_working["iso3"].isna()
    ].copy()
    _rate_column, _, _, _ = metric_columns(country_metric)
    if mapped_country_summary.empty:
        country_map = None
    else:
        _color_max = max(float(mapped_country_summary[_rate_column].max()), 0.01)
        country_map = px.choropleth(
            mapped_country_summary,
            locations="iso3",
            color=_rate_column,
            hover_name="group_value",
            hover_data={
                "iso3": False,
                "candidate_spells": ":,",
                "matched_spells": ":,",
                "matched_users": ":,",
                _rate_column: ":.2%",
            },
            labels={
                "candidate_spells": "Candidate user-company observations",
                "matched_spells": "Matched user-company observations",
                "matched_users": "Matched users",
                _rate_column: f"{metric_label(country_metric).title()} match rate",
            },
            color_continuous_scale="Blues",
            range_color=(0.0, _color_max),
            projection="natural earth",
            title=(
                f"Inventor {metric_label(country_metric)}-level match rates across countries"
            ),
        )
        country_map.update_geos(showframe=False, showcoastlines=True)
        country_map.update_layout(
            height=560,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )
    return country_chart, country_map, country_summary, unmapped_country_summary


@app.cell()
def country_output(
    country_chart,
    country_map,
    country_metric_selector,
    country_show_table,
    country_summary,
    country_top_n_selector,
    mo,
    unmapped_country_summary,
):
    _bar = (
        country_chart
        if country_chart is not None
        else mo.callout(mo.md("No country has a nonmissing denominator."), kind="warn")
    )
    _map = (
        country_map
        if country_map is not None
        else mo.callout(mo.md("No country can be mapped."), kind="warn")
    )
    _table = mo.md("")
    if country_show_table.value:
        _table = mo.accordion(
            {
                "All country match-rate statistics": mo.ui.table(
                    country_summary,
                    pagination=True,
                    page_size=20,
                    selection="multi",
                    show_column_summaries=False,
                ),
                "Country labels not mapped to ISO-3": mo.ui.table(
                    unmapped_country_summary,
                    pagination=True,
                    page_size=20,
                    show_column_summaries=False,
                ),
            }
        )
    mo.vstack(
        [
            mo.md("## 4. Other results"),
            mo.md("### 4.1. Match rates across countries"),
            country_metric_selector,
            country_top_n_selector,
            _bar,
            _map,
            country_show_table,
            _table,
        ],
        gap=1,
    )
    return


@app.cell()
def state_controls(METRIC_OPTIONS, mo):
    state_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition for U.S.-state results",
        full_width=True,
    )
    state_show_table = mo.ui.checkbox(
        value=False,
        label="Display the U.S.-state table",
    )
    return state_metric_selector, state_show_table


@app.cell()
def state_rates(
    MISSING_LABEL,
    US_LABEL,
    classification_match_rates,
    fnh,
    metric_columns,
    metric_label,
    pl,
    px,
    state_metric_selector,
    us_state_code,
):
    state_metric = state_metric_selector.value
    us_fnh = fnh.filter(pl.col("country") == US_LABEL)
    state_summary = classification_match_rates(us_fnh, "state")
    _state_map_working = state_summary.loc[
        state_summary["group_value"] != MISSING_LABEL
    ].copy()
    _state_map_working["state_code"] = _state_map_working["group_value"].map(
        us_state_code
    )
    state_map_data = _state_map_working.dropna(subset=["state_code"]).copy()
    unmapped_state_summary = _state_map_working.loc[
        _state_map_working["state_code"].isna()
    ].copy()
    _rate_column, _, _, _ = metric_columns(state_metric)
    if state_map_data.empty:
        state_map = None
    else:
        _color_max = max(float(state_map_data[_rate_column].max()), 0.01)
        state_map = px.choropleth(
            state_map_data,
            locations="state_code",
            locationmode="USA-states",
            scope="usa",
            color=_rate_column,
            hover_name="group_value",
            hover_data={
                "state_code": False,
                "candidate_spells": ":,",
                "matched_spells": ":,",
                "matched_users": ":,",
                _rate_column: ":.2%",
            },
            labels={
                "candidate_spells": "Candidate user-company observations",
                "matched_spells": "Matched user-company observations",
                "matched_users": "Matched users",
                _rate_column: f"{metric_label(state_metric).title()} match rate",
            },
            color_continuous_scale="Purples",
            range_color=(0.0, _color_max),
            title=(
                f"Inventor {metric_label(state_metric)}-level match rates across U.S. states"
            ),
        )
        state_map.update_geos(scope="usa", visible=False)
        state_map.update_layout(
            height=600,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )
    return state_map, state_summary, unmapped_state_summary


@app.cell()
def state_output(
    state_map,
    state_metric_selector,
    state_show_table,
    state_summary,
    mo,
    unmapped_state_summary,
):
    _map = (
        state_map
        if state_map is not None
        else mo.callout(mo.md("No U.S. state can be mapped."), kind="warn")
    )
    _table = mo.md("")
    if state_show_table.value:
        _table = mo.accordion(
            {
                "All U.S.-state match-rate statistics": mo.ui.table(
                    state_summary,
                    pagination=True,
                    page_size=20,
                    selection="multi",
                    show_column_summaries=False,
                ),
                "State labels not mapped to USPS codes": mo.ui.table(
                    unmapped_state_summary,
                    pagination=True,
                    show_column_summaries=False,
                ),
            }
        )
    mo.vstack(
        [
            mo.md("### 4.2. Match rates across U.S. states"),
            state_metric_selector,
            _map,
            state_show_table,
            _table,
        ],
        gap=1,
    )
    return


@app.cell()
def seniority_controls(METRIC_OPTIONS, SCOPE_OPTIONS, available_countries, fnh, mo):
    seniority_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition for seniority results",
        full_width=True,
    )
    seniority_country_selector = mo.ui.multiselect(
        options=available_countries(fnh),
        value=[],
        label="Countries replacing the non-U.S. seniority series (optional)",
        full_width=True,
    )
    seniority_table_scope_selector = mo.ui.dropdown(
        options=SCOPE_OPTIONS,
        value="All countries",
        label="Country scope for the seniority table",
        full_width=True,
    )
    seniority_show_table = mo.ui.checkbox(
        value=False,
        label="Display the seniority table",
    )
    return (
        seniority_country_selector,
        seniority_metric_selector,
        seniority_show_table,
        seniority_table_scope_selector,
    )


@app.cell()
def seniority_rates(
    fnh,
    make_seniority_rate_chart,
    metric_label,
    pd,
    seniority_country_selector,
    seniority_metric_selector,
    seniority_table_scope_selector,
    scoped_classification_match_rates,
):
    seniority_metric = seniority_metric_selector.value
    seniority_summary = scoped_classification_match_rates(
        fnh,
        "seniority",
        selected_countries=seniority_country_selector.value,
    )
    seniority_chart, _seniority_display = make_seniority_rate_chart(
        seniority_summary,
        f"Inventor {metric_label(seniority_metric)}-level match rates by seniority",
        seniority_metric,
        ["#7C3AED", "#2563EB", "#0F766E"],
    )
    seniority_summary["seniority_order"] = pd.to_numeric(
        seniority_summary["group_value"], errors="coerce"
    )
    seniority_table = seniority_summary.loc[
        seniority_summary["scope_key"] == seniority_table_scope_selector.value
    ].sort_values(["seniority_order", "group_value"], na_position="last")
    return seniority_chart, seniority_summary, seniority_table


@app.cell()
def seniority_output(
    mo,
    seniority_chart,
    seniority_country_selector,
    seniority_metric_selector,
    seniority_show_table,
    seniority_table,
    seniority_table_scope_selector,
):
    _figure = (
        seniority_chart
        if seniority_chart is not None
        else mo.callout(
            mo.md("No seniority level has a nonmissing denominator."), kind="warn"
        )
    )
    _table = mo.md("")
    if seniority_show_table.value:
        _table = mo.vstack(
            [
                seniority_table_scope_selector,
                mo.ui.table(
                    seniority_table,
                    pagination=True,
                    selection="multi",
                    show_column_summaries=False,
                ),
            ],
            gap=1,
        )
    mo.vstack(
        [
            mo.md("### 4.3. Match rates across seniority levels"),
            seniority_metric_selector,
            seniority_country_selector,
            _figure,
            seniority_show_table,
            mo.accordion({"Seniority table (select rows and country scope)": _table}),
        ],
        gap=1,
    )
    return


@app.cell()
def start_month_controls(METRIC_OPTIONS, SCOPE_OPTIONS, available_countries, fnh, mo):
    start_month_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition for start-month results",
        full_width=True,
    )
    start_month_country_selector = mo.ui.multiselect(
        options=available_countries(fnh),
        value=[],
        label="Countries replacing the non-U.S. start-month series (optional)",
        full_width=True,
    )
    start_month_table_scope_selector = mo.ui.dropdown(
        options=SCOPE_OPTIONS,
        value="All countries",
        label="Country scope for the start-month table",
        full_width=True,
    )
    start_month_show_table = mo.ui.checkbox(
        value=False,
        label="Display the start-month table",
    )
    return (
        start_month_country_selector,
        start_month_metric_selector,
        start_month_show_table,
        start_month_table_scope_selector,
    )


@app.cell()
def start_month_rates(
    fnh,
    make_time_chart,
    metric_label,
    pd,
    pl,
    scoped_classification_match_rates,
    start_month_country_selector,
    start_month_metric_selector,
    start_month_table_scope_selector,
):
    start_month_metric = start_month_metric_selector.value
    start_month_summary = scoped_classification_match_rates(
        fnh.filter(pl.col("start_month").is_not_null()),
        "start_month",
        selected_countries=start_month_country_selector.value,
    )
    start_month_summary["month"] = pd.to_datetime(
        start_month_summary["start_month"],
        errors="coerce",
    )
    start_month_chart, _start_month_display = make_time_chart(
        start_month_summary,
        (
            f"Monthly inventor {metric_label(start_month_metric)}-level match rates "
            "by focal-hire start month"
        ),
        start_month_metric,
        ["#DC2626", "#2563EB", "#0F766E"],
    )
    start_month_table = start_month_summary.loc[
        start_month_summary["scope_key"] == start_month_table_scope_selector.value
    ].copy()
    return start_month_chart, start_month_summary, start_month_table


@app.cell()
def start_month_output(
    mo,
    start_month_chart,
    start_month_country_selector,
    start_month_metric_selector,
    start_month_show_table,
    start_month_table,
    start_month_table_scope_selector,
):
    _figure = (
        start_month_chart
        if start_month_chart is not None
        else mo.callout(
            mo.md("No valid focal-hire start month is available."), kind="warn"
        )
    )
    _table = mo.md("")
    if start_month_show_table.value:
        _table = mo.vstack(
            [
                start_month_table_scope_selector,
                mo.ui.table(
                    start_month_table,
                    pagination=True,
                    page_size=20,
                    selection="multi",
                    show_column_summaries=False,
                ),
            ],
            gap=1,
        )
    mo.vstack(
        [
            mo.md("### 4.4. Match rates across focal-hire start months"),
            start_month_metric_selector,
            start_month_country_selector,
            _figure,
            mo.md(
                "The chart is monthly and shows the all-country, U.S., and non-U.S. "
                "or selected-country series separately."
            ),
            start_month_show_table,
            mo.accordion({"Start-month table (select rows and country scope)": _table}),
        ],
        gap=1,
    )
    return


@app.cell()
def interpretation(mo):
    mo.callout(
        mo.md(
            "**Interpretation boundary.** A nonmatch combines never patenting, patenting "
            "outside the available PatentsView coverage, and failed user-to-inventor linkage. "
            "Differences across countries, occupations, industries, states, seniority, and "
            "time can therefore reflect coverage and linkage quality as well as real inventive "
            "activity."
        ),
        kind="warn",
    )
    return


if __name__ == "__main__":
    app.run()

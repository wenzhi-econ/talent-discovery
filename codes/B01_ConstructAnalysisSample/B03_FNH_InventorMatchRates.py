# ruff: noqa: PLR1711

"""
Task:
    Summarize inventor-linkage rates in the candidate focal-new-hire sample.

Inputs:
(a) data/b_temp_data/B01_ConstructAnalysisSample/FocalNewHires_AllIndustries/*.parquet
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv

Outputs:
(a) Five reactive marimo sections covering basic counts, occupations, industries,
    industry-occupation cells, and other geographic, seniority, and start-month results.

Notes:
(1) The focal-new-hire Parquet dataset has one retained row per user-company pair.
(2) The inventor crosswalk is reduced to users with a nonmissing inventor ID before matching.
(3) Every figure has its own local controls; there is no global analysis-control panel.
(4) User-company rates use focal-new-hire rows as the denominator. User rates use distinct
    users within each displayed group as the denominator.
(5) Figure accordions expose the exact plotted rows and the corresponding full tables.
(6) Bar-chart reference lines are pooled rates within the all-country, U.S., and selected
    (or default non-U.S.) scopes, rather than unweighted averages across countries.
(7) Country results retain economies with at least 1,000 candidate user-company observations,
    rank the eligible set by the selected match rate, and apply Top-N only after ranking.
(8) Analysis tables behind Top-N charts are constructed before Top-N is applied.
(9) Industry results retain categories with at least 1,000 all-country candidate user-company
    observations, rank the eligible set once, and apply Top-N only for display.

Run:
    $fnh_match_notebook = "codes/B01_ConstructAnalysisSample/B03_FNH_InventorMatchRates.py"
    conda run -s -n Talent marimo edit $fnh_match_notebook

Wang Wenzhi, with the help of Codex
Time: 2026-08-24
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    layout_file="layouts/B03_FNH_InventorMatchRates.slides.json",
    auto_download=["html"],
)


@app.cell
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


@app.cell
def title(mo):
    mo.vstack(
        [
            mo.md("""
                # Inventor match rates among candidate focal new hires

                The inventor database:
                - Gaurav sent me a list of users that can be matched to an inventor ID (or multiple inventor IDs).
                    - It is at user-level (if we temporarily ignore the small set of users who have multiple inventor IDs). I will call it the **inventor database**.
                    - I merge the universe sample of candidate focal new hires with the inventor database.
                - This section is mainly about heterogeneous match rates across occupations, industries, and countries.
                    - Match rates can be calculated at two levels: user-company level, and user level. The results are similar for these two different levels.
                    - As expected, there is huge difference in match rates across countries.
                """)
        ]
    )
    return


@app.cell
def helpers(alt, pd, pl, pycountry, re):
    MISSING_LABEL = "<Missing>"
    MIN_ALL_COUNTRY_INDUSTRY_HIRES = 1_000
    MIN_COUNTRY_CANDIDATE_SPELLS = 1_000
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
    RATE_TABLE_FORMATS = {
        "Match rate": "{:.2%}",
        "spell_match_rate": "{:.2%}",
        "user_match_rate": "{:.2%}",
        "spell_ci_low": "{:.2%}",
        "spell_ci_high": "{:.2%}",
        "user_ci_low": "{:.2%}",
        "user_ci_high": "{:.2%}",
        "reference_rate": "{:.2%}",
        "95% CI lower": "{:.2%}",
        "95% CI upper": "{:.2%}",
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
            pl.col("user_id").filter(pl.col("inventor_match")).n_unique().alias("matched_users"),
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
            _margin = _z * (_p * (1.0 - _p) / _n + _z**2 / (4.0 * _n**2)) ** 0.5 / _denominator
            return (_center - _margin).clip(lower=0.0), (_center + _margin).clip(upper=1.0)

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
        _result["spell_match_rate"] = (_result["matched_spells"] / _spell_n).where(_spell_n > 0)
        _result["user_match_rate"] = (_result["matched_users"] / _user_n).where(_user_n > 0)
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
        _summary["group_value"] = _summary[value_column].map(_display_value).astype("string")
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
            pl.col("user_id").filter(pl.col("inventor_match")).n_unique().alias("matched_users"),
        ).to_dicts()[0]
        _row["spell_match_rate"] = (
            _row["matched_spells"] / _row["candidate_spells"]
            if _row["candidate_spells"]
            else float("nan")
        )
        _row["user_match_rate"] = (
            _row["matched_users"] / _row["unique_users"] if _row["unique_users"] else float("nan")
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
                    else ((pl.col("country") != US_LABEL) & (pl.col("country") != MISSING_LABEL))
                ),
            ),
        )

    def reference_match_rates(data, metric, selected_countries=()):
        """Calculate exact overall rates for the three chart reference scopes."""

        _reference_labels = {
            "all": "All-country average",
            "us": "U.S. average",
            "non_us": "Non-U.S. average",
            "custom": "Selected-country average",
        }
        _rate_key = "spell_match_rate" if metric == "spell" else "user_match_rate"
        _rows = []
        for _scope_detail, _scope_key, _scope_data in country_scope_frames(
            data,
            selected_countries,
        ):
            _statistics = sample_statistics(_scope_data)
            _rows.append(
                {
                    "reference_scope": _reference_labels[_scope_key],
                    "scope_detail": _scope_detail,
                    "scope_key": _scope_key,
                    "reference_rate": _statistics[_rate_key],
                }
            )
        return pd.DataFrame(_rows)

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
        _rate_column, _low_column, _high_column, _matched_column = metric_columns(metric)
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

    def _rate_axis():
        """Use ticks precise enough to avoid duplicate percent labels."""

        return alt.Axis(format=".1%", tickCount=6, labelOverlap=False)

    def _reference_subtitle(reference_rates):
        _valid = reference_rates.dropna(subset=["reference_rate"])
        if _valid.empty:
            return None
        _values = "  |  ".join(
            f"{_row.reference_scope}: {_row.reference_rate:.2%}" for _row in _valid.itertuples()
        )
        return f"Reference lines — {_values}"

    def _add_reference_rules(bar_chart, reference_rates, rate_upper):
        """Layer styled vertical rules for all-country, U.S., and comparison rates."""

        _references = reference_rates.dropna(subset=["reference_rate"]).copy()
        if _references.empty:
            return bar_chart
        _scope_order = _references["reference_scope"].tolist()
        _rules = (
            alt.Chart(_references)
            .mark_rule(strokeWidth=2.25, opacity=0.95)
            .encode(
                x=alt.X(
                    "reference_rate:Q",
                    scale=alt.Scale(domain=[0, rate_upper]),
                ),
                color=alt.Color(
                    "reference_scope:N",
                    title="Reference match rate",
                    scale=alt.Scale(
                        domain=_scope_order,
                        range=["#111827", "#DC2626", "#7C3AED"],
                    ),
                    legend=alt.Legend(orient="top", direction="horizontal"),
                ),
                strokeDash=alt.StrokeDash(
                    "reference_scope:N",
                    scale=alt.Scale(
                        domain=_scope_order,
                        range=[[1, 0], [8, 4], [2, 3]],
                    ),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("reference_scope:N", title="Reference scope"),
                    alt.Tooltip("scope_detail:N", title="Included countries"),
                    alt.Tooltip(
                        "reference_rate:Q",
                        title="Reference match rate",
                        format=".2%",
                    ),
                ],
            )
        )
        return alt.layer(bar_chart, _rules).resolve_scale(color="independent")

    def make_grouped_rate_chart(
        summary,
        title,
        metric,
        top_n,
        color_range,
        reference_rates,
        minimum_candidate_spells=0,
    ):
        """Rank an eligible all-country table once, then display its Top-N prefix."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _all_country = summary.loc[
            (summary["scope_key"] == "all")
            & (summary["group_value"] != MISSING_LABEL)
            & (summary["candidate_spells"] >= int(minimum_candidate_spells))
        ].copy()
        _largest = _all_country.sort_values(
            ["candidate_spells", "display_label"],
            ascending=[False, True],
        )
        _rate_ranking = _all_country.sort_values(
            [_rate_column, "candidate_spells", "display_label"],
            ascending=[False, False, True],
            na_position="last",
        )
        _ranked = _rate_ranking.head(int(top_n)).copy()
        _top_values = _ranked["group_value"].tolist()
        _shown = summary.loc[summary["group_value"].isin(_top_values)].copy()
        _candidate_rank = dict(
            zip(_largest["group_value"], range(1, len(_largest) + 1), strict=True)
        )
        _rate_rank = dict(
            zip(
                _rate_ranking["group_value"],
                range(1, len(_rate_ranking) + 1),
                strict=True,
            )
        )
        _eligible_values = _all_country["group_value"].tolist()
        _analysis_table = summary.loc[summary["group_value"].isin(_eligible_values)].copy()
        _analysis_table["all_country_candidate_rank"] = _analysis_table["group_value"].map(
            _candidate_rank
        )
        _analysis_table["all_country_match_rate_rank"] = _analysis_table["group_value"].map(
            _rate_rank
        )
        _scope_rank = dict(
            zip(
                summary["scope_key"].drop_duplicates(),
                range(summary["scope_key"].nunique()),
                strict=True,
            )
        )
        _analysis_table["_scope_rank"] = _analysis_table["scope_key"].map(_scope_rank)
        _analysis_table = _analysis_table.sort_values(
            ["all_country_match_rate_rank", "_scope_rank"]
        ).drop(columns="_scope_rank")
        if _shown.empty:
            return None, _analysis_table

        _category_order = _ranked["display_label"].tolist()
        _scope_order = [_scope for _scope in summary["country_scope"].drop_duplicates().tolist()]
        _shown["country_scope"] = pd.Categorical(
            _shown["country_scope"],
            categories=_scope_order,
            ordered=True,
        )
        _reference_max = reference_rates["reference_rate"].max(skipna=True)
        _rate_upper = min(
            1.0,
            max(
                float(_shown[_high_column].max()) * 1.08,
                float(_reference_max) * 1.08,
                0.01,
            ),
        )
        _bars = (
            alt.Chart(_shown)
            .mark_bar(opacity=0.84)
            .encode(
                x=alt.X(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=_rate_axis(),
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
                    legend=alt.Legend(orient="bottom", direction="horizontal"),
                ),
                tooltip=_chart_tooltips(metric, include_scope=True),
            )
        )
        _chart = (
            _add_reference_rules(_bars, reference_rates, _rate_upper)
            .properties(
                width="container",
                height=max(300, len(_category_order) * 28),
                title=alt.TitleParams(
                    text=title,
                    subtitle=_reference_subtitle(reference_rates),
                    anchor="start",
                    subtitleColor="#4B5563",
                    subtitlePadding=6,
                ),
            )
            .configure_view(stroke=None)
        )
        return _chart, _analysis_table

    def make_seniority_rate_chart(
        summary,
        title,
        metric,
        color_range,
        reference_rates,
    ):
        """Plot seniority as grouped horizontal bars in numeric order."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _shown = summary.loc[summary["group_value"] != MISSING_LABEL].copy()
        if _shown.empty:
            return None, _shown
        _shown["seniority_order"] = pd.to_numeric(_shown["group_value"], errors="coerce")
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
        _reference_max = reference_rates["reference_rate"].max(skipna=True)
        _rate_upper = min(
            1.0,
            max(
                float(_shown[_high_column].max()) * 1.08,
                float(_reference_max) * 1.08,
                0.01,
            ),
        )
        _bars = (
            alt.Chart(_shown)
            .mark_bar(opacity=0.84)
            .encode(
                x=alt.X(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=_rate_axis(),
                ),
                y=alt.Y(
                    "display_label:O",
                    sort=_category_order,
                    title="Seniority level",
                    axis=alt.Axis(labelPadding=6),
                ),
                yOffset=alt.YOffset("country_scope:N", sort=_scope_order),
                color=alt.Color(
                    "country_scope:N",
                    title="Country scope",
                    scale=alt.Scale(domain=_scope_order, range=color_range),
                    legend=alt.Legend(orient="bottom", direction="horizontal"),
                ),
                tooltip=_chart_tooltips(metric, include_scope=True),
            )
        )
        _chart = (
            _add_reference_rules(_bars, reference_rates, _rate_upper)
            .properties(
                width="container",
                height=max(300, len(_category_order) * 34),
                title=alt.TitleParams(
                    text=title,
                    subtitle=_reference_subtitle(reference_rates),
                    anchor="start",
                    subtitleColor="#4B5563",
                    subtitlePadding=6,
                ),
            )
            .configure_view(stroke=None)
        )
        return _chart, _shown

    def make_single_rate_chart(
        summary,
        title,
        metric,
        top_n,
        color,
        reference_rates,
        minimum_candidate_spells=0,
    ):
        """Rank eligible categories once, then display the requested prefix."""

        _rate_column, _, _high_column, _ = metric_columns(metric)
        _analysis_table = summary.loc[
            (summary["group_value"] != MISSING_LABEL)
            & (summary["candidate_spells"] >= int(minimum_candidate_spells))
        ].copy()
        _analysis_table = _analysis_table.sort_values(
            [_rate_column, "candidate_spells", "display_label"],
            ascending=[False, False, True],
            na_position="last",
        )
        _analysis_table["match_rate_rank"] = range(1, len(_analysis_table) + 1)
        _shown = _analysis_table.head(int(top_n)).copy()
        if _shown.empty:
            return None, _analysis_table
        _order = _shown["display_label"].tolist()
        _reference_max = reference_rates["reference_rate"].max(skipna=True)
        _rate_upper = min(
            1.0,
            max(
                float(_shown[_high_column].max()) * 1.08,
                float(_reference_max) * 1.08,
                0.01,
            ),
        )
        _bars = (
            alt.Chart(_shown)
            .mark_bar(color=color, opacity=0.86)
            .encode(
                x=alt.X(
                    f"{_rate_column}:Q",
                    title=f"{metric_label(metric).title()} match rate",
                    scale=alt.Scale(domain=[0, _rate_upper]),
                    axis=_rate_axis(),
                ),
                y=alt.Y(
                    "display_label:N",
                    sort=_order,
                    title=None,
                    axis=alt.Axis(labelLimit=420, labelPadding=6),
                ),
                tooltip=_chart_tooltips(metric),
            )
        )
        _chart = (
            _add_reference_rules(_bars, reference_rates, _rate_upper)
            .properties(
                width="container",
                height=max(300, len(_shown) * 25),
                title=alt.TitleParams(
                    text=title,
                    subtitle=_reference_subtitle(reference_rates),
                    anchor="start",
                    subtitleColor="#4B5563",
                    subtitlePadding=6,
                ),
            )
            .configure_view(stroke=None)
        )
        return _chart, _analysis_table

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
                    axis=_rate_axis(),
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

    def make_industry_occupation_heatmap(
        summary,
        industry_order,
        occupation_order,
        industry_title,
        occupation_title,
        metric,
        scope_label,
    ):
        """Plot match rates and metric-consistent counts over a complete cell grid."""

        if summary.empty:
            return None
        _valid = summary.dropna(subset=["match_rate"]).copy()
        _color_max = max(float(_valid["match_rate"].max()), 0.01) if not _valid.empty else 0.01
        _text_threshold = _color_max * 0.55
        _candidate_title = (
            "Candidate user-company observations" if metric == "spell" else "Candidate users"
        )
        _matched_title = (
            "Matched user-company observations" if metric == "spell" else "Matched users"
        )
        _x = alt.X(
            "industry_label:N",
            sort=industry_order,
            title=industry_title,
            axis=alt.Axis(labelAngle=-25, labelLimit=260, labelPadding=8),
        )
        _y = alt.Y(
            "occupation_label:N",
            sort=occupation_order,
            title=occupation_title,
            axis=alt.Axis(labelLimit=360, labelPadding=8),
        )
        _tooltips = [
            alt.Tooltip("industry_label:N", title="Industry"),
            alt.Tooltip("occupation_label:N", title="Occupation"),
            alt.Tooltip("candidate_count:Q", title=_candidate_title, format=","),
            alt.Tooltip("matched_count:Q", title=_matched_title, format=","),
            alt.Tooltip("match_rate:Q", title="Match rate", format=".2%"),
            alt.Tooltip("ci_low:Q", title="95% CI lower", format=".2%"),
            alt.Tooltip("ci_high:Q", title="95% CI upper", format=".2%"),
        ]
        _base = (
            alt.Chart(summary)
            .mark_rect(color="#F3F4F6", stroke="#FFFFFF", strokeWidth=1.5)
            .encode(x=_x, y=_y, tooltip=_tooltips)
        )
        _colored_cells = (
            alt.Chart(_valid)
            .mark_rect(stroke="#FFFFFF", strokeWidth=1.5)
            .encode(
                x=_x,
                y=_y,
                color=alt.Color(
                    "match_rate:Q",
                    title="Match rate",
                    scale=alt.Scale(
                        domain=[0.0, _color_max],
                        scheme="blues",
                    ),
                    legend=alt.Legend(format=".1%"),
                ),
                tooltip=_tooltips,
            )
        )
        _rate_labels = (
            alt.Chart(_valid)
            .mark_text(dy=-8, fontSize=12, fontWeight=600)
            .encode(
                x=_x,
                y=_y,
                text=alt.Text("match_rate:Q", format=".1%"),
                color=alt.condition(
                    f"datum.match_rate >= {_text_threshold}",
                    alt.value("#FFFFFF"),
                    alt.value("#111827"),
                ),
            )
        )
        _count_labels = (
            alt.Chart(summary)
            .mark_text(dy=10, fontSize=10)
            .encode(
                x=_x,
                y=_y,
                text=alt.Text("count_label:N"),
                color=alt.condition(
                    f"datum.match_rate >= {_text_threshold}",
                    alt.value("#FFFFFF"),
                    alt.value("#4B5563"),
                ),
                tooltip=_tooltips,
            )
        )
        return (
            alt.layer(_base, _colored_cells, _rate_labels, _count_labels)
            .properties(
                width="container",
                height=max(360, len(occupation_order) * 70),
                title=alt.TitleParams(
                    text=(
                        f"Inventor {metric_label(metric)}-level match rates across "
                        "industry-occupation cells"
                    ),
                    subtitle=(
                        f"Country scope: {scope_label}. Cell labels report match rate "
                        "and matched / candidate counts."
                    ),
                    anchor="start",
                    subtitleColor="#4B5563",
                    subtitlePadding=6,
                ),
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
        MIN_ALL_COUNTRY_INDUSTRY_HIRES,
        MIN_COUNTRY_CANDIDATE_SPELLS,
        MISSING_LABEL,
        RATE_TABLE_FORMATS,
        SCOPE_OPTIONS,
        US_LABEL,
        available_countries,
        classification_match_rates,
        country_iso3,
        grouped_match_rates,
        hierarchy_number,
        make_grouped_rate_chart,
        make_industry_occupation_heatmap,
        make_seniority_rate_chart,
        make_single_rate_chart,
        make_time_chart,
        metric_columns,
        metric_label,
        reference_match_rates,
        sample_statistics,
        scoped_classification_match_rates,
        us_state_code,
    )


@app.cell
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
    )


@app.cell
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
                "Missing inventor IDs": int(_patent_links["pv_inventor_id"].null_count()),
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
    _matched_users = _patent_users.lazy().with_columns(pl.lit(True).alias("inventor_match"))
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


@app.cell
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


@app.cell
def basic_numbers(
    MISSING_LABEL,
    RATE_TABLE_FORMATS,
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
            fnh.filter((pl.col("country") != US_LABEL) & (pl.col("country") != MISSING_LABEL)),
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
            mo.md("""
                # 1. Basic numbers

                - Baseline all-country match rate for all candidate focal new hires is around 1.4%.
                - US match rate is around 3.9%; while non-US match rate is around 0.7%.
                """),
            mo.ui.table(
                basic_numbers_table,
                pagination=False,
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
            mo.accordion(
                {
                    "Inventor-crosswalk diagnostics": mo.ui.table(
                        link_diagnostics,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Country coverage": mo.ui.table(
                        pd.DataFrame({"Nonmissing countries": [len(available_countries(fnh))]}),
                        pagination=False,
                        show_column_summaries=False,
                    ),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def occupation_controls(
    METRIC_OPTIONS,
    OCCUPATION_LABELS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
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
    return (
        occupation_country_selector,
        occupation_metric_selector,
        occupation_selector,
        occupation_table_scope_selector,
    )


@app.cell
def occupation_top_n_control(MISSING_LABEL, fnh, mo, occupation_selector, pl):
    _occupation_column = occupation_selector.value
    _category_count = int(
        fnh.filter(pl.col(_occupation_column) != MISSING_LABEL)
        .select(pl.col(_occupation_column).n_unique())
        .item()
    )
    _max_categories = max(1, _category_count)
    _default_categories = (
        _max_categories if _occupation_column == "onet_code" else min(50, _max_categories)
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
    return (occupation_top_n_selector,)


@app.cell
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
    reference_match_rates,
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
    occupation_reference_table = reference_match_rates(
        fnh,
        occupation_metric,
        occupation_country_selector.value,
    )
    occupation_chart, occupation_plot_table = make_grouped_rate_chart(
        occupation_summary,
        (f"Inventor {metric_label(occupation_metric)}-level match rates by {occupation_title}"),
        occupation_metric,
        occupation_top_n_selector.value,
        ["#2563EB", "#0F766E", "#B45309"],
        occupation_reference_table,
    )
    occupation_plot_table = occupation_plot_table.drop(
        columns=["scope_key"],
        errors="ignore",
    )
    occupation_table = occupation_summary.loc[
        occupation_summary["scope_key"] == occupation_table_scope_selector.value
    ].copy()
    occupation_table = occupation_table.drop(columns=["scope_key"], errors="ignore")
    occupation_note = "Match rates are higher within U.S. new hires than non-U.S. new hires for almost all occupations."

    return (
        occupation_chart,
        occupation_note,
        occupation_plot_table,
        occupation_reference_table,
        occupation_table,
    )


@app.cell
def occupation_output(
    RATE_TABLE_FORMATS,
    mo,
    occupation_chart,
    occupation_country_selector,
    occupation_metric_selector,
    occupation_note,
    occupation_plot_table,
    occupation_reference_table,
    occupation_selector,
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
    _all_categories_table = mo.vstack(
        [
            occupation_table_scope_selector,
            mo.ui.table(
                occupation_table,
                pagination=True,
                page_size=20,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        ],
        gap=1,
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 2. Match rates across different occupations

                - As discussed before, several ONET occupations arise from problematic Revelio role mappings.
                - Therefore, it is necessary to compare classification systems and inspect the exact denominators before interpreting a high match rate substantively.
                """
            ),
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
                {
                    "View analysis table behind the figure": mo.ui.table(
                        occupation_plot_table,
                        pagination=True,
                        page_size=20,
                        selection="multi",
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View reference match rates": mo.ui.table(
                        occupation_reference_table,
                        pagination=False,
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View all occupation statistics by country scope": (_all_categories_table),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def industry_controls(
    DEFAULT_INDUSTRY,
    INDUSTRY_LABELS,
    METRIC_OPTIONS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
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
    return (
        industry_country_selector,
        industry_metric_selector,
        industry_selector,
        industry_table_scope_selector,
    )


@app.cell
def industry_top_n_control(
    MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    MISSING_LABEL,
    fnh,
    industry_selector,
    mo,
    pl,
):
    _industry_column = industry_selector.value
    _category_count = (
        fnh.filter(pl.col(_industry_column) != MISSING_LABEL)
        .group_by(_industry_column)
        .len()
        .filter(pl.col("len") >= MIN_ALL_COUNTRY_INDUSTRY_HIRES)
        .height
    )
    _max_categories = max(1, _category_count)
    industry_top_n_selector = mo.ui.slider(
        start=1,
        stop=_max_categories,
        value=min(50, _max_categories),
        step=1,
        show_value=True,
        label="Number of eligible industry categories in the bar chart",
        full_width=True,
    )
    return (industry_top_n_selector,)


@app.cell
def industry_rates(
    INDUSTRY_LABELS,
    INDUSTRY_TITLES,
    MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    fnh,
    industry_country_selector,
    industry_metric_selector,
    industry_selector,
    industry_table_scope_selector,
    industry_top_n_selector,
    make_grouped_rate_chart,
    metric_label,
    reference_match_rates,
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
    industry_reference_table = reference_match_rates(
        fnh,
        industry_metric,
        industry_country_selector.value,
    )
    industry_chart, industry_plot_table = make_grouped_rate_chart(
        industry_summary,
        (f"Inventor {metric_label(industry_metric)}-level match rates by {industry_title}"),
        industry_metric,
        industry_top_n_selector.value,
        ["#B45309", "#2563EB", "#0F766E"],
        industry_reference_table,
        minimum_candidate_spells=MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    )
    industry_plot_table = industry_plot_table.drop(
        columns=["scope_key"],
        errors="ignore",
    )
    _eligible_values = industry_summary.loc[
        (industry_summary["scope_key"] == "all")
        & (industry_summary["candidate_spells"] >= MIN_ALL_COUNTRY_INDUSTRY_HIRES),
        "group_value",
    ]
    industry_table = industry_summary.loc[
        (industry_summary["scope_key"] == industry_table_scope_selector.value)
        & industry_summary["group_value"].isin(_eligible_values)
    ].copy()
    industry_table = industry_table.drop(columns=["scope_key"], errors="ignore")
    industry_note = f"""
        Industries must have at least {MIN_ALL_COUNTRY_INDUSTRY_HIRES:,} candidate new hires in the all-country sample.
        This is to avoid some small-sized industries have an extremely higher match rates.
        """
    return (
        industry_chart,
        industry_note,
        industry_plot_table,
        industry_reference_table,
        industry_table,
    )


@app.cell
def industry_output(
    RATE_TABLE_FORMATS,
    industry_chart,
    industry_country_selector,
    industry_metric_selector,
    industry_note,
    industry_plot_table,
    industry_reference_table,
    industry_selector,
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
    _all_categories_table = mo.vstack(
        [
            industry_table_scope_selector,
            mo.ui.table(
                industry_table,
                pagination=True,
                page_size=20,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        ],
        gap=1,
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 3. Match rates across different industries

                - The results are quite aligned with our expectations.
                    - Electronics and BioPharma are industries with relatively high match rates.
                """
            ),
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
                {
                    "View analysis table behind the figure": mo.ui.table(
                        industry_plot_table,
                        pagination=True,
                        page_size=20,
                        selection="multi",
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View reference match rates": mo.ui.table(
                        industry_reference_table,
                        pagination=False,
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View all industry statistics by country scope": (_all_categories_table),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def industry_occupation_controls(
    DEFAULT_INDUSTRY,
    INDUSTRY_LABELS,
    METRIC_OPTIONS,
    OCCUPATION_LABELS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
):
    io_industry_variable_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in INDUSTRY_LABELS.items()},
        value=INDUSTRY_LABELS[DEFAULT_INDUSTRY],
        label="Industry classification",
        full_width=True,
    )
    io_occupation_variable_selector = mo.ui.dropdown(
        options={_label: _column for _column, _label in OCCUPATION_LABELS.items()},
        value=OCCUPATION_LABELS["onet_code"],
        label="Occupation classification",
        full_width=True,
    )
    io_country_scope_selector = mo.ui.dropdown(
        options=SCOPE_OPTIONS,
        value="All countries",
        label="Country scope",
        full_width=True,
    )
    io_custom_country_selector = mo.ui.multiselect(
        options=available_countries(fnh),
        value=[],
        label="Countries used when the country scope is Selected countries",
        full_width=True,
    )
    io_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition",
        full_width=True,
    )
    return (
        io_country_scope_selector,
        io_custom_country_selector,
        io_industry_variable_selector,
        io_metric_selector,
        io_occupation_variable_selector,
    )


@app.cell
def industry_occupation_category_controls(
    INDUSTRY_LABELS,
    INDUSTRY_TITLES,
    MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    MISSING_LABEL,
    OCCUPATION_LABELS,
    OCCUPATION_TITLES,
    classification_match_rates,
    fnh,
    io_industry_variable_selector,
    io_occupation_variable_selector,
    mo,
):
    io_industry_column = io_industry_variable_selector.value
    io_occupation_column = io_occupation_variable_selector.value
    io_industry_title = INDUSTRY_LABELS[io_industry_column]
    io_occupation_title = OCCUPATION_LABELS[io_occupation_column]

    _industry_title_column = INDUSTRY_TITLES.get(io_industry_column)
    _occupation_title_column = OCCUPATION_TITLES.get(io_occupation_column)
    io_industry_category_table = classification_match_rates(
        fnh,
        io_industry_column,
        _industry_title_column,
    )
    io_industry_category_table = io_industry_category_table.loc[
        (io_industry_category_table["group_value"] != MISSING_LABEL)
        & (io_industry_category_table["candidate_spells"] >= MIN_ALL_COUNTRY_INDUSTRY_HIRES)
    ].copy()
    io_occupation_category_table = classification_match_rates(
        fnh,
        io_occupation_column,
        _occupation_title_column,
    )
    io_occupation_category_table = io_occupation_category_table.loc[
        io_occupation_category_table["group_value"] != MISSING_LABEL
    ].copy()

    _industry_options = dict(
        zip(
            io_industry_category_table["display_label"],
            io_industry_category_table["group_value"],
            strict=True,
        )
    )
    _occupation_options = dict(
        zip(
            io_occupation_category_table["display_label"],
            io_occupation_category_table["group_value"],
            strict=True,
        )
    )
    _industry_label_by_value = dict(
        zip(
            io_industry_category_table["group_value"],
            io_industry_category_table["display_label"],
            strict=True,
        )
    )
    _default_industry_values = (
        [
            "Biotechnology and Life Sciences",
            "Pharmaceutical Manufacturing",
            "Pharmaceuticals",
        ]
        if io_industry_column == "rics_k400"
        else io_industry_category_table.head(3)["group_value"].tolist()
    )
    _default_industry_labels = [
        _industry_label_by_value[_value]
        for _value in _default_industry_values
        if _value in _industry_label_by_value
    ]
    if not _default_industry_labels:
        _default_industry_labels = io_industry_category_table.head(3)["display_label"].tolist()

    _default_occupation_titles = [
        "Microbiologists",
        "Chemical Engineers",
        "Bioengineers and Biomedical Engineers",
        "Biochemists and Biophysicists",
        "Chemists",
        "Animal Scientists",
    ]
    if io_occupation_column == "onet_code":
        _default_occupation_labels = []
        for _title in _default_occupation_titles:
            _matches = io_occupation_category_table.loc[
                io_occupation_category_table[_occupation_title_column] == _title,
                "display_label",
            ]
            if not _matches.empty:
                _default_occupation_labels.append(_matches.iloc[0])
    else:
        _default_occupation_labels = io_occupation_category_table.head(6)["display_label"].tolist()
    if not _default_occupation_labels:
        _default_occupation_labels = io_occupation_category_table.head(6)["display_label"].tolist()

    io_industry_values_selector = mo.ui.multiselect(
        options=_industry_options,
        value=_default_industry_labels,
        label="Industries shown on the x-axis",
        full_width=True,
    )
    io_occupation_values_selector = mo.ui.multiselect(
        options=_occupation_options,
        value=_default_occupation_labels,
        label="Occupations shown on the y-axis",
        full_width=True,
    )
    return (
        io_industry_category_table,
        io_industry_column,
        io_industry_title,
        io_industry_values_selector,
        io_occupation_category_table,
        io_occupation_column,
        io_occupation_title,
        io_occupation_values_selector,
    )


@app.cell
def industry_occupation_rates(
    MISSING_LABEL,
    SCOPE_OPTIONS,
    US_LABEL,
    fnh,
    grouped_match_rates,
    io_country_scope_selector,
    io_custom_country_selector,
    io_industry_category_table,
    io_industry_column,
    io_industry_title,
    io_industry_values_selector,
    io_metric_selector,
    io_occupation_category_table,
    io_occupation_column,
    io_occupation_title,
    io_occupation_values_selector,
    make_industry_occupation_heatmap,
    metric_columns,
    metric_label,
    pd,
    pl,
    sample_statistics,
):
    _industry_values = [str(_value) for _value in io_industry_values_selector.value]
    _occupation_values = [str(_value) for _value in io_occupation_values_selector.value]
    _scope_key = io_country_scope_selector.value
    _custom_countries = tuple(io_custom_country_selector.value or ())
    _scope_labels = {_value: _label for _label, _value in SCOPE_OPTIONS.items()}
    _scope_label = _scope_labels[_scope_key]
    if _scope_key == "custom":
        _scope_label = (
            "Selected countries: " + ", ".join(_custom_countries)
            if _custom_countries
            else "Selected countries: none"
        )

    _scope_data = fnh
    if _scope_key == "us":
        _scope_data = fnh.filter(pl.col("country") == US_LABEL)
    elif _scope_key == "non_us":
        _scope_data = fnh.filter(
            (pl.col("country") != US_LABEL) & (pl.col("country") != MISSING_LABEL)
        )
    elif _scope_key == "custom":
        _scope_data = fnh.filter(pl.col("country").is_in(_custom_countries))

    industry_occupation_heatmap = None
    industry_occupation_table = pd.DataFrame(
        columns=[
            "Industry",
            "Occupation",
            "Candidate observations",
            "Matched observations",
            "Match rate",
            "95% CI lower",
            "95% CI upper",
        ]
    )
    industry_occupation_totals = pd.DataFrame(
        columns=[
            "Country scope",
            "Observation level",
            "Selected industries",
            "Selected occupations",
            "Populated cells",
            "Candidate observations",
            "Matched observations",
            "Match rate",
        ]
    )
    industry_occupation_note = (
        "Select at least one industry and one occupation to construct the heatmap."
    )
    if _industry_values and _occupation_values:
        _cell_data = _scope_data.filter(
            pl.col(io_industry_column).cast(pl.String).is_in(_industry_values)
            & pl.col(io_occupation_column).cast(pl.String).is_in(_occupation_values)
        )
        _summary = grouped_match_rates(
            _cell_data,
            [io_industry_column, io_occupation_column],
        ).rename(
            columns={
                io_industry_column: "industry_value",
                io_occupation_column: "occupation_value",
            }
        )
        if not _summary.empty:
            _summary["industry_value"] = _summary["industry_value"].astype("string")
            _summary["occupation_value"] = _summary["occupation_value"].astype("string")

        _grid = pd.MultiIndex.from_product(
            [_industry_values, _occupation_values],
            names=["industry_value", "occupation_value"],
        ).to_frame(index=False)
        _grid = _grid.merge(
            _summary,
            on=["industry_value", "occupation_value"],
            how="left",
        )
        _count_columns = [
            "candidate_spells",
            "matched_spells",
            "unique_users",
            "matched_users",
        ]
        _grid[_count_columns] = _grid[_count_columns].fillna(0).astype("int64")

        _industry_labels = dict(
            zip(
                io_industry_category_table["group_value"].astype(str),
                io_industry_category_table["display_label"],
                strict=True,
            )
        )
        _occupation_labels = dict(
            zip(
                io_occupation_category_table["group_value"].astype(str),
                io_occupation_category_table["display_label"],
                strict=True,
            )
        )
        _grid["industry_label"] = _grid["industry_value"].map(_industry_labels)
        _grid["occupation_label"] = _grid["occupation_value"].map(_occupation_labels)
        _industry_order = [_industry_labels.get(_value, _value) for _value in _industry_values]
        _occupation_order = [
            _occupation_labels.get(_value, _value) for _value in _occupation_values
        ]

        _metric = io_metric_selector.value
        _rate_column, _low_column, _high_column, _matched_column = metric_columns(_metric)
        _candidate_column = "candidate_spells" if _metric == "spell" else "unique_users"
        _grid["candidate_count"] = _grid[_candidate_column]
        _grid["matched_count"] = _grid[_matched_column]
        _grid["match_rate"] = _grid[_rate_column]
        _grid["ci_low"] = _grid[_low_column]
        _grid["ci_high"] = _grid[_high_column]
        _grid["count_label"] = (
            _grid["matched_count"].map("{:,}".format)
            + " / "
            + _grid["candidate_count"].map("{:,}".format)
        )
        industry_occupation_heatmap = make_industry_occupation_heatmap(
            _grid,
            _industry_order,
            _occupation_order,
            io_industry_title,
            io_occupation_title,
            _metric,
            _scope_label,
        )
        industry_occupation_table = _grid[
            [
                "industry_label",
                "occupation_label",
                "candidate_count",
                "matched_count",
                "match_rate",
                "ci_low",
                "ci_high",
            ]
        ].rename(
            columns={
                "industry_label": "Industry",
                "occupation_label": "Occupation",
                "candidate_count": "Candidate observations",
                "matched_count": "Matched observations",
                "match_rate": "Match rate",
                "ci_low": "95% CI lower",
                "ci_high": "95% CI upper",
            }
        )

        _total_statistics = sample_statistics(_cell_data)
        _total_candidate_key = "candidate_spells" if _metric == "spell" else "unique_users"
        _total_matched_key = "matched_spells" if _metric == "spell" else "matched_users"
        _total_rate_key = "spell_match_rate" if _metric == "spell" else "user_match_rate"
        industry_occupation_totals = pd.DataFrame(
            [
                {
                    "Country scope": _scope_label,
                    "Observation level": metric_label(_metric).title(),
                    "Selected industries": len(_industry_values),
                    "Selected occupations": len(_occupation_values),
                    "Populated cells": int((_grid["candidate_count"] > 0).sum()),
                    "Candidate observations": _total_statistics[_total_candidate_key],
                    "Matched observations": _total_statistics[_total_matched_key],
                    "Match rate": _total_statistics[_total_rate_key],
                }
            ]
        )
        industry_occupation_note = ""
        if _metric == "user":
            industry_occupation_note += (
                " A user appearing in multiple industry-occupation cells is distinct within "
                "each cell but counted once in the pooled total below."
            )
    return (
        industry_occupation_heatmap,
        industry_occupation_note,
        industry_occupation_table,
        industry_occupation_totals,
    )


@app.cell
def industry_occupation_output(
    RATE_TABLE_FORMATS,
    industry_occupation_heatmap,
    industry_occupation_note,
    industry_occupation_table,
    industry_occupation_totals,
    io_country_scope_selector,
    io_custom_country_selector,
    io_industry_values_selector,
    io_industry_variable_selector,
    io_metric_selector,
    io_occupation_values_selector,
    io_occupation_variable_selector,
    mo,
):
    _figure = (
        industry_occupation_heatmap
        if industry_occupation_heatmap is not None
        else mo.callout(
            mo.md("Select at least one industry and one occupation."),
            kind="warn",
        )
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Match rates across industry-occupation cells

                - In the heatmap, I plot the match rates in selected industry-occupation cells.
                - As discussed before, occupation names can be misleading sometimes.
                - A reasonable workflow is to first determine the focal industry, and then select occupations based on occupation distribution and match rates within the selected industry.
                """
            ),
            mo.md("Heatmap cells show the match rate and matched / candidate counts. "),
            io_industry_variable_selector,
            io_industry_values_selector,
            io_occupation_variable_selector,
            io_occupation_values_selector,
            io_country_scope_selector,
            io_custom_country_selector,
            io_metric_selector,
            _figure,
            mo.md(industry_occupation_note),
            mo.ui.table(
                industry_occupation_totals,
                pagination=False,
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
            mo.accordion(
                {
                    "View industry-occupation cell statistics": mo.ui.table(
                        industry_occupation_table,
                        pagination=True,
                        page_size=20,
                        selection="multi",
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def country_controls(
    METRIC_OPTIONS,
    MIN_COUNTRY_CANDIDATE_SPELLS,
    MISSING_LABEL,
    fnh,
    mo,
    pl,
):
    _country_count = int(
        fnh.filter(pl.col("country") != MISSING_LABEL)
        .group_by("country")
        .len(name="candidate_spells")
        .filter(pl.col("candidate_spells") >= MIN_COUNTRY_CANDIDATE_SPELLS)
        .height
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
        label=(
            "Number of highest-match-rate countries in the bar chart "
            f"(eligibility: at least {MIN_COUNTRY_CANDIDATE_SPELLS:,} candidates)"
        ),
        full_width=True,
    )
    return country_metric_selector, country_top_n_selector


@app.cell
def country_rates(
    MIN_COUNTRY_CANDIDATE_SPELLS,
    classification_match_rates,
    country_iso3,
    country_metric_selector,
    country_top_n_selector,
    fnh,
    make_single_rate_chart,
    metric_columns,
    metric_label,
    px,
    reference_match_rates,
):
    country_metric = country_metric_selector.value
    country_summary = classification_match_rates(fnh, "country")
    country_reference_table = reference_match_rates(fnh, country_metric)
    country_chart, country_plot_table = make_single_rate_chart(
        country_summary,
        (
            f"Inventor {metric_label(country_metric)}-level match rates across economies "
            f"with at least {MIN_COUNTRY_CANDIDATE_SPELLS:,} candidate observations"
        ),
        country_metric,
        country_top_n_selector.value,
        "#0F766E",
        country_reference_table,
        MIN_COUNTRY_CANDIDATE_SPELLS,
    )
    country_summary = country_plot_table.copy()
    _country_map_working = country_summary.copy()
    _country_map_working["iso3"] = _country_map_working["group_value"].map(country_iso3)
    mapped_country_summary = _country_map_working.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = _country_map_working.loc[_country_map_working["iso3"].isna()].copy()
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
                "match_rate_rank": ":,",
                _rate_column: ":.2%",
            },
            labels={
                "candidate_spells": "Candidate user-company observations",
                "matched_spells": "Matched user-company observations",
                "matched_users": "Matched users",
                "match_rate_rank": "Match-rate rank",
                _rate_column: f"{metric_label(country_metric).title()} match rate",
            },
            color_continuous_scale="Blues",
            range_color=(0.0, _color_max),
            projection="natural earth",
            title=(
                f"Inventor {metric_label(country_metric)}-level match rates across economies "
                f"with at least {MIN_COUNTRY_CANDIDATE_SPELLS:,} candidate observations"
            ),
        )
        country_map.update_geos(showframe=False, showcoastlines=True)
        country_map.update_layout(
            height=560,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )
    return (
        country_chart,
        country_map,
        country_plot_table,
        country_reference_table,
        unmapped_country_summary,
    )


@app.cell
def country_output(
    MIN_COUNTRY_CANDIDATE_SPELLS,
    RATE_TABLE_FORMATS,
    country_chart,
    country_map,
    country_metric_selector,
    country_plot_table,
    country_reference_table,
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
    _table = mo.accordion(
        {
            "View eligible-country ranking behind the bar chart": mo.ui.table(
                country_plot_table,
                pagination=True,
                page_size=20,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
            "View reference match rates": mo.ui.table(
                country_reference_table,
                pagination=False,
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
            "View country labels not mapped to ISO-3": mo.ui.table(
                unmapped_country_summary,
                pagination=True,
                page_size=20,
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        }
    )
    mo.vstack(
        [
            mo.md("## 5. Other results"),
            mo.md("### 5.1. Match rates across economies"),
            mo.md(f"""
                - Countries must have at least {MIN_COUNTRY_CANDIDATE_SPELLS:,} candidate focal new hires.
                - This is to avoid some small-sized countries have unusually high match rates.
                - US is not the country with the highest rate, though this partially reflects there are substantially more new hires in US (3,491,981) when comparing with e.g., South Korea (59,025).
            """),
            country_metric_selector,
            country_top_n_selector,
            _bar,
            _map,
            _table,
        ],
        gap=1,
    )
    return


@app.cell
def state_controls(METRIC_OPTIONS, mo):
    state_metric_selector = mo.ui.dropdown(
        options=METRIC_OPTIONS,
        value="User-company level",
        label="Match-rate definition for U.S.-state results",
        full_width=True,
    )
    return (state_metric_selector,)


@app.cell
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
    _state_map_working = state_summary.loc[state_summary["group_value"] != MISSING_LABEL].copy()
    _state_map_working["state_code"] = _state_map_working["group_value"].map(us_state_code)
    state_map_data = _state_map_working.dropna(subset=["state_code"]).copy()
    unmapped_state_summary = _state_map_working.loc[_state_map_working["state_code"].isna()].copy()
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
            title=(f"Inventor {metric_label(state_metric)}-level match rates across U.S. states"),
        )
        state_map.update_geos(scope="usa", visible=False)
        state_map.update_layout(
            height=600,
            margin={"l": 0, "r": 0, "t": 55, "b": 0},
            coloraxis_colorbar={"tickformat": ".1%"},
        )
    return state_map, state_summary, unmapped_state_summary


@app.cell
def state_output(
    RATE_TABLE_FORMATS,
    mo,
    state_map,
    state_metric_selector,
    state_summary,
    unmapped_state_summary,
):
    _map = (
        state_map
        if state_map is not None
        else mo.callout(mo.md("No U.S. state can be mapped."), kind="warn")
    )
    _table = mo.accordion(
        {
            "View all U.S.-state match-rate statistics": mo.ui.table(
                state_summary,
                pagination=True,
                page_size=20,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
            "View state labels not mapped to USPS codes": mo.ui.table(
                unmapped_state_summary,
                pagination=True,
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        }
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.2. Match rates across U.S. states
                """
            ),
            state_metric_selector,
            _map,
            _table,
        ],
        gap=1,
    )
    return


@app.cell
def seniority_controls(
    METRIC_OPTIONS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
):
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
    return (
        seniority_country_selector,
        seniority_metric_selector,
        seniority_table_scope_selector,
    )


@app.cell
def seniority_rates(
    fnh,
    make_seniority_rate_chart,
    metric_label,
    pd,
    reference_match_rates,
    scoped_classification_match_rates,
    seniority_country_selector,
    seniority_metric_selector,
    seniority_table_scope_selector,
):
    seniority_metric = seniority_metric_selector.value
    seniority_summary = scoped_classification_match_rates(
        fnh,
        "seniority",
        selected_countries=seniority_country_selector.value,
    )
    seniority_reference_table = reference_match_rates(
        fnh,
        seniority_metric,
        seniority_country_selector.value,
    )
    seniority_chart, seniority_plot_table = make_seniority_rate_chart(
        seniority_summary,
        f"Inventor {metric_label(seniority_metric)}-level match rates by seniority",
        seniority_metric,
        ["#7C3AED", "#2563EB", "#0F766E"],
        seniority_reference_table,
    )
    seniority_summary["seniority_order"] = pd.to_numeric(
        seniority_summary["group_value"], errors="coerce"
    )
    seniority_table = seniority_summary.loc[
        seniority_summary["scope_key"] == seniority_table_scope_selector.value
    ].sort_values(["seniority_order", "group_value"], na_position="last")
    seniority_plot_table = seniority_plot_table.sort_values(
        ["seniority_order", "country_scope"],
        na_position="last",
    ).drop(columns=["scope_key"], errors="ignore")
    seniority_table = seniority_table.drop(columns=["scope_key"], errors="ignore")
    return (
        seniority_chart,
        seniority_plot_table,
        seniority_reference_table,
        seniority_table,
    )


@app.cell
def seniority_output(
    RATE_TABLE_FORMATS,
    mo,
    seniority_chart,
    seniority_country_selector,
    seniority_metric_selector,
    seniority_plot_table,
    seniority_reference_table,
    seniority_table,
    seniority_table_scope_selector,
):
    _figure = (
        seniority_chart
        if seniority_chart is not None
        else mo.callout(mo.md("No seniority level has a nonmissing denominator."), kind="warn")
    )
    _all_categories_table = mo.vstack(
        [
            seniority_table_scope_selector,
            mo.ui.table(
                seniority_table,
                pagination=True,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        ],
        gap=1,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.3. Match rates across seniority levels

                - **New hires who are more senior are more likely to show up in the inventor database.**
                - This is intuitive but has important implications for our sample construction: 
                    - Are we measuring productivity or seniority among the new hires?
                    - A company that prefers to hire more senior people will have higher observed productivity based on patents. However, it is not necessarily more productive.
                - We need to keep this differential match rates across the seniority distribution in mind.
                """
            ),
            seniority_metric_selector,
            seniority_country_selector,
            _figure,
            mo.accordion(
                {
                    "View statistics plotted in the figure": mo.ui.table(
                        seniority_plot_table,
                        pagination=True,
                        page_size=20,
                        selection="multi",
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View reference match rates": mo.ui.table(
                        seniority_reference_table,
                        pagination=False,
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View all seniority statistics by country scope": (_all_categories_table),
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def start_month_controls(
    METRIC_OPTIONS,
    SCOPE_OPTIONS,
    available_countries,
    fnh,
    mo,
):
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
    return (
        start_month_country_selector,
        start_month_metric_selector,
        start_month_table_scope_selector,
    )


@app.cell
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
    start_month_chart, start_month_plot_table = make_time_chart(
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
    start_month_plot_table = start_month_plot_table.sort_values(["month", "country_scope"]).drop(
        columns=["scope_key"], errors="ignore"
    )
    start_month_table = start_month_table.sort_values("month").drop(
        columns=["scope_key"],
        errors="ignore",
    )
    return start_month_chart, start_month_plot_table, start_month_table


@app.cell
def start_month_output(
    RATE_TABLE_FORMATS,
    mo,
    start_month_chart,
    start_month_country_selector,
    start_month_metric_selector,
    start_month_plot_table,
    start_month_table,
    start_month_table_scope_selector,
):
    _figure = (
        start_month_chart
        if start_month_chart is not None
        else mo.callout(mo.md("No valid focal-hire start month is available."), kind="warn")
    )
    _all_months_table = mo.vstack(
        [
            start_month_table_scope_selector,
            mo.ui.table(
                start_month_table,
                pagination=True,
                page_size=20,
                selection="multi",
                show_column_summaries=False,
                format_mapping=RATE_TABLE_FORMATS,
            ),
        ],
        gap=1,
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.4. Match rates across focal-hire start months
                """
            ),
            start_month_metric_selector,
            start_month_country_selector,
            _figure,
            mo.accordion(
                {
                    "View statistics plotted in the figure": mo.ui.table(
                        start_month_plot_table,
                        pagination=True,
                        page_size=20,
                        selection="multi",
                        show_column_summaries=False,
                        format_mapping=RATE_TABLE_FORMATS,
                    ),
                    "View all start-month statistics by country scope": (_all_months_table),
                }
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()

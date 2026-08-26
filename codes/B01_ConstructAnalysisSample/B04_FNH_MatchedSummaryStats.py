# ruff: noqa: PLR1711

"""Summary statistics for the inventor-matched candidate focal new hires."""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="full",
    layout_file="layouts/B04_FNH_MatchedSummaryStats.slides.json",
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
    # Inventor-matched candidate focal new hires: Summary statistics

    - In this report, I will offer summary statistics for the matched sample of candidate new hires.
    - When possible, I will always constrast them with the corresponding statistics for the universe sample of candidate new hires.
    - **One key message from such a comparison is that the inventor-matched sample is different from the universe sample of candidate new hires.**
    """)
    return


@app.cell
def _(alt, math, pd, pycountry, re):
    MISSING_LABEL = "<Missing>"
    MIN_ALL_COUNTRY_INDUSTRY_HIRES = 1_000
    US_LABEL = "United States"
    ALL_COUNTRIES_SCOPE = "__all__"
    US_SCOPE = "__us__"
    NON_US_SCOPE = "__non_us__"

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

    def available_country_options(data):
        countries = sorted(
            str(country)
            for country in data["country"].dropna().astype("string").unique()
            if str(country) not in {MISSING_LABEL, US_LABEL}
        )
        return {
            "All countries": ALL_COUNTRIES_SCOPE,
            US_LABEL: US_SCOPE,
            "Non-U.S. countries": NON_US_SCOPE,
            **{country: country for country in countries},
        }

    def filter_country_scope(data, selections):
        selected = tuple(selections or ())
        if not selected or ALL_COUNTRIES_SCOPE in selected:
            return data, "All countries"
        mask = pd.Series(False, index=data.index)
        labels = []
        if US_SCOPE in selected:
            mask |= data["country"].eq(US_LABEL)
            labels.append(US_LABEL)
        if NON_US_SCOPE in selected:
            mask |= data["country"].notna() & data["country"].ne(US_LABEL)
            labels.append("Non-U.S. countries")
        explicit_countries = [
            value
            for value in selected
            if value not in {ALL_COUNTRIES_SCOPE, US_SCOPE, NON_US_SCOPE}
        ]
        if explicit_countries:
            mask |= data["country"].isin(explicit_countries)
            labels.extend(explicit_countries)
        return data.loc[mask].copy(), ", ".join(labels)

    def category_selector_options(data, value_column, title_column=None):
        summary = distribution_table(data, value_column, title_column)
        summary = summary.loc[summary["value"].ne(MISSING_LABEL)]
        return dict(
            zip(
                summary["display_label"].astype(str),
                summary["value"].astype(str),
                strict=True,
            )
        )

    def restrict_to_eligible_industries(
        summary,
        eligible_values,
        value_column="value",
    ):
        result = summary.loc[summary[value_column].astype("string").isin(eligible_values)].copy()
        result = result.reset_index(drop=True)
        result["rank"] = range(1, len(result) + 1)
        return result

    def make_share_chart(
        summary,
        title,
        top_n=None,
        baseline=None,
        share_axis_title="Share within selected country scope",
    ):
        top = summary.head(top_n).copy() if top_n else summary.copy()
        top = top.drop(
            columns=["baseline_count", "baseline_share"],
            errors="ignore",
        )
        if baseline is not None:
            baseline_values = baseline[["display_label", "count", "share"]].rename(
                columns={"count": "baseline_count", "share": "baseline_share"}
            )
            top = top.merge(baseline_values, on="display_label", how="left")
            top[["baseline_count", "baseline_share"]] = top[
                ["baseline_count", "baseline_share"]
            ].fillna(0.0)
        else:
            top["baseline_count"] = math.nan
            top["baseline_share"] = math.nan
        if top.empty:
            return alt.Chart(pd.DataFrame({"display_label": []})).mark_bar()
        order = top["display_label"].tolist()
        maximum = float(top[["share", "baseline_share"]].max().max())
        domain = [0.0, maximum * 1.05 if maximum else 1.0]
        matched_sample = "Inventor-matched sample (bars)"
        universe_sample = "Universe sample (diamonds)"
        sample_domain = [matched_sample, universe_sample]
        sample_scale = alt.Scale(
            domain=sample_domain,
            range=["#2563EB", "#B91C1C"],
        )
        sample_legend = alt.Legend(
            title=None,
            orient="top",
            direction="horizontal",
        )
        tooltip = [
            alt.Tooltip("display_label:N", title="Category"),
            alt.Tooltip("count:Q", title="Inventor-matched hires", format=","),
            alt.Tooltip("share:Q", title="Matched share", format=".2%"),
            alt.Tooltip("baseline_count:Q", title="Universe hires", format=","),
            alt.Tooltip("baseline_share:Q", title="Universe share", format=".2%"),
            alt.Tooltip("rank:Q", title="Matched rank", format="d"),
        ]
        base = alt.Chart(top).encode(
            y=alt.Y(
                "display_label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=620, labelPadding=6),
            ),
            tooltip=tooltip,
        )
        bars = (
            base.transform_calculate(sample=f"'{matched_sample}'")
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X(
                    "share:Q",
                    title=share_axis_title,
                    axis=alt.Axis(format=".1%"),
                    scale=alt.Scale(domain=domain),
                ),
                color=alt.Color(
                    "sample:N",
                    scale=sample_scale,
                    legend=sample_legend,
                ),
            )
        )
        diamonds = (
            base.transform_calculate(sample=f"'{universe_sample}'")
            .mark_point(
                shape="diamond",
                filled=True,
                size=100,
            )
            .encode(
                x=alt.X("baseline_share:Q"),
                color=alt.Color(
                    "sample:N",
                    scale=sample_scale,
                    legend=sample_legend,
                ),
            )
        )
        return (
            alt.layer(bars, diamonds)
            .properties(
                width="container",
                height=max(280, len(top) * 24),
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
        distribution_table,
        filter_country_scope,
        hierarchy_number,
        joint_distribution_table,
        make_share_chart,
        restrict_to_eligible_industries,
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
        universe_fnh["start_month"] = pd.to_datetime(universe_fnh[date_column], errors="coerce")
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
        EXPECTED_RICS_COLUMNS,
        EXPECTED_ROLE_COLUMNS,
        available_rics_columns,
        available_role_columns,
        fnh,
        link_diagnostics,
        universe_fnh,
    )


@app.cell
def _(
    EXPECTED_RICS_COLUMNS,
    EXPECTED_ROLE_COLUMNS,
    MIN_ALL_COUNTRY_INDUSTRY_HIRES,
    MISSING_LABEL,
    available_country_options,
    available_rics_columns,
    available_role_columns,
    distribution_table,
    fnh,
    hierarchy_number,
    pd,
    universe_fnh,
):
    DEFAULT_BIOPHARM_INDUSTRIES = (
        "Biotechnology and Life Sciences",
        "Pharmaceutical Manufacturing",
        "Pharmaceuticals",
    )
    DEFAULT_FOCAL_OCCUPATION_TITLES = (
        "Microbiologists",
        "Chemical Engineers",
        "Bioengineers and Biomedical Engineers",
        "Biochemists and Biophysicists",
        "Chemists",
        "Animal Scientists",
    )
    CLASSIFICATION_LABELS = {
        "onet_code": "ONET code and title",
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
    _industry_columns = ("naics_code", *available_rics_columns)

    def _eligible_industry_values(column):
        _counts = (
            universe_fnh[column].fillna(MISSING_LABEL).astype("string").value_counts(dropna=False)
        )
        return frozenset(
            str(value)
            for value, count in _counts.items()
            if value != MISSING_LABEL and count >= MIN_ALL_COUNTRY_INDUSTRY_HIRES
        )

    eligible_industries_by_column = {
        column: _eligible_industry_values(column) for column in _industry_columns
    }
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
                    summary.loc[summary["value"] == MISSING_LABEL, "count"].sum() / len(fnh)
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
    expected.extend(["naics_code", "naics_description", "country", "state", "seniority"])
    schema_report = pd.DataFrame(
        [
            {
                "Variable": column,
                "Status": ("Available" if column in fnh.columns else "Absent from input schema"),
                "Missing rows": (int(fnh[column].isna().sum()) if column in fnh.columns else pd.NA),
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
    basic_numbers = pd.DataFrame(
        [
            {
                "Measure": "Number of focal new hires (user-company level)",
                "Inventor-matched sample": len(fnh),
                "Universe sample": len(universe_fnh),
            },
            {
                "Measure": "Number of distinct users",
                "Inventor-matched sample": int(fnh["user_id"].nunique(dropna=True)),
                "Universe sample": int(universe_fnh["user_id"].nunique(dropna=True)),
            },
            {
                "Measure": "Number of companies",
                "Inventor-matched sample": int(fnh["rcid"].nunique(dropna=True)),
                "Universe sample": int(universe_fnh["rcid"].nunique(dropna=True)),
            },
            {
                "Measure": "Number of countries",
                "Inventor-matched sample": int(fnh["country"].nunique(dropna=True)),
                "Universe sample": int(universe_fnh["country"].nunique(dropna=True)),
            },
        ]
    )
    basic_numbers["Match rate"] = (
        basic_numbers["Inventor-matched sample"] / basic_numbers["Universe sample"]
    )
    country_selector_options = available_country_options(universe_fnh)
    industry_selector_options = {
        CLASSIFICATION_LABELS[column]: column for column in ("naics_code", *available_rics_columns)
    }
    occupation_selector_options = {
        CLASSIFICATION_LABELS[column]: column for column in ("onet_code", *available_role_columns)
    }
    default_industry_column = (
        "rics_k400"
        if "rics_k400" in available_rics_columns
        else available_rics_columns[-1]
        if available_rics_columns
        else "naics_code"
    )
    return (
        CLASSIFICATION_LABELS,
        DEFAULT_BIOPHARM_INDUSTRIES,
        DEFAULT_FOCAL_OCCUPATION_TITLES,
        basic_numbers,
        classification_stats,
        country_selector_options,
        default_industry_column,
        eligible_industries_by_column,
        industry_selector_options,
        naics_title_diagnostic,
        occupation_selector_options,
        onet_title_diagnostic,
        schema_report,
        title_columns,
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
    _table_rows = "\n".join(
        "| "
        + str(measure)
        + " | "
        + f"{int(matched):,}"
        + " | "
        + f"{int(universe):,}"
        + " | "
        + f"{match_rate:.2%}"
        + " |"
        for measure, matched, universe, match_rate in basic_numbers.itertuples(
            index=False,
            name=None,
        )
    )
    mo.vstack(
        [
            mo.md("## 1. Basic numbers"),
            mo.md(
                """
                
                Recall the construction of the universe sample of candidate new hires:
                - Retaining employment spells in two-digit occupation groups 17 (Architecture and Engineering) and 19 (Life, Physical, and Social Science).
                - Retaining starts from January 2021 through December 2023.
                - Excluding missing geography or job-title information.
                - Excluding internships.
                - Retaining one spell per user-company cell.

                
                This matched notebook adds one restriction: 
                - Matched sample consists of those new hires who has at least one inventor ID in the  inventor database. 
                - Multiple inventor IDs will not duplicate the user-company observation. So the matched sample is still at user-company level. 
                - All summary statistics are simple averages over matched user-company observations.
                """
            ),
            mo.md(
                "| Measure | Inventor-matched sample | Universe sample | Match rate |\n"
                "|---|---:|---:|---:|\n" + _table_rows
            ),
            mo.accordion(
                {
                    "Classification coverage": mo.ui.table(
                        classification_stats,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Requested-variable coverage": mo.ui.table(
                        schema_report,
                        pagination=False,
                        show_column_summaries=False,
                    ),
                    "Inventor crosswalk diagnostics": mo.ui.table(
                        link_diagnostics,
                        pagination=False,
                        show_column_summaries=False,
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
def _(CLASSIFICATION_LABELS, country_selector_options, mo, occupation_selector_options):
    occupation_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    occupation_variable_selector = mo.ui.dropdown(
        options=occupation_selector_options,
        value=CLASSIFICATION_LABELS["onet_code"],
        label="Occupation variable",
        full_width=True,
    )
    occupation_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of top occupations",
    )
    return (
        occupation_country_selector,
        occupation_top_n_selector,
        occupation_variable_selector,
    )


@app.cell
def _(
    CLASSIFICATION_LABELS,
    distribution_table,
    filter_country_scope,
    fnh,
    make_share_chart,
    mo,
    occupation_country_selector,
    occupation_top_n_selector,
    occupation_variable_selector,
    title_columns,
    universe_fnh,
):
    _variable = occupation_variable_selector.value
    _classification_label = CLASSIFICATION_LABELS[_variable]
    _top_n = max(1, int(occupation_top_n_selector.value))
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        occupation_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        occupation_country_selector.value,
    )
    _summary = distribution_table(
        _matched_scope,
        _variable,
        title_columns.get(_variable),
    )
    _baseline = distribution_table(
        _universe_scope,
        _variable,
        title_columns.get(_variable),
    )
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} occupations by {_classification_label} — {_scope_label}",
        _top_n,
        baseline=_baseline,
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 2. The occupation distribution

                Choose either ONET or a Revelio occupation hierarchy. Bars report the
                inventor-matched sample; red diamonds report the universe baseline for the
                same categories.
                """
            ),
            mo.hstack(
                [
                    occupation_country_selector,
                    occupation_variable_selector,
                    occupation_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View matched occupation categories": mo.ui.table(
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
):
    industry_country_selector = mo.ui.multiselect(
        options=country_selector_options,
        value=["All countries"],
        label="Countries",
        full_width=True,
    )
    industry_variable_selector = mo.ui.dropdown(
        options=industry_selector_options,
        value=CLASSIFICATION_LABELS[default_industry_column],
        label="Industry variable",
        full_width=True,
    )
    industry_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=50,
        label="Number of top eligible industries",
    )
    return industry_country_selector, industry_top_n_selector, industry_variable_selector


@app.cell
def _(
    CLASSIFICATION_LABELS,
    distribution_table,
    eligible_industries_by_column,
    filter_country_scope,
    fnh,
    industry_country_selector,
    industry_top_n_selector,
    industry_variable_selector,
    make_share_chart,
    mo,
    restrict_to_eligible_industries,
    title_columns,
    universe_fnh,
):
    _variable = industry_variable_selector.value
    _classification_label = CLASSIFICATION_LABELS[_variable]
    _top_n = max(1, int(industry_top_n_selector.value))
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        industry_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        industry_country_selector.value,
    )
    _summary = distribution_table(
        _matched_scope,
        _variable,
        title_columns.get(_variable),
    )
    _baseline = distribution_table(
        _universe_scope,
        _variable,
        title_columns.get(_variable),
    )
    _eligible_values = eligible_industries_by_column[_variable]
    _summary = restrict_to_eligible_industries(_summary, _eligible_values)
    _baseline = restrict_to_eligible_industries(_baseline, _eligible_values)
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industries by {_classification_label} — {_scope_label}",
        _top_n,
        baseline=_baseline,
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 3. The industry distribution

                Choose either NAICS or a Revelio industry hierarchy. Only industries with at
                least 1,000 candidate focal new hires across all countries are shown. Bars
                report the inventor-matched sample; red diamonds report the universe baseline.
                """
            ),
            mo.hstack(
                [
                    industry_country_selector,
                    industry_variable_selector,
                    industry_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _chart,
            mo.accordion(
                {
                    "View eligible matched industry categories": mo.ui.table(
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
def _(
    eligible_industries_by_column,
    filter_country_scope,
    fnh,
    industry_occupation_country_selector,
    industry_occupation_industry_selector,
    industry_occupation_occupation_selector,
    industry_occupation_top_n_selector,
    joint_distribution_table,
    make_share_chart,
    mo,
    restrict_to_eligible_industries,
    title_columns,
    universe_fnh,
):
    _industry_column = industry_occupation_industry_selector.value
    _occupation_column = industry_occupation_occupation_selector.value
    _top_n = max(1, int(industry_occupation_top_n_selector.value))
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        industry_occupation_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        industry_occupation_country_selector.value,
    )
    _summary = joint_distribution_table(
        _matched_scope,
        _industry_column,
        _occupation_column,
        title_columns,
    )
    _baseline = joint_distribution_table(
        _universe_scope,
        _industry_column,
        _occupation_column,
        title_columns,
    )
    _eligible_values = eligible_industries_by_column[_industry_column]
    _summary = restrict_to_eligible_industries(
        _summary,
        _eligible_values,
        value_column="industry_value",
    )
    _baseline = restrict_to_eligible_industries(
        _baseline,
        _eligible_values,
        value_column="industry_value",
    )
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industry-occupation combinations — {_scope_label}",
        _top_n,
        baseline=_baseline,
    )
    mo.vstack(
        [
            mo.md(
                """
                ## 4. Industry-occupation distribution

                - In this section, I document the joint and marginal distribution of industry-occupation combinations among the universe sample of candidate focal new hires.
                """
            ),
            mo.md(
                """
                ### 4.1. Joint industry-occupation distribution
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
                    "View eligible matched industry-occupation combinations": mo.ui.table(
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
        value=25,
        label="Number of occupations",
    )
    return (
        occupation_within_industry_country_selector,
        occupation_within_industry_industry_variable_selector,
        occupation_within_industry_occupation_variable_selector,
        occupation_within_industry_top_n_selector,
    )


@app.cell
def _(
    DEFAULT_BIOPHARM_INDUSTRIES,
    category_selector_options,
    eligible_industries_by_column,
    mo,
    occupation_within_industry_industry_variable_selector,
    title_columns,
    universe_fnh,
):
    _industry_column = occupation_within_industry_industry_variable_selector.value
    _options = category_selector_options(
        universe_fnh,
        _industry_column,
        title_columns.get(_industry_column),
    )
    _eligible_values = eligible_industries_by_column[_industry_column]
    _options = {label: value for label, value in _options.items() if value in _eligible_values}
    _default_labels = [
        _label
        for _industry in DEFAULT_BIOPHARM_INDUSTRIES
        for _label, _value in _options.items()
        if _value == _industry
    ]
    if not _default_labels and _options:
        _default_labels = [next(iter(_options))]
    occupation_within_industry_industries_selector = mo.ui.multiselect(
        options=_options,
        value=_default_labels,
        label="Industries",
        full_width=True,
    )
    return (occupation_within_industry_industries_selector,)


@app.cell
def _(
    distribution_table,
    filter_country_scope,
    fnh,
    make_share_chart,
    mo,
    occupation_within_industry_country_selector,
    occupation_within_industry_industries_selector,
    occupation_within_industry_industry_variable_selector,
    occupation_within_industry_occupation_variable_selector,
    occupation_within_industry_top_n_selector,
    title_columns,
    universe_fnh,
):
    _industry_column = occupation_within_industry_industry_variable_selector.value
    _occupation_column = occupation_within_industry_occupation_variable_selector.value
    _industry_values = tuple(occupation_within_industry_industries_selector.value or ())
    _top_n = max(1, int(occupation_within_industry_top_n_selector.value))
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        occupation_within_industry_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        occupation_within_industry_country_selector.value,
    )
    if _industry_values:
        _matched_conditioned = _matched_scope.loc[
            _matched_scope[_industry_column].astype("string").isin(_industry_values)
        ]
        _universe_conditioned = _universe_scope.loc[
            _universe_scope[_industry_column].astype("string").isin(_industry_values)
        ]
    else:
        _matched_conditioned = _matched_scope.iloc[0:0]
        _universe_conditioned = _universe_scope.iloc[0:0]
    _summary = distribution_table(
        _matched_conditioned,
        _occupation_column,
        title_columns.get(_occupation_column),
    )
    _baseline = distribution_table(
        _universe_conditioned,
        _occupation_column,
        title_columns.get(_occupation_column),
    )
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} occupations within selected industries — {_scope_label}",
        _top_n,
        baseline=_baseline,
        share_axis_title="Share within selected industry set",
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 4.2. Occupation distribution within an industry
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
            occupation_within_industry_industries_selector,
            _chart,
            mo.accordion(
                {
                    "View matched conditional occupation categories": mo.ui.table(
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
        value=25,
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
    DEFAULT_FOCAL_OCCUPATION_TITLES,
    category_selector_options,
    industry_within_occupation_occupation_variable_selector,
    mo,
    title_columns,
    universe_fnh,
):
    _occupation_column = industry_within_occupation_occupation_variable_selector.value
    _options = category_selector_options(
        universe_fnh,
        _occupation_column,
        title_columns.get(_occupation_column),
    )
    _default_labels = [
        _label
        for _title in DEFAULT_FOCAL_OCCUPATION_TITLES
        for _label in _options
        if _label.endswith(f" — {_title}")
    ]
    if not _default_labels and _options:
        _default_labels = [next(iter(_options))]
    industry_within_occupation_occupations_selector = mo.ui.multiselect(
        options=_options,
        value=_default_labels,
        label="Occupations",
        full_width=True,
    )
    return (industry_within_occupation_occupations_selector,)


@app.cell
def _(
    distribution_table,
    eligible_industries_by_column,
    filter_country_scope,
    fnh,
    industry_within_occupation_country_selector,
    industry_within_occupation_industry_variable_selector,
    industry_within_occupation_occupation_variable_selector,
    industry_within_occupation_occupations_selector,
    industry_within_occupation_top_n_selector,
    make_share_chart,
    mo,
    restrict_to_eligible_industries,
    title_columns,
    universe_fnh,
):
    _industry_column = industry_within_occupation_industry_variable_selector.value
    _occupation_column = industry_within_occupation_occupation_variable_selector.value
    _occupation_values = tuple(industry_within_occupation_occupations_selector.value or ())
    _top_n = max(1, int(industry_within_occupation_top_n_selector.value))
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        industry_within_occupation_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        industry_within_occupation_country_selector.value,
    )
    if _occupation_values:
        _matched_conditioned = _matched_scope.loc[
            _matched_scope[_occupation_column].astype("string").isin(_occupation_values)
        ]
        _universe_conditioned = _universe_scope.loc[
            _universe_scope[_occupation_column].astype("string").isin(_occupation_values)
        ]
    else:
        _matched_conditioned = _matched_scope.iloc[0:0]
        _universe_conditioned = _universe_scope.iloc[0:0]
    _summary = distribution_table(
        _matched_conditioned,
        _industry_column,
        title_columns.get(_industry_column),
    )
    _baseline = distribution_table(
        _universe_conditioned,
        _industry_column,
        title_columns.get(_industry_column),
    )
    _eligible_values = eligible_industries_by_column[_industry_column]
    _summary = restrict_to_eligible_industries(_summary, _eligible_values)
    _baseline = restrict_to_eligible_industries(_baseline, _eligible_values)
    _chart = make_share_chart(
        _summary,
        f"Top {_top_n} industries within selected occupations — {_scope_label}",
        _top_n,
        baseline=_baseline,
        share_axis_title="Share within selected occupation set",
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 4.3. Industry distribution within an occupation
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
            industry_within_occupation_occupations_selector,
            _chart,
            mo.accordion(
                {
                    "View eligible matched conditional industry categories": mo.ui.table(
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
    country_iso3,
    distribution_table,
    fnh,
    math,
    universe_fnh,
    us_state_code,
):
    country_summary = distribution_table(fnh, "country")
    baseline_country_summary = distribution_table(universe_fnh, "country")
    _baseline_country_values = baseline_country_summary[
        ["country", "display_label", "count", "share"]
    ].rename(
        columns={
            "country": "baseline_country",
            "count": "baseline_count",
            "share": "baseline_share",
        }
    )
    country_summary = country_summary.merge(
        _baseline_country_values,
        on="display_label",
        how="outer",
    )
    country_summary["country"] = country_summary["country"].fillna(
        country_summary["baseline_country"]
    )
    country_summary[["count", "share", "baseline_count", "baseline_share"]] = country_summary[
        ["count", "share", "baseline_count", "baseline_share"]
    ].fillna(0.0)
    country_summary = country_summary.sort_values(
        ["share", "baseline_share", "display_label"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    country_summary["rank"] = range(1, len(country_summary) + 1)
    country_summary["value"] = country_summary["display_label"]
    country_summary["iso3"] = country_summary["country"].map(country_iso3)
    country_summary["log10_count"] = country_summary["count"].map(
        lambda count: math.log10(count) if count > 0 else 0
    )
    mapped_country_summary = country_summary.dropna(subset=["iso3"]).copy()
    unmapped_country_summary = country_summary.loc[country_summary["iso3"].isna()].copy()
    state_working = fnh.loc[fnh["country"] == "United States", ["state"]].copy()
    state_baseline_working = universe_fnh.loc[
        universe_fnh["country"] == "United States", ["state"]
    ].copy()
    for working in [state_working, state_baseline_working]:
        working["state"] = working["state"].fillna("<Missing>")
    us_state_summary = (
        state_working.groupby("state", observed=True).size().rename("count").reset_index()
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
        baseline_state_summary["baseline_count"] / baseline_state_summary["baseline_count"].sum()
    )
    us_state_summary = us_state_summary.merge(
        baseline_state_summary,
        on="state",
        how="left",
    )
    us_state_summary["baseline_share_within_country"] = us_state_summary[
        "baseline_share_within_country"
    ].fillna(0.0)
    us_state_summary["state_code"] = us_state_summary["state"].map(us_state_code)
    state_map_data = us_state_summary.dropna(subset=["state_code"]).copy()
    unmatched_state_data = us_state_summary.loc[us_state_summary["state_code"].isna()].copy()
    state_map_coverage = (
        state_map_data["count"].sum() / us_state_summary["count"].sum()
        if not us_state_summary.empty
        else 0.0
    )
    return (
        baseline_country_summary,
        country_summary,
        mapped_country_summary,
        state_map_coverage,
        state_map_data,
        unmapped_country_summary,
        unmatched_state_data,
    )


@app.cell
def _(
    baseline_country_summary,
    country_summary,
    make_share_chart,
    mapped_country_summary,
    mo,
    px,
    unmapped_country_summary,
):
    _map = px.choropleth(
        mapped_country_summary,
        locations="iso3",
        color="log10_count",
        hover_name="country",
        hover_data={
            "iso3": False,
            "log10_count": False,
            "count": ":,",
            "share": ":.2%",
            "baseline_count": ":,",
            "baseline_share": ":.2%",
        },
        labels={
            "count": "Inventor-matched hires",
            "share": "Matched share",
            "baseline_count": "Universe hires",
            "baseline_share": "Universe share",
            "log10_count": "Log10 inventor-matched hires",
        },
        color_continuous_scale="Blues",
        projection="natural earth",
        title="Inventor-matched candidate focal new hires by country",
    ).update_geos(showframe=False, showcoastlines=True)
    _country_bars = make_share_chart(
        country_summary,
        "Country distribution: matched sample versus universe",
        baseline=baseline_country_summary,
    )
    mo.vstack(
        [
            mo.md("## 5. Other results\n\n### 5.1. Geography distribution"),
            mo.md(
                "In the inventor-matched sample, US dominates with a share of around 60%; followed by Germany, France, UK, India."
            ),
            _country_bars,
            _map,
            mo.accordion(
                {
                    "View country statistics": mo.ui.table(
                        country_summary,
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
            _figure,
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
    alt,
    distribution_table,
    filter_country_scope,
    fnh,
    mo,
    pd,
    seniority_country_selector,
    universe_fnh,
):
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        seniority_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        seniority_country_selector.value,
    )
    matched = distribution_table(_matched_scope, "seniority")[
        ["display_label", "count", "share"]
    ].copy()
    baseline = distribution_table(_universe_scope, "seniority")[["display_label", "share"]].copy()
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
    plot_data["seniority_order"] = pd.to_numeric(plot_data["display_label"], errors="coerce")
    order = (
        plot_data[["display_label", "seniority_order"]]
        .drop_duplicates()
        .sort_values(["seniority_order", "display_label"], na_position="last")["display_label"]
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
        .properties(
            width="container",
            height=380,
            title=alt.TitleParams(
                text=f"Seniority distribution — {_scope_label}",
                anchor="start",
            ),
        )
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md(
                """
                ### 5.2. Seniority distribution

                - Matched sample has a different seniority distribution compared with all candidate new hires.
                - Again, as I have said before, this could have serious implications for our interpretations of the patents-based productivity results.
                - Firms that hire workers at later career stages may inherit larger pre-hire patent stocks.
                """
            ),
            seniority_country_selector,
            _figure,
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
    alt,
    filter_country_scope,
    fnh,
    mo,
    pd,
    time_series_country_selector,
    universe_fnh,
):
    _matched_scope, _scope_label = filter_country_scope(
        fnh,
        time_series_country_selector.value,
    )
    _universe_scope, _ = filter_country_scope(
        universe_fnh,
        time_series_country_selector.value,
    )
    matched_series = (
        _matched_scope.dropna(subset=["start_month"])
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
        _universe_scope.dropna(subset=["start_month"])
        .assign(
            start_month=lambda data: (
                pd.to_datetime(data["start_month"]).dt.to_period("M").dt.to_timestamp()
            )
        )
        .groupby("start_month", observed=True)
        .size()
        .rename("universe_count")
    )
    time_series = pd.concat([matched_series, baseline_series], axis=1).fillna(0).reset_index()
    time_series["matched_share_of_universe"] = time_series["matched_count"] / time_series[
        "universe_count"
    ].replace(0, pd.NA)
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
    _share_plot_data = pd.concat([_matched_month_plot, _baseline_month_plot], ignore_index=True)
    _count_figure = (
        alt.Chart(time_series)
        .mark_line(point=True, color="#7C3AED")
        .encode(
            x=alt.X("start_month:T", title="Employment start month"),
            y=alt.Y("matched_count:Q", title="Inventor-matched candidate new hires"),
            tooltip=[
                alt.Tooltip("start_month:T", title="Start month", format="%b %Y"),
                alt.Tooltip("matched_count:Q", title="Inventor-matched hires", format=","),
                alt.Tooltip(
                    "matched_share_of_universe:Q",
                    title="Matched share of universe",
                    format=".2%",
                ),
            ],
        )
        .properties(
            width="container",
            height=320,
            title=alt.TitleParams(
                text=f"Inventor-matched hires over time — {_scope_label}",
                anchor="start",
            ),
        )
        .configure_view(stroke=None)
    )
    _share_figure = (
        alt.Chart(_share_plot_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("start_month:T", title="Employment start month"),
            y=alt.Y("share:Q", title="Share of each sample", axis=alt.Axis(format=".1%")),
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
        .properties(
            width="container",
            height=360,
            title=alt.TitleParams(
                text=f"Monthly composition by sample — {_scope_label}",
                anchor="start",
            ),
        )
        .configure_view(stroke=None)
    )
    mo.vstack(
        [
            mo.md("### 5.3. Time-series"),
            time_series_country_selector,
            _count_figure,
            _share_figure,
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()

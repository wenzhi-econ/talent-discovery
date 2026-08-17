# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair>=6.2,<7",
#     "marimo>=0.23.14,<0.24",
#     "pandas>=3.0,<4",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _():
    import altair as alt
    import marimo as mo
    import pandas as pd
    import sys

    return alt, mo, pd, sys


@app.cell(hide_code=True)
def title(mo):
    mo.md(r"""
    # LinkedIn occupation coverage in 2022

    This notebook compares the occupation and age distributions of US LinkedIn
    users with three national benchmarks. **ACS is the main benchmark**; CPS and
    OEWS provide robustness checks.

    Use the controls to examine three questions:

    1. Which occupations are most prominent or overrepresented in LinkedIn?
    2. Are younger workers represented more strongly than older workers?
    3. Do conclusions change across benchmarks or when active profiles are used?
    """)
    return


@app.cell(hide_code=True)
def load_data(mo, pd, sys):
    data_dir = mo.notebook_location() / "public" / "data"

    def read_csv(filename, **kwargs):
        data_path = data_dir / filename
        if sys.platform == "emscripten":
            from pyodide.http import open_url

            return pd.read_csv(open_url(str(data_path)), **kwargs)
        return pd.read_csv(str(data_path), **kwargs)

    acs_raw = read_csv(
        "LinkedIn_vs_ACS_2022_OccAgeShares.csv",
        dtype={"acs_occupation_code": "string"},
    )
    cps_raw = read_csv(
        "LinkedIn_vs_CPS_2022_OccAgeShares.csv",
        dtype={"cps_occupation_code": "string"},
    )
    oews_raw = read_csv(
        "LinkedIn_vs_OEWS_2022_OccShares.csv",
        dtype={"oews_occupation_code": "string"},
    )

    acs_crosswalk = read_csv(
        "Crosswalk_ONET_to_ACS_2018PUMSOcc.csv",
        dtype="string",
    )
    cps_crosswalk = read_csv(
        "Crosswalk_ONET_to_CPS_2018CensusOcc.csv",
        dtype="string",
    )
    oews_crosswalk = read_csv(
        "Crosswalk_ONET_to_OEWS_2018SOC.csv",
        dtype="string",
    )
    return (
        acs_crosswalk,
        acs_raw,
        cps_crosswalk,
        cps_raw,
        oews_crosswalk,
        oews_raw,
    )


@app.cell(hide_code=True)
def standardize_data(
    acs_crosswalk,
    acs_raw,
    cps_crosswalk,
    cps_raw,
    oews_crosswalk,
    oews_raw,
    pd,
):
    def standardize_occupation_shares(
        data,
        source,
        code_column,
        title_column,
    ):
        selected_columns = [
            code_column,
            title_column,
            "linkedin_user_count",
            "linkedin_active_user_count",
            "benchmark_worker_count",
            "linkedin_user_share",
            "linkedin_active_user_share",
            "benchmark_worker_share",
        ]
        standardized = data.loc[:, selected_columns].rename(
            columns={
                code_column: "occupation_code",
                title_column: "occupation_title",
            }
        )
        standardized["benchmark_source"] = source
        return standardized


    _acs_all_ages = standardize_occupation_shares(
        acs_raw.loc[acs_raw["age_scope"] == "All ages"],
        "ACS",
        "acs_occupation_code",
        "acs_occupation_title",
    )
    _cps_all_ages = standardize_occupation_shares(
        cps_raw.loc[cps_raw["age_scope"] == "All ages"],
        "CPS",
        "cps_occupation_code",
        "cps_occupation_title",
    )
    _oews_all_ages = standardize_occupation_shares(
        oews_raw,
        "OEWS",
        "oews_occupation_code",
        "oews_occupation_title",
    )

    occupation_shares = pd.concat(
        [_acs_all_ages, _cps_all_ages, _oews_all_ages],
        ignore_index=True,
    )

    _crosswalk_frames = []
    for _source, _data, _code, _title in [
        (
            "ACS",
            acs_crosswalk,
            "acs_occupation_code",
            "acs_occupation_title",
        ),
        (
            "CPS",
            cps_crosswalk,
            "cps_occupation_code",
            "cps_occupation_title",
        ),
        (
            "OEWS",
            oews_crosswalk,
            "oews_occupation_code",
            "oews_occupation_title",
        ),
    ]:
        _frame = _data[
            ["onet_code", "onet_title", _code, _title, "mapping_status"]
        ].rename(
            columns={
                _code: "occupation_code",
                _title: "occupation_title",
            }
        )
        _frame["benchmark_source"] = _source
        _crosswalk_frames.append(_frame)

    crosswalk_long = pd.concat(_crosswalk_frames, ignore_index=True)
    return crosswalk_long, occupation_shares


@app.cell(hide_code=True)
def occupation_section(mo):
    mo.md(r"""
    ## 1. Occupations with the strongest LinkedIn presence

    ### 1.1. Largest and most overrepresented occupations

    Bars show LinkedIn's share and diamonds show the selected benchmark share.
    Ranking by the LinkedIn-to-benchmark ratio identifies overrepresentation;
    ranking by LinkedIn share identifies the largest occupations on the platform.
    """)
    return


@app.cell(hide_code=True)
def top_controls(mo):
    benchmark_selector = mo.ui.dropdown(
        options={
            "ACS - main benchmark": "ACS",
            "CPS - robustness check": "CPS",
            "OEWS - robustness check": "OEWS",
        },
        value="ACS - main benchmark",
        label="US benchmark",
    )
    linkedin_sample_selector = mo.ui.radio(
        options={
            "All LinkedIn users": "linkedin_user_share",
            "Active LinkedIn users": "linkedin_active_user_share",
        },
        value="All LinkedIn users",
        inline=True,
        label="LinkedIn sample",
    )
    ranking_selector = mo.ui.dropdown(
        options={
            "Largest LinkedIn share": "linkedin_share_desc",
            "Most overrepresented": "ratio_desc",
            "Most underrepresented": "ratio_asc",
        },
        value="Largest LinkedIn share",
        label="Ordering",
    )
    top_n_selector = mo.ui.slider(
        start=10,
        stop=50,
        step=5,
        value=30,
        show_value=True,
        label="Number of occupations",
    )

    mo.hstack(
        [
            benchmark_selector,
            linkedin_sample_selector,
            ranking_selector,
            top_n_selector,
        ],
        justify="start",
        align="end",
        gap=2,
        wrap=True,
    )
    return (
        benchmark_selector,
        linkedin_sample_selector,
        ranking_selector,
        top_n_selector,
    )


@app.cell(hide_code=True)
def rank_occupations(
    benchmark_selector,
    linkedin_sample_selector,
    occupation_shares,
    ranking_selector,
    top_n_selector,
):
    selected_source = benchmark_selector.value
    selected_share_column = linkedin_sample_selector.value
    selected_count_column = (
        "linkedin_user_count"
        if selected_share_column == "linkedin_user_share"
        else "linkedin_active_user_count"
    )
    selected_sample_label = (
        "All LinkedIn users"
        if selected_share_column == "linkedin_user_share"
        else "Active LinkedIn users"
    )

    selected_source_data = occupation_shares.loc[
        occupation_shares["benchmark_source"] == selected_source
    ].copy()
    selected_source_data["linkedin_share"] = selected_source_data[
        selected_share_column
    ]
    selected_source_data["linkedin_count"] = selected_source_data[
        selected_count_column
    ]
    selected_source_data["representation_ratio"] = (
        selected_source_data["linkedin_share"]
        / selected_source_data["benchmark_worker_share"]
    ).where(selected_source_data["benchmark_worker_share"] > 0)

    _ranking_value = ranking_selector.value
    if _ranking_value == "linkedin_share_desc":
        _rank_column = "linkedin_share"
        _rank_ascending = False
        selected_ranking_label = "largest LinkedIn share"
    elif _ranking_value == "ratio_desc":
        _rank_column = "representation_ratio"
        _rank_ascending = False
        selected_ranking_label = "highest LinkedIn-to-benchmark ratio"
    else:
        _rank_column = "representation_ratio"
        _rank_ascending = True
        selected_ranking_label = "lowest LinkedIn-to-benchmark ratio"

    _rankable = selected_source_data.dropna(subset=[_rank_column]).copy()
    top_occupations = (
        _rankable.sort_values(
            [_rank_column, "occupation_code"],
            ascending=[_rank_ascending, True],
        )
        .head(top_n_selector.value)
        .copy()
    )
    occupation_order = top_occupations["occupation_title"].tolist()
    return (
        occupation_order,
        selected_count_column,
        selected_ranking_label,
        selected_sample_label,
        selected_share_column,
        selected_source,
        top_occupations,
    )


@app.cell(hide_code=True)
def ranking_summary(mo, pd, selected_source, top_occupations):
    _first_occupation = top_occupations.iloc[0]
    _first_ratio = _first_occupation["representation_ratio"]
    _ratio_text = (
        f"{_first_ratio:,.2f}"
        if pd.notna(_first_ratio)
        else "not defined"
    )

    mo.callout(
        mo.md(
            f"""
            **First occupation under the current ranking:**
            {_first_occupation['occupation_title']}

            - LinkedIn share: **{_first_occupation['linkedin_share']:.2%}**
            - {selected_source} share:
              **{_first_occupation['benchmark_worker_share']:.2%}**
            - LinkedIn / benchmark: **{_ratio_text}**
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def coverage_chart(
    alt,
    occupation_order,
    selected_ranking_label,
    selected_sample_label,
    selected_source,
    top_n_selector,
    top_occupations,
):
    _top_base = alt.Chart(top_occupations).encode(
        y=alt.Y(
            "occupation_title:N",
            sort=occupation_order,
            title=None,
            axis=alt.Axis(
                labelLimit=390,
                labelFontSize=11,
                labelPadding=6,
            ),
        ),
        tooltip=[
            alt.Tooltip("occupation_code:N", title="Occupation code"),
            alt.Tooltip("occupation_title:N", title="Occupation"),
            alt.Tooltip("linkedin_count:Q", title=selected_sample_label, format=","),
            alt.Tooltip(
                "benchmark_worker_count:Q",
                title=f"{selected_source} workers/jobs",
                format=",",
            ),
            alt.Tooltip(
                "linkedin_share:Q",
                title="LinkedIn share",
                format=".2%",
            ),
            alt.Tooltip(
                "benchmark_worker_share:Q",
                title=f"{selected_source} share",
                format=".2%",
            ),
            alt.Tooltip(
                "representation_ratio:Q",
                title="LinkedIn / benchmark",
                format=".2f",
            ),
        ],
    )

    _top_bars = _top_base.mark_bar(
        color="#2563EB",
        opacity=0.82,
        cornerRadiusEnd=3,
    ).encode(
        x=alt.X(
            "linkedin_share:Q",
            title="Share of workers or users",
            axis=alt.Axis(format=".1%", gridColor="#E5E7EB"),
        )
    )
    _top_markers = _top_base.mark_point(
        color="#F97316",
        filled=True,
        shape="diamond",
        size=110,
        stroke="white",
        strokeWidth=1,
    ).encode(x=alt.X("benchmark_worker_share:Q"))

    coverage_chart = (
        (_top_bars + _top_markers)
        .properties(
            width="container",
            height=max(360, top_n_selector.value * 18),
            title=alt.TitleParams(
                text=f"Top {top_n_selector.value}: {selected_ranking_label}",
                subtitle=[
                    f"Blue bars: {selected_sample_label}",
                    f"Orange diamonds: {selected_source} benchmark",
                ],
                anchor="start",
                fontSize=18,
                subtitleFontSize=12,
                offset=12,
            ),
        )
        .configure_view(stroke=None)
        .configure_axis(
            titleFontSize=12,
            labelColor="#374151",
            titleColor="#111827",
        )
    )

    coverage_chart
    return


@app.cell(hide_code=True)
def ranking_table(mo, selected_sample_label, selected_source, top_occupations):
    top_display_table = top_occupations[
        [
            "occupation_code",
            "occupation_title",
            "linkedin_share",
            "benchmark_worker_share",
            "representation_ratio",
            "linkedin_count",
            "benchmark_worker_count",
        ]
    ].copy()
    top_display_table["linkedin_share"] = top_display_table[
        "linkedin_share"
    ].map("{:.2%}".format)
    top_display_table["benchmark_worker_share"] = top_display_table[
        "benchmark_worker_share"
    ].map("{:.2%}".format)
    top_display_table["representation_ratio"] = top_display_table[
        "representation_ratio"
    ].map("{:.2f}".format)
    top_display_table = top_display_table.rename(
        columns={
            "occupation_code": "Occupation code",
            "occupation_title": "Occupation",
            "linkedin_share": "LinkedIn share",
            "benchmark_worker_share": f"{selected_source} share",
            "representation_ratio": "LinkedIn / benchmark",
            "linkedin_count": selected_sample_label,
            "benchmark_worker_count": f"{selected_source} workers/jobs",
        }
    )

    mo.accordion(
        {
            "View data behind the ranking": mo.ui.table(
                top_display_table,
                pagination=True,
                page_size=10,
                show_column_summaries=False,
            )
        }
    )
    return


@app.cell(hide_code=True)
def focus_definitions(pd):
    BIOPHARM_ONET = {
        "19-1011.00": "Animal Scientists",
        "19-1021.00": "Biochemists and Biophysicists",
        "19-1022.00": "Microbiologists",
        "19-1029.01": "Bioinformatics Scientists",
        "19-1029.02": "Molecular and Cellular Biologists",
        "19-1029.03": "Geneticists",
        "19-1029.04": "Biologists",
        "19-2031.00": "Chemists",
        "19-1042.00": "Medical Scientists, Except Epidemiologists",
        "17-2031.00": "Bioengineers and Biomedical Engineers",
        "17-2041.00": "Chemical Engineers",
    }
    AUTOMATION_ONET = {
        "17-2071.00": "Electrical Engineers",
        "17-2072.00": "Electronics Engineers, Except Computer",
        "17-2141.00": "Mechanical Engineers",
        "17-2199.05": "Mechatronics Engineers",
        "17-2199.06": "Microsystems Engineers",
        "17-2199.08": "Robotics Engineers",
        "17-2199.09": "Nanosystems Engineers",
    }
    ELECTRONICS_ONET = {
        "17-2061.00": "Computer Hardware Engineers",
        "17-2071.00": "Electrical Engineers",
        "17-2072.00": "Electronics Engineers, Except Computer",
        "17-2072.01": (
            "Radio Frequency Identification Device Specialists"
        ),
        "17-2131.00": "Materials Engineers",
        "17-2199.06": "Microsystems Engineers",
        "17-2199.07": "Photonics Engineers",
        "17-2199.09": "Nanosystems Engineers",
        "19-2012.00": "Physicists",
        "19-2032.00": "Materials Scientists",
    }

    FOCUS_INDUSTRIES = {
        "BioPharm": BIOPHARM_ONET,
        "Automation": AUTOMATION_ONET,
        "Electronics": ELECTRONICS_ONET,
    }
    _focus_rows = []
    for _industry, _occupations in FOCUS_INDUSTRIES.items():
        for _onet_code, _requested_title in _occupations.items():
            _focus_rows.append(
                {
                    "industry": _industry,
                    "onet_code": _onet_code,
                    "requested_onet_title": _requested_title,
                }
            )
    focus_onet_catalog = pd.DataFrame(_focus_rows)
    return FOCUS_INDUSTRIES, focus_onet_catalog


@app.cell(hide_code=True)
def focus_section(mo):
    mo.md(r"""
    ### 1.2. Priority occupations: BioPharm, Automation, and Electronics

    The requested O*NET codes are mapped to the selected benchmark's
    occupation classification. The chart contains one row per unique
    benchmark occupation, preventing double counting when multiple detailed
    O*NET occupations collapse into one broader category.
    """)
    return


@app.cell(hide_code=True)
def focus_controls(FOCUS_INDUSTRIES, mo):
    focus_industry_selector = mo.ui.dropdown(
        options=list(FOCUS_INDUSTRIES),
        value="BioPharm",
        label="Priority group",
    )
    focus_source_selector = mo.ui.dropdown(
        options={
            "ACS - main benchmark": "ACS",
            "CPS - robustness check": "CPS",
            "OEWS - robustness check": "OEWS",
        },
        value="ACS - main benchmark",
        label="US benchmark",
    )
    mo.hstack(
        [focus_industry_selector, focus_source_selector],
        justify="start",
        align="end",
        gap=2,
    )
    return focus_industry_selector, focus_source_selector


@app.cell(hide_code=True)
def prepare_focus_data(
    crosswalk_long,
    focus_industry_selector,
    focus_onet_catalog,
    focus_source_selector,
    occupation_shares,
    selected_count_column,
    selected_share_column,
):
    selected_focus_industry = focus_industry_selector.value
    selected_focus_source = focus_source_selector.value
    _focus_source_data = occupation_shares.loc[
        occupation_shares["benchmark_source"] == selected_focus_source
    ].copy()
    _focus_source_data["linkedin_share"] = _focus_source_data[
        selected_share_column
    ]
    _focus_source_data["linkedin_count"] = _focus_source_data[
        selected_count_column
    ]
    _focus_source_data["representation_ratio"] = (
        _focus_source_data["linkedin_share"]
        / _focus_source_data["benchmark_worker_share"]
    ).where(_focus_source_data["benchmark_worker_share"] > 0)
    _selected_focus_codes = focus_onet_catalog.loc[
        focus_onet_catalog["industry"] == selected_focus_industry,
        ["industry", "onet_code", "requested_onet_title"],
    ]
    _selected_focus_crosswalk = crosswalk_long.loc[
        crosswalk_long["benchmark_source"] == selected_focus_source
    ]
    focus_mapping = _selected_focus_codes.merge(
        _selected_focus_crosswalk,
        on="onet_code",
        how="left",
        validate="one_to_one",
    )

    _focus_map_summary = (
        focus_mapping.dropna(subset=["occupation_code"])
        .groupby(
            ["occupation_code", "occupation_title"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mapped_onet_count=("onet_code", "nunique"),
            mapped_onet_codes=(
                "onet_code",
                lambda values: ", ".join(sorted(values)),
            ),
        )
    )
    focus_occupation_data = _focus_source_data.merge(
        _focus_map_summary,
        on=["occupation_code", "occupation_title"],
        how="inner",
        validate="one_to_one",
    ).sort_values("linkedin_share", ascending=False)
    focus_occupation_order = focus_occupation_data[
        "occupation_title"
    ].tolist()
    return (
        focus_mapping,
        focus_occupation_data,
        focus_occupation_order,
        selected_focus_industry,
        selected_focus_source,
    )


@app.cell(hide_code=True)
def focus_chart(
    alt,
    focus_occupation_data,
    focus_occupation_order,
    selected_focus_industry,
    selected_focus_source,
    selected_sample_label,
):
    _focus_base = alt.Chart(focus_occupation_data).encode(
        y=alt.Y(
            "occupation_title:N",
            sort=focus_occupation_order,
            title=None,
            axis=alt.Axis(labelLimit=390, labelFontSize=11),
        ),
        tooltip=[
            alt.Tooltip("occupation_code:N", title="Benchmark code"),
            alt.Tooltip("occupation_title:N", title="Benchmark occupation"),
            alt.Tooltip(
                "mapped_onet_codes:N",
                title="Requested O*NET codes",
            ),
            alt.Tooltip(
                "linkedin_share:Q",
                title="LinkedIn share",
                format=".2%",
            ),
            alt.Tooltip(
                "benchmark_worker_share:Q",
                title=f"{selected_focus_source} share",
                format=".2%",
            ),
            alt.Tooltip(
                "representation_ratio:Q",
                title="LinkedIn / benchmark",
                format=".2f",
            ),
        ],
    )
    _focus_bars = _focus_base.mark_bar(
        color="#0F766E",
        opacity=0.84,
        cornerRadiusEnd=3,
    ).encode(
        x=alt.X(
            "linkedin_share:Q",
            title="Share of workers or users",
            axis=alt.Axis(format=".2%", gridColor="#E5E7EB"),
        )
    )
    _focus_markers = _focus_base.mark_point(
        color="#F97316",
        filled=True,
        shape="diamond",
        size=120,
        stroke="white",
        strokeWidth=1,
    ).encode(x=alt.X("benchmark_worker_share:Q"))

    focus_coverage_chart = (
        (_focus_bars + _focus_markers)
        .properties(
            width="container",
            height=max(220, len(focus_occupation_data) * 34),
            title=alt.TitleParams(
                text=f"{selected_focus_industry} occupations",
                subtitle=[
                    f"Teal bars: {selected_sample_label}",
                    f"Orange diamonds: {selected_focus_source} benchmark",
                ],
                anchor="start",
                fontSize=18,
                subtitleFontSize=12,
            ),
        )
        .configure_view(stroke=None)
    )
    focus_coverage_chart
    return


@app.cell(hide_code=True)
def focus_tables(
    focus_mapping,
    focus_occupation_data,
    mo,
    selected_focus_source,
):
    _mapping_display = focus_mapping[
        [
            "onet_code",
            "requested_onet_title",
            "occupation_code",
            "occupation_title",
            "mapping_status",
        ]
    ].rename(
        columns={
            "onet_code": "Requested O*NET code",
            "requested_onet_title": "Requested O*NET occupation",
            "occupation_code": f"{selected_focus_source} code",
            "occupation_title": f"{selected_focus_source} occupation",
            "mapping_status": "Mapping status",
        }
    )
    _focus_share_display = focus_occupation_data[
        [
            "occupation_code",
            "occupation_title",
            "mapped_onet_count",
            "linkedin_share",
            "benchmark_worker_share",
            "representation_ratio",
        ]
    ].copy()
    _focus_share_display["linkedin_share"] = _focus_share_display[
        "linkedin_share"
    ].map("{:.3%}".format)
    _focus_share_display["benchmark_worker_share"] = _focus_share_display[
        "benchmark_worker_share"
    ].map("{:.3%}".format)
    _focus_share_display["representation_ratio"] = _focus_share_display[
        "representation_ratio"
    ].map("{:.2f}".format)
    _focus_share_display = _focus_share_display.rename(
        columns={
            "occupation_code": f"{selected_focus_source} code",
            "occupation_title": f"{selected_focus_source} occupation",
            "mapped_onet_count": "Requested O*NET codes mapped",
            "linkedin_share": "LinkedIn share",
            "benchmark_worker_share": f"{selected_focus_source} share",
            "representation_ratio": "LinkedIn / benchmark",
        }
    )

    mo.accordion(
        {
            "Comparison shares": mo.ui.table(
                _focus_share_display,
                pagination=False,
                show_column_summaries=False,
            ),
            "O*NET-to-benchmark mapping": mo.ui.table(
                _mapping_display,
                pagination=True,
                page_size=12,
                show_column_summaries=False,
            ),
        }
    )
    return


@app.cell(hide_code=True)
def age_section(mo):
    mo.md(r"""
    ## 2. Are younger cohorts better represented?

    ACS provides exact ages and is the default. CPS provides published age
    bins as a robustness check. Select either all mapped occupations or one
    occupation to compare its age profile in LinkedIn with the benchmark.

    The lower panel reports the ratio of LinkedIn's age share to the
    benchmark age share. Values above one indicate overrepresentation.
    """)
    return


@app.cell(hide_code=True)
def age_controls(mo):
    age_source_selector = mo.ui.dropdown(
        options={
            "ACS exact ages - main": "ACS",
            "CPS published age bins - robustness": "CPS",
        },
        value="ACS exact ages - main",
        label="Age benchmark",
    )
    age_sample_selector = mo.ui.radio(
        options={
            "All LinkedIn users": "linkedin_user_count",
            "Active LinkedIn users": "linkedin_active_user_count",
        },
        value="All LinkedIn users",
        inline=True,
        label="LinkedIn sample",
    )
    age_range_selector = mo.ui.range_slider(
        start=16,
        stop=99,
        step=1,
        value=[18, 80],
        debounce=True,
        show_value=True,
        label="Displayed ACS ages",
    )
    mo.hstack(
        [age_source_selector, age_sample_selector, age_range_selector],
        justify="start",
        align="end",
        gap=2,
        wrap=True,
    )
    return age_range_selector, age_sample_selector, age_source_selector


@app.cell(hide_code=True)
def age_occupation_control(age_source_selector, mo, occupation_shares):
    selected_age_source = age_source_selector.value
    _age_source_occupations = occupation_shares.loc[
        occupation_shares["benchmark_source"] == selected_age_source,
        ["occupation_code", "occupation_title"],
    ].drop_duplicates()
    _age_option_pairs = _age_source_occupations.sort_values(
        ["occupation_title", "occupation_code"]
    )
    age_occupation_options = {"All mapped occupations": "__ALL__"}
    for _row in _age_option_pairs.itertuples(index=False):
        _label = f"{_row.occupation_title} [{_row.occupation_code}]"
        age_occupation_options[_label] = _row.occupation_code

    age_occupation_selector = mo.ui.dropdown(
        options=age_occupation_options,
        value="All mapped occupations",
        searchable=True,
        label="Occupation",
        full_width=True,
    )
    age_occupation_selector
    return age_occupation_selector, selected_age_source


@app.cell(hide_code=True)
def prepare_age_data(
    acs_raw,
    age_occupation_selector,
    age_range_selector,
    age_sample_selector,
    cps_raw,
    pd,
    selected_age_source,
):
    selected_age_occupation = age_occupation_selector.value
    selected_age_count_column = age_sample_selector.value
    selected_age_sample_label = (
        "All LinkedIn users"
        if selected_age_count_column == "linkedin_user_count"
        else "Active LinkedIn users"
    )

    if selected_age_source == "ACS":
        _age_source_data = acs_raw.loc[
            acs_raw["age_scope"] == "Exact age"
        ].copy()
        _age_code_column = "acs_occupation_code"
        _age_title_column = "acs_occupation_title"
        _age_group_columns = ["age"]
    else:
        _age_source_data = cps_raw.loc[
            cps_raw["age_scope"] == "Published age bin"
        ].copy()
        _age_code_column = "cps_occupation_code"
        _age_title_column = "cps_occupation_title"
        _age_group_columns = ["age_bin", "age_min", "age_max"]

    if selected_age_occupation != "__ALL__":
        _age_source_data = _age_source_data.loc[
            _age_source_data[_age_code_column]
            == selected_age_occupation
        ]
        _age_selected_title = _age_source_data[
            _age_title_column
        ].dropna().iloc[0]
        selected_age_occupation_label = (
            f"{_age_selected_title} [{selected_age_occupation}]"
        )
    else:
        selected_age_occupation_label = "All mapped occupations"

    age_distribution = (
        _age_source_data.groupby(
            _age_group_columns,
            as_index=False,
            dropna=False,
        )[
            [
                "linkedin_user_count",
                "linkedin_active_user_count",
                "benchmark_worker_count",
            ]
        ]
        .sum()
    )
    age_distribution["linkedin_age_count"] = age_distribution[
        selected_age_count_column
    ]
    age_distribution["linkedin_age_share"] = (
        age_distribution["linkedin_age_count"]
        / age_distribution["linkedin_age_count"].sum()
    )
    age_distribution["benchmark_age_share"] = (
        age_distribution["benchmark_worker_count"]
        / age_distribution["benchmark_worker_count"].sum()
    )
    age_distribution["representation_ratio"] = (
        age_distribution["linkedin_age_share"]
        / age_distribution["benchmark_age_share"]
    ).where(age_distribution["benchmark_age_share"] > 0)

    if selected_age_source == "ACS":
        _age_min, _age_max = age_range_selector.value
        age_chart_data = age_distribution.loc[
            age_distribution["age"].between(_age_min, _age_max)
        ].copy()
    else:
        _cps_age_order = [
            "16-19",
            "20-24",
            "25-34",
            "35-44",
            "45-54",
            "55-64",
            "65+",
        ]
        age_chart_data = age_distribution.copy()
        age_chart_data["age_bin"] = pd.Categorical(
            age_chart_data["age_bin"],
            categories=_cps_age_order,
            ordered=True,
        )
        age_chart_data = age_chart_data.sort_values("age_bin")
    return (
        age_chart_data,
        age_distribution,
        selected_age_occupation_label,
        selected_age_sample_label,
    )


@app.cell(hide_code=True)
def age_chart(
    age_chart_data,
    alt,
    pd,
    selected_age_occupation_label,
    selected_age_sample_label,
    selected_age_source,
):
    if selected_age_source == "ACS":
        _age_x = alt.X(
            "age:Q",
            title="Age in 2022",
            axis=alt.Axis(tickMinStep=5, gridColor="#E5E7EB"),
        )
        _age_tooltip = alt.Tooltip("age:Q", title="Age")
    else:
        _age_x = alt.X(
            "age_bin:N",
            sort=[
                "16-19",
                "20-24",
                "25-34",
                "35-44",
                "45-54",
                "55-64",
                "65+",
            ],
            title="Age bin in 2022",
            axis=alt.Axis(labelAngle=0, gridColor="#E5E7EB"),
        )
        _age_tooltip = alt.Tooltip("age_bin:N", title="Age bin")

    _age_tidy = age_chart_data.melt(
        id_vars=[
            column
            for column in age_chart_data.columns
            if column not in ["linkedin_age_share", "benchmark_age_share"]
        ],
        value_vars=["linkedin_age_share", "benchmark_age_share"],
        var_name="series",
        value_name="share",
    )
    _age_tidy["series"] = _age_tidy["series"].map(
        {
            "linkedin_age_share": selected_age_sample_label,
            "benchmark_age_share": f"{selected_age_source} benchmark",
        }
    )

    _age_share_chart = (
        alt.Chart(_age_tidy)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=_age_x,
            y=alt.Y(
                "share:Q",
                title="Share within selected occupation(s)",
                axis=alt.Axis(format=".1%"),
            ),
            color=alt.Color(
                "series:N",
                title=None,
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                ),
                scale=alt.Scale(
                    domain=[
                        selected_age_sample_label,
                        f"{selected_age_source} benchmark",
                    ],
                    range=["#2563EB", "#F97316"],
                ),
            ),
            tooltip=[
                _age_tooltip,
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
            ],
        )
        .properties(width=850, height=250)
    )

    _ratio_line = alt.Chart(pd.DataFrame({"ratio": [1.0]})).mark_rule(
        color="#6B7280",
        strokeDash=[5, 4],
    ).encode(y="ratio:Q")
    _age_ratio_points = (
        alt.Chart(age_chart_data)
        .mark_line(point=True, color="#7C3AED", strokeWidth=2.3)
        .encode(
            x=_age_x,
            y=alt.Y(
                "representation_ratio:Q",
                title="LinkedIn / benchmark",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                _age_tooltip,
                alt.Tooltip(
                    "representation_ratio:Q",
                    title="LinkedIn / benchmark",
                    format=".2f",
                ),
                alt.Tooltip(
                    "linkedin_age_count:Q",
                    title=selected_age_sample_label,
                    format=",",
                ),
                alt.Tooltip(
                    "benchmark_worker_count:Q",
                    title=f"{selected_age_source} workers",
                    format=",",
                ),
            ],
        )
        .properties(width=850, height=170)
    )

    age_coverage_chart = (
        alt.vconcat(_age_share_chart, _ratio_line + _age_ratio_points)
        .resolve_scale(x="shared")
        .properties(
            title=alt.TitleParams(
                text=selected_age_occupation_label,
                subtitle=(
                    "Age distributions and relative representation; "
                    "the dashed line marks parity"
                ),
                anchor="start",
                fontSize=18,
                subtitleFontSize=12,
            )
        )
        .configure_view(stroke=None)
        .configure_axis(labelColor="#374151", titleColor="#111827")
    )
    age_coverage_chart
    return


@app.cell(hide_code=True)
def age_summary(
    age_distribution,
    mo,
    selected_age_occupation_label,
    selected_age_source,
):
    if selected_age_source == "ACS":
        _young_rows = age_distribution["age"].between(20, 34)
        _older_rows = age_distribution["age"].between(45, 64)
    else:
        _young_rows = age_distribution["age_bin"].isin(["20-24", "25-34"])
        _older_rows = age_distribution["age_bin"].isin(["45-54", "55-64"])

    def _group_representation_ratio(mask):
        _linkedin_share = (
            age_distribution.loc[mask, "linkedin_age_count"].sum()
            / age_distribution["linkedin_age_count"].sum()
        )
        _benchmark_share = (
            age_distribution.loc[mask, "benchmark_worker_count"].sum()
            / age_distribution["benchmark_worker_count"].sum()
        )
        return (
            _linkedin_share / _benchmark_share
            if _benchmark_share > 0
            else float("nan")
        )

    young_representation_ratio = _group_representation_ratio(_young_rows)
    older_representation_ratio = _group_representation_ratio(_older_rows)
    _age_conclusion = (
        "stronger"
        if young_representation_ratio > older_representation_ratio
        else "weaker"
    )
    mo.callout(
        mo.md(
            f"""
            For **{selected_age_occupation_label}**, relative representation
            is **{young_representation_ratio:.2f}** at ages 20-34 and
            **{older_representation_ratio:.2f}** at ages 45-64.

            On this comparison, younger-worker representation is
            **{_age_conclusion}** than older-worker representation.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def age_table(
    age_distribution,
    mo,
    selected_age_sample_label,
    selected_age_source,
):
    _age_display = age_distribution.copy()
    _age_display["LinkedIn share"] = _age_display[
        "linkedin_age_share"
    ].map("{:.2%}".format)
    _age_display[f"{selected_age_source} share"] = _age_display[
        "benchmark_age_share"
    ].map("{:.2%}".format)
    _age_display["LinkedIn / benchmark"] = _age_display[
        "representation_ratio"
    ].map("{:.2f}".format)
    _age_columns = (
        ["age"]
        if selected_age_source == "ACS"
        else ["age_bin", "age_min", "age_max"]
    )
    _age_display = _age_display[
        _age_columns
        + [
            "linkedin_age_count",
            "benchmark_worker_count",
            "LinkedIn share",
            f"{selected_age_source} share",
            "LinkedIn / benchmark",
        ]
    ].rename(
        columns={
            "age": "Age",
            "age_bin": "Age bin",
            "age_min": "Minimum age",
            "age_max": "Maximum age",
            "linkedin_age_count": selected_age_sample_label,
            "benchmark_worker_count": f"{selected_age_source} workers",
        }
    )
    mo.accordion(
        {
            "View age-profile data": mo.ui.table(
                _age_display,
                pagination=True,
                page_size=12,
                show_column_summaries=False,
            )
        }
    )
    return


@app.cell(hide_code=True)
def robustness_section(mo):
    mo.md(r"""
    ## 3. Robustness across benchmarks and active-user definitions

    The first display aggregates each priority group to the union of its
    unique mapped benchmark occupations. This avoids double counting when
    several requested O*NET codes share one benchmark category.

    The scatter plot compares occupation-level representation ratios using
    all LinkedIn users versus active users. Points on the diagonal are
    insensitive to the active-profile restriction.
    """)
    return


@app.cell(hide_code=True)
def robustness_controls(FOCUS_INDUSTRIES, mo):
    robustness_industry_selector = mo.ui.dropdown(
        options=list(FOCUS_INDUSTRIES),
        value="BioPharm",
        label="Priority group",
    )
    robustness_source_selector = mo.ui.dropdown(
        options=["ACS", "CPS", "OEWS"],
        value="ACS",
        label="Scatter benchmark",
    )
    mo.hstack(
        [robustness_industry_selector, robustness_source_selector],
        justify="start",
        align="end",
        gap=2,
    )
    return robustness_industry_selector, robustness_source_selector


@app.cell(hide_code=True)
def prepare_robustness_data(
    FOCUS_INDUSTRIES,
    crosswalk_long,
    occupation_shares,
    pd,
    robustness_industry_selector,
    robustness_source_selector,
):
    selected_robustness_industry = robustness_industry_selector.value
    _robustness_focus_codes = set(
        FOCUS_INDUSTRIES[selected_robustness_industry]
    )
    _robustness_rows = []
    for _source in ["ACS", "CPS", "OEWS"]:
        _source_codes = (
            crosswalk_long.loc[
                (crosswalk_long["benchmark_source"] == _source)
                & crosswalk_long["onet_code"].isin(_robustness_focus_codes),
                "occupation_code",
            ]
            .dropna()
            .unique()
        )
        _source_group = occupation_shares.loc[
            (occupation_shares["benchmark_source"] == _source)
            & occupation_shares["occupation_code"].isin(_source_codes)
        ]
        _all_share = _source_group["linkedin_user_share"].sum()
        _active_share = _source_group["linkedin_active_user_share"].sum()
        _benchmark_share = _source_group["benchmark_worker_share"].sum()
        _robustness_rows.append(
            {
                "benchmark_source": _source,
                "mapped_occupation_count": len(_source_codes),
                "linkedin_user_share": _all_share,
                "linkedin_active_user_share": _active_share,
                "benchmark_worker_share": _benchmark_share,
                "all_user_ratio": (
                    _all_share / _benchmark_share
                    if _benchmark_share > 0
                    else float("nan")
                ),
                "active_user_ratio": (
                    _active_share / _benchmark_share
                    if _benchmark_share > 0
                    else float("nan")
                ),
            }
        )
    robustness_group_summary = pd.DataFrame(_robustness_rows)

    selected_robustness_source = robustness_source_selector.value
    robustness_scatter_data = occupation_shares.loc[
        occupation_shares["benchmark_source"]
        == selected_robustness_source
    ].copy()
    robustness_scatter_data["all_user_ratio"] = (
        robustness_scatter_data["linkedin_user_share"]
        / robustness_scatter_data["benchmark_worker_share"]
    )
    robustness_scatter_data["active_user_ratio"] = (
        robustness_scatter_data["linkedin_active_user_share"]
        / robustness_scatter_data["benchmark_worker_share"]
    )
    robustness_scatter_data = robustness_scatter_data.loc[
        (robustness_scatter_data["all_user_ratio"] > 0)
        & (robustness_scatter_data["active_user_ratio"] > 0)
    ].copy()
    robustness_scatter_data["active_effect"] = (
        robustness_scatter_data["active_user_ratio"]
        / robustness_scatter_data["all_user_ratio"]
    )
    robustness_scatter_data["active_effect_label"] = (
        robustness_scatter_data["active_effect"] >= 1
    ).map({True: "Higher with active users", False: "Lower with active users"})
    return (
        robustness_group_summary,
        robustness_scatter_data,
        selected_robustness_industry,
        selected_robustness_source,
    )


@app.cell(hide_code=True)
def robustness_source_chart(
    alt,
    pd,
    robustness_group_summary,
    selected_robustness_industry,
):
    _robustness_tidy = robustness_group_summary.melt(
        id_vars=[
            "benchmark_source",
            "mapped_occupation_count",
            "benchmark_worker_share",
        ],
        value_vars=["all_user_ratio", "active_user_ratio"],
        var_name="linkedin_sample",
        value_name="representation_ratio",
    )
    _robustness_tidy["linkedin_sample"] = _robustness_tidy[
        "linkedin_sample"
    ].map(
        {
            "all_user_ratio": "All LinkedIn users",
            "active_user_ratio": "Active LinkedIn users",
        }
    )
    _robustness_parity = alt.Chart(
        pd.DataFrame({"ratio": [1.0]})
    ).mark_rule(
        color="#6B7280",
        strokeDash=[5, 4],
    ).encode(y="ratio:Q")
    _robustness_points = (
        alt.Chart(_robustness_tidy)
        .mark_line(point=alt.OverlayMarkDef(size=100), strokeWidth=2.5)
        .encode(
            x=alt.X(
                "benchmark_source:N",
                sort=["ACS", "CPS", "OEWS"],
                title="US benchmark",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "representation_ratio:Q",
                title="LinkedIn / benchmark",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "linkedin_sample:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "All LinkedIn users",
                        "Active LinkedIn users",
                    ],
                    range=["#2563EB", "#0F766E"],
                ),
            ),
            tooltip=[
                alt.Tooltip("benchmark_source:N", title="Benchmark"),
                alt.Tooltip("linkedin_sample:N", title="LinkedIn sample"),
                alt.Tooltip(
                    "representation_ratio:Q",
                    title="LinkedIn / benchmark",
                    format=".2f",
                ),
                alt.Tooltip(
                    "benchmark_worker_share:Q",
                    title="Benchmark share",
                    format=".3%",
                ),
                alt.Tooltip(
                    "mapped_occupation_count:Q",
                    title="Mapped benchmark occupations",
                    format="d",
                ),
            ],
        )
        .properties(height=280)
    )
    robustness_source_chart = (
        (_robustness_parity + _robustness_points)
        .properties(
            width="container",
            title=alt.TitleParams(
                text=(
                    f"{selected_robustness_industry}: "
                    "relative representation across sources"
                ),
                subtitle="Dashed line marks parity with the benchmark",
                anchor="start",
                fontSize=18,
                subtitleFontSize=12,
            ),
        )
        .configure_view(stroke=None)
    )
    robustness_source_chart
    return


@app.cell(hide_code=True)
def robustness_active_chart(
    alt,
    pd,
    robustness_scatter_data,
    selected_robustness_source,
):
    _scatter_min = min(
        robustness_scatter_data["all_user_ratio"].min(),
        robustness_scatter_data["active_user_ratio"].min(),
    )
    _scatter_max = max(
        robustness_scatter_data["all_user_ratio"].max(),
        robustness_scatter_data["active_user_ratio"].max(),
    )
    _scatter_parity = alt.Chart(
        pd.DataFrame(
            {
                "all_user_ratio": [_scatter_min, _scatter_max],
                "active_user_ratio": [_scatter_min, _scatter_max],
            }
        )
    ).mark_line(color="#6B7280", strokeDash=[5, 4]).encode(
        x="all_user_ratio:Q",
        y="active_user_ratio:Q",
    )
    _scatter_points = (
        alt.Chart(robustness_scatter_data)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X(
                "all_user_ratio:Q",
                title="Representation ratio: all LinkedIn users",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "active_user_ratio:Q",
                title="Representation ratio: active LinkedIn users",
                scale=alt.Scale(type="log"),
            ),
            color=alt.Color(
                "active_effect_label:N",
                title=None,
                scale=alt.Scale(
                    domain=[
                        "Higher with active users",
                        "Lower with active users",
                    ],
                    range=["#0F766E", "#DC2626"],
                ),
            ),
            tooltip=[
                alt.Tooltip("occupation_code:N", title="Occupation code"),
                alt.Tooltip("occupation_title:N", title="Occupation"),
                alt.Tooltip(
                    "all_user_ratio:Q",
                    title="All-user ratio",
                    format=".2f",
                ),
                alt.Tooltip(
                    "active_user_ratio:Q",
                    title="Active-user ratio",
                    format=".2f",
                ),
                alt.Tooltip(
                    "active_effect:Q",
                    title="Active / all ratio",
                    format=".3f",
                ),
            ],
        )
        .properties(height=430)
    )
    robustness_active_chart = (
        (_scatter_parity + _scatter_points)
        .properties(
            width="container",
            title=alt.TitleParams(
                text=(
                    f"Active-profile sensitivity by occupation: "
                    f"{selected_robustness_source}"
                ),
                subtitle=(
                    "Log scales; points on the diagonal are unchanged "
                    "by the active-user restriction"
                ),
                anchor="start",
                fontSize=18,
                subtitleFontSize=12,
            ),
        )
        .configure_view(stroke=None)
    )
    robustness_active_chart
    return


@app.cell(hide_code=True)
def robustness_table(mo, robustness_group_summary):
    _robustness_display = robustness_group_summary.copy()
    for _column in [
        "linkedin_user_share",
        "linkedin_active_user_share",
        "benchmark_worker_share",
    ]:
        _robustness_display[_column] = _robustness_display[_column].map(
            "{:.3%}".format
        )
    for _column in ["all_user_ratio", "active_user_ratio"]:
        _robustness_display[_column] = _robustness_display[_column].map(
            "{:.2f}".format
        )
    _robustness_display = _robustness_display.rename(
        columns={
            "benchmark_source": "Benchmark",
            "mapped_occupation_count": "Unique benchmark occupations",
            "linkedin_user_share": "All-user LinkedIn share",
            "linkedin_active_user_share": "Active-user LinkedIn share",
            "benchmark_worker_share": "Benchmark share",
            "all_user_ratio": "All-user ratio",
            "active_user_ratio": "Active-user ratio",
        }
    )
    mo.accordion(
        {
            "View robustness summary": mo.ui.table(
                _robustness_display,
                pagination=False,
                show_column_summaries=False,
            )
        }
    )
    return


@app.cell(hide_code=True)
def methodological_notes(mo):
    mo.md(r"""
    ## Methodological notes

    - LinkedIn age is delivered with a 2025 reference year. C05 subtracts three
      years before comparisons with 2022 ACS and CPS ages.
    - Active users have a profile update date on or after January 1, 2024.
    - Shares are calculated within each benchmark's successfully mapped
      occupation universe. Source changes can therefore change denominators.
    - Priority O*NET occupations are displayed at the benchmark occupation grain.
      Several detailed O*NET codes may map to one broader benchmark category.
    - ACS and CPS count workers. OEWS counts wage-and-salary jobs and excludes
      self-employment, so OEWS is not a like-for-like worker total.
    - Extreme representation ratios can arise from very small benchmark cells.
      Always inspect the underlying counts before interpreting outliers.
    """)
    return


if __name__ == "__main__":
    app.run()

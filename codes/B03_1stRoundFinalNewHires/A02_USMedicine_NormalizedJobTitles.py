"""
Task:
    Explore normalized job-title distributions within and across ONET occupations in the US medicine-spell baseline sample.

Inputs:
(a) data/b_temp_data/B03_1stRoundFinalNewHires/US_NAICS3254_NonIntern.parquet
    <== Constructed by "codes/B03_1stRoundFinalNewHires/A01_ExamineNewEmpSpells.py"

Outputs:
    Not applicable. Results are displayed in the notebook; no files are written.

Descriptions of outputs:
    Not applicable.

Run:
    conda run --live-stream --name Talent --cwd "E:/Dropbox/E_Projects/TalentDiscovery" marimo edit --watch --no-token --skip-update-check codes/B03_1stRoundFinalNewHires/A02_USMedicine_NormalizedJobTitles.py


Notes:
(1) The baseline sample contains US employment spells in NAICS 3254 after excluding internship spells; each row is an employment spell.
(2) ONET occupation shares use all spells in the baseline sample as the denominator.
(3) Normalized-title shares use nonmissing normalized titles in the selected ONET occupations as the denominator; selecting multiple occupations pools their title counts.
(4) The word cloud displays at most the 100 most frequent normalized titles and encodes frequency through text size; positions and colors have no substantive meaning.


Wang Wenzhi
Time: 2026-09-08
"""

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="full", app_title="US medicine job titles")


@app.cell
def import_libraries():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    from wordcloud import WordCloud

    return (
        Path,
        plt,
        mo,
        pd,
        WordCloud,
    )


@app.cell
def define_settings(
    Path,
):
    INPUT_FILE = Path("data/b_temp_data/B03_1stRoundFinalNewHires/US_NAICS3254_NonIntern.parquet")
    INPUT_COLUMNS = [
        "onet_code",
        "onet_title",
        "job_title_normalized",
    ]
    MAX_WORDS = 100
    RANDOM_SEED = 123
    return (
        INPUT_COLUMNS,
        INPUT_FILE,
        MAX_WORDS,
        RANDOM_SEED,
    )


@app.cell
def load_baseline_sample(
    INPUT_COLUMNS,
    INPUT_FILE,
    pd,
):
    baseline_spells = pd.read_parquet(
        INPUT_FILE,
        columns=INPUT_COLUMNS,
    )

    occupation_distribution = (
        baseline_spells
        .groupby(
            [
                "onet_code",
                "onet_title",
            ],
            as_index=False,
            dropna=False,
        )
        .agg(
            spell_count=("job_title_normalized", "size"),
            title_nonmissing_count=("job_title_normalized", "count"),
            distinct_title_count=("job_title_normalized", "nunique"),
        )
        .sort_values(
            [
                "spell_count",
                "onet_code",
                "onet_title",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        )
        .reset_index(drop=True)
    )
    occupation_distribution["baseline_share"] = occupation_distribution["spell_count"] / len(
        baseline_spells
    )
    occupation_distribution["occupation_rank"] = range(
        1,
        len(occupation_distribution) + 1,
    )
    return (
        baseline_spells,
        occupation_distribution,
    )


@app.cell
def introduce_report(
    baseline_spells,
    mo,
    occupation_distribution,
):
    _n_spells = len(baseline_spells)
    _n_occupations = len(occupation_distribution)
    _n_nonmissing_titles = baseline_spells["job_title_normalized"].notna().sum()
    _n_distinct_titles = baseline_spells["job_title_normalized"].nunique(dropna=True)
    _contents = f"""
    # Normalized job titles in US medicine employment spells

    Use the control below to examine one ONET occupation or pool several occupations. The baseline contains **{_n_spells:,} employment spells**, **{_n_occupations:,} ONET occupations**, **{_n_nonmissing_titles:,} nonmissing normalized titles**, and **{_n_distinct_titles:,} distinct normalized-title values**.
    """

    mo.md(_contents)
    return


@app.cell
def define_occupation_control(
    mo,
    occupation_distribution,
):
    _section_title = """
    ## Select ONET occupations
    """
    _occupation_label = """
    ONET occupations, ordered by baseline share
    """.strip()
    _occupation_options = {
        (
            f"{row.occupation_rank:02.0f}. {row.onet_title} "
            f"({row.onet_code}; {row.baseline_share:.1%})"
        ): row.onet_code
        for row in occupation_distribution.itertuples(index=False)
    }
    _default_label = next(iter(_occupation_options))

    ui_onet_occupations = mo.ui.multiselect(
        options=_occupation_options,
        value=[_default_label],
        label=_occupation_label,
        full_width=True,
    )
    return (ui_onet_occupations,)


@app.cell
def calculate_selected_distributions(
    baseline_spells,
    occupation_distribution,
    ui_onet_occupations,
):
    _selected_codes = ui_onet_occupations.value
    _flag_selected = baseline_spells["onet_code"].isin(_selected_codes)
    _selected_spells = baseline_spells.loc[_flag_selected].copy()

    selected_occupation_distribution = (
        occupation_distribution
        .loc[occupation_distribution["onet_code"].isin(_selected_codes)]
        .sort_values("occupation_rank")
        .reset_index(drop=True)
    )

    job_title_distribution = (
        _selected_spells
        .dropna(subset=["job_title_normalized"])
        .groupby("job_title_normalized", as_index=False)
        .size()
        .rename(columns={"size": "title_count"})
        .sort_values(
            [
                "title_count",
                "job_title_normalized",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )
    _n_titles = job_title_distribution["title_count"].sum()
    job_title_distribution["title_share"] = job_title_distribution["title_count"] / _n_titles
    job_title_distribution = job_title_distribution.sort_values(
        [
            "title_share",
            "job_title_normalized",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)
    job_title_distribution["title_rank"] = range(
        1,
        len(job_title_distribution) + 1,
    )

    title_by_occupation_distribution = (
        _selected_spells
        .dropna(subset=["job_title_normalized"])
        .groupby(
            [
                "onet_code",
                "onet_title",
                "job_title_normalized",
            ],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "title_count"})
        .sort_values(
            [
                "onet_code",
                "title_count",
                "job_title_normalized",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )
    title_by_occupation_distribution["occupation_title_count"] = (
        title_by_occupation_distribution.groupby("onet_code")["title_count"].transform("sum")
    )
    title_by_occupation_distribution["within_occupation_share"] = (
        title_by_occupation_distribution["title_count"]
        / title_by_occupation_distribution["occupation_title_count"]
    )
    title_by_occupation_distribution["within_occupation_rank"] = (
        title_by_occupation_distribution.groupby("onet_code").cumcount() + 1
    )
    return (
        job_title_distribution,
        selected_occupation_distribution,
        title_by_occupation_distribution,
    )


@app.cell
def render_job_title_report(
    MAX_WORDS,
    RANDOM_SEED,
    WordCloud,
    job_title_distribution,
    mo,
    plt,
    selected_occupation_distribution,
    title_by_occupation_distribution,
    ui_onet_occupations,
):
    _empty_contents = """
    Select at least one ONET occupation with a nonmissing normalized job title.
    """
    _empty_title = """
    Check the occupation selection
    """.strip()
    mo.stop(
        job_title_distribution.empty,
        mo.md(_empty_contents).callout(kind="warn", title=_empty_title),
    )

    _n_occupations = len(selected_occupation_distribution)
    _n_selected_spells = selected_occupation_distribution["spell_count"].sum()
    _n_nonmissing_titles = job_title_distribution["title_count"].sum()
    _n_distinct_titles = len(job_title_distribution)
    _word_cloud_table = job_title_distribution.head(MAX_WORDS).copy()
    _displayed_share = _word_cloud_table["title_share"].sum()
    _top_title = job_title_distribution.iloc[0]
    _within_occupation_table = title_by_occupation_distribution.loc[
        title_by_occupation_distribution["within_occupation_rank"].le(50)
    ].copy()
    _frequencies = _word_cloud_table.set_index("job_title_normalized")["title_count"].to_dict()

    _figure_title = f"""
    Normalized job titles in {_n_occupations:,} selected ONET occupation{"s" if _n_occupations != 1 else ""}
    """.strip()
    _method_title = """
    How is the word cloud constructed?
    """.strip()
    _method_contents = f"""
    1. Keep the **{_n_selected_spells:,} employment spells** in the selected ONET occupation set.
    2. Exclude missing `job_title_normalized` values and count exact normalized-title values; **{_n_nonmissing_titles:,} titles** remain.
    3. Pool title counts when multiple occupations are selected and display the **{len(_word_cloud_table):,} most frequent titles**. Text size represents the title count; position and color are decorative.
    """
    _occupation_table_label = """
    Selected ONET occupations and their shares in the full baseline sample
    """.strip()
    _occupation_accordion_label = """
    View the selected ONET occupations
    """.strip()
    _title_table_label = """
    All normalized-title values, ordered by share in the selected occupations
    """.strip()
    _title_accordion_label = """
    View the full normalized-title distribution
    """.strip()
    _within_occupation_table_label = """
    Top 50 normalized titles within each selected ONET occupation
    """.strip()
    _within_occupation_accordion_label = """
    Compare normalized-title frequencies across selected ONET occupations
    """.strip()
    _interpretation_title = """
    What does the selected distribution show?
    """.strip()
    _interpretation_contents = f"""
    - The selected occupations contain **{_n_distinct_titles:,} distinct normalized titles**. The word cloud covers **{_displayed_share:.1%}** of nonmissing title observations in this selection.
    - The most frequent normalized title is **{_top_title["job_title_normalized"]}**, representing **{_top_title["title_share"]:.1%}** of nonmissing titles in the pooled selection.
    - When several occupations are selected, the pooled word cloud gives more visual weight to larger occupations. Use the within-occupation table to compare title shares with separate denominators.
    """
    _occupation_format_mapping = {
        "occupation_rank": "{:,.0f}",
        "spell_count": "{:,.0f}",
        "title_nonmissing_count": "{:,.0f}",
        "distinct_title_count": "{:,.0f}",
        "baseline_share": "{:.1%}",
    }
    _title_format_mapping = {
        "title_rank": "{:,.0f}",
        "title_count": "{:,.0f}",
        "title_share": "{:.1%}",
    }
    _within_occupation_format_mapping = {
        "title_count": "{:,.0f}",
        "occupation_title_count": "{:,.0f}",
        "within_occupation_share": "{:.1%}",
        "within_occupation_rank": "{:,.0f}",
    }

    _word_cloud = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        colormap="gnuplot",
        max_words=MAX_WORDS,
        prefer_horizontal=1.0,
        random_state=RANDOM_SEED,
    ).generate_from_frequencies(_frequencies)
    _figure, _axis = plt.subplots(figsize=(16, 9))
    _axis.imshow(_word_cloud, interpolation="bilinear")
    _axis.set_title(_figure_title, fontsize=16)
    _axis.axis("off")
    _figure.tight_layout()

    _occupation_table = mo.ui.table(
        data=selected_occupation_distribution,
        label=_occupation_table_label,
        format_mapping=_occupation_format_mapping,
        show_column_summaries=False,
        pagination=True,
        page_size=10,
        selection=None,
    )
    _title_table = mo.ui.table(
        data=job_title_distribution,
        label=_title_table_label,
        format_mapping=_title_format_mapping,
        show_column_summaries=False,
        pagination=True,
        page_size=20,
        selection=None,
    )
    _within_occupation_table_ui = mo.ui.table(
        data=_within_occupation_table,
        label=_within_occupation_table_label,
        format_mapping=_within_occupation_format_mapping,
        show_column_summaries=False,
        pagination=True,
        page_size=20,
        selection=None,
    )

    mo.vstack(
        [
            ui_onet_occupations,
            mo.md(_method_contents).callout(kind="info", title=_method_title),
            _figure,
            mo.accordion({
                _occupation_accordion_label: _occupation_table,
                _title_accordion_label: _title_table,
                _within_occupation_accordion_label: _within_occupation_table_ui,
            }),
            mo.md(_interpretation_contents).callout(
                kind="danger",
                title=_interpretation_title,
            ),
        ],
        gap=1,
    )
    return


if __name__ == "__main__":
    app.run()

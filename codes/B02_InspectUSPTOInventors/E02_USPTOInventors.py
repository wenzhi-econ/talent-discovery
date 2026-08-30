# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==6.2.2; sys_platform != 'emscripten'",
#     "altair; sys_platform == 'emscripten'",
#     "matplotlib==3.11.1; sys_platform != 'emscripten'",
#     "pandas==3.0.5; sys_platform != 'emscripten'",
#     "pandas; sys_platform == 'emscripten'",
#     "pyarrow==25.0.0; sys_platform != 'emscripten'",
#     "pyarrow; sys_platform == 'emscripten'",
#     "wordcloud==1.9.6; sys_platform != 'emscripten'",
# ]
# ///

# ruff: noqa: PLR1711

"""
Task:
    Inspect employment spells of Revelio users linked to USPTO inventors through an
    interactive marimo notebook, so that the subsample behind every figure can be changed
    without editing code.

Inputs:
(a) data/b_temp_data/B02_InspectUSPTOInventors/Inventors_USPTO_UserPositions/*.parquet
    The notebook falls back to the B01_ConstructAnalysisSample copy of the same extract
    when the B02 directory is absent. The directory can also be overridden in the notebook.

Outputs:
    Not applicable. The notebook keeps every result in memory and writes no files.

Notes:
(1) The notebook does not import ``codes.main``; it resolves the project root from the
    notebook location and reads the Parquet parts with ``pyarrow.dataset``.
(2) The observation unit is the employment spell. An inventor with more recorded
    positions receives more weight.
(3) Employment spells with a missing Revelio company identifier (``rcid``) are dropped,
    and the delivered missing-country string ``empty`` is converted to a missing value.
(4) Every figure has its own controls. Country scope is either all countries or the
    United States; the seniority scope is any set of seniority levels.
(5) Top-N only truncates the number of categories drawn in a figure. Distribution tables
    behind the figures are always constructed in full before Top-N is applied.
(6) ``construct_distribution_table`` builds the spell-level distributions and
    ``normalize_job_titles`` standardizes self-reported job titles. Neither uses fuzzy
    matching, stemming, or token reordering.
(7) Job-title normalization is expensive on this sample, so the appendix computes it only
    after the corresponding switch is turned on.

Run from the project root with the Talent environment:

    conda run -s -n Talent marimo edit codes/B02_InspectUSPTOInventors/E02_USPTOInventors.py

Wang Wenzhi, with the help of Codex
Time: 2026-08-28
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", auto_download=["html"])


@app.cell
def imports():
    import re
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import pandas as pd
    import pyarrow.dataset as ds
    from matplotlib.figure import Figure
    from wordcloud import WordCloud

    return Figure, Path, WordCloud, alt, ds, mo, pd, re


@app.cell
def _(mo):
    mo.md(r"""
    # USPTO-LinkedIn inventors: Summary statistics

    Descriptions of the sample:

    - The underlying sample for this report is all employment spells among the USPTO inventors who are also in Revelio's profile database.
    - The difference between this report and the 3 previous reports is that **the inventors are not necessarily new hires**.
        - Remember that the universe sample of focal new hires are those employees **who report a new position (in occupation 17: Architecture and Engineering occupations and 19: Life, Physical, and Social Science occupations) during 2021-23**.
        - Instead, the sample in this report is the whole sample of USPTO-LinkedIn merged inventors.
            - They don't necessarily start a new position during 2021-23.
            - Nor do they have to report an employment spell in 2-digit occupation codes 17 or 19.
    - The observation unit is at **inventor $\times$ employment spell level**, not the inventor.
        - A potential issue is that an inventor with more recorded positions (who are potentially older) receives more weight in every figure below.
        - To somehow deal with this issue, I allow users to restrict employment spells to any specific seniority levels.

    Two purposes of this report:
    1. We want to have a better understanding of the USPTO-LinkedIn inventors, even though they are not new hires -- so they won't be in our analysis sample.
    2. **We want to know the self-reported job titles of these USPTO inventors in case we want to construct the analysis sample using job titles instead of ONET occupation codes.**
        - In previous reports, I have shown that the Revelio's occupation classifications are quite messy.
        - This implies that we may want to construct the final analysis sample using users' self-reported job titles, instead of Revelio's occupation variables.
        - This report aims to offer us a good prior about the job titles inventors may report in their employment spells.
    """)
    return


@app.cell
def helpers(pd):
    MISSING_LABEL = "<Missing>"
    US_LABEL = "United States"
    ALL_COUNTRIES_SCOPE = "all"
    US_SCOPE = "us"
    COUNTRY_SCOPE_OPTIONS = {"All countries": ALL_COUNTRIES_SCOPE, US_LABEL: US_SCOPE}
    DEFAULT_SENIORITY_LEVELS = (2, 3)
    DEFAULT_TOP_N = 30
    DEFAULT_HEATMAP_CATEGORIES = 8
    DEFAULT_BIOPHARM_INDUSTRIES = (
        "Biotechnology and Life Sciences",
        "Pharmaceutical Manufacturing",
        "Pharmaceuticals",
    )
    DEFAULT_BIOPHARM_OCCUPATIONS = (
        "Microbiologists",
        "Chemical Engineers",
        "Bioengineers and Biomedical Engineers",
        "Biochemists and Biophysicists",
        "Chemists",
        "Animal Scientists",
    )

    def construct_distribution_table(
        category_series: pd.Series,
        include_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Construct a descending frequency distribution for a categorical series.

        Parameters
        ----------
        category_series : pd.Series
            Categories measured at the employment-spell level.
        include_missing : bool, default False
            Whether missing values should appear as a separate category.

        Returns
        -------
        pd.DataFrame
            Category-level counts, shares, and descending frequency ranks.
        """

        distribution = (
            category_series.value_counts(dropna=not include_missing, sort=True)
            .rename("counts")
            .rename_axis("value")
            .reset_index()
        )
        distribution["share"] = distribution["counts"] / distribution["counts"].sum()
        distribution["rank"] = range(1, len(distribution) + 1)

        return distribution[["value", "counts", "share", "rank"]]

    TITLE_ABBREVIATION_REPLACEMENTS = (
        (r"\bsnr\b", "senior"),
        (r"\bsr\b", "senior"),
        (r"\bjr\b", "junior"),
        (r"\basst\b", "assistant"),
        (r"\bassoc\b", "associate"),
        (r"\bcoord\b", "coordinator"),
        (r"\bdir\b", "director"),
        (r"\bengr\b", "engineer"),
        (r"\bexec\b", "executive"),
        (r"\bmngr\b", "manager"),
        (r"\bmgr\b", "manager"),
        (r"\bsupv\b", "supervisor"),
        (r"\bdept\b", "department"),
        (r"\bintl\b", "international"),
        (r"\bmfg\b", "manufacturing"),
        (r"\bmktg\b", "marketing"),
        (r"\bmgmt\b", "management"),
        (r"\bops\b", "operations"),
        (r"\bsvp\b", "senior vice president"),
        (r"\bevp\b", "executive vice president"),
        (r"\bavp\b", "assistant vice president"),
        (r"\bvp\b", "vice president"),
        (r"\bceo\b", "chief executive officer"),
        (r"\bcfo\b", "chief financial officer"),
        (r"\bchro\b", "chief human resources officer"),
        (r"\bcio\b", "chief information officer"),
        (r"\bcoo\b", "chief operating officer"),
        (r"\bcto\b", "chief technology officer"),
        (r"\bhr\b", "human resources"),
        (r"\bqa\b", "quality assurance"),
        (r"\bqc\b", "quality control"),
        (r"\bcofounder\b", "co founder"),
    )

    def normalize_job_titles(job_titles: pd.Series) -> pd.Series:
        """
        Normalize superficial variation in self-reported job titles.

        Parameters
        ----------
        job_titles : pd.Series
            Raw job titles. Missing values are permitted and preserved.

        Returns
        -------
        pd.Series
            Normalized titles with pandas string dtype and the original index.

        Notes
        -----
        (1) Blank titles become missing values.
        (2) The rules standardize text form but do not infer occupations or seniority.
        (3) Fuzzy matching, stemming, spell correction, and token reordering are excluded
            to avoid merging substantively different titles.
        """

        normalized_titles = (
            job_titles.astype("string")
            .str.normalize("NFKC")
            .str.casefold()
            .str.replace("&amp;", "&", regex=False)
            .str.strip()
        )

        # Normalize research-and-development variants before punctuation becomes a separator.
        normalized_titles = normalized_titles.str.replace(
            r"\br\s*(?:&|/|\+)\s*d\b",
            "research and development",
            regex=True,
        )
        normalized_titles = normalized_titles.str.replace(
            r"\br\s+and\s+d\b",
            "research and development",
            regex=True,
        )
        normalized_titles = normalized_titles.str.replace(r"\s+\+\s+", " and ", regex=True)
        normalized_titles = normalized_titles.str.replace("&", " and ", regex=False)
        normalized_titles = normalized_titles.str.replace(".", "", regex=False)
        normalized_titles = normalized_titles.str.replace(
            "['\u2018\u2019\u02bc]",
            "",
            regex=True,
        )
        normalized_titles = normalized_titles.str.replace(
            "[-\u2010-\u2015\u2212/|,;:_()\\[\\]{}]+",
            " ",
            regex=True,
        )
        normalized_titles = normalized_titles.str.replace('"', " ", regex=False)
        normalized_titles = normalized_titles.str.replace("\\", " ", regex=False)
        normalized_titles = normalized_titles.str.replace(
            r"[!?@%^*=<>~`$]+",
            " ",
            regex=True,
        )
        normalized_titles = normalized_titles.str.replace(r"\s+", " ", regex=True).str.strip()

        for pattern, replacement in TITLE_ABBREVIATION_REPLACEMENTS:
            normalized_titles = normalized_titles.str.replace(pattern, replacement, regex=True)

        normalized_titles = normalized_titles.str.replace(r"\s+", " ", regex=True).str.strip()
        return normalized_titles.mask(normalized_titles.eq(""), pd.NA)

    return (
        ALL_COUNTRIES_SCOPE,
        COUNTRY_SCOPE_OPTIONS,
        DEFAULT_BIOPHARM_INDUSTRIES,
        DEFAULT_BIOPHARM_OCCUPATIONS,
        DEFAULT_HEATMAP_CATEGORIES,
        DEFAULT_SENIORITY_LEVELS,
        DEFAULT_TOP_N,
        MISSING_LABEL,
        US_LABEL,
        US_SCOPE,
        construct_distribution_table,
        normalize_job_titles,
    )


@app.cell
def helpers_analysis(
    ALL_COUNTRIES_SCOPE,
    MISSING_LABEL,
    US_LABEL,
    US_SCOPE,
    construct_distribution_table,
    pd,
    re,
):
    def hierarchy_number(column_name):
        """Return the numeric resolution in a Revelio hierarchy column name."""

        match = re.search(r"_k(\d+)$", column_name)
        return int(match.group(1)) if match else -1

    def scope_label(country_scope):
        """Return the display label of a country scope."""

        return US_LABEL if country_scope == US_SCOPE else "All countries"

    def filter_spells(data, country_scope, seniority_levels):
        """
        Restrict employment spells to one country scope and a set of seniority levels.

        An empty seniority selection keeps every level, so the control can never produce an
        empty figure. Spells with a missing seniority value are dropped once at least one
        numeric level is selected, unless the missing level itself is selected.
        """

        if country_scope == ALL_COUNTRIES_SCOPE:
            selected = data
        else:
            selected = data.loc[data["country"].eq(US_LABEL)]

        levels = tuple(seniority_levels or ())
        if levels:
            numeric_levels = pd.to_numeric(selected["seniority"], errors="coerce")
            if MISSING_LABEL in levels:
                numeric_values = [level for level in levels if level != MISSING_LABEL]
                keep = numeric_levels.isna() | numeric_levels.isin(numeric_values)
            else:
                keep = numeric_levels.isin(levels)
            selected = selected.loc[keep]
        return selected

    def seniority_options(data):
        """Return selectable seniority levels, adding a missing level only when needed."""

        levels = pd.to_numeric(data["seniority"], errors="coerce")
        levels_present = tuple(int(level) for level in sorted(levels.dropna().unique().tolist()))
        options = {f"Level {level}": level for level in levels_present}
        if levels.isna().any():
            options[MISSING_LABEL] = MISSING_LABEL
        return options

    def default_option_labels(options, values):
        """Translate selected option values back into the option names marimo expects."""

        labels_by_value = {value: label for label, value in options.items()}
        return [labels_by_value[value] for value in values if value in labels_by_value]

    def default_category_labels(distribution, preferred_categories, fallback_limit):
        """Select preferred categories by raw value, display label, or appended title."""

        available = distribution.loc[
            distribution["value"].astype(str).ne(MISSING_LABEL),
            ["value", "display_label"],
        ]
        aliases = {}
        for row in available.itertuples(index=False):
            value = str(row.value)
            display_label = str(row.display_label)
            aliases.setdefault(value, display_label)
            aliases.setdefault(display_label, display_label)
            if " — " in display_label:
                aliases.setdefault(display_label.split(" — ", maxsplit=1)[1], display_label)

        selected = [
            aliases[str(category)] for category in preferred_categories if str(category) in aliases
        ]
        selected = list(dict.fromkeys(selected))
        if selected:
            return selected
        return available["display_label"].head(fallback_limit).tolist()

    def category_lookup(data, column, title_column=None):
        """Return one row per distinct category with its display label and spell count."""

        group_columns = [column] if title_column is None else [column, title_column]
        cells = (
            data.groupby(group_columns, observed=True, dropna=False)
            .size()
            .rename("counts")
            .reset_index()
        )
        values = cells[column].astype("string").fillna(MISSING_LABEL)
        labels = values.copy()
        if title_column is not None:
            titles = cells[title_column].astype("string").fillna("")
            flagged = titles.ne("")
            labels.loc[flagged] = values.loc[flagged] + " — " + titles.loc[flagged]
        lookup = pd.DataFrame(
            {
                "value": values,
                "display_label": labels.fillna(MISSING_LABEL),
                "counts": cells["counts"],
            }
        )
        return lookup.drop_duplicates("display_label", ignore_index=True)

    def category_series(data, column, title_column=None):
        """Return the per-spell display label of one classification variable."""

        values = data[column].astype("string")
        labels = values.copy()
        if title_column is not None:
            titles = data[title_column].astype("string")
            flagged = values.notna() & titles.notna() & titles.ne("")
            labels.loc[flagged] = values.loc[flagged] + " — " + titles.loc[flagged]
        return labels.fillna(MISSING_LABEL)

    def classification_distribution(data, column, title_column=None):
        """
        Construct the spell-level distribution of one classification variable.

        The returned table carries the raw category in ``value``, its readable form in
        ``display_label``, and the counts, shares, and descending ranks produced by
        ``construct_distribution_table``.
        """

        labels = category_series(data, column, title_column)
        distribution = construct_distribution_table(labels, include_missing=False)
        distribution = distribution.rename(columns={"value": "display_label"})
        lookup = category_lookup(data, column, title_column).loc[:, ["value", "display_label"]]
        distribution = distribution.merge(lookup, on="display_label", how="left")
        return distribution.loc[:, ["value", "display_label", "counts", "share", "rank"]]

    def joint_distribution(
        data,
        industry_column,
        occupation_column,
        industry_title_column=None,
        occupation_title_column=None,
    ):
        """Construct the joint industry-occupation distribution of employment spells."""

        industry_lookup = category_lookup(data, industry_column, industry_title_column)
        occupation_lookup = category_lookup(data, occupation_column, occupation_title_column)
        cells = (
            data.groupby([industry_column, occupation_column], observed=True, dropna=False)
            .size()
            .rename("counts")
            .reset_index()
        )
        cells["industry_value"] = cells[industry_column].astype("string").fillna(MISSING_LABEL)
        cells["occupation_value"] = cells[occupation_column].astype("string").fillna(MISSING_LABEL)
        cells = cells.merge(
            industry_lookup.rename(
                columns={"value": "industry_value", "display_label": "industry_label"}
            ).loc[:, ["industry_value", "industry_label"]],
            on="industry_value",
            how="left",
        )
        cells = cells.merge(
            occupation_lookup.rename(
                columns={"value": "occupation_value", "display_label": "occupation_label"}
            ).loc[:, ["occupation_value", "occupation_label"]],
            on="occupation_value",
            how="left",
        )
        cells["industry_label"] = cells["industry_label"].fillna(MISSING_LABEL)
        cells["occupation_label"] = cells["occupation_label"].fillna(MISSING_LABEL)
        cells["display_label"] = cells["industry_label"] + " × " + cells["occupation_label"]
        cells = cells.sort_values(
            ["counts", "display_label"],
            ascending=[False, True],
            ignore_index=True,
        )
        cells["share"] = cells["counts"] / cells["counts"].sum()
        cells["rank"] = range(1, len(cells) + 1)
        return cells.loc[
            :,
            [
                "industry_value",
                "industry_label",
                "occupation_value",
                "occupation_label",
                "display_label",
                "counts",
                "share",
                "rank",
            ],
        ]

    def distribution_from_counts(cells, value_column, label_column):
        """Collapse joint cell counts into a marginal distribution within the pooled cells."""

        grouped = (
            cells.groupby([value_column, label_column], dropna=False)["counts"]
            .sum()
            .rename("counts")
            .reset_index()
            .sort_values(["counts", label_column], ascending=[False, True], ignore_index=True)
        )
        grouped["share"] = grouped["counts"] / grouped["counts"].sum()
        grouped["rank"] = range(1, len(grouped) + 1)
        grouped = grouped.rename(columns={value_column: "value", label_column: "display_label"})
        return grouped.loc[:, ["value", "display_label", "counts", "share", "rank"]]

    def subset_cells(cells, value_column, selected_values):
        """Keep joint cells whose category value belongs to the selected set."""

        selected = tuple(str(value) for value in (selected_values or ()))
        return cells.loc[cells[value_column].astype("string").isin(selected)].copy()

    return (
        classification_distribution,
        default_category_labels,
        default_option_labels,
        distribution_from_counts,
        filter_spells,
        hierarchy_number,
        joint_distribution,
        scope_label,
        seniority_options,
        subset_cells,
    )


@app.cell
def helpers_charts(Figure, WordCloud, alt, pd):
    def make_title(text, subtitle=None):
        """Build chart title parameters and drop an absent subtitle."""

        parameters = {
            "text": text,
            "anchor": "start",
            "subtitleColor": "#4B5563",
            "subtitlePadding": 6,
        }
        if subtitle is not None:
            parameters["subtitle"] = subtitle
        return alt.TitleParams(**parameters)

    def make_share_chart(
        summary,
        title,
        subtitle=None,
        top_n=None,
        x_title="Share of employment spells",
        color="#2563EB",
    ):
        """Draw a descending horizontal share chart and truncate it only for display."""

        columns = ["display_label", "counts", "share", "rank"]
        top = summary.loc[:, columns].copy()
        if top_n:
            top = top.head(top_n)
        if top.empty:
            return alt.Chart(pd.DataFrame({"display_label": []})).mark_bar()
        order = top["display_label"].tolist()
        maximum = float(top["share"].max())
        domain = [0.0, maximum * 1.16 if maximum else 1.0]
        base = alt.Chart(top).encode(
            y=alt.Y(
                "display_label:N",
                sort=order,
                title=None,
                axis=alt.Axis(labelLimit=460, labelPadding=6),
            ),
            tooltip=[
                alt.Tooltip("display_label:N", title="Category"),
                alt.Tooltip("counts:Q", title="Employment spells", format=","),
                alt.Tooltip("share:Q", title="Share", format=".2%"),
                alt.Tooltip("rank:Q", title="Rank", format="d"),
            ],
        )
        bars = base.mark_bar(color=color, opacity=0.85).encode(
            x=alt.X(
                "share:Q",
                title=x_title,
                axis=alt.Axis(format=".1%"),
                scale=alt.Scale(domain=domain),
            )
        )
        labels = base.mark_text(align="left", baseline="middle", dx=4).encode(
            x=alt.X("share:Q"),
            text=alt.Text("share:Q", format=".1%"),
        )
        return (
            alt.layer(bars, labels)
            .properties(
                width="container",
                height=max(280, len(top) * 21),
                title=make_title(title, subtitle),
            )
            .configure_view(stroke=None)
        )

    def make_seniority_chart(summary, title, subtitle=None):
        """Draw the seniority distribution with numeric levels on the x-axis."""

        columns = ["display_label", "counts", "share", "rank"]
        data = summary.loc[:, columns].copy()
        if data.empty:
            return alt.Chart(pd.DataFrame({"display_label": []})).mark_bar()
        order = data["display_label"].tolist()
        return (
            alt.Chart(data)
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
                    title="Share of employment spells",
                    axis=alt.Axis(format=".1%"),
                ),
                tooltip=[
                    alt.Tooltip("display_label:N", title="Seniority level"),
                    alt.Tooltip("counts:Q", title="Employment spells", format=","),
                    alt.Tooltip("share:Q", title="Share", format=".2%"),
                ],
            )
            .properties(
                width="container",
                height=380,
                title=make_title(title, subtitle),
            )
            .configure_view(stroke=None)
        )

    def make_heatmap(
        cells,
        industry_order,
        occupation_order,
        industry_title,
        occupation_title,
        title,
        subtitle=None,
    ):
        """Draw a share heatmap with occupations on the y-axis and industries on the x-axis."""

        chart_data = cells.loc[:, ["industry_label", "occupation_label", "counts", "share"]].copy()
        chart_data["share_label"] = chart_data["share"].map(
            lambda value: "" if not value or value <= 0 else f"{value:.2%}"
        )
        positive = chart_data.loc[chart_data["share"].gt(0)].copy()
        color_max = float(positive["share"].max()) if not positive.empty else 0.01
        threshold = color_max * 0.55
        x_encoding = alt.X(
            "industry_label:N",
            sort=industry_order,
            title=industry_title,
            axis=alt.Axis(labelAngle=-25, labelLimit=280, labelPadding=8),
        )
        y_encoding = alt.Y(
            "occupation_label:N",
            sort=occupation_order,
            title=occupation_title,
            axis=alt.Axis(labelLimit=380, labelPadding=8),
        )
        tooltips = [
            alt.Tooltip("industry_label:N", title="Industry"),
            alt.Tooltip("occupation_label:N", title="Occupation"),
            alt.Tooltip("counts:Q", title="Employment spells", format=","),
            alt.Tooltip("share:Q", title="Share", format=".2%"),
        ]
        base = (
            alt.Chart(chart_data)
            .mark_rect(color="#F3F4F6", stroke="#FFFFFF", strokeWidth=1.5)
            .encode(x=x_encoding, y=y_encoding, tooltip=tooltips)
        )
        colored_cells = (
            alt.Chart(positive)
            .mark_rect(stroke="#FFFFFF", strokeWidth=1.5)
            .encode(
                x=x_encoding,
                y=y_encoding,
                color=alt.Color(
                    "share:Q",
                    title="Share of employment spells",
                    scale=alt.Scale(domain=[0.0, color_max], scheme="blues"),
                    legend=alt.Legend(format=".1%"),
                ),
                tooltip=tooltips,
            )
        )
        text_labels = (
            alt.Chart(chart_data)
            .mark_text(fontSize=11, fontWeight=600)
            .encode(
                x=x_encoding,
                y=y_encoding,
                text=alt.Text("share_label:N"),
                color=alt.condition(
                    f"datum.share >= {threshold}",
                    alt.value("#FFFFFF"),
                    alt.value("#111827"),
                ),
            )
        )
        return (
            alt.layer(base, colored_cells, text_labels)
            .properties(
                width="container",
                height=max(340, len(occupation_order) * 52),
                title=make_title(title, subtitle),
            )
            .configure_view(stroke=None)
        )

    def make_word_cloud(summary, title, top_n):
        """Draw a deterministic word cloud whose font sizes increase with title shares."""

        top = summary.loc[:, ["display_label", "share"]].head(top_n)
        frequencies = {
            str(row.display_label): float(row.share)
            for row in top.itertuples(index=False)
            if float(row.share) > 0
        }
        cloud = WordCloud(
            width=1600,
            height=800,
            background_color="white",
            colormap="viridis",
            collocations=False,
            margin=4,
            max_words=len(frequencies),
            min_font_size=10,
            prefer_horizontal=0.9,
            random_state=20260828,
            relative_scaling=0.6,
        ).generate_from_frequencies(frequencies)
        figure = Figure(figsize=(16, 8), layout="constrained")
        axis = figure.subplots()
        axis.imshow(cloud, interpolation="bilinear")
        axis.set_axis_off()
        axis.set_title(title, loc="left", fontsize=16, pad=14)
        return figure

    return (
        make_heatmap,
        make_seniority_chart,
        make_share_chart,
        make_word_cloud,
    )


@app.cell
def paths(Path, mo):
    def resolve_project_root(start):
        """Walk upward until a directory contains both ``codes`` and ``data``."""

        for candidate in (start, *start.parents):
            if (candidate / "codes").is_dir() and (candidate / "data").is_dir():
                return candidate
        return start

    NOTEBOOK_DIR = Path(mo.notebook_dir() or Path.cwd())
    PROJECT_ROOT = resolve_project_root(NOTEBOOK_DIR)
    INPUT_DIR_CANDIDATES = (
        PROJECT_ROOT
        / "data"
        / "b_temp_data"
        / "B02_InspectUSPTOInventors"
        / "Inventors_USPTO_UserPositions",
        PROJECT_ROOT
        / "data"
        / "b_temp_data"
        / "B01_ConstructAnalysisSample"
        / "Inventors_USPTO_UserPositions",
    )
    DEFAULT_INPUT_DIR = next(
        (path for path in INPUT_DIR_CANDIDATES if path.is_dir()),
        INPUT_DIR_CANDIDATES[0],
    )
    INPUT_COLUMNS = (
        "user_id",
        "position_id",
        "rcid",
        "country",
        "seniority",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        "role_k50",
        "role_k150",
        "role_k300",
        "role_k500",
        "role_k1000",
        "role_k1500",
        "rics_k50",
        "rics_k200",
        "rics_k400",
        "title_raw",
    )
    TEXT_COLUMNS = (
        "country",
        "onet_code",
        "onet_title",
        "naics_code",
        "naics_description",
        "role_k50",
        "role_k150",
        "role_k300",
        "role_k500",
        "role_k1000",
        "role_k1500",
        "rics_k50",
        "rics_k200",
        "rics_k400",
        "title_raw",
    )
    return DEFAULT_INPUT_DIR, INPUT_COLUMNS, TEXT_COLUMNS


@app.cell
def load_data(DEFAULT_INPUT_DIR, INPUT_COLUMNS, TEXT_COLUMNS, ds, mo, pd):
    parquet_files = tuple(sorted(DEFAULT_INPUT_DIR.glob("*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No Parquet files found in {DEFAULT_INPUT_DIR}")

    with mo.status.spinner(
        f"Reading {len(parquet_files):,} Parquet files from {DEFAULT_INPUT_DIR} ..."
    ):
        spells = (
            ds.dataset(DEFAULT_INPUT_DIR, format="parquet")
            .to_table(columns=list(INPUT_COLUMNS))
            .to_pandas()
        )

    with mo.status.spinner("Preparing employment spells ..."):
        for column in TEXT_COLUMNS:
            spells[column] = spells[column].astype("category")
        spells = spells.dropna(subset=["rcid"])
        spells["rcid"] = spells["rcid"].astype("int64")
        country = spells["country"].astype("string")
        spells["country"] = country.mask(country.eq("empty"), pd.NA).astype("category")
        spells = spells.reset_index(drop=True)

    # load_note = mo.md(
    #     f"""
    #     - Parquet parts read: **{len(parquet_files):,}**.
    #     - Employment spells kept after dropping missing `rcid`: **{len(spells):,}**.
    #     - Distinct inventors (`user_id`): **{spells["user_id"].nunique():,}**.
    #     """
    # )
    # mo.vstack([load_note], gap=1)
    return (spells,)


@app.cell
def basic_numbers(US_LABEL, mo, pd, spells):
    def _counts(data):
        return {
            "Employment spells": len(data),
            "Distinct inventors": int(data["user_id"].nunique()),
            "Distinct companies": int(data["rcid"].nunique()),
            "Distinct countries": int(data.loc[data["country"].notna(), "country"].nunique()),
            "Distinct ONET occupations": int(
                data.loc[data["onet_code"].notna(), "onet_code"].nunique()
            ),
            "Distinct NAICS industries": int(
                data.loc[data["naics_code"].notna(), "naics_code"].nunique()
            ),
        }

    _all_country_counts = _counts(spells)
    _us_counts = _counts(spells.loc[spells["country"].eq(US_LABEL)])
    _us_shares = {
        measure: (us_value / all_value * 100)
        for (measure, all_value), us_value in zip(
            _all_country_counts.items(), _us_counts.values(), strict=True
        )
    }
    _basic_numbers = pd.DataFrame(
        [
            {
                "Measure": measure,
                "All countries": all_value,
                "US sample": us_value,
                "US share": us_share,
            }
            for (measure, all_value), us_value, us_share in zip(
                _all_country_counts.items(), _us_counts.values(), _us_shares.values(), strict=True
            )
        ]
    )
    _quality = pd.DataFrame(
        [
            {
                "Diagnostic": "Employment-spell rows",
                "Value": len(spells),
            },
            {
                "Diagnostic": "Distinct position_id values",
                "Value": int(spells["position_id"].nunique()),
            },
            {
                "Diagnostic": "Rows with a missing position_id",
                "Value": int(spells["position_id"].isna().sum()),
            },
            {
                "Diagnostic": "Rows with a missing country",
                "Value": int(spells["country"].isna().sum()),
            },
            {
                "Diagnostic": "Rows with a missing ONET occupation",
                "Value": int(spells["onet_code"].isna().sum()),
            },
            {
                "Diagnostic": "Rows with a missing NAICS industry",
                "Value": int(spells["naics_code"].isna().sum()),
            },
        ]
    )
    mo.vstack(
        [
            mo.md(
                R"""
                ## 1. Basic numbers

                - Remember that the underlying data is at **inventor $\times$ employment spell level**.
                - I will simply call them **inventors**, but they are actually only a subset of USPTO inventors who can be matched to Revelio's profile data.
                    - A small subsample of LinkedIn users are matched to multiple inventor ID in Gaurav's list.
                    - A user linked to several inventor IDs is counted only once.
                - The two columns are the all-country sample and the U.S. sample.
                    - The country restrictions also apply to employment spells.
                    - An Indian inventor who starts his job in India but moves to US at some point will be counted as one distinct inventor in the US sample, but only his US employment spell will be counted in US employment spells.
                """
            ),
            mo.ui.table(
                _basic_numbers,
                pagination=False,
                show_column_summaries=False,
                format_mapping={"All countries": "{:i,}", US_LABEL: "{:i,}", "US share": "{:.2f}%"},
            ),
            mo.accordion(
                {
                    "Data-quality diagnostics": mo.ui.table(
                        _quality,
                        pagination=False,
                        show_column_summaries=False,
                        format_mapping={"Value": "{:,}"},
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def classification_variables(hierarchy_number, spells):
    TITLE_COLUMNS = {"onet_code": "onet_title", "naics_code": "naics_description"}
    role_columns = tuple(
        sorted(
            (column for column in spells.columns if column.startswith("role_k")),
            key=hierarchy_number,
        )
    )
    rics_columns = tuple(
        sorted(
            (column for column in spells.columns if column.startswith("rics_k")),
            key=hierarchy_number,
        )
    )
    OCCUPATION_VARIABLE_OPTIONS = {
        "ONET occupation": "onet_code",
        **{f"Revelio role K{hierarchy_number(column):,}": column for column in role_columns},
    }
    INDUSTRY_VARIABLE_OPTIONS = {
        "NAICS industry (naics_code)": "naics_code",
        **{f"Revelio industry K{hierarchy_number(column):,}": column for column in rics_columns},
    }
    DEFAULT_OCCUPATION_COLUMN = "onet_code"
    DEFAULT_INDUSTRY_COLUMN = "rics_k400" if rics_columns else "naics_code"

    def label_of(options, column, fallback=None):
        """Return the option name marimo expects for one classification column."""

        for label, value in options.items():
            if value == column:
                return label
        return fallback if fallback is not None else next(iter(options))

    DEFAULT_OCCUPATION_LABEL = label_of(OCCUPATION_VARIABLE_OPTIONS, DEFAULT_OCCUPATION_COLUMN)
    DEFAULT_INDUSTRY_LABEL = label_of(INDUSTRY_VARIABLE_OPTIONS, DEFAULT_INDUSTRY_COLUMN)
    return (
        DEFAULT_INDUSTRY_LABEL,
        DEFAULT_OCCUPATION_LABEL,
        INDUSTRY_VARIABLE_OPTIONS,
        OCCUPATION_VARIABLE_OPTIONS,
        TITLE_COLUMNS,
    )


@app.cell
def seniority_choices(seniority_options, spells):
    SENIORITY_OPTIONS = seniority_options(spells)
    return (SENIORITY_OPTIONS,)


@app.cell
def occupation_controls(
    COUNTRY_SCOPE_OPTIONS,
    DEFAULT_OCCUPATION_LABEL,
    DEFAULT_SENIORITY_LEVELS,
    DEFAULT_TOP_N,
    OCCUPATION_VARIABLE_OPTIONS,
    SENIORITY_OPTIONS,
    US_LABEL,
    default_option_labels,
    mo,
):
    occupation_country_selector = mo.ui.dropdown(
        options=COUNTRY_SCOPE_OPTIONS,
        value=US_LABEL,
        label="Country scope",
    )
    occupation_seniority_selector = mo.ui.multiselect(
        options=SENIORITY_OPTIONS,
        value=default_option_labels(SENIORITY_OPTIONS, DEFAULT_SENIORITY_LEVELS),
        label="Seniority levels",
    )
    occupation_variable_selector = mo.ui.dropdown(
        options=OCCUPATION_VARIABLE_OPTIONS,
        value=DEFAULT_OCCUPATION_LABEL,
        label="Occupation variable",
    )
    occupation_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of occupations in the figure",
    )
    return (
        occupation_country_selector,
        occupation_seniority_selector,
        occupation_top_n_selector,
        occupation_variable_selector,
    )


@app.cell
def occupation_output(
    TITLE_COLUMNS,
    classification_distribution,
    filter_spells,
    make_share_chart,
    mo,
    occupation_country_selector,
    occupation_seniority_selector,
    occupation_top_n_selector,
    occupation_variable_selector,
    scope_label,
    spells,
):
    _country_scope = occupation_country_selector.value
    _seniority_levels = tuple(occupation_seniority_selector.value or ())
    _column = occupation_variable_selector.value
    _top_n = max(1, int(occupation_top_n_selector.value))
    _scope = scope_label(_country_scope)
    _subset = filter_spells(spells, _country_scope, _seniority_levels)
    _summary = classification_distribution(_subset, _column, TITLE_COLUMNS.get(_column))
    _subtitle = (
        f"{len(_subset):,} employment spells — {_scope}"
        + (
            f", seniority {', '.join(str(level) for level in _seniority_levels)}"
            if _seniority_levels
            else ", all seniority levels"
        )
        + f". Top {_top_n} of {len(_summary):,} categories."
    )
    _figure = (
        make_share_chart(
            _summary,
            f"Occupation distribution in {_column} — {_scope}",
            subtitle=_subtitle,
            top_n=_top_n,
            color="#2563EB",
        )
        if not _summary.empty
        else mo.callout(
            mo.md("No employment spell remains in the selected subsample."), kind="warn"
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 2. Occupation distribution

                Explanations of the statistics:
                - Again, the underlying data used to calculate the share is at **inventor $\times$ employment spell level**.
                - An inventor with more employment spells will have higher weights.
                    - To somehow alleviate this issue, I allow users to count only those employment spells at certain seniority levels.

                Some short observations:
                - **Software developers have the largest share among all inventors' employment spells.**
                    - This occupation is omitted from the universe sample of focal new hires in previous reports, because all programming-related occupations are under 2-digit code 15: Computer and Mathematical Occupations.
                        - Recall the universe sample of focal new hires is constructed from occupation codes 17: Architecture and Engineering occupations and 19: Life, Physical, and Social Science occupations.
                    - Later (in Section 4.3.), I will show that software developers are not a large share in our baseline BioPharma industry.
                - Most inventors' employment spells are under occupation 17 or 19. 
                    - **Revelio's ONET occupations have measurement errors, but they seem to be able to capture most inventors that we are interested in.**
                """
            ),
            mo.hstack(
                [
                    occupation_country_selector,
                    occupation_seniority_selector,
                    occupation_variable_selector,
                    occupation_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _figure,
            mo.accordion(
                {
                    "View the complete occupation distribution": mo.ui.table(
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
def industry_controls(
    COUNTRY_SCOPE_OPTIONS,
    DEFAULT_INDUSTRY_LABEL,
    DEFAULT_SENIORITY_LEVELS,
    DEFAULT_TOP_N,
    INDUSTRY_VARIABLE_OPTIONS,
    SENIORITY_OPTIONS,
    US_LABEL,
    default_option_labels,
    mo,
):
    industry_country_selector = mo.ui.dropdown(
        options=COUNTRY_SCOPE_OPTIONS,
        value=US_LABEL,
        label="Country scope",
    )
    industry_seniority_selector = mo.ui.multiselect(
        options=SENIORITY_OPTIONS,
        value=default_option_labels(SENIORITY_OPTIONS, DEFAULT_SENIORITY_LEVELS),
        label="Seniority levels",
    )
    industry_variable_selector = mo.ui.dropdown(
        options=INDUSTRY_VARIABLE_OPTIONS,
        value=DEFAULT_INDUSTRY_LABEL,
        label="Industry variable",
    )
    industry_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of industries in the figure",
    )
    return (
        industry_country_selector,
        industry_seniority_selector,
        industry_top_n_selector,
        industry_variable_selector,
    )


@app.cell
def industry_output(
    TITLE_COLUMNS,
    classification_distribution,
    filter_spells,
    industry_country_selector,
    industry_seniority_selector,
    industry_top_n_selector,
    industry_variable_selector,
    make_share_chart,
    mo,
    scope_label,
    spells,
):
    _country_scope = industry_country_selector.value
    _seniority_levels = tuple(industry_seniority_selector.value or ())
    _column = industry_variable_selector.value
    _top_n = max(1, int(industry_top_n_selector.value))
    _scope = scope_label(_country_scope)
    _subset = filter_spells(spells, _country_scope, _seniority_levels)
    _summary = classification_distribution(_subset, _column, TITLE_COLUMNS.get(_column))
    _subtitle = (
        f"{len(_subset):,} employment spells — {_scope}"
        + (
            f", seniority {', '.join(str(level) for level in _seniority_levels)}"
            if _seniority_levels
            else ", all seniority levels"
        )
        + f". Top {_top_n} of {len(_summary):,} categories."
    )
    _figure = (
        make_share_chart(
            _summary,
            f"Industry distribution in {_column} — {_scope}",
            subtitle=_subtitle,
            top_n=_top_n,
            color="#B45309",
        )
        if not _summary.empty
        else mo.callout(
            mo.md("No employment spell remains in the selected subsample."), kind="warn"
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 3. Industry distribution

                - Again, the underlying data used to calculate the share is at **inventor $\times$ employment spell level**.
                    - An inventor with more employment spells will have higher weights.
                    - To somehow alleviate this issue, I allow users to count only those employment spells at certain seniority levels.
                - Around 7% of inventors' employment spells are in "Research Universities" (highest industry among inventors' employment spells).
                """
            ),
            mo.hstack(
                [
                    industry_country_selector,
                    industry_seniority_selector,
                    industry_variable_selector,
                    industry_top_n_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            _figure,
            mo.accordion(
                {
                    "View the complete industry distribution": mo.ui.table(
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
def io_controls(
    COUNTRY_SCOPE_OPTIONS,
    DEFAULT_INDUSTRY_LABEL,
    DEFAULT_OCCUPATION_LABEL,
    DEFAULT_SENIORITY_LEVELS,
    DEFAULT_TOP_N,
    INDUSTRY_VARIABLE_OPTIONS,
    OCCUPATION_VARIABLE_OPTIONS,
    SENIORITY_OPTIONS,
    US_LABEL,
    default_option_labels,
    mo,
):
    io_country_selector = mo.ui.dropdown(
        options=COUNTRY_SCOPE_OPTIONS,
        value=US_LABEL,
        label="Country scope",
    )
    io_seniority_selector = mo.ui.multiselect(
        options=SENIORITY_OPTIONS,
        value=default_option_labels(SENIORITY_OPTIONS, DEFAULT_SENIORITY_LEVELS),
        label="Seniority levels",
    )
    io_industry_variable_selector = mo.ui.dropdown(
        options=INDUSTRY_VARIABLE_OPTIONS,
        value=DEFAULT_INDUSTRY_LABEL,
        label="Industry variable",
    )
    io_occupation_variable_selector = mo.ui.dropdown(
        options=OCCUPATION_VARIABLE_OPTIONS,
        value=DEFAULT_OCCUPATION_LABEL,
        label="Occupation variable",
    )
    io_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of industry-occupation cells in the figure",
    )
    return (
        io_country_selector,
        io_industry_variable_selector,
        io_occupation_variable_selector,
        io_seniority_selector,
        io_top_n_selector,
    )


@app.cell
def io_data(
    TITLE_COLUMNS,
    classification_distribution,
    filter_spells,
    io_country_selector,
    io_industry_variable_selector,
    io_occupation_variable_selector,
    io_seniority_selector,
    joint_distribution,
    spells,
):
    io_country_scope = io_country_selector.value
    io_seniority_levels = tuple(io_seniority_selector.value or ())
    io_industry_column = io_industry_variable_selector.value
    io_occupation_column = io_occupation_variable_selector.value
    io_subset = filter_spells(spells, io_country_scope, io_seniority_levels)
    io_industry_distribution = classification_distribution(
        io_subset, io_industry_column, TITLE_COLUMNS.get(io_industry_column)
    )
    io_occupation_distribution = classification_distribution(
        io_subset, io_occupation_column, TITLE_COLUMNS.get(io_occupation_column)
    )
    io_cells = joint_distribution(
        io_subset,
        io_industry_column,
        io_occupation_column,
        TITLE_COLUMNS.get(io_industry_column),
        TITLE_COLUMNS.get(io_occupation_column),
    )
    return (
        io_cells,
        io_country_scope,
        io_industry_column,
        io_occupation_column,
        io_seniority_levels,
        io_subset,
    )


@app.cell
def io_joint_output(
    io_cells,
    io_country_scope,
    io_country_selector,
    io_industry_column,
    io_industry_variable_selector,
    io_occupation_column,
    io_occupation_variable_selector,
    io_seniority_levels,
    io_seniority_selector,
    io_subset,
    io_top_n_selector,
    make_share_chart,
    mo,
    scope_label,
):
    _top_n = max(1, int(io_top_n_selector.value))
    _scope = scope_label(io_country_scope)
    _subtitle = (
        f"{len(io_subset):,} employment spells — {_scope}"
        + (
            f", seniority {', '.join(str(level) for level in io_seniority_levels)}"
            if io_seniority_levels
            else ", all seniority levels"
        )
        + f". Top {_top_n} of {len(io_cells):,} populated cells."
    )
    _figure = (
        make_share_chart(
            io_cells,
            (f"Joint {io_industry_column} × {io_occupation_column} distribution — {_scope}"),
            subtitle=_subtitle,
            top_n=_top_n,
            color="#0F766E",
        )
        if not io_cells.empty
        else mo.callout(
            mo.md("No employment spell remains in the selected subsample."), kind="warn"
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 4. Industry-occupation distribution

                ### 4.1. Joint industry-occupation distribution in horizontal bar charts
                """
            ),
            mo.hstack(
                [
                    io_country_selector,
                    io_seniority_selector,
                    io_industry_variable_selector,
                    io_occupation_variable_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            io_top_n_selector,
            _figure,
            mo.accordion(
                {
                    "View the complete joint distribution": mo.ui.table(
                        io_cells,
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
def heatmap_variable_controls(
    DEFAULT_INDUSTRY_LABEL,
    DEFAULT_OCCUPATION_LABEL,
    INDUSTRY_VARIABLE_OPTIONS,
    OCCUPATION_VARIABLE_OPTIONS,
    mo,
):
    heatmap_industry_variable_selector = mo.ui.dropdown(
        options=INDUSTRY_VARIABLE_OPTIONS,
        value=DEFAULT_INDUSTRY_LABEL,
        label="Industry variable",
    )
    heatmap_occupation_variable_selector = mo.ui.dropdown(
        options=OCCUPATION_VARIABLE_OPTIONS,
        value=DEFAULT_OCCUPATION_LABEL,
        label="Occupation variable",
    )
    return (
        heatmap_industry_variable_selector,
        heatmap_occupation_variable_selector,
    )


@app.cell
def heatmap_data(
    TITLE_COLUMNS,
    classification_distribution,
    heatmap_industry_variable_selector,
    heatmap_occupation_variable_selector,
    io_subset,
    joint_distribution,
):
    heatmap_industry_column = heatmap_industry_variable_selector.value
    heatmap_occupation_column = heatmap_occupation_variable_selector.value
    heatmap_industry_distribution = classification_distribution(
        io_subset,
        heatmap_industry_column,
        TITLE_COLUMNS.get(heatmap_industry_column),
    )
    heatmap_occupation_distribution = classification_distribution(
        io_subset,
        heatmap_occupation_column,
        TITLE_COLUMNS.get(heatmap_occupation_column),
    )
    heatmap_cells = joint_distribution(
        io_subset,
        heatmap_industry_column,
        heatmap_occupation_column,
        TITLE_COLUMNS.get(heatmap_industry_column),
        TITLE_COLUMNS.get(heatmap_occupation_column),
    )
    return (
        heatmap_cells,
        heatmap_industry_column,
        heatmap_industry_distribution,
        heatmap_occupation_column,
        heatmap_occupation_distribution,
    )


@app.cell
def heatmap_category_controls(
    DEFAULT_BIOPHARM_INDUSTRIES,
    DEFAULT_BIOPHARM_OCCUPATIONS,
    DEFAULT_HEATMAP_CATEGORIES,
    default_category_labels,
    heatmap_industry_distribution,
    heatmap_occupation_distribution,
    mo,
):
    def _option_pairs(distribution):
        return dict(
            zip(
                distribution["display_label"],
                distribution["value"].astype(str),
                strict=False,
            )
        )

    heatmap_industry_options = _option_pairs(heatmap_industry_distribution)
    heatmap_occupation_options = _option_pairs(heatmap_occupation_distribution)
    heatmap_industry_selector = mo.ui.multiselect(
        options=heatmap_industry_options,
        value=default_category_labels(
            heatmap_industry_distribution,
            DEFAULT_BIOPHARM_INDUSTRIES,
            DEFAULT_HEATMAP_CATEGORIES,
        ),
        label="Industries on the x-axis",
        full_width=True,
    )
    heatmap_occupation_selector = mo.ui.multiselect(
        options=heatmap_occupation_options,
        value=default_category_labels(
            heatmap_occupation_distribution,
            DEFAULT_BIOPHARM_OCCUPATIONS,
            DEFAULT_HEATMAP_CATEGORIES,
        ),
        label="Occupations on the y-axis",
        full_width=True,
    )
    return heatmap_industry_selector, heatmap_occupation_selector


@app.cell
def heatmap_output(
    heatmap_cells,
    heatmap_industry_column,
    heatmap_industry_distribution,
    heatmap_industry_selector,
    heatmap_industry_variable_selector,
    heatmap_occupation_column,
    heatmap_occupation_distribution,
    heatmap_occupation_selector,
    heatmap_occupation_variable_selector,
    io_country_scope,
    io_seniority_levels,
    io_subset,
    make_heatmap,
    mo,
    pd,
    scope_label,
):
    _industry_values = tuple(heatmap_industry_selector.value or ())
    _occupation_values = tuple(heatmap_occupation_selector.value or ())
    _industry_labels = dict(
        zip(
            heatmap_industry_distribution["value"].astype(str),
            heatmap_industry_distribution["display_label"],
            strict=False,
        )
    )
    _occupation_labels = dict(
        zip(
            heatmap_occupation_distribution["value"].astype(str),
            heatmap_occupation_distribution["display_label"],
            strict=False,
        )
    )
    _industry_order = [
        _industry_labels[value] for value in _industry_values if value in _industry_labels
    ]
    _occupation_order = [
        _occupation_labels[value] for value in _occupation_values if value in _occupation_labels
    ]
    if _industry_values and _occupation_values:
        _grid = pd.MultiIndex.from_product(
            [list(_industry_values), list(_occupation_values)],
            names=["industry_value", "occupation_value"],
        ).to_frame(index=False)
        _grid = _grid.merge(
            heatmap_cells.loc[:, ["industry_value", "occupation_value", "counts", "share"]],
            on=["industry_value", "occupation_value"],
            how="left",
        )
        _grid["counts"] = _grid["counts"].fillna(0).astype("int64")
        _grid["share"] = _grid["share"].fillna(0.0)
        _grid["industry_label"] = _grid["industry_value"].map(_industry_labels)
        _grid["occupation_label"] = _grid["occupation_value"].map(_occupation_labels)
        _spells_total = len(io_subset)
        _seniority_text = (
            f", seniority {', '.join(str(level) for level in io_seniority_levels)}"
            if io_seniority_levels
            else ", all seniority levels"
        )
        _coverage_text = (
            f". Selected cells cover {float(_grid['counts'].sum()) / _spells_total:.2%} "
            "of the subsample."
            if _spells_total
            else "."
        )
        _subtitle = (
            f"{_spells_total:,} employment spells — {scope_label(io_country_scope)}"
            + _seniority_text
            + _coverage_text
        )
        _figure = make_heatmap(
            _grid,
            _industry_order,
            _occupation_order,
            heatmap_industry_column,
            heatmap_occupation_column,
            (
                f"Joint {heatmap_industry_column} × {heatmap_occupation_column} shares — "
                f"{scope_label(io_country_scope)}"
            ),
            subtitle=_subtitle,
        )
        _table = mo.ui.table(
            _grid.loc[
                :,
                [
                    "industry_label",
                    "occupation_label",
                    "counts",
                    "share",
                ],
            ],
            pagination=True,
            page_size=20,
            show_column_summaries=False,
        )
    else:
        _figure = mo.callout(mo.md("Select at least one industry and one occupation."), kind="warn")
        _table = mo.md("")
    mo.vstack(
        [
            mo.md(
                r"""
                ### 4.2. Joint distribution as a heatmap

                - Occupations are on the y-axis, industries are on the x-axis, and the darkness of a cell reports its share of the selected subsample.
                    - Cells with no employment spell stay light gray.
                - The shares are small -- which is normal.
                    - Recall from the last bar chart, the industry-occupation that has the largest share in the inventors' employment spells is Research University $\times$ Animal Scientists -- which is only 2.5% of the full employment spells.
                """
            ),
            mo.hstack(
                [
                    heatmap_industry_variable_selector,
                    heatmap_occupation_variable_selector,
                ],
                justify="start",
                gap=2,
                widths="equal",
            ),
            heatmap_industry_selector,
            heatmap_occupation_selector,
            _figure,
            mo.accordion({"View the heatmap cells": _table}),
        ],
        gap=1,
    )
    return


@app.cell
def occupation_within_industry_variable_controls(
    DEFAULT_INDUSTRY_LABEL,
    DEFAULT_OCCUPATION_LABEL,
    INDUSTRY_VARIABLE_OPTIONS,
    OCCUPATION_VARIABLE_OPTIONS,
    mo,
):
    occupation_within_industry_industry_variable_selector = mo.ui.dropdown(
        options=INDUSTRY_VARIABLE_OPTIONS,
        value=DEFAULT_INDUSTRY_LABEL,
        label="Industry variable",
    )
    occupation_within_industry_occupation_variable_selector = mo.ui.dropdown(
        options=OCCUPATION_VARIABLE_OPTIONS,
        value=DEFAULT_OCCUPATION_LABEL,
        label="Occupation variable",
    )
    return (
        occupation_within_industry_industry_variable_selector,
        occupation_within_industry_occupation_variable_selector,
    )


@app.cell
def occupation_within_industry_data(
    TITLE_COLUMNS,
    classification_distribution,
    io_subset,
    joint_distribution,
    occupation_within_industry_industry_variable_selector,
    occupation_within_industry_occupation_variable_selector,
):
    occupation_within_industry_industry_column = (
        occupation_within_industry_industry_variable_selector.value
    )
    occupation_within_industry_occupation_column = (
        occupation_within_industry_occupation_variable_selector.value
    )
    occupation_within_industry_industry_distribution = classification_distribution(
        io_subset,
        occupation_within_industry_industry_column,
        TITLE_COLUMNS.get(occupation_within_industry_industry_column),
    )
    occupation_within_industry_cells = joint_distribution(
        io_subset,
        occupation_within_industry_industry_column,
        occupation_within_industry_occupation_column,
        TITLE_COLUMNS.get(occupation_within_industry_industry_column),
        TITLE_COLUMNS.get(occupation_within_industry_occupation_column),
    )
    return (
        occupation_within_industry_cells,
        occupation_within_industry_industry_column,
        occupation_within_industry_industry_distribution,
        occupation_within_industry_occupation_column,
    )


@app.cell
def occupation_within_industry_controls(
    DEFAULT_BIOPHARM_INDUSTRIES,
    DEFAULT_TOP_N,
    default_category_labels,
    mo,
    occupation_within_industry_industry_distribution,
):
    _industry_options = dict(
        zip(
            occupation_within_industry_industry_distribution["display_label"],
            occupation_within_industry_industry_distribution["value"].astype(str),
            strict=False,
        )
    )
    occupation_within_industry_selector = mo.ui.multiselect(
        options=_industry_options,
        value=default_category_labels(
            occupation_within_industry_industry_distribution,
            DEFAULT_BIOPHARM_INDUSTRIES,
            3,
        ),
        label="Industries pooled before the occupation shares are calculated",
        full_width=True,
    )
    occupation_within_industry_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of occupations in the figure",
    )
    return (
        occupation_within_industry_selector,
        occupation_within_industry_top_n_selector,
    )


@app.cell
def occupation_within_industry_output(
    distribution_from_counts,
    io_country_scope,
    make_share_chart,
    mo,
    occupation_within_industry_cells,
    occupation_within_industry_industry_column,
    occupation_within_industry_industry_variable_selector,
    occupation_within_industry_occupation_column,
    occupation_within_industry_occupation_variable_selector,
    occupation_within_industry_selector,
    occupation_within_industry_top_n_selector,
    scope_label,
    subset_cells,
):
    _selected_industries = tuple(occupation_within_industry_selector.value or ())
    _top_n = max(1, int(occupation_within_industry_top_n_selector.value))
    _selected_cells = subset_cells(
        occupation_within_industry_cells,
        "industry_value",
        _selected_industries,
    )
    _summary = distribution_from_counts(_selected_cells, "occupation_value", "occupation_label")
    _spells_in_selection = int(_selected_cells["counts"].sum())
    _subtitle = (
        f"{_spells_in_selection:,} employment spells in the selected industries — "
        f"{scope_label(io_country_scope)}. Top {_top_n} of {len(_summary):,} occupations."
    )
    _figure = (
        make_share_chart(
            _summary,
            (
                f"Occupation distribution ({occupation_within_industry_occupation_column}) "
                f"within selected {occupation_within_industry_industry_column} industries — "
                f"{scope_label(io_country_scope)}"
            ),
            subtitle=_subtitle,
            top_n=_top_n,
            x_title="Share within the selected industries",
            color="#1D4ED8",
        )
        if _selected_industries
        else mo.callout(mo.md("Select at least one industry."), kind="warn")
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 4.3. Occupation distribution within selected industries

                - The selected industries are pooled before the occupation shares are calculated, so the denominator is the number of employment spells in the selected industries.
                    - This is the step that turns a focal industry into a set of focal occupations.
                - **In the BioPharma industry of interest, around 50% of inventors' employment spells are classified as "Animal Scientists" in ONET, and most ONET occupations are under 2-digit code 17 or 19**
                    - The only ONET occupation of inventors' in the BioPharma industry that could be missing is "Software Developers", which is only 2.3%.
                    - Anyway, I don't think we should include the sample of programmers in our analysis sample.
                """
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
            occupation_within_industry_selector,
            occupation_within_industry_top_n_selector,
            _figure,
            mo.accordion(
                {
                    "View the complete conditional distribution": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View the selected joint cells": mo.ui.table(
                        _selected_cells,
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
def industry_within_occupation_variable_controls(
    DEFAULT_INDUSTRY_LABEL,
    DEFAULT_OCCUPATION_LABEL,
    INDUSTRY_VARIABLE_OPTIONS,
    OCCUPATION_VARIABLE_OPTIONS,
    mo,
):
    industry_within_occupation_industry_variable_selector = mo.ui.dropdown(
        options=INDUSTRY_VARIABLE_OPTIONS,
        value=DEFAULT_INDUSTRY_LABEL,
        label="Industry variable",
    )
    industry_within_occupation_occupation_variable_selector = mo.ui.dropdown(
        options=OCCUPATION_VARIABLE_OPTIONS,
        value=DEFAULT_OCCUPATION_LABEL,
        label="Occupation variable",
    )
    return (
        industry_within_occupation_industry_variable_selector,
        industry_within_occupation_occupation_variable_selector,
    )


@app.cell
def industry_within_occupation_data(
    TITLE_COLUMNS,
    classification_distribution,
    industry_within_occupation_industry_variable_selector,
    industry_within_occupation_occupation_variable_selector,
    io_subset,
    joint_distribution,
):
    industry_within_occupation_industry_column = (
        industry_within_occupation_industry_variable_selector.value
    )
    industry_within_occupation_occupation_column = (
        industry_within_occupation_occupation_variable_selector.value
    )
    industry_within_occupation_occupation_distribution = classification_distribution(
        io_subset,
        industry_within_occupation_occupation_column,
        TITLE_COLUMNS.get(industry_within_occupation_occupation_column),
    )
    industry_within_occupation_cells = joint_distribution(
        io_subset,
        industry_within_occupation_industry_column,
        industry_within_occupation_occupation_column,
        TITLE_COLUMNS.get(industry_within_occupation_industry_column),
        TITLE_COLUMNS.get(industry_within_occupation_occupation_column),
    )
    return (
        industry_within_occupation_cells,
        industry_within_occupation_industry_column,
        industry_within_occupation_occupation_column,
        industry_within_occupation_occupation_distribution,
    )


@app.cell
def industry_within_occupation_controls(
    DEFAULT_BIOPHARM_OCCUPATIONS,
    DEFAULT_TOP_N,
    default_category_labels,
    industry_within_occupation_occupation_distribution,
    mo,
):
    _occupation_options = dict(
        zip(
            industry_within_occupation_occupation_distribution["display_label"],
            industry_within_occupation_occupation_distribution["value"].astype(str),
            strict=False,
        )
    )
    industry_within_occupation_selector = mo.ui.multiselect(
        options=_occupation_options,
        value=default_category_labels(
            industry_within_occupation_occupation_distribution,
            DEFAULT_BIOPHARM_OCCUPATIONS,
            3,
        ),
        label="Occupations pooled before the industry shares are calculated",
        full_width=True,
    )
    industry_within_occupation_top_n_selector = mo.ui.number(
        start=1,
        stop=2000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of industries in the figure",
    )
    return (
        industry_within_occupation_selector,
        industry_within_occupation_top_n_selector,
    )


@app.cell
def industry_within_occupation_output(
    distribution_from_counts,
    industry_within_occupation_cells,
    industry_within_occupation_industry_column,
    industry_within_occupation_industry_variable_selector,
    industry_within_occupation_occupation_column,
    industry_within_occupation_occupation_variable_selector,
    industry_within_occupation_selector,
    industry_within_occupation_top_n_selector,
    io_country_scope,
    make_share_chart,
    mo,
    scope_label,
    subset_cells,
):
    _selected_occupations = tuple(industry_within_occupation_selector.value or ())
    _top_n = max(1, int(industry_within_occupation_top_n_selector.value))
    _selected_cells = subset_cells(
        industry_within_occupation_cells,
        "occupation_value",
        _selected_occupations,
    )
    _summary = distribution_from_counts(_selected_cells, "industry_value", "industry_label")
    _spells_in_selection = int(_selected_cells["counts"].sum())
    _subtitle = (
        f"{_spells_in_selection:,} employment spells in the selected occupations — "
        f"{scope_label(io_country_scope)}. Top {_top_n} of {len(_summary):,} industries."
    )
    _figure = (
        make_share_chart(
            _summary,
            (
                f"Industry distribution ({industry_within_occupation_industry_column}) "
                f"within selected {industry_within_occupation_occupation_column} "
                f"occupations — {scope_label(io_country_scope)}"
            ),
            subtitle=_subtitle,
            top_n=_top_n,
            x_title="Share within the selected occupations",
            color="#C2410C",
        )
        if _selected_occupations
        else mo.callout(mo.md("Select at least one occupation."), kind="warn")
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 4.4. Industry distribution within selected occupations

                - The selected occupations are pooled before the industry shares are calculated, so the denominator is the number of employment spells in the selected occupations.
                - A dispersed industry mix suggests that the occupation is not specific to the industries we care about.
                """
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
            industry_within_occupation_selector,
            industry_within_occupation_top_n_selector,
            _figure,
            mo.accordion(
                {
                    "View the complete conditional distribution": mo.ui.table(
                        _summary,
                        pagination=True,
                        page_size=20,
                        show_column_summaries=False,
                    ),
                    "View the selected joint cells": mo.ui.table(
                        _selected_cells,
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
def seniority_controls(COUNTRY_SCOPE_OPTIONS, US_LABEL, mo):
    seniority_country_selector = mo.ui.dropdown(
        options=COUNTRY_SCOPE_OPTIONS,
        value=US_LABEL,
        label="Country scope",
    )
    return (seniority_country_selector,)


@app.cell
def seniority_output(
    MISSING_LABEL,
    construct_distribution_table,
    filter_spells,
    make_seniority_chart,
    mo,
    pd,
    scope_label,
    seniority_country_selector,
    spells,
):
    _country_scope = seniority_country_selector.value
    _subset = filter_spells(spells, _country_scope, ())
    _labels = _subset["seniority"].astype("string").fillna(MISSING_LABEL)
    _summary = construct_distribution_table(_labels, include_missing=False)
    _summary = _summary.rename(columns={"value": "display_label"})
    _summary["seniority_order"] = pd.to_numeric(_summary["display_label"], errors="coerce")
    _summary = _summary.sort_values(
        ["seniority_order", "display_label"], na_position="last", ignore_index=True
    )
    _subtitle = f"{len(_subset):,} employment spells — {scope_label(_country_scope)}. "
    _figure = (
        make_seniority_chart(
            _summary,
            f"Seniority distribution — {scope_label(_country_scope)}",
            subtitle=_subtitle,
        )
        if not _summary.empty
        else mo.callout(
            mo.md("No employment spell remains in the selected subsample."), kind="warn"
        )
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ## 5. Other statistics

                ### 5.1. Seniority distribution
                """
            ),
            seniority_country_selector,
            _figure,
            mo.accordion(
                {
                    "View the complete seniority distribution": mo.ui.table(
                        _summary.loc[:, ["display_label", "counts", "share", "rank"]],
                        pagination=False,
                        show_column_summaries=False,
                    )
                }
            ),
        ],
        gap=1,
    )
    return


@app.cell
def country_controls(DEFAULT_TOP_N, mo):
    country_top_n_selector = mo.ui.number(
        start=1,
        stop=1000,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of countries in the figure",
    )
    return (country_top_n_selector,)


@app.cell
def country_output(
    MISSING_LABEL,
    construct_distribution_table,
    country_top_n_selector,
    make_share_chart,
    mo,
    spells,
):
    _labels = spells["country"].astype("string").fillna(MISSING_LABEL)
    _summary = construct_distribution_table(_labels, include_missing=False)
    _summary = _summary.rename(columns={"value": "display_label"})
    _summary["value"] = _summary["display_label"]
    _top_n = max(1, int(country_top_n_selector.value))
    _subtitle = (
        f"{len(spells):,} employment spells in all countries. "
        f"Top {_top_n} of {len(_summary):,} countries."
    )
    _figure = (
        make_share_chart(
            _summary,
            "Country distribution across inventor employment spells",
            subtitle=_subtitle,
            top_n=_top_n,
            color="#0F766E",
        )
        if not _summary.empty
        else mo.callout(mo.md("No employment spell has a country value."), kind="warn")
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 5.2. Country distribution

                - Notice that the inventors are those LinkedIn users who can be matched to an inventor ID in USPTO.
                - Inventor coverage in Revelio is far stronger in the United States than elsewhere.
                - Therefore, the default subsample elsewhere in this report is restricted to U.S. spells.
                """
            ),
            country_top_n_selector,
            _figure,
            mo.accordion(
                {
                    "View the complete country distribution": mo.ui.table(
                        _summary.loc[:, ["value", "display_label", "counts", "share", "rank"]],
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
def title_controls(
    DEFAULT_BIOPHARM_INDUSTRIES,
    DEFAULT_BIOPHARM_OCCUPATIONS,
    DEFAULT_SENIORITY_LEVELS,
    DEFAULT_TOP_N,
    MISSING_LABEL,
    SENIORITY_OPTIONS,
    US_LABEL,
    classification_distribution,
    default_category_labels,
    default_option_labels,
    mo,
    pd,
    spells,
):
    _title_seniority_options = {
        label: value
        for label, value in SENIORITY_OPTIONS.items()
        if value != MISSING_LABEL and int(value) < 5
    }
    _seniority = pd.to_numeric(spells["seniority"], errors="coerce")
    _options_subset = spells.loc[spells["country"].eq(US_LABEL) & _seniority.lt(5)]
    _occupation_distribution = classification_distribution(
        _options_subset,
        "onet_code",
        "onet_title",
    )
    _occupation_distribution = _occupation_distribution.loc[
        _occupation_distribution["value"].astype(str).ne(MISSING_LABEL)
    ]
    _industry_distribution = classification_distribution(
        _options_subset,
        "rics_k400",
    )
    _industry_distribution = _industry_distribution.loc[
        _industry_distribution["value"].astype(str).ne(MISSING_LABEL)
    ]
    _occupation_options_by_label = dict(
        zip(
            _occupation_distribution["display_label"],
            _occupation_distribution["value"].astype(str),
            strict=False,
        )
    )
    _default_occupation_labels = default_category_labels(
        _occupation_distribution,
        DEFAULT_BIOPHARM_OCCUPATIONS,
        6,
    )
    _occupation_options = {
        label: _occupation_options_by_label[label] for label in _default_occupation_labels
    }
    _occupation_options.update(_occupation_options_by_label)
    _industry_options_by_label = dict(
        zip(
            _industry_distribution["display_label"],
            _industry_distribution["value"].astype(str),
            strict=False,
        )
    )
    _default_industry_labels = default_category_labels(
        _industry_distribution,
        DEFAULT_BIOPHARM_INDUSTRIES,
        3,
    )
    _industry_options = {
        label: _industry_options_by_label[label] for label in _default_industry_labels
    }
    _industry_options.update(_industry_options_by_label)

    _title_seniority_selector = mo.ui.multiselect(
        options=_title_seniority_options,
        value=default_option_labels(
            _title_seniority_options,
            DEFAULT_SENIORITY_LEVELS,
        ),
        label="Seniority levels below 5",
    )
    _title_industry_selector = mo.ui.multiselect(
        options=_industry_options,
        value=_default_industry_labels,
        label="rics_k400 industries",
        full_width=True,
    )
    _title_occupation_selector = mo.ui.multiselect(
        options=_occupation_options,
        value=list(_occupation_options),
        label="ONET occupations",
        full_width=True,
    )
    _title_top_n_selector = mo.ui.number(
        start=0,
        stop=300,
        step=1,
        value=DEFAULT_TOP_N,
        label="Number of normalized titles in the bar chart",
    )
    _title_controls = mo.ui.dictionary(
        {
            "seniority_levels": _title_seniority_selector,
            "bar_chart_top_n": _title_top_n_selector,
            "industries": _title_industry_selector,
            "occupations": _title_occupation_selector,
        },
        label="Sample and figure settings",
    )
    title_filter_form = _title_controls.form(
        label="Section 6 settings",
        submit_button_label="Calculate and normalize job titles",
        submit_button_tooltip=("Apply the current selections and run the job-title normalization."),
    )
    return (title_filter_form,)


@app.cell
def title_output(
    US_LABEL,
    make_share_chart,
    make_word_cloud,
    mo,
    normalize_job_titles,
    pd,
    spells,
    title_filter_form,
):
    _intro = mo.md(
        r"""
        ## 6. Self-reported job titles

        - Recall the two purposes of this report:
            1. We want to have a better understanding of the USPTO inventors.
            2. **We want to know the self-reported job titles of these USPTO inventors in case we want to construct the analysis sample using job titles instead of ONET occupation codes.**
        - This section will target purpose 2 -- the self-reported job titles among USPTO inventors.
            - I restrict to inventors' employment spells in US.
            - Occupation variable is fixed to ONET and industry variable is fixed to `rics_k400`.
        - Changing a form field does not run the job-title calculation. Click the **Calculate and normalize job titles** button after finalizing all selections.
        - Visualization:
            - In the bar chart, shares are calculated against all nonmissing normalized titles in the selected subsample. 
            - The word cloud always keeps the top 300 normalized job titles. A title with a higher share is drawn with a larger font in the word cloud.
        """
    )
    _selection = title_filter_form.value
    mo.stop(
        _selection is None,
        mo.vstack(
            [
                _intro,
                title_filter_form,
                mo.callout(
                    mo.md("Finalize the settings above, then click the calculation button."),
                    kind="info",
                ),
            ],
            gap=1,
        ),
    )

    _seniority_levels = tuple(_selection["seniority_levels"] or ())
    _selected_industries = tuple(_selection["industries"] or ())
    _selected_occupations = tuple(_selection["occupations"] or ())
    _top_n = min(300, max(0, int(_selection["bar_chart_top_n"])))

    _seniority = pd.to_numeric(spells["seniority"], errors="coerce")
    _keep = spells["country"].eq(US_LABEL) & _seniority.lt(5)
    if _seniority_levels:
        _keep &= _seniority.isin(_seniority_levels)
    _subset = spells.loc[_keep]
    _subset = _subset.loc[_subset["rics_k400"].astype("string").isin(_selected_industries)]
    _subset = _subset.loc[_subset["onet_code"].astype("string").isin(_selected_occupations)]

    with mo.status.spinner("Normalizing self-reported job titles ..."):
        _raw_titles = _subset["title_raw"].astype("string")
        _lookup = (
            _raw_titles.value_counts(dropna=True)
            .rename("employment_spell_count")
            .rename_axis("title_raw")
            .reset_index()
        )
        _lookup["title_normalized"] = normalize_job_titles(_lookup["title_raw"])
        _normalized = (
            _lookup.dropna(subset=["title_normalized"])
            .groupby("title_normalized", as_index=False, sort=False)
            .agg(
                counts=("employment_spell_count", "sum"),
                raw_title_variants=("title_raw", "size"),
            )
            .rename(columns={"title_normalized": "display_label"})
            .sort_values(
                ["counts", "display_label"],
                ascending=[False, True],
                ignore_index=True,
            )
        )
        _normalized["share"] = _normalized["counts"] / _normalized["counts"].sum()
        _normalized["rank"] = range(1, len(_normalized) + 1)
        _normalized["value"] = _normalized["display_label"]

    _seniority_text = (
        ", ".join(str(level) for level in _seniority_levels)
        if _seniority_levels
        else "all available levels below 5"
    )
    _industry_text = f"{len(_selected_industries):,} selected industries"
    _occupation_text = f"{len(_selected_occupations):,} selected occupations"
    _subtitle = (
        f"{len(_subset):,} employment spells; Seniority level {_seniority_text}; "
        f"{_industry_text}; {_occupation_text}. "
        f"Top {_top_n} of {len(_normalized):,} normalized titles."
    )
    _displayed_titles = _normalized.head(_top_n)

    if _normalized.empty:
        _content = mo.callout(
            mo.md("No nonmissing normalized job title remains in the selected subsample."),
            kind="warn",
        )
    else:
        _bar_chart = (
            make_share_chart(
                _normalized,
                f"Normalized self-reported job titles — {US_LABEL}",
                subtitle=_subtitle,
                top_n=_top_n,
                color="#6D28D9",
            )
            if _top_n > 0
            else mo.callout(
                mo.md("Set the Top-N control above zero to display the bar chart."),
                kind="warn",
            )
        )
        _content = mo.vstack(
            [
                _bar_chart,
                make_word_cloud(
                    _normalized,
                    "Top 300 normalized self-reported job titles — word cloud",
                    300,
                ),
                mo.accordion(
                    {
                        "View the displayed normalized titles": mo.ui.table(
                            _displayed_titles.loc[
                                :,
                                [
                                    "value",
                                    "display_label",
                                    "counts",
                                    "share",
                                    "rank",
                                    "raw_title_variants",
                                ],
                            ],
                            pagination=True,
                            page_size=20,
                            show_column_summaries=False,
                        )
                    }
                ),
            ],
            gap=1,
        )
    _diagnostics = pd.DataFrame(
        [
            {
                "Diagnostic": "Employment spells after all sample filters",
                "Value": len(_subset),
            },
            {
                "Diagnostic": "Distinct nonmissing raw titles",
                "Value": len(_lookup),
            },
            {
                "Diagnostic": "Distinct normalized titles",
                "Value": len(_normalized),
            },
            {
                "Diagnostic": "Normalized titles with multiple raw variants",
                "Value": int((_normalized["raw_title_variants"] > 1).sum()),
            },
            {
                "Diagnostic": "Spells missing a normalized title",
                "Value": int(_raw_titles.isna().sum()),
            },
        ]
    )
    _items = [
        _intro,
        title_filter_form,
        _content,
        mo.accordion(
            {
                "Job-title normalization diagnostics": mo.ui.table(
                    _diagnostics,
                    pagination=False,
                    show_column_summaries=False,
                    format_mapping={"Value": "{:,}"},
                )
            }
        ),
    ]
    mo.vstack(_items, gap=1)
    return


if __name__ == "__main__":
    app.run()

"""
Task:
    Inspect employment spells of Revelio users linked to USPTO inventors and normalize their
    self-reported US job titles.

Inputs:
(a) data/b_temp_data/B02_InspectUSPTOInventors/Inventors_USPTO_UserPositions/*.parquet

Outputs:
    Not applicable.

Descriptions of outputs:
    Not applicable.

Run:
    conda run -s -n Talent python -m codes.B02_InspectUSPTOInventors.E01_USPTOInventors

Notes:
(1) The analysis drops employment spells with a missing Revelio company identifier (`rcid`).
(2) Counts and distributions use employment spells, rather than inventors, as the unit.
(3) Job-title normalization applies Unicode compatibility normalization, case folding,
    punctuation and whitespace standardization, and selected low-ambiguity abbreviation rules.
(4) Each distinct nonmissing raw title is normalized once and then mapped back to US spells.
(5) The normalizer does not use fuzzy matching, stemming, token reordering, or ambiguous
    abbreviation expansions. The script constructs all distributions in memory and saves no files.


Wang Wenzhi, with the help of Codex
Time: 2026-08-28
"""

import pandas as pd

from codes import main

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define settings and the distribution helper
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_DIR = main.DIR_TEMPDATA / "B02_InspectUSPTOInventors" / "Inventors_USPTO_UserPositions"
INPUT_COLUMNS = [
    "user_id",
    "position_id",
    "position_number",
    "rcid",
    "country",
    "role_k50",
    "role_k150",
    "role_k300",
    "role_k500",
    "role_k1000",
    "role_k1500",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
    "rics_k50",
    "rics_k200",
    "rics_k400",
    "title_raw",
    "title_translated",
    "seniority",
]
MISSING_COUNTRY_VALUE = "empty"
UNITED_STATES_LABEL = "United States"


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


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read inventor employment spells
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


input_files = tuple(sorted(INPUT_DIR.glob("*.parquet")))

main.report_status(f"Number of individual Parquet files: {len(input_files)}", level="info")

inventor_spells = pd.concat(
    [pd.read_parquet(input_file, columns=INPUT_COLUMNS) for input_file in input_files],
    ignore_index=True,
)
main.report_status(
    "Finished reading and appending all Parquet files for inventor employment spells.",
    level="info",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Clean missing identifiers
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


main.report_status("Drop employment spells with missing company IDs.", level="info")
inventor_spells = inventor_spells.dropna(subset=["rcid"])

flag_missing_country = inventor_spells["country"].eq(MISSING_COUNTRY_VALUE)
inventor_spells.loc[flag_missing_country, "country"] = pd.NA


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Report basic employment-spell statistics
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


n_spells = len(inventor_spells)
n_inventors = inventor_spells["user_id"].nunique()
n_companies = inventor_spells["rcid"].nunique()
n_countries = inventor_spells.loc[
    inventor_spells["country"].notna(),
    "country",
].nunique()
n_naics = inventor_spells.loc[
    inventor_spells["naics_code"].notna(),
    "naics_code",
].nunique()
n_onet = inventor_spells.loc[
    inventor_spells["onet_code"].notna(),
    "onet_code",
].nunique()

main.report_status(
    (
        "Basic numbers:\n"
        f"  - Number of employment spells: {n_spells:,}\n"
        f"  - Number of distinct inventors: {n_inventors:,}\n"
        f"  - Number of distinct companies: {n_companies:,}\n"
        f"  - Number of distinct countries: {n_countries:,}\n"
        f"  - Number of distinct NAICS industries: {n_naics:,}\n"
        f"  - Number of distinct ONET occupations: {n_onet:,}"
    ),
    level="info",
)

flag_us = inventor_spells["country"].eq(UNITED_STATES_LABEL)
inventor_positions_us = inventor_spells.loc[flag_us]

n_spells_us = len(inventor_positions_us)
n_inventors_us = inventor_positions_us["user_id"].nunique()
n_companies_us = inventor_positions_us["rcid"].nunique()
n_naics_us = inventor_positions_us.loc[
    inventor_positions_us["naics_code"].notna(),
    "naics_code",
].nunique()
n_onet_us = inventor_positions_us.loc[
    inventor_positions_us["onet_code"].notna(),
    "onet_code",
].nunique()

main.report_status(
    (
        "Basic numbers within US:\n"
        f"  - Number of employment spells: {n_spells_us:,}\n"
        f"  - Number of distinct inventors: {n_inventors_us:,}\n"
        f"  - Number of distinct companies: {n_companies_us:,}\n"
        f"  - Number of distinct NAICS industries: {n_naics_us:,}\n"
        f"  - Number of distinct ONET occupations: {n_onet_us:,}"
    ),
    level="info",
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Construct employment-spell distributions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


country_distribution = construct_distribution_table(inventor_spells["country"])

us_onet_title_distribution = construct_distribution_table(
    inventor_spells.loc[flag_us, "onet_title"]
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 6. Investigate self-reported job titles within US
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-6-1. Define helpers to normalize job titles
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


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
    (3) Fuzzy matching, stemming, spell correction, and token reordering are excluded to avoid
        merging substantively different titles.
    """

    normalized_titles = (
        job_titles.astype("string")
        .str.normalize("NFKC")
        .str.casefold()
        .str.replace("&amp;", "&", regex=False)
        .str.strip()
    )

    # Normalize research-and-development variants before treating punctuation as separators.
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


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-6-2. Investigate the job titles
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

flag_desired_seniority = inventor_spells["seniority"].isin([2, 3])
inventor_spells_us = inventor_spells.loc[flag_us & flag_desired_seniority].copy()
raw_job_titles = inventor_spells_us["title_raw"].astype("string")
n_missing_raw_job_titles = raw_job_titles.isna().sum()
main.report_status(
    f"Number of missing raw job titles within US: {n_missing_raw_job_titles:,}",
    level="warning",
)

# Normalize only distinct titles because the raw data contain many repeated employment-spell values.
raw_job_title_counts = raw_job_titles.value_counts(dropna=True)
job_title_normalization_lookup = (
    raw_job_title_counts.rename("employment_spell_count").rename_axis("title_raw").reset_index()
)
job_title_normalization_lookup["title_normalized"] = normalize_job_titles(
    job_title_normalization_lookup["title_raw"]
)
job_title_normalization_lookup = job_title_normalization_lookup[
    [
        "title_raw",
        "title_normalized",
        "employment_spell_count",
    ]
]

title_normalization_map = job_title_normalization_lookup.set_index("title_raw")["title_normalized"]
inventor_spells_us["title_normalized"] = raw_job_titles.map(title_normalization_map).astype(
    "string"
)

# Aggregate the lookup rather than recounting all US spells after normalization.
us_normalized_job_title_distribution = (
    job_title_normalization_lookup.dropna(subset=["title_normalized"])
    .groupby("title_normalized", as_index=False, sort=False)
    .agg(
        counts=("employment_spell_count", "sum"),
        raw_title_variants=("title_raw", "size"),
    )
    .rename(columns={"title_normalized": "value"})
    .sort_values(["counts", "value"], ascending=[False, True], ignore_index=True)
)
us_normalized_job_title_distribution["share"] = (
    us_normalized_job_title_distribution["counts"]
    / us_normalized_job_title_distribution["counts"].sum()
)
us_normalized_job_title_distribution["rank"] = range(
    1,
    len(us_normalized_job_title_distribution) + 1,
)
us_normalized_job_title_distribution = us_normalized_job_title_distribution[
    [
        "value",
        "counts",
        "share",
        "rank",
        "raw_title_variants",
    ]
]

# Keep all many-to-one mappings available for auditing potentially overaggressive rules.
flag_normalization_collision = (job_title_normalization_lookup["title_normalized"].notna()) & (
    job_title_normalization_lookup["title_normalized"].duplicated(keep=False)
)
job_title_normalization_collisions = job_title_normalization_lookup.loc[
    flag_normalization_collision
].sort_values(
    [
        "title_normalized",
        "employment_spell_count",
        "title_raw",
    ],
    ascending=[True, False, True],
    ignore_index=True,
)

n_distinct_raw_job_titles = len(job_title_normalization_lookup)
n_blank_job_title_values = job_title_normalization_lookup["title_normalized"].isna().sum()
n_distinct_nonblank_job_titles = n_distinct_raw_job_titles - n_blank_job_title_values
n_distinct_normalized_job_titles = len(us_normalized_job_title_distribution)
n_merged_job_title_values = n_distinct_nonblank_job_titles - n_distinct_normalized_job_titles
n_normalized_titles_with_multiple_variants = (
    us_normalized_job_title_distribution["raw_title_variants"].gt(1).sum()
)
normalization_reduction_pct = (
    n_merged_job_title_values / n_distinct_nonblank_job_titles * 100
    if n_distinct_nonblank_job_titles
    else 0.0
)
n_missing_normalized_job_titles = inventor_spells_us["title_normalized"].isna().sum()

main.report_status(
    (
        "Self-reported job-title normalization within US:\n"
        f"  - Distinct nonmissing raw titles: {n_distinct_raw_job_titles:,}\n"
        f"  - Raw title values normalized to missing: {n_blank_job_title_values:,}\n"
        f"  - Distinct normalized titles: {n_distinct_normalized_job_titles:,}\n"
        f"  - Raw title values merged by the rules: {n_merged_job_title_values:,}\n"
        "  - Normalized titles with multiple raw variants: "
        f"{n_normalized_titles_with_multiple_variants:,}\n"
        f"  - Reduction among distinct nonblank titles: {normalization_reduction_pct:0.3f}%\n"
        f"  - Spells missing a normalized title: {n_missing_normalized_job_titles:,}"
    ),
    level="info",
)

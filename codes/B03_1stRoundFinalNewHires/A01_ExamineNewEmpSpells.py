"""
Task:
    Restrict employment spells to US, medicine industry (NAICS = 3254), non-internship spell.

Run:
    conda run --live-stream --name Talent --cwd "E:/Dropbox/E_Projects/TalentDiscovery" python -m codes.B03_1stRoundFinalNewHires.A01_ExamineNewEmpSpells

"""

from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from codes import main


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Global settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_DIR = main.DIR_TEMPDATA / "B03_1stRoundFinalNewHires" / "NewEmpSpells_TwoOccGroups_AllInd"
INPUT_FILES = tuple(sorted(INPUT_DIR.glob("*.parquet")))
OUTPUT_FILE = main.DIR_TEMPDATA / "B03_1stRoundFinalNewHires" / "US_NAICS3254_NonIntern.parquet"

COLUMNS = [
    "user_id",
    "position_number",
    "rcid",
    "startdate",
    "country",
    "state",
    "title_raw",
    "title_translated",
    "seniority",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Do sample restrictions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Restrict to US, medicine spells
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


candidate_spells = (
    pd
    .concat(
        [pd.read_parquet(input_file, columns=COLUMNS) for input_file in INPUT_FILES],
        ignore_index=True,
    )
    .sort_values(["user_id", "position_number"])
    .reset_index(drop=True)
)

us_medicine_spells = (
    candidate_spells
    .loc[
        (candidate_spells["naics_code"].str.startswith("3254"))
        & (candidate_spells["country"] == "United States")
    ]
    .sort_values(["user_id", "position_number"])
    .reset_index(drop=True)
)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Exclude internship spells
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


INTERNSHIP_TITLE_REGEX = (
    r"(?i)(?:\bintern(ship)?s?\b|\bsummer[\s-]+interns?\b|"
    r"\bsummer[\s-]+students?\b|\bstudent[\s-]+interns?\b|"
    r"\bworking[\s-]+students?\b|\bplacement[\s-]+students?\b|"
    r"\bwork[\s-]+placements?\b|\bindustrial[\s-]+placements?\b|"
    r"\bco[\s-]?ops?\b|\bco[\s-]?operative students?\b)"
)

flag_internship = (
    us_medicine_spells[["title_raw", "title_translated"]]
    .apply(
        lambda title: title.str.contains(
            INTERNSHIP_TITLE_REGEX,
            na=False,
        )
    )
    .any(axis="columns")
)

us_medicine_spells = us_medicine_spells.loc[~flag_internship].reset_index(drop=True)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-3. Normalize job titles
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
        job_titles
        .astype("string")
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


us_medicine_spells["job_title_normalized"] = normalize_job_titles(us_medicine_spells["title_raw"])
us_medicine_spells.to_parquet(OUTPUT_FILE, index=False)
main.report_status(f"Dataset saved: {main.relative_path(OUTPUT_FILE)}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Examine normalized job titles
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


onet_occupations = (
    us_medicine_spells[["onet_code", "onet_title"]]
    .drop_duplicates()
    .sort_values(["onet_code", "onet_title"])
    .reset_index(drop=True)
)
print(onet_occupations.to_string(index=False))

TARGET_ONET_CODE = "19-1011.00"

occupation_sample = us_medicine_spells.loc[us_medicine_spells["onet_code"].eq(TARGET_ONET_CODE)]
occupation_title = occupation_sample["onet_title"].dropna().mode().iat[0]

job_title_distribution = (
    occupation_sample["job_title_normalized"]
    .dropna()
    .value_counts()
    .rename_axis("job_title_normalized")
    .rename("count")
    .to_frame()
    .assign(
        share=lambda data: data["count"] / data["count"].sum(),
        percent=lambda data: 100 * data["share"],
    )
)

print(f"\nONET occupation: {TARGET_ONET_CODE} — {occupation_title}")
print(f"Number of observations with a nonmissing title: {job_title_distribution['count'].sum():,}")
print(f"Number of distinct normalized titles: {len(job_title_distribution):,}")
# print(job_title_distribution.to_string())

title_frequencies = job_title_distribution["count"].to_dict()

word_cloud = WordCloud(
    width=1600,
    height=900,
    background_color="white",
    colormap="gnuplot",
    max_words=100,
    prefer_horizontal=1.0,
    random_state=123,
).generate_from_frequencies(title_frequencies)

fig, ax = plt.subplots(figsize=(16, 9))
ax.imshow(word_cloud, interpolation="bilinear")
ax.set_title(
    f"{occupation_title}\nONET {TARGET_ONET_CODE}",
    fontsize=16,
)
ax.axis("off")
fig.tight_layout()
plt.show()

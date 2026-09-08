"""
Task:
    Construct the US pharmaceutical-manufacturing and biotechnology-R&D focal-hire baseline.

Inputs:
(a) data/b_temp_data/B03_1stRoundFinalNewHires/NewEmpSpells_TwoOccGroups_AllInd/*.parquet
    <== Constructed by "A_NewEmpSpells_TwoOccGroups_AllInd.ipynb" on Fabric.

Outputs:
(a) data/c_final_data/B03_1stRoundFinalNewHires/FocalNewHires_Med_US.parquet
(b) data/b_temp_data/B03_1stRoundFinalNewHires/US_BioPharm_SpellAudit.parquet
(c) outputs/B03_1stRoundFinalNewHires/B_SampleFlow.parquet
(d) outputs/B03_1stRoundFinalNewHires/B_TitleDecisions.parquet
(e) outputs/B03_1stRoundFinalNewHires/B_RunMetadata.json

Descriptions of outputs:
(1) Output (a) contains one selected spell per user-company pair, including review flags.
(2) Output (b) contains every US spell in the two industry groups, including excluded spells,
    title-rule flags, and selection status. It permits reconstruction of every restriction.
(3) Output (c) reports sequential sample sizes. Distinct counts before the industry restriction
    are unavailable by design: those stages count rows without loading all identifiers.
(4) Output (d) reports title decisions by normalized raw/translated title and delivered O*NET
    code, with spell counts and selected-pair counts. All titles are included, not only common ones.
(5) Output (e) records settings, code hash, input-part metadata, output counts, and limitations.

Run:
    conda run -s -n Talent python -m codes.B03_1stRoundFinalNewHires.B_FocalNewHires_Med_US

Notes:
(1) Keep US positions with NAICS beginning 3254 or equal to 541714. Retain the upstream
    2021-2023 start window and O*NET major groups 17/19 without further O*NET exclusions.
(2) Normalize both title fields before screening internships, co-ops, and explicit internships.
    Postdoctoral roles and research fellowships are not internships. Other trainee titles are
    flagged for review rather than automatically removed.
(3) Exclude explicit data-science, clinical-trial operations, routine quality/compliance,
    nonresearch operational, commercial, and administrative roles. Clinical scientists,
    analytical chemists, and research/process-development roles are retained by default.
(4) Exclude management/advisory titles without an explicit researcher role. Mixed titles such
    as "associate director clinical scientist" remain with a review flag. Senior/principal
    scientific grades are not exclusions. No numeric seniority threshold is imposed.
(5) Use conservative title exclusions, not an affirmative-title whitelist. Unresolved titles
    remain flagged, except missing or punctuation-only titles. An optional stricter indicator
    identifies retained titles with scientific evidence and no title-review flags.
(6) Choose the first qualifying spell within the window at each (user_id, rcid), then the
    lowest seniority, then the lowest position_number. Missing seniority sorts last. Reject
    unresolved ties rather than relying on input order. Internal transfers remain eligible.
(7) This window-based selection does not establish first-ever company or scientific employment.
    Full histories will be retrieved later on Fabric. Source dates may have month precision.
(8) Patents, inventor matches, posting availability, tenure, and subsequent outcomes never
    determine eligibility. Missing state and inconsistent end dates are diagnostic flags only.
(9) Existing outputs from this script are replaced when OVERWRITE_OUTPUTS is True. Each Parquet
    output is staged and checked before replacement; the run metadata is written last.
(10) Title rules are research assumptions, not validated occupational labels or inventor status.


Wang Wenzhi
Time: 2026-09-08
"""

import hashlib
import html
import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from codes import main


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths, settings, and reporting helpers
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_DIR = main.DIR_TEMPDATA / "B03_1stRoundFinalNewHires" / "NewEmpSpells_TwoOccGroups_AllInd"
OUTPUT_DIR = main.DIR_OUTPUTS / "B03_1stRoundFinalNewHires"
OUTPUT_BASELINE = main.DIR_FINALDATA / "B03_1stRoundFinalNewHires" / "FocalNewHires_Med_US.parquet"
OUTPUT_AUDIT = main.DIR_TEMPDATA / "B03_1stRoundFinalNewHires" / "US_BioPharm_SpellAudit.parquet"
OUTPUT_FLOW = OUTPUT_DIR / "B_SampleFlow.parquet"
OUTPUT_TITLES = OUTPUT_DIR / "B_TitleDecisions.parquet"
OUTPUT_METADATA = OUTPUT_DIR / "B_RunMetadata.json"
OUTPUT_FILES = (OUTPUT_BASELINE, OUTPUT_AUDIT, OUTPUT_FLOW, OUTPUT_TITLES, OUTPUT_METADATA)

OVERWRITE_OUTPUTS = True
RULE_VERSION = "2026-09-08.1"
COUNTRY = "United States"
NAICS_PREFIX = "3254"
BIOTECH_NAICS = "541714"
COHORT_START = pd.Timestamp("2021-01-01")
COHORT_END = pd.Timestamp("2024-01-01")
ONET_MAJOR_GROUPS = ("17", "19")
PAIR_COLUMNS = ["user_id", "rcid"]
MISSING_TEXT = ("", "empty", "null", "none", "nan", "na", "n/a")
SOURCE_COLUMNS = [
    "user_id",
    "rcid",
    "position_id",
    "position_number",
    "company_name",
    "company_raw",
    "company_cleaned",
    "startdate",
    "enddate",
    "start_date",
    "country",
    "state",
    "city",
    "location_raw",
    "title_raw",
    "title_translated",
    "seniority",
    "onet_code",
    "onet_title",
    "naics_code",
    "naics_description",
    "rics_k400",
]


def sample_counts(spells: pd.DataFrame) -> dict[str, int]:
    """
    Count spells, users, user-company pairs, and companies without mutating the input.
    """
    return {
        "spell_count": len(spells),
        "user_count": spells["user_id"].nunique(),
        "user_company_count": len(spells[PAIR_COLUMNS].drop_duplicates()),
        "company_count": spells["rcid"].nunique(),
    }


def report_sample(
    stage: str,
    counts: dict[str, int],
    previous_count: int | None,
) -> dict[str, object]:
    """
    Report a sequential restriction through the project helper and return its audit record.

    Parameters
    ----------
    stage : str
        Human-readable restriction label.
    counts : dict[str, int]
        Spell count and, when available, distinct identifier counts.
    previous_count : int or None
        Number of spells immediately before this restriction; None for the input universe.

    Returns
    -------
    dict[str, object]
        One row of the sample-flow report. Missing distinct counts mean not computed, not zero.
    """
    n_spells = counts["spell_count"]
    n_removed = None if previous_count is None else previous_count - n_spells
    if n_removed is not None and n_removed < 0:
        raise ValueError(f"Sample size increased during {stage}.")
    removed_pct = None if previous_count in (None, 0) else 100 * n_removed / previous_count
    message = f"{stage}: {n_spells:,} spells"
    if "user_company_count" in counts:
        message += (
            f"; {counts['user_count']:,} users; {counts['user_company_count']:,} user-company"
            f" pairs; {counts['company_count']:,} companies"
        )
    if n_removed is not None:
        message += f"; removed {n_removed:,}"
        if removed_pct is not None:
            message += f" ({removed_pct:.2f}% of previous stage)"
    main.report_status(message)
    return {"stage": stage, **counts, "spells_removed": n_removed, "removed_pct": removed_pct}


run = main.start_run(__file__)
pa.set_cpu_count(min(8, pa.cpu_count()))
input_files = tuple(sorted(INPUT_DIR.glob("*.parquet")))
if not input_files:
    raise FileNotFoundError(f"No Parquet parts found in {main.relative_path(INPUT_DIR)}.")
if not OVERWRITE_OUTPUTS and any(path.exists() for path in OUTPUT_FILES):
    raise FileExistsError("An output already exists. Review OVERWRITE_OUTPUTS before rerunning.")
main.report_status(f"Reading {len(input_files):,} parts; overwrite outputs: {OVERWRITE_OUTPUTS}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Restrict country and industry before materializing the candidate spells
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


source_dataset = ds.dataset([str(path) for path in input_files], format="parquet")
missing_columns = sorted(set(SOURCE_COLUMNS) - set(source_dataset.schema.names))
if missing_columns:
    raise ValueError(f"The intermediate extract is missing columns: {missing_columns}.")

# The upstream schema supplies trimmed country values and six-digit text NAICS codes.
# Row-count scans read restriction columns only; full records are read after both restrictions.
country_filter = ds.field("country") == COUNTRY
industry_filter = pc.starts_with(ds.field("naics_code"), pattern=NAICS_PREFIX) | (
    ds.field("naics_code") == BIOTECH_NAICS
)
n_input = source_dataset.count_rows()
n_us = source_dataset.count_rows(filter=country_filter)
sample_flow_records = [
    report_sample("Input: upstream 2021-2023, O*NET 17/19", {"spell_count": n_input}, None),
    report_sample("Keep United States", {"spell_count": n_us}, n_input),
]
candidate_spells = source_dataset.to_table(
    columns=SOURCE_COLUMNS,
    filter=country_filter & industry_filter,
).to_pandas()
sample_flow_records.append(
    report_sample("Keep NAICS 3254 or 541714", sample_counts(candidate_spells), n_us)
)

# Fail on a wrong or incompatible extract rather than silently changing the upstream universe.
required_keys = PAIR_COLUMNS + ["position_id", "position_number"]
if candidate_spells[required_keys].isna().any().any():
    raise ValueError("Missing source identifiers in the industry-restricted sample.")
if candidate_spells["position_id"].duplicated().any():
    raise ValueError("Repeated position_id in the extract; resolve source duplicates first.")
candidate_spells["start_date"] = pd.to_datetime(candidate_spells["start_date"], errors="raise")
flag_valid_window = candidate_spells["start_date"].ge(COHORT_START) & candidate_spells[
    "start_date"
].lt(COHORT_END)
flag_valid_onet = candidate_spells["onet_code"].str[:2].isin(ONET_MAJOR_GROUPS)
if not flag_valid_window.all() or not flag_valid_onet.all():
    raise ValueError("The local extract violates the upstream cohort or broad O*NET restriction.")

candidate_spells["industry_group"] = candidate_spells["naics_code"].map(
    lambda code: "pharmaceutical_manufacturing" if code.startswith(NAICS_PREFIX) else "biotech_rd"
)
candidate_spells["start_year"] = candidate_spells["start_date"].dt.year.astype("int16")
candidate_spells["start_month"] = candidate_spells["start_date"].dt.to_period("M").dt.to_timestamp()
candidate_spells["state_raw"] = candidate_spells["state"]
state_cleaned = candidate_spells["state"].astype("string").str.strip()
candidate_spells["state"] = state_cleaned.mask(state_cleaned.str.casefold().isin(MISSING_TEXT))
candidate_spells["is_state_missing"] = candidate_spells["state"].isna()
candidate_spells["end_date"] = pd.to_datetime(candidate_spells["enddate"], errors="coerce")
candidate_spells["is_end_before_start"] = candidate_spells["end_date"].lt(
    candidate_spells["start_date"]
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Normalize both title fields and identify internship positions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


TITLE_ABBREVIATIONS = {
    "snr": "senior",
    "sr": "senior",
    "jr": "junior",
    "asst": "assistant",
    "assoc": "associate",
    "coord": "coordinator",
    "dir": "director",
    "engr": "engineer",
    "engg": "engineering",
    "exec": "executive",
    "mngr": "manager",
    "mgr": "manager",
    "supv": "supervisor",
    "dept": "department",
    "intl": "international",
    "mfg": "manufacturing",
    "mktg": "marketing",
    "mgmt": "management",
    "ops": "operations",
    "svp": "senior vice president",
    "evp": "executive vice president",
    "avp": "assistant vice president",
    "vp": "vice president",
    "ceo": "chief executive officer",
    "cfo": "chief financial officer",
    "coo": "chief operating officer",
    "cto": "chief technology officer",
    "cso": "chief scientific officer",
    "hr": "human resources",
    "qa": "quality assurance",
    "qc": "quality control",
    "cofounder": "co founder",
}
INTERNSHIP_PATTERN = (
    r"\b(?:interns?|internships?|externs?|externships?|co ops?|coops?|"
    r"co operative students?|cooperative students?|summer students?|working students?|"
    r"placement students?|work placements?|industrial placements?)\b"
)


def normalize_job_titles(job_titles: pd.Series) -> pd.Series:
    """
    Normalize title form while preserving meaningful words and missing values.

    Parameters
    ----------
    job_titles : pd.Series
        Raw or translated self-reported titles, potentially containing missing values.

    Returns
    -------
    pd.Series
        Normalized titles with the original index and pandas string dtype.

    Notes
    -----
    (1) Work on distinct values to avoid repeating text transformations for common titles.
    (2) Keep word order, scientific qualifiers, and seniority. No stemming or fuzzy matching.
    (3) Expand only explicit abbreviations. Ambiguous RA, AD, and PD are not expanded.
    """
    original = job_titles.astype("string")
    unique_titles = pd.Series(original.dropna().unique(), dtype="string")
    # Python's Unicode regex semantics preserve non-English letters in the raw title.
    normalized = unique_titles.map(html.unescape).astype(pd.StringDtype(storage="python"))
    normalized = normalized.str.normalize("NFKC").str.casefold().str.strip()
    normalized = normalized.mask(normalized.isin(MISSING_TEXT))
    normalized = normalized.str.replace("_", " ", regex=False)
    normalized = normalized.str.replace(
        r"\br\s*(?:[&/+.\-]|and|\s)\s*d\b", "research and development", regex=True
    )
    for dotted, expanded in (
        (r"\bq\.\s*a\.", "quality assurance"),
        (r"\bq\.\s*c\.", "quality control"),
        (r"\bc\.\s*r\.\s*a\.", "cra"),
        (r"\bv\.\s*p\.", "vice president"),
    ):
        normalized = normalized.str.replace(dotted, expanded, regex=True)
    normalized = normalized.str.replace("&", " and ", regex=False)
    normalized = normalized.str.replace(r"['\u2018\u2019\u02bc]", "", regex=True)
    normalized = normalized.str.replace(r"[^\w\s]", " ", regex=True)
    normalized = normalized.str.replace(r"\s+", " ", regex=True).str.strip()
    for abbreviation, replacement in TITLE_ABBREVIATIONS.items():
        normalized = normalized.str.replace(rf"\b{abbreviation}\b", replacement, regex=True)
    normalized = normalized.mask(normalized.eq(""))
    lookup = pd.Series(normalized.array, index=unique_titles)
    return original.map(lookup).astype("string")


candidate_spells["title_raw_normalized"] = normalize_job_titles(candidate_spells["title_raw"])
candidate_spells["title_translated_normalized"] = normalize_job_titles(
    candidate_spells["title_translated"]
)
title_raw = candidate_spells["title_raw_normalized"]
title_translated = candidate_spells["title_translated_normalized"]
candidate_spells["job_title_normalized"] = title_raw.fillna(title_translated)
candidate_spells["title_source"] = "raw"
candidate_spells.loc[title_raw.isna() & title_translated.notna(), "title_source"] = "translated"
candidate_spells.loc[title_raw.isna() & title_translated.isna(), "title_source"] = "missing"
candidate_spells["is_title_translation_different"] = (
    title_raw.notna() & title_translated.notna() & title_raw.ne(title_translated).fillna(False)
)
candidate_spells["is_internship"] = title_raw.str.contains(
    INTERNSHIP_PATTERN, na=False
) | title_translated.str.contains(INTERNSHIP_PATTERN, na=False)
flag_remaining = ~candidate_spells["is_internship"]
sample_flow_records.append(
    report_sample(
        "Exclude internships, co-ops, and externships",
        sample_counts(candidate_spells.loc[flag_remaining, PAIR_COLUMNS]),
        len(candidate_spells),
    )
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Apply transparent functional exclusions and retain review flags
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# These are title-function rules. Delivered O*NET labels never override a title decision.
# The order below determines the first exclusion reason; all overlapping flags are also saved.
EXCLUSION_PATTERNS = {
    "data_science": r"\bdata scien\w*\b",
    "clinical_operations": (
        r"\bclinical research (?:project |study )?(?:associates?|assistants?|coordinators?|"
        r"monitors?|managers?)\b|"
        r"\bclinical (?:trials?|study) (?:associates?|assistants?|coordinators?|managers?|"
        r"specialists?|administrators?|monitors?|leads?|leaders?|management|operations)\b|"
        r"\bclinical (?:operations|project|supply)\b|"
        r"\bclinical data (?:management|managers?|coordinators?|specialists?|analysts?)\b|"
        r"\b(?:trial|study|site) (?:monitors?|coordinators?|managers?|management)\b|"
        r"\btrials? (?:leads?|leaders?|operations)\b|"
        r"\b(?:country approval|study start up|study startup|site activation|"
        r"site start up|patient recruitment|trial master file)\b|"
        r"\b(?:cra|cta|crc)\b"
    ),
    "quality_compliance": (
        r"\bquality (?:assurance|control|systems?|compliance|audit\w*|inspect\w*|engineer\w*|"
        r"specialists?|associates?|technicians?|analysts?|operations|laboratory|scientists?|"
        r"chemists?|leads?|leaders?)\b|\b(?:product|supplier) quality\b|"
        r"\bcompliance (?:specialists?|officers?|analysts?|engineers?)\b|\bgmp auditors?\b"
    ),
    "regulatory_medical_affairs": (
        r"\bregulatory (?:affairs|operations|submissions?|specialists?|associates?|scientists?|"
        r"coordinators?|managers?|writers?|writing)\b|\bmedical affairs\b|"
        r"\bmedical (?:science|scientific) liaisons?\b|\bmsl\b|\bpharmacovigilance\b|"
        r"\bdrug safety\b|\b(?:medical|scientific) (?:writers?|writing|communications)\b"
    ),
    "operational_engineering": (
        r"\bcomputer systems? validation\b|\bcommissioning\b|\bqualification engineers?\b|"
        r"\b(?:equipment validation|facilities|maintenance|field service|technical support)\b|"
        r"\breliability engineers?\b|\b(?:manufacturing|production) "
        r"(?:associates?|technicians?|operators?|specialists?)\b|"
        r"\b(?:process|plant|chemical|bioprocess) operators?\b|"
        r"\b(?:supply chain|logistics|procurement|warehouse)\b"
    ),
    "commercial_administration": (
        r"\b(?:sales|marketing|business development|business analysts?|human resources|"
        r"talent acquisition|recruiters?|accountants?|finance|legal|customer service|"
        r"customer success)\b|\bpatent (?:attorneys?|agents?|counsel)\b|"
        r"\b(?:administrative|executive) assistants?\b|\bsecretar\w*\b"
    ),
    "safety_healthcare": (
        r"\b(?:environmental health|occupational health|health and safety|ehs|hse)\b|"
        r"\bsafety (?:engineers?|specialists?|officers?)\b|\bindustrial hygienists?\b|"
        r"\b(?:nurses?|nursing|phlebotom\w*|pharmacists?)\b|"
        r"\b(?:medical|clinical) (?:laboratory )?(?:technologists?|technicians?)\b"
    ),
    "it_business_analytics": (
        r"\bsoftware (?:development |test )?(?:engineers?|developers?|architects?)\b|"
        r"\b(?:data|cloud|network) (?:engineers?|developers?|architects?)\b|"
        r"\b(?:devops|cybersecurity|information technology|business intelligence)\b|"
        r"\bit (?:support|specialists?|analysts?|consultants?)\b|\bsystems? administrators?\b"
    ),
    "nonemployment": (
        r"\b(?:retired|retirement|board members?|board observers?|advisory board|"
        r"chairman|chairwoman|chairperson|unemployed|open to work|"
        r"student ambassador|campus ambassador|volunteer)\b"
    ),
}
RESEARCH_ROLE_PATTERN = (
    r"\b(?:scientists?|chemists?|biochemists?|biologists?|microbiologists?|immunologists?|"
    r"virologists?|geneticists?|pharmacologists?|toxicologists?|physicists?|bioengineers?|"
    r"engineers?|investigators?|"
    r"researchers?|bioinformaticians?|postdocs?|postdoctoral)\b|"
    r"\bresearch (?:associates?|assistants?|fellows?|fellowships?|technicians?|specialists?|"
    r"scholars?|affiliates?)\b"
)
MANAGEMENT_PATTERN = (
    r"\b(?:managers?|directors?|head|vice president|president|chief|founders?|owners?|"
    r"supervisors?|consultants?|advisors?|advisers?)\b|\b(?:team|group) leaders?\b"
)
RESEARCH_FUNCTION_PATTERNS = {
    "clinical_science": r"\bclinical (?:research )?scien\w*\b",
    "analytical_science": r"\b(?:analytical|bioanalytical)\b",
    "process_development": r"\b(?:process|bioprocess|cell line) development\b",
    "computational_biology": (
        r"\b(?:bioinformatics?|bioinformaticians?|cheminformatics?|chemoinformatics?)\b|"
        r"\bcomputational (?:biology|biologists?|chemistry|chemists?|genomics)\b"
    ),
    "discovery_development": (
        r"\b(?:drug discovery|medicinal chemistry|formulation|protein engineering|"
        r"assay development|molecular biology|cell biology|pharmacology|toxicology|"
        r"immunology|research and development)\b"
    ),
}
QUALITY_DEVELOPMENT_PATTERN = (
    r"\b(?:analytical|bioanalytical|assay|method|methods|process|formulation) development\b"
)
AMBIGUOUS_TECHNICAL_PATTERN = (
    r"\b(?:msat|cmc|validation|trainees?|apprentices?|students?|technicians?|technologists?)\b|"
    r"\b(?:process|manufacturing|production|automation|project) engineers?\b|"
    r"^(?:(?:senior|principal|staff|associate|lead) )?engineer(?: [ivx]+| \d+)?$"
)
EXCLUSION_ORDER = ["missing_title", *EXCLUSION_PATTERNS, "management_without_research_role"]


def classify_titles(normalized_titles: pd.Series) -> pd.DataFrame:
    """
    Classify normalized titles using functional evidence and explicit exceptions.

    Parameters
    ----------
    normalized_titles : pd.Series
        Normalized titles. Missing values are permitted.

    Returns
    -------
    pd.DataFrame
        Boolean exclusion/evidence flags indexed like the input; the input is not modified.

    Notes
    -----
    (1) Mixed analytical-development/QC work remains reviewable. QA, audit, and quality-system
        jobs receive no such exception. Method validation alone is not operational validation.
    (2) Computing roles explicitly situated in computational biology/bioinformatics are not
        excluded as generic IT. Explicit data-science wording remains an exclusion.
    (3) A scientific job grade is different from a managerial title. Mixed researcher/manager
        titles remain reviewable, while management without a researcher role is excluded.
    """
    flags = pd.DataFrame(index=normalized_titles.index)
    for reason, pattern in EXCLUSION_PATTERNS.items():
        flags[f"exclude_{reason}"] = normalized_titles.str.contains(pattern, na=False)
    flags["has_research_role"] = normalized_titles.str.contains(RESEARCH_ROLE_PATTERN, na=False)
    flags["has_management_title"] = normalized_titles.str.contains(MANAGEMENT_PATTERN, na=False)
    for function, pattern in RESEARCH_FUNCTION_PATTERNS.items():
        flags[f"is_{function}"] = normalized_titles.str.contains(pattern, na=False)
    flags["has_research_function"] = flags[
        [f"is_{function}" for function in RESEARCH_FUNCTION_PATTERNS]
    ].any(axis=1)
    flags["is_mixed_management_title"] = flags["has_management_title"] & flags["has_research_role"]
    flags["exclude_management_without_research_role"] = (
        flags["has_management_title"] & ~flags["has_research_role"]
    )
    flag_quality_development = normalized_titles.str.contains(QUALITY_DEVELOPMENT_PATTERN, na=False)
    flag_quality_assurance = normalized_titles.str.contains(
        r"\bquality (?:assurance|systems?|audit\w*|compliance)\b", na=False
    )
    flags["is_mixed_quality_development"] = (
        flags["exclude_quality_compliance"] & flag_quality_development & ~flag_quality_assurance
    )
    flags["exclude_quality_compliance"] &= ~flags["is_mixed_quality_development"]
    # Word order varies: "engineer II validation" is the same function as "validation engineer".
    # Analytical/assay/method validation remains eligible scientific work and is not equated
    # with equipment, computer-system, or plant qualification.
    flag_validation = normalized_titles.str.contains(r"\bvalidation\b", na=False)
    flag_scientific_validation = flag_validation & normalized_titles.str.contains(
        r"\b(?:analytical|bioanalytical|assay|method|methods)\b", na=False
    )
    flag_validation_support = normalized_titles.str.contains(
        r"\b(?:engineers?|specialists?|analysts?|associates?|technicians?|consultants?|"
        r"coordinators?|contractors?|leads?|leaders?)\b",
        na=False,
    )
    flags["exclude_operational_engineering"] |= (
        flag_validation & flag_validation_support & ~flag_scientific_validation
    )
    flags["exclude_it_business_analytics"] &= ~flags["is_computational_biology"]
    flags["exclude_safety_healthcare"] &= ~normalized_titles.str.contains(
        r"\b(?:physician|nurse) scientists?\b", na=False
    )
    flags["is_technical_title_ambiguous"] = normalized_titles.str.contains(
        AMBIGUOUS_TECHNICAL_PATTERN, na=False
    ) & ~(
        normalized_titles.str.contains(QUALITY_DEVELOPMENT_PATTERN, na=False)
        | flag_scientific_validation
    )
    return flags.astype(bool)


# Evaluate each unique title once. Apply each field's contextual exceptions before combining
# decisions; concatenating two languages could incorrectly create a cross-title exception.
unique_normalized_titles = pd.Series(
    pd.unique(pd.concat([title_raw, title_translated]).dropna()), dtype="string"
)
title_flags = classify_titles(unique_normalized_titles)
title_flags.index = unique_normalized_titles
flag_translation_conflict = pd.Series(False, index=candidate_spells.index)
for column in title_flags.columns:
    raw_flag = title_raw.map(title_flags[column]).fillna(False).astype(bool)
    translated_flag = title_translated.map(title_flags[column]).fillna(False).astype(bool)
    candidate_spells[column] = raw_flag | translated_flag
    if column.startswith("exclude_"):
        flag_translation_conflict |= raw_flag.ne(translated_flag)
candidate_spells["is_title_translation_conflict"] = (
    candidate_spells["is_title_translation_different"] & flag_translation_conflict
)
candidate_spells["exclude_missing_title"] = candidate_spells["job_title_normalized"].isna()
candidate_spells["is_title_uninformative"] = ~(
    candidate_spells["has_research_role"] | candidate_spells["has_research_function"]
)
REVIEW_COLUMNS = [
    "is_mixed_management_title",
    "is_mixed_quality_development",
    "is_technical_title_ambiguous",
    "is_title_translation_conflict",
    "is_title_uninformative",
]
candidate_spells["needs_title_review"] = candidate_spells[REVIEW_COLUMNS].any(axis=1)
candidate_spells["exclusion_reason"] = "retained"
candidate_spells.loc[candidate_spells["is_internship"], "exclusion_reason"] = "internship"
for reason in EXCLUSION_ORDER:
    previous_count = int(flag_remaining.sum())
    flag_exclude = candidate_spells[f"exclude_{reason}"]
    candidate_spells.loc[flag_remaining & flag_exclude, "exclusion_reason"] = reason
    flag_remaining &= ~flag_exclude
    sample_flow_records.append(
        report_sample(
            f"Exclude {reason.replace('_', ' ')}",
            sample_counts(candidate_spells.loc[flag_remaining, PAIR_COLUMNS]),
            previous_count,
        )
    )
candidate_spells["is_occupation_eligible"] = flag_remaining
candidate_spells["is_strict_research_title"] = (
    flag_remaining & ~candidate_spells["needs_title_review"]
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Select the first qualifying spell per user-company pair
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def select_first_spells(eligible_spells: pd.DataFrame) -> pd.DataFrame:
    """
    Select the earliest date, lowest seniority, and lowest position number within each pair.

    Parameters
    ----------
    eligible_spells : pd.DataFrame
        Eligible source spells with identifiers, parsed start_date, seniority, and position_number.

    Returns
    -------
    pd.DataFrame
        One intact source row per user-company pair; the supplied frame is not modified.

    Raises
    ------
    ValueError
        Raised for missing keys/dates, invalid numeric order fields, or unresolved winning ties.

    Notes
    -----
    (1) Missing seniority ranks after observed levels. Position number must be nonmissing.
    (2) The full selection key is checked only among winning rows. Losing ties do not affect
        selection. No additional tie-breaker or synthesized combination of rows is used.
    """
    ordered = eligible_spells.copy()
    required = PAIR_COLUMNS + ["position_id", "start_date", "position_number"]
    if ordered[required].isna().any().any():
        raise ValueError("Missing identifiers, start date, or position number in eligible spells.")
    for column in ["seniority", "position_number"]:
        numeric = pd.to_numeric(ordered[column], errors="raise")
        if (numeric.dropna() % 1).ne(0).any():
            raise ValueError(f"Nonintegral values found in {column}.")
        ordered[column] = numeric.astype("Int64")
    if not ordered["seniority"].dropna().between(1, 7).all():
        raise ValueError("Observed seniority must be in the delivered 1-7 scale.")

    order_columns = ["start_date", "seniority", "position_number"]
    for column in order_columns:
        minimum = ordered.groupby(PAIR_COLUMNS, sort=False)[column].transform("min")
        # An all-missing seniority group proceeds to position number without losing the pair.
        flag_minimum = ordered[column].eq(minimum).fillna(False) | (
            ordered[column].isna() & minimum.isna()
        )
        ordered = ordered.loc[flag_minimum].copy()
    if ordered.duplicated(PAIR_COLUMNS).any():
        raise ValueError(
            "Unresolved first-date/seniority/position-number tie; inspect source rows."
        )
    return ordered.sort_values(PAIR_COLUMNS).reset_index(drop=True)


eligible_spells = candidate_spells.loc[flag_remaining].copy()
focal_new_hires = select_first_spells(eligible_spells)
sample_flow_records.append(
    report_sample(
        "Select first qualifying spell per user-company",
        sample_counts(focal_new_hires),
        len(eligible_spells),
    )
)
candidate_spells["is_selected"] = candidate_spells["position_id"].isin(
    focal_new_hires["position_id"]
)
candidate_spells["selection_status"] = "excluded"
candidate_spells.loc[flag_remaining, "selection_status"] = "eligible_not_selected"
candidate_spells.loc[candidate_spells["is_selected"], "selection_status"] = "selected"
focal_new_hires["selection_scope"] = "first_qualifying_spell_in_2021_2023"
focal_new_hires["history_validation_status"] = "not_yet_checked"
focal_new_hires["title_rule_version"] = RULE_VERSION


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 6. Validate, save the baseline and audit outputs, and report completion
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def replace_staged_file(temporary_file: Path, output_file: Path) -> None:
    """
    Atomically replace an output, retrying Windows sharing locks for up to 12 seconds.

    Parameters
    ----------
    temporary_file : Path
        Fully written and validated staged file.
    output_file : Path
        Intended destination. Its existing contents remain intact until replacement succeeds.

    Raises
    ------
    PermissionError
        Raised immediately for permission errors other than sharing locks, or after six attempts.
    """
    for attempt in range(6):
        try:
            temporary_file.replace(output_file)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (32, 33) or attempt == 5:
                raise
            time.sleep(min(0.5 * 2**attempt, 4.0))


def save_parquet(frame: pd.DataFrame, output_file: Path) -> None:
    """
    Stage a Parquet output, verify its round trip, and replace the destination atomically.

    Parameters
    ----------
    frame : pd.DataFrame
        Intended output, with a deterministic row order and no meaningful DataFrame index.
    output_file : Path
        Destination controlled by the script's overwrite setting.
    """
    main.ensure_parent(output_file)
    temporary_file = output_file.with_suffix(".parquet.incomplete")
    frame.to_parquet(temporary_file, index=False)
    with temporary_file.open("rb") as source:
        restored = pd.read_parquet(source)
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), restored, check_dtype=False)
    replace_staged_file(temporary_file, output_file)
    main.report_status(f"Saved {len(frame):,} rows: {main.relative_path(output_file)}.")


exclusion_columns = [f"exclude_{reason}" for reason in EXCLUSION_ORDER]
if focal_new_hires[exclusion_columns + ["is_internship"]].any().any():
    raise AssertionError("An excluded spell reached the final baseline.")
if len(focal_new_hires) != len(eligible_spells[PAIR_COLUMNS].drop_duplicates()):
    raise AssertionError(
        "Selection did not preserve exactly one row per eligible user-company pair."
    )
if not focal_new_hires["country"].eq(COUNTRY).all():
    raise AssertionError("A non-US position reached the final baseline.")
if not (
    focal_new_hires["naics_code"].str.startswith(NAICS_PREFIX)
    | focal_new_hires["naics_code"].eq(BIOTECH_NAICS)
).all():
    raise AssertionError("An out-of-scope industry reached the final baseline.")

sample_flow = pd.DataFrame(sample_flow_records)
for column in [
    "spell_count",
    "user_count",
    "user_company_count",
    "company_count",
    "spells_removed",
]:
    sample_flow[column] = sample_flow[column].astype("Int64")
title_group_columns = [
    "title_raw_normalized",
    "title_translated_normalized",
    "onet_code",
    "onet_title",
    "industry_group",
    "exclusion_reason",
    "needs_title_review",
    *REVIEW_COLUMNS,
]
title_decisions = (
    candidate_spells
    .groupby(title_group_columns, dropna=False, as_index=False)
    .agg(
        spell_count=("position_id", "size"),
        selected_pair_count=("is_selected", "sum"),
        user_count=("user_id", "nunique"),
        company_count=("rcid", "nunique"),
    )
    .sort_values(
        ["spell_count", *title_group_columns],
        ascending=[False, *([True] * len(title_group_columns))],
        na_position="last",
    )
    .reset_index(drop=True)
)
candidate_spells = candidate_spells.sort_values(
    PAIR_COLUMNS + ["start_date", "position_number", "position_id"]
).reset_index(drop=True)

for label, column in (
    ("Selected pairs needing title review", "needs_title_review"),
    ("Selected pairs with missing state", "is_state_missing"),
    ("Selected pairs with end before start", "is_end_before_start"),
):
    main.report_status(f"{label}: {int(focal_new_hires[column].sum()):,}.")
main.report_status(
    "Selected pairs by industry: " + str(focal_new_hires["industry_group"].value_counts().to_dict())
)
main.report_status(
    "Full-history and inventor-linkage validation remain pending; title review flags are retained.",
    level="warning",
)

save_parquet(candidate_spells, OUTPUT_AUDIT)
save_parquet(sample_flow, OUTPUT_FLOW)
save_parquet(title_decisions, OUTPUT_TITLES)
save_parquet(focal_new_hires, OUTPUT_BASELINE)

input_manifest = [
    {"name": path.name, "bytes": path.stat().st_size, "modified_ns": path.stat().st_mtime_ns}
    for path in input_files
]
run_metadata = {
    "rule_version": RULE_VERSION,
    "started_utc": run.started_at.isoformat(),
    "script": main.relative_path(__file__),
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_directory": main.relative_path(INPUT_DIR),
    "input_parts": input_manifest,
    "country": COUNTRY,
    "naics_prefix": NAICS_PREFIX,
    "additional_naics": BIOTECH_NAICS,
    "cohort_start": COHORT_START.isoformat(),
    "cohort_end_exclusive": COHORT_END.isoformat(),
    "onet_major_groups": ONET_MAJOR_GROUPS,
    "exclusion_order": EXCLUSION_ORDER,
    "selected_counts": {key: int(value) for key, value in sample_counts(focal_new_hires).items()},
    "selected_review_count": int(focal_new_hires["needs_title_review"].sum()),
    "validation_scope": "Local extract only; no Fabric history or inventor-linkage validation.",
    "outputs": [main.relative_path(path) for path in OUTPUT_FILES],
    "pandas_version": pd.__version__,
    "pyarrow_version": pa.__version__,
}
temporary_metadata = OUTPUT_METADATA.with_suffix(".json.incomplete")
temporary_metadata.write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
replace_staged_file(temporary_metadata, OUTPUT_METADATA)
main.finish_run(
    run,
    f"Constructed {len(focal_new_hires):,} focal user-company observations.",
    f"Baseline: {main.relative_path(OUTPUT_BASELINE)}.",
)

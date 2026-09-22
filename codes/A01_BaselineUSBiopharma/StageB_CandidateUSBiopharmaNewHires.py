"""
Task:
    Select the full candidate US biopharma new-hire sample using pandas.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageA_BroadestCandidateEmpSpells/
    <== Download the Parquet directory produced by StageA_BroadestCandidateEmpSpells.ipynb.

Outputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_CandidateNewHires.parquet
(b) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
(c) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SampleFlow.parquet
(d) data/b_temp_data/A01_BaselineUSBiopharma/StageB_TitleAuditSummary.parquet
(e) data/b_temp_data/A01_BaselineUSBiopharma/StageB_RunManifest.json

Descriptions of outputs:
(1) Output (a) has one intact qualifying source spell per user-company in 2021-2023.
(2) Output (b) has one row per US biopharma source spell, including title and selection flags.
(3) Output (c) has one row per sequential sample restriction.
(4) Output (d) reports title exclusions and review flags.
(5) Output (e) identifies the source inventory, code, environment, and output row counts.

Run:
    conda run -s -n Talent python -m codes.A01_BaselineUSBiopharma.StageB_CandidateUSBiopharmaNewHires
    Add --overwrite only after reviewing any existing Stage B outputs.

Notes:
(1) Keep US spells with NAICS beginning 3254 or equal to 541714, then apply reference titles.
(2) Internal transfers and nonexcluded ambiguous titles remain eligible; no seniority cutoff.
(3) Select earliest start, lowest observed seniority, then lowest position number. Reject ties.
(4) Source fields are preserved; standardized identifiers are added in id_* fields.
(5) Existing outputs are replaced only when the --overwrite command-line option is supplied.


Wang Wenzhi
Time: 2026-09-08
"""

import argparse
import hashlib
import html
import json
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths and a common helper function to clean string columns
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "b_temp_data" / "A01_BaselineUSBiopharma"
INPUT_DIR = OUTPUT_DIR / "StageA_BroadestCandidateEmpSpells"
OUTPUT_HIRES = OUTPUT_DIR / "StageB_CandidateNewHires.parquet"
OUTPUT_AUDIT = OUTPUT_DIR / "StageB_SpellAudit.parquet"
OUTPUT_FLOW = OUTPUT_DIR / "StageB_SampleFlow.parquet"
OUTPUT_TITLE_SUMMARY = OUTPUT_DIR / "StageB_TitleAuditSummary.parquet"
OUTPUT_MANIFEST = OUTPUT_DIR / "StageB_RunManifest.json"
OUTPUT_FILES = [OUTPUT_HIRES, OUTPUT_AUDIT, OUTPUT_FLOW, OUTPUT_TITLE_SUMMARY, OUTPUT_MANIFEST]
MISSING_TEXT = ("", "empty", "null", "none", "nan", "na", "n/a")
REQUIRED_INPUT_COLUMNS = [
    "country",
    "naics_code",
    "onet_code",
    "user_id",
    "rcid",
    "position_id",
    "position_number",
    "seniority",
    "startdate",
    "parsed_start_date",
    "title_raw",
    "title_translated",
]


def clean_text(
    values: pd.Series,
    text_treated_as_na: tuple[str] = MISSING_TEXT,
) -> pd.Series:
    """
    Trim text and replace explicit missing tokens with pandas missing values.
    """
    cleaned = values.astype("string").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin(text_treated_as_na))


def require_columns(frame: pd.DataFrame, columns: list[str], *, source: Path | None = None) -> None:
    """Require a unique set of input columns and report the source on failure."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing or not frame.columns.is_unique:
        location = f" in {source}" if source is not None else ""
        raise ValueError(
            f"Invalid schema{location}: missing columns {missing}; column names must be unique."
        )


def standardize_id(values: pd.Series) -> pd.Series:
    """Canonicalize signed integral identifiers without passing through floats."""
    if pd.api.types.is_float_dtype(values) and values.dropna().abs().gt(2**53 - 1).any():
        raise ValueError("Unsafe floating identifier; retrieve an integer or string source.")
    cleaned = clean_text(values)
    valid = cleaned.str.fullmatch(r"-?\d+(?:\.0+)?", na=True)
    if not valid.all():
        bad_values = cleaned.loc[~valid].drop_duplicates().head(5).tolist()
        raise ValueError(f"Identifiers must be signed integers: {bad_values}")
    cleaned = cleaned.str.replace(r"\.0+$", "", regex=True)
    is_negative = cleaned.str.startswith("-").fillna(False)
    digits = cleaned.str.removeprefix("-").str.lstrip("0")
    digits = digits.mask(digits.eq("").fillna(False), "0")
    return digits.mask(is_negative & digits.ne("0").fillna(False), "-" + digits)


def identify_biopharma(naics_values: pd.Series) -> pd.Series:
    """Identify the stated NAICS scope while safely handling missing values."""
    naics = clean_text(naics_values).str.replace(r"\.0+$", "", regex=True)
    begins_3254 = naics.str.startswith("3254").fillna(False)
    is_biotech_r_and_d = naics.eq("541714").fillna(False)
    return (begins_3254 | is_biotech_r_and_d).astype(bool)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read Parquet parts and restrict country and industry with pandas
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def inspect_input_files(input_dir: Path) -> tuple[list[Path], dict[str, int | str]]:
    """Validate the multipart input schema and fingerprint its file inventory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Stage A input directory does not exist: {input_dir}")
    input_files = sorted(input_dir.rglob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No Parquet parts found under: {input_dir}")

    expected_schema = None
    inventory_hash = hashlib.sha256()
    total_rows = 0
    total_bytes = 0
    for input_file in input_files:
        parquet_file = pq.ParquetFile(input_file)
        schema = parquet_file.schema_arrow.remove_metadata()
        if len(schema.names) != len(set(schema.names)):
            raise ValueError(f"Duplicate Parquet field names in {input_file}.")
        missing = sorted(set(REQUIRED_INPUT_COLUMNS) - set(schema.names))
        if missing:
            raise ValueError(f"Missing required fields in {input_file}: {missing}")
        if expected_schema is None:
            expected_schema = schema
        elif not schema.equals(expected_schema):
            raise ValueError(f"Parquet schema differs from the first input part: {input_file}")
        relative_name = input_file.relative_to(input_dir).as_posix()
        file_size = input_file.stat().st_size
        file_rows = parquet_file.metadata.num_rows
        inventory_hash.update(f"{relative_name}\0{file_size}\0{file_rows}\n".encode())
        total_rows += file_rows
        total_bytes += file_size

    inventory = {
        "file_count": len(input_files),
        "total_rows_from_metadata": total_rows,
        "total_bytes": total_bytes,
        "sha256_of_path_size_rows": inventory_hash.hexdigest(),
    }
    return input_files, inventory


def read_candidate_spells(
    input_files: list[Path],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Read only full rows passing the US-biopharma restriction from each Parquet part."""
    restricted_parts = []
    count_input = 0
    count_us = 0
    count_biopharma_all_countries = 0
    count_us_biopharma = 0
    for input_file in input_files:
        filters = pd.read_parquet(input_file, columns=["country", "naics_code"])
        is_us = clean_text(filters["country"]).eq("United States").fillna(False)
        is_biopharma = identify_biopharma(filters["naics_code"])
        is_us_biopharma = is_us & is_biopharma
        count_input += len(filters)
        count_us += int(is_us.sum())
        count_biopharma_all_countries += int(is_biopharma.sum())
        count_us_biopharma += int(is_us_biopharma.sum())
        if is_us_biopharma.any():
            spells = pd.read_parquet(input_file)
            require_columns(spells, REQUIRED_INPUT_COLUMNS, source=input_file)
            if len(spells) != len(filters):
                raise ValueError(f"Filter and full reads have different row counts: {input_file}")
            restricted_parts.append(spells.loc[is_us_biopharma.to_numpy()].copy())

    if restricted_parts:
        candidate_spells = pd.concat(restricted_parts, ignore_index=True)
    else:
        candidate_spells = pd.read_parquet(input_files[0]).iloc[0:0].copy()
    counts = {
        "stage_a_spells": count_input,
        "us_spells": count_us,
        "biopharma_spells_all_countries": count_biopharma_all_countries,
        "us_biopharma_spells": count_us_biopharma,
    }
    if len(candidate_spells) != count_us_biopharma:
        raise AssertionError("US-biopharma row count differs from the retained rows.")
    return candidate_spells, counts


def prepare_stage_a_fields(candidate_spells: pd.DataFrame) -> pd.DataFrame:
    """Validate Stage A restrictions and add canonical dates and linkage identifiers."""
    require_columns(candidate_spells, REQUIRED_INPUT_COLUMNS)
    prepared = candidate_spells.copy()
    if prepared["position_id"].isna().any():
        raise ValueError("The source position_id field contains missing values.")
    if prepared.duplicated(["position_id"]).any():
        duplicate_count = int(prepared.duplicated(["position_id"], keep=False).sum())
        raise ValueError(f"The source position_id field has {duplicate_count:,} duplicate rows.")
    missing_pair_ids = prepared[["user_id", "rcid"]].isna().any(axis=1)
    if missing_pair_ids.any():
        raise ValueError(f"Missing user_id or rcid on {int(missing_pair_ids.sum()):,} rows.")

    parsed_dates = pd.to_datetime(prepared["parsed_start_date"], errors="raise")
    if parsed_dates.isna().any():
        raise ValueError("The Stage A parsed_start_date field contains missing values.")
    prepared["start_date"] = parsed_dates
    valid_restrictions = (
        prepared["start_date"].ge("2021-01-01")
        & prepared["start_date"].lt("2024-01-01")
        & clean_text(prepared["onet_code"]).str[:2].isin(["17", "19"])
    )
    if not valid_restrictions.all():
        raise ValueError(
            f"Stage A cohort or occupation restrictions fail on "
            f"{int((~valid_restrictions).sum()):,} US-biopharma rows."
        )

    for source, standardized in [
        ("user_id", "id_user"),
        ("rcid", "id_rcid"),
        ("position_id", "id_position"),
    ]:
        prepared[standardized] = standardize_id(prepared[source])
        if prepared[standardized].isna().any():
            raise ValueError(f"The standardized field {standardized} contains missing values.")
    if prepared["id_position"].duplicated().any():
        raise ValueError("Standardization creates duplicate id_position values.")
    return prepared


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Apply shared title rules and select the first qualifying spell
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-1. Define a helper function to normalize job titles
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


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


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-2. Define a helper function to classify job titles
# >>        (based on some pre-defined exclusion patterns)
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


INTERNSHIP_PATTERN = (
    r"\b(?:interns?|internships?|externs?|externships?|co ops?|coops?|"
    r"co operative students?|cooperative students?|summer students?|working students?|"
    r"placement students?|work placements?|industrial placements?)\b"
)

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


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-3. Define a final coordinator function to screen job titles
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


REVIEW_COLUMNS = [
    "is_mixed_management_title",
    "is_mixed_quality_development",
    "is_technical_title_ambiguous",
    "is_title_translation_conflict",
    "is_title_uninformative",
]
RULE_VERSION = "2026-09-21.1"


def title_rule_hash() -> str:
    """Return a stable hash of the title-rule configuration saved in every output row."""
    configuration = {
        "abbreviations": TITLE_ABBREVIATIONS,
        "internship": INTERNSHIP_PATTERN,
        "exclusions": EXCLUSION_PATTERNS,
        "research_role": RESEARCH_ROLE_PATTERN,
        "management": MANAGEMENT_PATTERN,
        "research_functions": RESEARCH_FUNCTION_PATTERNS,
        "quality_development": QUALITY_DEVELOPMENT_PATTERN,
        "ambiguous_technical": AMBIGUOUS_TECHNICAL_PATTERN,
        "exclusion_order": EXCLUSION_ORDER,
        "review_columns": REVIEW_COLUMNS,
    }
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def screen_titles(raw_titles: pd.Series, translated_titles: pd.Series) -> pd.DataFrame:
    """
    Return title decisions without modifying either source Series.

    Parameters
    ----------
    raw_titles, translated_titles : pd.Series
        Aligned source title fields; use missing strings for unavailable translations.

    Returns
    -------
    pd.DataFrame
        Normalized titles, exclusions, review flags, and optional strict eligibility.
    """
    raw = normalize_job_titles(raw_titles)
    translated = normalize_job_titles(translated_titles)
    flags = pd.DataFrame(index=raw.index)
    flags["title_raw_normalized"] = raw
    flags["title_translated_normalized"] = translated
    flags["job_title_normalized"] = raw.fillna(translated)
    flags["is_translated_title_missing"] = translated.isna()
    different = raw.notna() & translated.notna() & raw.ne(translated).fillna(False)
    flags["is_title_translation_different"] = different
    raw_flags = classify_titles(raw)
    translated_flags = classify_titles(translated)
    conflict = pd.Series(False, index=raw.index)
    for column in raw_flags:
        flags[column] = raw_flags[column] | translated_flags[column]
        if column.startswith("exclude_"):
            conflict |= raw_flags[column].ne(translated_flags[column])
    raw_internship = raw.str.contains(INTERNSHIP_PATTERN, na=False)
    translated_internship = translated.str.contains(INTERNSHIP_PATTERN, na=False)
    conflict |= raw_internship.ne(translated_internship)
    flags["is_title_translation_conflict"] = different & conflict
    flags["exclude_missing_title"] = flags["job_title_normalized"].isna()
    flags["is_title_uninformative"] = ~(flags["has_research_role"] | flags["has_research_function"])
    flags["is_internship"] = raw_internship | translated_internship
    flags["needs_title_review"] = flags[REVIEW_COLUMNS].any(axis=1)
    flags["exclusion_reason"] = pd.Series("retained", index=raw.index, dtype="string")
    remaining = ~flags["is_internship"]
    flags.loc[~remaining, "exclusion_reason"] = "internship"
    for reason in EXCLUSION_ORDER:
        excluded = flags[f"exclude_{reason}"]
        flags.loc[remaining & excluded, "exclusion_reason"] = reason
        remaining &= ~excluded
    flags["is_occupation_eligible"] = remaining
    flags["is_strict_research_title"] = remaining & ~flags["needs_title_review"]
    flags["title_rule_version"] = RULE_VERSION
    flags["title_rule_hash"] = title_rule_hash()
    return flags


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Select the first qualifying spell
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-4-1. Define a helper function to select the first qualifying spell
# >>        (within a user-company cell)
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def select_first_spells(eligible_spells: pd.DataFrame) -> pd.DataFrame:
    """
    Select intact source rows by the stated ordered minima; reject unresolved winning ties.

    Parameters
    ----------
    eligible_spells : pd.DataFrame
        Qualifying spells; source ordering fields are preserved without mutation.

    Returns
    -------
    pd.DataFrame
        One row per standardized user-company, sorted by those keys.
    """
    ordered = eligible_spells.copy()
    for column in ["seniority", "position_number"]:
        numeric = pd.to_numeric(clean_text(ordered[column]), errors="raise")
        if (numeric.dropna() % 1).ne(0).any():
            raise ValueError(f"Nonintegral ordering values in {column}.")
        ordered[f"_{column}_order"] = numeric.astype("Int64")
    if ordered["_position_number_order"].isna().any():
        raise ValueError("Missing position_number.")
    for column in ["start_date", "_seniority_order", "_position_number_order"]:
        minimum = ordered.groupby(["id_user", "id_rcid"], sort=False)[column].transform("min")
        winning = ordered[column].eq(minimum).fillna(False) | (
            ordered[column].isna() & minimum.isna()
        )
        ordered = ordered.loc[winning].copy()
    return (
        ordered
        .drop(columns=["_seniority_order", "_position_number_order"])
        .sort_values(["id_user", "id_rcid"])
        .reset_index(drop=True)
    )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Save candidate hires, the spell audit, and sample counts
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def validate_selection(
    candidate_spells: pd.DataFrame,
    eligible_spells: pd.DataFrame,
    candidate_hires: pd.DataFrame,
) -> None:
    """Require exactly one intact selected row for every eligible user-company pair."""
    pair_columns = ["id_user", "id_rcid"]
    expected_pairs = len(eligible_spells[pair_columns].drop_duplicates())
    if len(candidate_hires) > expected_pairs:
        duplicate_pairs = int(candidate_hires.duplicated(pair_columns, keep=False).sum())
        raise AssertionError(
            f"Ordered minima left {duplicate_pairs:,} rows in unresolved user-company ties."
        )
    if len(candidate_hires) < expected_pairs:
        raise AssertionError(
            f"Selection returned {len(candidate_hires):,} rows for {expected_pairs:,} "
            "eligible user-company pairs."
        )
    if candidate_hires.duplicated(pair_columns).any():
        raise AssertionError("Selected user-company identifiers are not unique.")

    selected_ids = set(candidate_hires["id_position"])
    selected = candidate_spells["id_position"].isin(selected_ids)
    if (selected & ~candidate_spells["is_occupation_eligible"]).any():
        raise AssertionError("An occupation-ineligible spell was selected.")
    selected_pairs = candidate_spells.loc[selected, pair_columns]
    if len(selected_pairs) != expected_pairs or selected_pairs.duplicated(pair_columns).any():
        raise AssertionError("The spell audit does not mark exactly one row per eligible pair.")


def build_sample_flow(
    counts: dict[str, int], eligible_count: int, selected_count: int
) -> pd.DataFrame:
    """Build a genuinely sequential flow and identify the unit at every step."""
    restrictions = [
        "Stage A",
        "United States",
        "US biopharma",
        "Title eligible",
        "Selected first qualifying spell",
    ]
    row_counts = [
        counts["stage_a_spells"],
        counts["us_spells"],
        counts["us_biopharma_spells"],
        eligible_count,
        selected_count,
    ]
    prior_steps = pd.Series([pd.NA, *restrictions[:-1]], dtype="string")
    prior_counts = [row_counts[0], *row_counts[:-1]]
    removed = [0, *[before - after for before, after in pairwise(row_counts)]]
    shares = [
        1.0 if before == 0 else after / before for before, after in zip(prior_counts, row_counts)
    ]
    return pd.DataFrame({
        "step": range(1, len(restrictions) + 1),
        "restriction": pd.Series(restrictions, dtype="string"),
        "unit": pd.Series(["employment_spell"] * 4 + ["user_company_observation"], dtype="string"),
        "prior_step": prior_steps,
        "row_count": row_counts,
        "removed_count": removed,
        "retained_share": shares,
    })


def build_title_summary(candidate_spells: pd.DataFrame) -> pd.DataFrame:
    """Summarize primary exclusions and overlapping review strata."""
    total_count = len(candidate_spells)
    eligible = candidate_spells["is_occupation_eligible"]
    eligible_count = int(eligible.sum())
    exclusion_counts = candidate_spells["exclusion_reason"].value_counts(dropna=False)
    rows = [
        {
            "summary_type": "exclusion_reason",
            "metric": str(metric),
            "row_count": int(row_count),
            "denominator_count": total_count,
        }
        for metric, row_count in exclusion_counts.items()
    ]
    for metric in [*REVIEW_COLUMNS, "needs_title_review", "is_strict_research_title"]:
        row_count = int((candidate_spells[metric] & eligible).sum())
        rows.append({
            "summary_type": "eligible_title_flag",
            "metric": metric,
            "row_count": row_count,
            "denominator_count": eligible_count,
        })
    summary = pd.DataFrame(rows)
    summary["summary_type"] = summary["summary_type"].astype("string")
    summary["metric"] = summary["metric"].astype("string")
    summary["share"] = summary["row_count"].div(summary["denominator_count"].replace(0, pd.NA))
    return summary.sort_values(["summary_type", "metric"]).reset_index(drop=True)


def run_regression_checks() -> None:
    """Exercise missing NAICS values, title boundaries, translation conflicts, and ties."""
    empty_titles = pd.Series([], dtype="string")
    if not screen_titles(empty_titles, empty_titles).empty:
        raise AssertionError("Empty-title input regression check failed.")

    naics = pd.Series(["325412", "541714", "541715", pd.NA], dtype="string")
    if identify_biopharma(naics).tolist() != [True, True, False, False]:
        raise AssertionError("NAICS regression check failed.")
    identifiers = pd.Series(["001", "-001", "-0", 2, pd.NA], dtype="object")
    if standardize_id(identifiers).tolist()[:4] != ["1", "-1", "0", "2"]:
        raise AssertionError("Signed-identifier regression check failed.")

    expected_reasons = {
        "data scientist": "data_science",
        "clinical research associate": "clinical_operations",
        "quality control analyst": "quality_compliance",
        "regulatory affairs scientist": "regulatory_medical_affairs",
        "manufacturing operator": "operational_engineering",
        "sales scientist": "commercial_administration",
        "safety engineer": "safety_healthcare",
        "software engineer": "it_business_analytics",
        "retired scientist": "nonemployment",
        "research intern": "internship",
        "chief scientific officer": "management_without_research_role",
        "QA analytical development scientist": "quality_compliance",
        "engineer II validation": "operational_engineering",
        "bioinformatics data scientist": "data_science",
    }
    retained = [
        "postdoctoral fellow",
        "clinical scientist",
        "QC analytical development scientist",
        "method validation scientist",
        "bioinformatics software engineer",
        "physician scientist",
        "nurse scientist",
        "scientist manager",
    ]
    titles = pd.Series([*expected_reasons, *retained, pd.NA], dtype="string")
    results = screen_titles(titles, pd.Series(pd.NA, index=titles.index, dtype="string"))
    for index, expected_reason in enumerate(expected_reasons.values()):
        if results.loc[index, "exclusion_reason"] != expected_reason:
            title = titles.iloc[index]
            actual = results.loc[index, "exclusion_reason"]
            raise AssertionError(f"Title check failed for {title!r}: {actual!r}")
    retained_start = len(expected_reasons)
    if not results.iloc[retained_start : retained_start + len(retained)][
        "is_occupation_eligible"
    ].all():
        raise AssertionError("A retained title boundary regression check failed.")
    if results.iloc[-1]["exclusion_reason"] != "missing_title":
        raise AssertionError("Missing-title regression check failed.")

    conflict = screen_titles(
        pd.Series(["scientist", "scientist"], dtype="string"),
        pd.Series(["clinical research associate", "research intern"], dtype="string"),
    )
    if not conflict["is_title_translation_conflict"].all():
        raise AssertionError("Translation-conflict regression check failed.")

    selection_input = pd.DataFrame({
        "id_user": ["1", "1", "2", "2"],
        "id_rcid": ["10", "10", "20", "20"],
        "start_date": pd.to_datetime(["2021-01-01"] * 4),
        "seniority": [5, 2, 3, 3],
        "position_number": [1, 2, 2, 1],
        "id_position": ["101", "102", "201", "202"],
    })
    selected = select_first_spells(selection_input)
    if selected["id_position"].tolist() != ["102", "202"]:
        raise AssertionError("First-spell ordered-minima regression check failed.")


def validate_round_trip(frame: pd.DataFrame, restored: pd.DataFrame, output_file: Path) -> None:
    """Validate values plus exact types for critical constructed fields."""
    pd.testing.assert_frame_equal(frame, restored, check_dtype=False)
    if output_file in [OUTPUT_HIRES, OUTPUT_AUDIT]:
        for column in ["id_user", "id_rcid", "id_position"]:
            if not pd.api.types.is_string_dtype(restored[column]):
                raise AssertionError(f"Parquet changed {column} to {restored[column].dtype}.")
        if not pd.api.types.is_datetime64_any_dtype(restored["start_date"]):
            raise AssertionError("Parquet changed the canonical start_date type.")
        boolean_columns = [
            column
            for column in frame
            if column.startswith(("exclude_", "is_", "has_", "needs_"))
            and pd.api.types.is_bool_dtype(frame[column])
        ]
        changed = [
            column for column in boolean_columns if not pd.api.types.is_bool_dtype(restored[column])
        ]
        if changed:
            raise AssertionError(f"Parquet changed boolean field types: {changed}")


def publish_outputs(
    frames: dict[Path, pd.DataFrame], manifest: dict[str, object], *, overwrite: bool
) -> None:
    """Stage and validate every output, then publish data files and the manifest last."""
    if not overwrite:
        existing = [path for path in OUTPUT_FILES if path.exists()]
        if existing:
            raise FileExistsError(f"Stage B outputs already exist: {existing}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    staged_files: dict[Path, Path] = {}
    try:
        for output_file, frame in frames.items():
            normalized = frame.reset_index(drop=True)
            staged = output_file.with_name(f"{output_file.name}.incomplete")
            normalized.to_parquet(staged, index=False)
            restored = pd.read_parquet(staged)
            validate_round_trip(normalized, restored, output_file)
            staged_files[output_file] = staged
        manifest["outputs"] = {
            output_file.name: {
                "row_count": len(frames[output_file]),
                "staged_bytes": staged_files[output_file].stat().st_size,
            }
            for output_file in frames
        }
        staged_manifest = OUTPUT_MANIFEST.with_name(f"{OUTPUT_MANIFEST.name}.incomplete")
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        staged_manifest.write_text(manifest_text, encoding="utf-8")
        if json.loads(staged_manifest.read_text(encoding="utf-8")) != manifest:
            raise AssertionError("Manifest JSON round trip failed.")
        staged_files[OUTPUT_MANIFEST] = staged_manifest
        for output_file in [*frames, OUTPUT_MANIFEST]:
            staged_files[output_file].replace(output_file)
    except Exception:
        for staged in staged_files.values():
            if staged.exists():
                staged.unlink()
        raise
    for output_file, frame in frames.items():
        print(f"Saved {len(frame):,} rows: {output_file}")
    print(f"Saved run manifest: {OUTPUT_MANIFEST}")


def parse_arguments() -> argparse.Namespace:
    """Parse the single destructive publication option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing Stage B outputs after all staged files pass validation",
    )
    return parser.parse_args()


def main(*, overwrite: bool = False) -> None:
    """Construct, validate, and publish the Stage B candidate sample."""
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    print(f"Started Stage B at {started_at:%Y-%m-%d %H:%M:%S} UTC.")
    if not overwrite:
        existing = [path for path in OUTPUT_FILES if path.exists()]
        if existing:
            raise FileExistsError(
                "Stage B outputs already exist; rerun with --overwrite only after review: "
                f"{existing}"
            )

    run_regression_checks()
    input_files, input_inventory = inspect_input_files(INPUT_DIR)
    candidate_spells, counts = read_candidate_spells(input_files)
    if counts["stage_a_spells"] != input_inventory["total_rows_from_metadata"]:
        raise AssertionError("Scanned Stage A count differs from Parquet metadata.")
    candidate_spells = prepare_stage_a_fields(candidate_spells)

    title_flags = screen_titles(candidate_spells["title_raw"], candidate_spells["title_translated"])
    collisions = sorted(set(title_flags) & set(candidate_spells))
    if collisions:
        raise ValueError(f"Source columns collide with constructed title fields: {collisions}")
    candidate_spells = pd.concat([candidate_spells, title_flags], axis=1)
    eligible_spells = candidate_spells.loc[candidate_spells["is_occupation_eligible"]].copy()
    candidate_hires = select_first_spells(eligible_spells)
    candidate_spells["is_selected"] = candidate_spells["id_position"].isin(
        candidate_hires["id_position"]
    )
    validate_selection(candidate_spells, eligible_spells, candidate_hires)

    sample_flow = build_sample_flow(counts, len(eligible_spells), len(candidate_hires))
    title_summary = build_title_summary(candidate_spells)
    identifier_columns = [
        "id_user",
        "id_rcid",
        "id_position",
        "user_id",
        "rcid",
        "position_id",
    ]
    candidate_hires = candidate_hires[
        identifier_columns
        + [column for column in candidate_hires if column not in identifier_columns]
    ]
    candidate_spells = candidate_spells.sort_values([
        "id_user",
        "id_rcid",
        "start_date",
        "id_position",
    ]).reset_index(drop=True)

    manifest = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_directory": INPUT_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "input_inventory": input_inventory,
        "marginal_counts": counts,
        "pandas_version": pd.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "title_rule_hash": title_rule_hash(),
        "title_rule_version": RULE_VERSION,
    }
    publish_outputs(
        {
            OUTPUT_AUDIT: candidate_spells,
            OUTPUT_FLOW: sample_flow,
            OUTPUT_TITLE_SUMMARY: title_summary,
            OUTPUT_HIRES: candidate_hires,
        },
        manifest,
        overwrite=overwrite,
    )
    elapsed = time.perf_counter() - started_clock
    print(
        f"Finished Stage B in {elapsed:,.1f} seconds with "
        f"{len(candidate_hires):,} candidate user-company observations."
    )


if __name__ == "__main__":
    arguments = parse_arguments()
    main(overwrite=arguments.overwrite)

"""
Task:
    Attribute changes in inventor representation to the saved Stage B title-exclusion rules.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv
(c) data/b_temp_data/A01_BaselineUSBiopharma/StageB_RunManifest.json
(d) codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py
    Read as text to check provenance; the executable pipeline is never imported.

Outputs:
    All outputs below are in data/b_temp_data/A01_BaselineUSBiopharma/.
(a) DiagnosticB01_SampleSummary.csv
(b) DiagnosticB01_PrimaryReasons.csv
(c) DiagnosticB01_SequentialRules.csv
(d) DiagnosticB01_RemoveOneRule.csv
(e) DiagnosticB01_TitleSummary.csv
(f) DiagnosticB01_ManagementStrata.csv
(g) DiagnosticB01_SpellAudit.parquet
(h) DiagnosticB01_RunManifest.json

Descriptions of outputs:
(1) Output (a) has one row per baseline or illustrative counterfactual sample.
(2) Output (b) has one row per mutually exclusive primary exclusion reason, including retained.
(3) Output (c) has one row per sequential rule, preceded by the unrestricted sample.
(4) Output (d) has one row per individually disabled rule; other rules remain active.
(5) Output (e) has one row per primary reason and normalized title, including missing titles.
(6) Output (f) has one row per research-function/role/translation stratum of management matches.
(7) Output (g) has one row per source spell with identifiers, titles, and all exclusion flags.
(8) Output (h) records input fingerprints, versions, output row counts, and interpretation notes.

Run:
    From the project root:
    conda activate Talent
    python -m codes.A01_BaselineUSBiopharma.DiagnosticB01_InventorsInJobTitleRestrictions
    Add --overwrite to replace this diagnostic's existing outputs.

Notes:
(1) Inventor means user_id membership in the deduplicated crosswalk, not patenting in this job.
(2) This reproduces ResultsB04/B05: all audited spells, before user-company spell selection.
(3) A user survives a sample restriction if at least one of that user's spells survives.
(4) Primary reasons follow Stage B precedence; all matching rules can overlap.
(5) Removing one rule restores only spells with no other exclusion, not every matching spell.
(6) Sequential user losses are disjoint but order dependent. Users within primary reasons
    overlap, so those reason-specific user counts must not be summed.
(7) Primary spell contributions sum to the overall change in percentage points: for reason r,
    contribution = 100 * (before_share * removed_spells_r - removed_inventors_r) / kept_spells.
    This is an accounting decomposition, not a causal effect or a remove-one-rule estimate.
(8) Saved research-function evidence is broader than the research-role wording used to exempt
    management titles. Illustrative relaxations do not change the production sample.
(9) This is a local audit of the downloaded Stage B sample, not a Fabric validation.


Wang Wenzhi
Time: 2026-09-22
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from codes import main


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define input paths, audited rule order, and output settings
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


OUTPUT_DIR = main.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
INPUT_AUDIT = OUTPUT_DIR / "StageB_SpellAudit.parquet"
INPUT_INVENTOR_LINKS = main.DIR_RAWDATA / "A_Revelio/revelio_user_id_patentsview_id.csv"
INPUT_MANIFEST = OUTPUT_DIR / "StageB_RunManifest.json"
INPUT_STAGE_B = main.DIR_CODES / "A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
RULE_VERSION = "2026-09-21.1"
RULE_HASH = "6a75213836ef3299d5519be8b3d55eda84fb87b579dd53368c4d56c111749690"
# Internship is applied before EXCLUSION_ORDER in Stage B's screen_titles function.
RULE_COLUMNS = {
    "internship": "is_internship",
    "missing_title": "exclude_missing_title",
    "data_science": "exclude_data_science",
    "clinical_operations": "exclude_clinical_operations",
    "quality_compliance": "exclude_quality_compliance",
    "regulatory_medical_affairs": "exclude_regulatory_medical_affairs",
    "operational_engineering": "exclude_operational_engineering",
    "commercial_administration": "exclude_commercial_administration",
    "safety_healthcare": "exclude_safety_healthcare",
    "it_business_analytics": "exclude_it_business_analytics",
    "nonemployment": "exclude_nonemployment",
    "management_without_research_role": "exclude_management_without_research_role",
}
AUDIT_COLUMNS = [
    "id_user",
    "id_position",
    "user_id",
    "title_raw",
    "title_translated",
    "title_raw_normalized",
    "title_translated_normalized",
    "job_title_normalized",
    "exclusion_reason",
    "is_occupation_eligible",
    "has_research_role",
    "has_research_function",
    "is_title_translation_conflict",
    "title_rule_version",
    "title_rule_hash",
    *RULE_COLUMNS.values(),
]
OUTPUT_NAMES = [
    "SampleSummary.csv",
    "PrimaryReasons.csv",
    "SequentialRules.csv",
    "RemoveOneRule.csv",
    "TitleSummary.csv",
    "ManagementStrata.csv",
    "SpellAudit.parquet",
    "RunManifest.json",
]
OUTPUT_PATHS = {name: OUTPUT_DIR / f"DiagnosticB01_{name}" for name in OUTPUT_NAMES}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--overwrite", action="store_true", help="replace diagnostic outputs")
arguments = parser.parse_args()
existing_outputs = [path for path in OUTPUT_PATHS.values() if path.exists()]
if existing_outputs and not arguments.overwrite:
    raise FileExistsError(f"Use --overwrite to replace diagnostic outputs: {existing_outputs}")
run_record = main.start_run(__file__)


def summarize_sample(spells: pd.DataFrame, sample_mask: pd.Series) -> dict[str, int | float]:
    """
    Count spells and distinct users within one Boolean sample mask.

    Parameters
    ----------
    spells : pd.DataFrame
        One row per source spell, with id_user and a nonmissing Boolean is_inventor flag.
    sample_mask : pd.Series
        Boolean sample membership aligned to spells.index.

    Returns
    -------
    dict[str, int | float]
        Counts and inventor shares at spell and user levels. Empty shares are undefined.
    """
    sample_spells = spells.loc[sample_mask, ["id_user", "is_inventor"]]
    sample_users = sample_spells.drop_duplicates("id_user")
    n_spells = len(sample_spells)
    n_inventor_spells = int(sample_spells["is_inventor"].sum())
    n_users = len(sample_users)
    n_inventor_users = int(sample_users["is_inventor"].sum())
    return {
        "spell_count": n_spells,
        "inventor_spell_count": n_inventor_spells,
        "inventor_spell_share": n_inventor_spells / n_spells if n_spells else float("nan"),
        "user_count": n_users,
        "inventor_user_count": n_inventor_users,
        "inventor_user_share": n_inventor_users / n_users if n_users else float("nan"),
    }


def file_sha256(input_path: Path) -> str:
    """
    Fingerprint an input file without loading the entire file into memory.
    """
    checksum = hashlib.sha256()
    with input_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Validate provenance, identifiers, and the saved exclusion decisions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


stage_b_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
if stage_b_manifest["script_sha256"] != file_sha256(INPUT_STAGE_B):
    raise ValueError("Stage B code differs from the code that produced the saved audit.")
spell_audit = pd.read_parquet(INPUT_AUDIT, columns=AUDIT_COLUMNS)
if len(spell_audit) != stage_b_manifest["outputs"][INPUT_AUDIT.name]["row_count"]:
    raise ValueError("The Stage B manifest and spell audit have different row counts.")
for column, expected in [("title_rule_version", RULE_VERSION), ("title_rule_hash", RULE_HASH)]:
    if not spell_audit[column].eq(expected).fillna(False).all():
        raise ValueError(f"Unexpected {column}; review this diagnostic against the updated rules.")
    if stage_b_manifest[column] != expected:
        raise ValueError(f"Stage B manifest has an unexpected {column}.")
for column in ["id_user", "id_position"]:
    spell_audit[column] = spell_audit[column].astype("string").str.strip()
    if not spell_audit[column].str.fullmatch(r"-?\d+", na=False).all():
        raise ValueError(f"Missing or invalid identifier in {column}.")
if spell_audit["id_position"].duplicated().any():
    raise ValueError("The audit must have exactly one row per source position.")
source_user_ids = spell_audit["user_id"].astype("string").str.strip()
if not source_user_ids.eq(spell_audit["id_user"]).fillna(False).all():
    raise ValueError("Source and standardized user identifiers disagree.")
boolean_columns = [
    "is_occupation_eligible",
    "has_research_role",
    "has_research_function",
    "is_title_translation_conflict",
    *RULE_COLUMNS.values(),
]
for column in boolean_columns:
    if spell_audit[column].isna().any() or not pd.api.types.is_bool_dtype(spell_audit[column]):
        raise ValueError(f"Expected nonmissing Boolean values in {column}.")

inventor_links = pd.read_csv(
    INPUT_INVENTOR_LINKS,
    usecols=["user_id"],
    dtype="string",
    low_memory=False,
)
inventor_links["user_id"] = inventor_links["user_id"].str.strip()
if not inventor_links["user_id"].str.fullmatch(r"\d+", na=False).all():
    raise ValueError("The inventor crosswalk has missing or invalid user identifiers.")
inventor_user_ids = inventor_links["user_id"].drop_duplicates()
spell_audit["is_inventor"] = spell_audit["id_user"].isin(inventor_user_ids)
rule_flags = spell_audit[list(RULE_COLUMNS.values())]
spell_audit["exclusion_rule_count"] = rule_flags.sum(axis=1).astype("int8")
flag_all = pd.Series(True, index=spell_audit.index)
flag_eligible = spell_audit["is_occupation_eligible"]
if not flag_eligible.eq(~rule_flags.any(axis=1)).all():
    raise AssertionError("The union of all saved rules does not reproduce title eligibility.")
reconstructed_reasons = pd.Series("retained", index=spell_audit.index, dtype="string")
flag_remaining = flag_all.copy()
for reason, column in RULE_COLUMNS.items():
    reconstructed_reasons.loc[flag_remaining & spell_audit[column]] = reason
    flag_remaining &= ~spell_audit[column]
if not reconstructed_reasons.eq(spell_audit["exclusion_reason"]).fillna(False).all():
    raise AssertionError("The stated rule order does not reproduce saved primary reasons.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Reproduce baseline shares and calculate illustrative management relaxations
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


baseline_before = summarize_sample(spell_audit, flag_all)
baseline_after = summarize_sample(spell_audit, flag_eligible)
flag_management = spell_audit["exclude_management_without_research_role"]
flag_management_only = flag_management & spell_audit["exclusion_rule_count"].eq(1)
sample_masks = {
    "before_title_restrictions": flag_all,
    "after_title_restrictions": flag_eligible,
    "all_excluded_spells": ~flag_eligible,
    "disable_management_rule": flag_eligible | flag_management_only,
    "exempt_management_with_saved_research_function": (
        flag_eligible | (flag_management_only & spell_audit["has_research_function"])
    ),
}
sample_summary = pd.DataFrame([
    {"sample": sample, **summarize_sample(spell_audit, sample_mask)}
    for sample, sample_mask in sample_masks.items()
])
# Users appearing in excluded spells can also have retained spells; count total user exits directly.
retained_user_ids = spell_audit.loc[flag_eligible, "id_user"].drop_duplicates()
spell_audit["has_retained_spell"] = spell_audit["id_user"].isin(retained_user_ids)
lost_users = spell_audit.loc[~spell_audit["has_retained_spell"]].drop_duplicates("id_user")
if len(lost_users) != baseline_before["user_count"] - baseline_after["user_count"]:
    raise AssertionError("Lost users do not reconcile to before and after user counts.")
if int(lost_users["is_inventor"].sum()) != (
    baseline_before["inventor_user_count"] - baseline_after["inventor_user_count"]
):
    raise AssertionError("Lost inventor users do not reconcile to the baseline counts.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Attribute primary, sequential, and remove-one-rule changes
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


primary_rows = []
sequential_rows = [{
    "step": 0,
    "rule": "before_title_restrictions",
    **baseline_before,
    "removed_spell_count": 0,
    "removed_inventor_spell_count": 0,
    "removed_user_count": 0,
    "removed_inventor_user_count": 0,
    "inventor_spell_share_change_pp": 0.0,
    "inventor_user_share_change_pp": 0.0,
}]
counterfactual_rows = []
flag_remaining = flag_all.copy()
prior_summary = baseline_before
n_removed_inventor_spells = (
    baseline_before["inventor_spell_count"] - baseline_after["inventor_spell_count"]
)
for step, (reason, column) in enumerate(RULE_COLUMNS.items(), start=1):
    primary_summary = summarize_sample(spell_audit, spell_audit["exclusion_reason"].eq(reason))
    primary_summary["share_of_removed_inventor_spells"] = (
        primary_summary["inventor_spell_count"] / n_removed_inventor_spells
    )
    primary_summary["contribution_to_spell_share_change_pp"] = 100 * (
        baseline_before["inventor_spell_share"] * primary_summary["spell_count"]
        - primary_summary["inventor_spell_count"]
    ) / baseline_after["spell_count"]
    primary_rows.append({"reason": reason, **primary_summary})

    flag_remaining &= ~spell_audit[column]
    remaining_summary = summarize_sample(spell_audit, flag_remaining)
    sequential_row = {"step": step, "rule": reason, **remaining_summary}
    for metric in ["spell_count", "inventor_spell_count", "user_count", "inventor_user_count"]:
        sequential_row[f"removed_{metric}"] = prior_summary[metric] - remaining_summary[metric]
    for unit in ["spell", "user"]:
        sequential_row[f"inventor_{unit}_share_change_pp"] = 100 * (
            remaining_summary[f"inventor_{unit}_share"] - prior_summary[f"inventor_{unit}_share"]
        )
    sequential_rows.append(sequential_row)
    prior_summary = remaining_summary

    flag_restored = spell_audit[column] & spell_audit["exclusion_rule_count"].eq(1)
    counterfactual_summary = summarize_sample(spell_audit, flag_eligible | flag_restored)
    counterfactual_row = {"disabled_rule": reason, **counterfactual_summary}
    matched_summary = summarize_sample(spell_audit, spell_audit[column])
    for metric in ["spell_count", "inventor_spell_count", "user_count", "inventor_user_count"]:
        counterfactual_row[f"matched_{metric}"] = matched_summary[metric]
        counterfactual_row[f"restored_{metric}"] = (
            counterfactual_summary[metric] - baseline_after[metric]
        )
    for unit in ["spell", "user"]:
        counterfactual_row[f"inventor_{unit}_share_change_pp"] = 100 * (
            counterfactual_summary[f"inventor_{unit}_share"]
            - baseline_after[f"inventor_{unit}_share"]
        )
    counterfactual_rows.append(counterfactual_row)

primary_rows.append({"reason": "retained", **baseline_after})
primary_reasons = pd.DataFrame(primary_rows)
sequential_rules = pd.DataFrame(sequential_rows)
remove_one_rule = pd.DataFrame(counterfactual_rows)
for metric in ["spell_count", "inventor_spell_count"]:
    if int(primary_reasons[metric].sum()) != baseline_before[metric]:
        raise AssertionError(f"Primary reasons fail to partition {metric}.")
for metric in ["spell_count", "inventor_spell_count", "user_count", "inventor_user_count"]:
    if prior_summary[metric] != baseline_after[metric]:
        raise AssertionError(f"Sequential rules fail to reproduce final {metric}.")
    if int(sequential_rules[f"removed_{metric}"].sum()) != (
        baseline_before[metric] - baseline_after[metric]
    ):
        raise AssertionError(f"Sequential losses fail to reconcile {metric}.")
expected_share_change = 100 * (
    baseline_after["inventor_spell_share"] - baseline_before["inventor_spell_share"]
)
actual_share_change = primary_reasons["contribution_to_spell_share_change_pp"].sum()
if abs(actual_share_change - expected_share_change) > 1e-10:
    raise AssertionError("Primary-reason contributions do not sum to the observed share change.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Summarize actual titles and management research-function evidence
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


title_summary = (
    spell_audit.groupby(["exclusion_reason", "job_title_normalized"], dropna=False, sort=True)
    .agg(
        spell_count=("id_position", "size"),
        inventor_spell_count=("is_inventor", "sum"),
        user_count=("id_user", "nunique"),
        research_function_spell_count=("has_research_function", "sum"),
        translation_conflict_spell_count=("is_title_translation_conflict", "sum"),
    )
    .reset_index()
)
title_summary["inventor_spell_share"] = (
    title_summary["inventor_spell_count"] / title_summary["spell_count"]
)
title_summary = title_summary.sort_values(
    ["exclusion_reason", "inventor_spell_count", "spell_count", "job_title_normalized"],
    ascending=[True, False, False, True],
).reset_index(drop=True)
management_rows = []
stratum_columns = [
    "has_research_function",
    "has_research_role",
    "is_title_translation_conflict",
]
for stratum, stratum_spells in spell_audit.loc[flag_management].groupby(stratum_columns):
    flag_stratum = spell_audit.index.isin(stratum_spells.index)
    management_rows.append({
        **dict(zip(stratum_columns, stratum)),
        **summarize_sample(spell_audit, pd.Series(flag_stratum, index=spell_audit.index)),
        "sole_management_spell_count": int(stratum_spells["exclusion_rule_count"].eq(1).sum()),
        "sole_management_inventor_spell_count": int(
            (stratum_spells["exclusion_rule_count"].eq(1) & stratum_spells["is_inventor"]).sum()
        ),
    })
management_strata = pd.DataFrame(management_rows)
spell_audit = spell_audit.sort_values(["id_user", "id_position"]).reset_index(drop=True)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 6. Save validated diagnostics and provenance without changing Stage B outputs
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


output_frames = {
    "SampleSummary.csv": (sample_summary, "sample"),
    "PrimaryReasons.csv": (primary_reasons, "primary_reason"),
    "SequentialRules.csv": (sequential_rules, "sequential_step"),
    "RemoveOneRule.csv": (remove_one_rule, "disabled_rule"),
    "TitleSummary.csv": (title_summary, "primary_reason_and_normalized_title"),
    "ManagementStrata.csv": (management_strata, "management_evidence_stratum"),
    "SpellAudit.parquet": (spell_audit, "source_spell"),
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for name, (output_frame, unit) in output_frames.items():
    output_path = OUTPUT_PATHS[name]
    temporary_output = output_path.with_suffix(output_path.suffix + ".incomplete")
    if output_path.suffix == ".parquet":
        output_frame.to_parquet(temporary_output, index=False)
        pd.testing.assert_frame_equal(output_frame, pd.read_parquet(temporary_output))
    else:
        output_frame.to_csv(temporary_output, index=False)
        restored = pd.read_csv(temporary_output, low_memory=False)
        if len(restored) != len(output_frame) or list(restored) != list(output_frame):
            raise AssertionError(f"CSV row count or column order changed: {name}")
    temporary_output.replace(output_path)
    print(f"Saved {len(output_frame):,} rows ({unit}): {output_path}")

diagnostic_manifest = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "script_sha256": file_sha256(Path(__file__)),
    "pandas_version": pd.__version__,
    "title_rule_version": RULE_VERSION,
    "title_rule_hash": RULE_HASH,
    "exclusion_precedence": list(RULE_COLUMNS),
    "input_sha256": {
        path.relative_to(main.PROJECT_ROOT).as_posix(): file_sha256(path)
        for path in [INPUT_AUDIT, INPUT_INVENTOR_LINKS, INPUT_MANIFEST, INPUT_STAGE_B]
    },
    "crosswalk_row_count": len(inventor_links),
    "crosswalk_unique_user_count": len(inventor_user_ids),
    "entirely_excluded_user_count": len(lost_users),
    "entirely_excluded_inventor_user_count": int(lost_users["is_inventor"].sum()),
    "output_row_counts": {name: len(frame) for name, (frame, unit) in output_frames.items()},
    "notes": [
        "Only local downloaded data were validated; no Fabric execution was performed.",
        "Crosswalk membership does not date patents or assign them to the observed employer.",
        "Primary-reason user counts overlap; sequential user exits are order dependent.",
        "Remove-one-rule restored user counts exclude users with any already-retained spell.",
        "Research-function exemption uses saved evidence from either raw or translated titles.",
        "No changes were made to production rules or the Stage B outputs.",
    ],
}
manifest_path = OUTPUT_PATHS["RunManifest.json"]
temporary_manifest = manifest_path.with_suffix(".json.incomplete")
temporary_manifest.write_text(json.dumps(diagnostic_manifest, indent=2) + "\n", encoding="utf-8")
temporary_manifest.replace(manifest_path)
print(sample_summary.to_string(index=False))
print(primary_reasons.to_string(index=False))
main.finish_run(
    run_record,
    "All local count, rule-union, precedence, and accounting checks passed.",
)

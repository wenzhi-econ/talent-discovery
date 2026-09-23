"""
Task:
    Report collective and individual title-rule changes in matched-inventor representation.

Inputs:
(a) data/b_temp_data/A01_BaselineUSBiopharma/StageB_SpellAudit.parquet
    <== Constructed by "codes/A01_BaselineUSBiopharma/StageB_CandidateUSBiopharmaNewHires.py"
(b) data/a_raw_data/A_Revelio/revelio_user_id_patentsview_id.csv
(c) data/b_temp_data/A01_BaselineUSBiopharma/StageB_RunManifest.json

Outputs:
    All outputs below are in outputs/A01_BaselineUSBiopharma/.
(a) ResultsB06_SampleComparisons.csv
(b) ResultsB06_RuleComparisons.csv
(c) ResultsB06_SeniorityByPrimaryReason.csv
(d) ResultsB06_UserSpellCounts.csv

Descriptions of outputs:
(1) Output (a) has one row per set of active rules, with spell and distinct-user statistics.
(2) Output (b) has one row per rule and comparison context: applied alone, in production order,
    or added last while holding all other rules active. Changes are after minus before.
(3) Output (c) has one row per primary reason and describes source-spell seniority scores.
(4) Output (d) has one row per mutually exclusive user survival group. Spell counts cover only
    the audited 2021-2023 US biopharma sample, not users' lifetime employment histories.

Run:
    From the project root:
    conda activate Talent
    python -m codes.A01_BaselineUSBiopharma.ResultsB06_InventorSharesInJobTitleRestrictions

Notes:
(1) Inventor means crosswalk membership. This does not date patents or identify their employer.
(2) A user survives if any source spell survives. User losses exclude users with retained spells.
(3) The sample precedes selection of one qualifying spell per user-company pair.
(4) Overlapping rules make stand-alone, sequential, and conditional comparisons different.
    Conditional effects must not be added together; sequential share changes do telescope.
(5) Positive changes indicate inventor-share enrichment, not causal effects or proof that every
    retained worker is a researcher. Negative changes are reported without suppression.
(6) Nonemployment and management restrictions can define scope even when they lower shares.
(7) Seniority scores are descriptive proxies, not ages or validated measures of research work.
(8) Existing ResultsB06 outputs are replaced after temporary CSV files pass round-trip checks.
(9) This script does not edit the report, production rules, or Stage B datasets.


Wang Wenzhi
Time: 2026-09-22
"""

import json

import pandas as pd

from codes import main


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Define paths, saved-rule precedence, and summary helpers
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


INPUT_DIR = main.DIR_TEMPDATA / "A01_BaselineUSBiopharma"
INPUT_AUDIT = INPUT_DIR / "StageB_SpellAudit.parquet"
INPUT_MANIFEST = INPUT_DIR / "StageB_RunManifest.json"
INPUT_INVENTORS = main.DIR_RAWDATA / "A_Revelio/revelio_user_id_patentsview_id.csv"
OUTPUT_DIR = main.DIR_OUTPUTS / "A01_BaselineUSBiopharma"
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
EXPECTED_RULE_VERSION = "2026-09-21.1"
EXPECTED_RULE_HASH = "6a75213836ef3299d5519be8b3d55eda84fb87b579dd53368c4d56c111749690"
AUDIT_COLUMNS = [
    "id_user",
    "id_position",
    "user_id",
    "job_title_normalized",
    "seniority",
    "exclusion_reason",
    "is_occupation_eligible",
    "title_rule_version",
    "title_rule_hash",
    *RULE_COLUMNS.values(),
]
COUNT_COLUMNS = ["spell_count", "inventor_spell_count", "user_count", "inventor_user_count"]
run_record = main.start_run(__file__)


def summarize_sample(spells: pd.DataFrame, sample_mask: pd.Series) -> dict[str, int | float]:
    """
    Count spells and users separately within the supplied Boolean sample mask.

    Parameters
    ----------
    spells : pd.DataFrame
        Source spells with id_user, is_inventor, and job_title_normalized columns.
    sample_mask : pd.Series
        Boolean membership aligned to spells.index.

    Returns
    -------
    dict[str, int | float]
        Counts, inventor shares, and nonmissing standardized-title counts within the sample.
    """
    sample_spells = spells.loc[sample_mask]
    sample_users = sample_spells.drop_duplicates("id_user")
    summary = {"standardized_title_count": sample_spells["job_title_normalized"].nunique()}
    for unit, sample_records in [("spell", sample_spells), ("user", sample_users)]:
        n_records = len(sample_records)
        n_inventors = int(sample_records["is_inventor"].sum())
        summary[f"{unit}_count"] = n_records
        summary[f"inventor_{unit}_count"] = n_inventors
        summary[f"inventor_{unit}_share"] = (
            n_inventors / n_records if n_records else float("nan")
        )
    return summary


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Read identifiers and validate the saved title-rule decisions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spell_audit = pd.read_parquet(INPUT_AUDIT, columns=AUDIT_COLUMNS)
manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
if len(spell_audit) != manifest["outputs"][INPUT_AUDIT.name]["row_count"]:
    raise ValueError("The Stage B manifest and audit row counts differ.")
for column, expected in [
    ("title_rule_version", EXPECTED_RULE_VERSION),
    ("title_rule_hash", EXPECTED_RULE_HASH),
]:
    if manifest[column] != expected or not spell_audit[column].eq(expected).fillna(False).all():
        raise ValueError(f"Review ResultsB06 against the updated Stage B {column}.")
for column in ["id_user", "id_position"]:
    spell_audit[column] = spell_audit[column].astype("string").str.strip()
    if not spell_audit[column].str.fullmatch(r"-?\d+", na=False).all():
        raise ValueError(f"Missing or invalid identifiers in {column}.")
if spell_audit["id_position"].duplicated().any():
    raise ValueError("Expected one row per source position.")
source_user_ids = spell_audit["user_id"].astype("string").str.strip()
if not source_user_ids.eq(spell_audit["id_user"]).fillna(False).all():
    raise ValueError("Source and standardized user identifiers disagree.")
for column in ["is_occupation_eligible", *RULE_COLUMNS.values()]:
    if spell_audit[column].isna().any() or not pd.api.types.is_bool_dtype(spell_audit[column]):
        raise ValueError(f"Expected nonmissing Boolean values in {column}.")

inventor_links = pd.read_csv(
    INPUT_INVENTORS, usecols=["user_id"], dtype="string", low_memory=False
)
inventor_links["user_id"] = inventor_links["user_id"].str.strip()
if not inventor_links["user_id"].str.fullmatch(r"\d+", na=False).all():
    raise ValueError("Invalid inventor crosswalk user identifiers.")
spell_audit["is_inventor"] = spell_audit["id_user"].isin(
    inventor_links["user_id"].drop_duplicates()
)
rule_flags = spell_audit[list(RULE_COLUMNS.values())]
flag_all = pd.Series(True, index=spell_audit.index)
flag_final = ~rule_flags.any(axis=1)
if not flag_final.eq(spell_audit["is_occupation_eligible"]).all():
    raise AssertionError("The saved rules do not reproduce final eligibility.")
primary_reason = pd.Series("retained", index=spell_audit.index, dtype="string")
flag_remaining = flag_all.copy()
for rule, column in RULE_COLUMNS.items():
    primary_reason.loc[flag_remaining & spell_audit[column]] = rule
    flag_remaining &= ~spell_audit[column]
if not primary_reason.eq(spell_audit["exclusion_reason"]).fillna(False).all():
    raise AssertionError("Rule precedence differs from the saved primary reasons.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Calculate sample comparisons for the main slide
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


management_column = RULE_COLUMNS["management_without_research_role"]
nonemployment_column = RULE_COLUMNS["nonemployment"]
flag_management = spell_audit[management_column]
flag_nonemployment = spell_audit[nonemployment_column]
flag_pass_other_rules = ~rule_flags.drop(columns=management_column).any(axis=1)
flag_pass_other_ten = ~rule_flags.drop(
    columns=[management_column, nonemployment_column]
).any(axis=1)
sample_masks = {
    "before_title_restrictions": flag_all,
    "all_rules_except_management": flag_pass_other_rules,
    "all_rules": flag_final,
    "management_only": ~flag_management,
    "management_and_nonemployment_only": ~(flag_management | flag_nonemployment),
    "all_rules_except_management_and_nonemployment": flag_pass_other_ten,
}
baseline_summary = summarize_sample(spell_audit, flag_all)
final_summary = summarize_sample(spell_audit, flag_final)
sample_rows = []
for sample, sample_mask in sample_masks.items():
    sample_summary = summarize_sample(spell_audit, sample_mask)
    sample_row = {"sample": sample, **sample_summary}
    for unit in ["spell", "user"]:
        sample_row[f"inventor_{unit}_retention_share"] = (
            sample_summary[f"inventor_{unit}_count"] / baseline_summary[f"inventor_{unit}_count"]
        )
        sample_row[f"noninventor_{unit}_retention_share"] = (
            sample_summary[f"{unit}_count"] - sample_summary[f"inventor_{unit}_count"]
        ) / (baseline_summary[f"{unit}_count"] - baseline_summary[f"inventor_{unit}_count"])
    sample_rows.append(sample_row)
sample_comparisons = pd.DataFrame(sample_rows)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Compare every rule alone, sequentially, and conditional on every other rule
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


comparison_rows = []
flag_remaining = flag_all.copy()
for rule_order, (rule, column) in enumerate(RULE_COLUMNS.items(), start=1):
    # Dropping one flag admits only spells that pass every other rule, even with overlaps.
    flag_pass_without_rule = ~rule_flags.drop(columns=column).any(axis=1)
    context_masks = {
        "rule_alone": (flag_all, ~spell_audit[column]),
        "production_sequence": (flag_remaining, flag_remaining & ~spell_audit[column]),
        "conditional_on_all_other_rules": (flag_pass_without_rule, flag_final),
    }
    for context, (flag_before, flag_after) in context_masks.items():
        if (flag_after & ~flag_before).any():
            raise AssertionError("An exclusion comparison unexpectedly adds spells.")
        before_summary = summarize_sample(spell_audit, flag_before)
        after_summary = summarize_sample(spell_audit, flag_after)
        comparison_row = {"context": context, "rule_order": rule_order, "rule": rule}
        for timing, summary in [("before", before_summary), ("after", after_summary)]:
            comparison_row.update({f"{timing}_{name}": value for name, value in summary.items()})
        for metric in COUNT_COLUMNS:
            comparison_row[f"removed_{metric}"] = before_summary[metric] - after_summary[metric]
        for unit in ["spell", "user"]:
            comparison_row[f"inventor_{unit}_share_change_pp"] = 100 * (
                after_summary[f"inventor_{unit}_share"]
                - before_summary[f"inventor_{unit}_share"]
            )
            n_removed = comparison_row[f"removed_{unit}_count"]
            n_removed_inventors = comparison_row[f"removed_inventor_{unit}_count"]
            comparison_row[f"removed_inventor_{unit}_share"] = (
                n_removed_inventors / n_removed if n_removed else float("nan")
            )
        comparison_rows.append(comparison_row)
    flag_remaining &= ~spell_audit[column]
rule_comparisons = pd.DataFrame(comparison_rows)
sequential_rows = rule_comparisons.loc[rule_comparisons["context"].eq("production_sequence")]
for metric in COUNT_COLUMNS:
    expected_removed_count = baseline_summary[metric] - final_summary[metric]
    if sequential_rows[f"removed_{metric}"].sum() != expected_removed_count:
        raise AssertionError(f"Sequential losses do not reconcile for {metric}.")
for unit in ["spell", "user"]:
    expected_change = 100 * (
        final_summary[f"inventor_{unit}_share"] - baseline_summary[f"inventor_{unit}_share"]
    )
    if abs(sequential_rows[f"inventor_{unit}_share_change_pp"].sum() - expected_change) > 1e-10:
        raise AssertionError(f"Sequential {unit} share changes do not telescope.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Examine seniority and observed spell counts without inferring age
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


spell_audit["seniority"] = pd.to_numeric(spell_audit["seniority"], errors="raise")
seniority_summary = spell_audit.groupby("exclusion_reason", sort=True).agg(
    spell_count=("id_position", "size"),
    inventor_spell_count=("is_inventor", "sum"),
    observed_seniority_count=("seniority", "count"),
    seniority_mean=("seniority", "mean"),
    seniority_median=("seniority", "median"),
).reset_index()
seniority_summary["inventor_spell_share"] = (
    seniority_summary["inventor_spell_count"] / seniority_summary["spell_count"]
)
# Group users by survival, then count all their spells in the original audit, including exclusions.
spell_audit["passes_nonmanagement_rules"] = flag_pass_other_rules
user_profiles = spell_audit.groupby("id_user", sort=True).agg(
    audited_spell_count=("id_position", "size"),
    has_retained_spell=("is_occupation_eligible", "max"),
    passes_nonmanagement_rules=("passes_nonmanagement_rules", "max"),
    is_inventor=("is_inventor", "max"),
)
user_profiles["user_group"] = "lost_before_management_step"
user_profiles.loc[user_profiles["passes_nonmanagement_rules"], "user_group"] = (
    "lost_at_management_step"
)
user_profiles.loc[user_profiles["has_retained_spell"], "user_group"] = "retained"
user_spell_counts = user_profiles.groupby("user_group", sort=True).agg(
    user_count=("audited_spell_count", "size"),
    inventor_user_count=("is_inventor", "sum"),
    audited_spell_count=("audited_spell_count", "sum"),
    audited_spells_per_user_mean=("audited_spell_count", "mean"),
    audited_spells_per_user_median=("audited_spell_count", "median"),
).reset_index()
user_spell_counts["inventor_user_share"] = (
    user_spell_counts["inventor_user_count"] / user_spell_counts["user_count"]
)
if user_spell_counts["user_count"].sum() != baseline_summary["user_count"]:
    raise AssertionError("User survival groups do not partition all users.")
if user_spell_counts["audited_spell_count"].sum() != baseline_summary["spell_count"]:
    raise AssertionError("User survival groups do not account for every audited spell.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 6. Validate and publish the summary tables
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


output_frames = {
    "SampleComparisons": (sample_comparisons, "active rule set"),
    "RuleComparisons": (rule_comparisons, "rule and comparison context"),
    "SeniorityByPrimaryReason": (seniority_summary, "primary exclusion reason"),
    "UserSpellCounts": (user_spell_counts, "mutually exclusive user survival group"),
}
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for name, (output_frame, unit) in output_frames.items():
    output_path = OUTPUT_DIR / f"ResultsB06_{name}.csv"
    temporary_output = output_path.with_suffix(".csv.incomplete")
    output_frame.to_csv(temporary_output, index=False)
    restored = pd.read_csv(temporary_output, low_memory=False)
    pd.testing.assert_frame_equal(output_frame, restored, check_dtype=False, rtol=1e-10, atol=1e-12)
    temporary_output.replace(output_path)
    print(f"Saved {len(output_frame):,} rows ({unit}): {output_path}")
print(sample_comparisons.to_string(index=False))
print(rule_comparisons.loc[
    rule_comparisons["context"].eq("conditional_on_all_other_rules"),
    ["rule", "inventor_spell_share_change_pp", "inventor_user_share_change_pp"],
].to_string(index=False))
main.finish_run(run_record, "Local rule, count, share-change, and CSV round-trip checks passed.")

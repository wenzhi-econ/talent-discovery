"""
Task:
    Compute transparent length, structure, corruption, and truncation diagnostics.

Inputs:
(a) data/b_temp_data/C01_PreProcessPostings/B_UniqueNormalized.parquet
    <== Constructed by "B_NormalizeDescriptions.py"

Outputs:
(a) data/b_temp_data/C01_PreProcessPostings/C_TextDiagnostics.parquet

Description of the output:
(1) The output is at normalized-text level, i.e., each row is one unique normalized job
    description.

Notes:
(1) This script measures text condition but never repairs or classifies text.
(2) Parser warnings remain in B_MapFromDescriptionsToNormalized.parquet because they describe raw
    descriptions and can differ across raw descriptions sharing one normalized text.
(3) Classification thresholds belong in the downstream classification script, not in this file.

Run:
    conda run -s -n Talent python codes/C01_PreProcessPostings/C_ComputeTextDiagnostics.py

Wang Wenzhi, with the help of Codex
Time: 2026-08-18
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict, cast

import _Utilities_C01 as util
import pyarrow as pa
import pyarrow.parquet as pq


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify global parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-1. General settings
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


BATCH_SIZE = 20_000
COMPRESSION = "zstd"
IF_OVERWRITE = True
IF_DELETETABLE = True


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-2. Input and output datasets
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


FOLDER_DATA: Final[Path] = util.FOLDER_DATA
INPUT: Final[Path] = FOLDER_DATA / "B_UniqueNormalized.parquet"
OUTPUT: Final[Path] = FOLDER_DATA / "C_TextDiagnostics.parquet"
TEMP_TABLE: Final[Path] = FOLDER_DATA / "C_Table.db"

SCHEMA: Final[pa.Schema] = pa.schema(
    [
        ("normalized_hash", pa.string(), False),
        ("count_character", pa.int64(), False),
        ("count_alphanumeric", pa.int64(), False),
        ("count_word", pa.int64(), False),
        ("count_line", pa.int64(), False),
        ("count_block", pa.int64(), False),
        ("count_heading", pa.int64(), False),
        ("count_paragraph", pa.int64(), False),
        ("count_list_item", pa.int64(), False),
        ("count_source_line", pa.int64(), False),
        ("count_table_row", pa.int64(), False),
        ("count_informative_unit", pa.int64(), False),
        ("count_url", pa.int64(), False),
        ("count_email", pa.int64(), False),
        ("count_replacement_character", pa.int64(), False),
        ("share_replacement_character", pa.float64(), False),
        ("count_nul_character", pa.int64(), False),
        ("share_non_latin_letter", pa.float64(), False),
        ("count_sentence_boundary", pa.int64(), False),
        ("flag_run_on_text", pa.bool_(), False),
        ("flag_possible_truncation", pa.bool_(), False),
        ("flag_final_block_is_heading", pa.bool_(), False),
        ("flag_final_text_ends_with_leadin", pa.bool_(), False),
        ("flag_unmatched_open_delimiter", pa.bool_(), False),
        ("flag_first_block_starts_lowercase", pa.bool_(), False),
    ]
)

LABEL: Final[dict[str, str]] = {
    "normalized_hash": "ID for normalized job descriptions",
    "count_character": "Number of visible characters, excluding inserted block separators",
    "count_alphanumeric": "Number of Unicode alphanumeric characters",
    "count_word": "Number of words under the Unicode-aware word proxy",
    "count_line": "Number of lines in the readable normalized representation",
    "count_block": "Number of normalized text blocks",
    "count_heading": "Number of heading blocks",
    "count_paragraph": "Number of paragraph blocks",
    "count_list_item": "Number of list-item blocks",
    "count_source_line": "Number of preserved plain-text source-line blocks",
    "count_table_row": "Number of table-row blocks",
    "count_informative_unit": "Number of informative blocks or paragraph units",
    "count_url": "Number of URL matches",
    "count_email": "Number of email-address matches",
    "count_replacement_character": "Number of Unicode replacement characters",
    "share_replacement_character": "Share of visible characters that are replacements",
    "count_nul_character": "Number of NUL characters remaining after normalization",
    "share_non_latin_letter": "Share of alphabetic characters outside the Latin script",
    "count_sentence_boundary": "Number of terminal-punctuation sentence-boundary proxies",
    "flag_run_on_text": "Indicator for long text with suspiciously little structure",
    "flag_possible_truncation": "Indicator for any strong truncation clue",
    "flag_final_block_is_heading": "Indicator that the final block is a heading",
    "flag_final_text_ends_with_leadin": "Indicator that final text ends with a lead-in",
    "flag_unmatched_open_delimiter": "Indicator for an unmatched opening delimiter",
    "flag_first_block_starts_lowercase": "Indicator for a lowercase first content block",
}


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-3. Useful settings for type hinting
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


T_BlockType: TypeAlias = Literal[
    "HEADING",
    "LIST_ITEM",
    "PARAGRAPH",
    "SOURCE_LINE",
    "TABLE_ROW",
]


class T_NormalizedDescription(TypedDict):
    """Fields read for one unique normalized description."""

    normalized_hash: str
    normalized: str
    normalized_in_json: str


class T_NormalizedBlock(TypedDict):
    """Canonical fields for one normalized text block."""

    block_type: T_BlockType
    heading_level: int | None
    text: str


class T_Diagnostics(TypedDict):
    """Diagnostic fields written for one unique normalized description."""

    normalized_hash: str
    count_character: int
    count_alphanumeric: int
    count_word: int
    count_line: int
    count_block: int
    count_heading: int
    count_paragraph: int
    count_list_item: int
    count_source_line: int
    count_table_row: int
    count_informative_unit: int
    count_url: int
    count_email: int
    count_replacement_character: int
    share_replacement_character: float
    count_nul_character: int
    share_non_latin_letter: float
    count_sentence_boundary: int
    flag_run_on_text: bool
    flag_possible_truncation: bool
    flag_final_block_is_heading: bool
    flag_final_text_ends_with_leadin: bool
    flag_unmatched_open_delimiter: bool
    flag_first_block_starts_lowercase: bool


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-4. Diagnostic patterns and characters
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


WORD_RE: Final[re.Pattern[str]] = re.compile(
    r"[^\W_]+(?:['’-][^\W_]+)*",
    re.UNICODE,
)
BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"[.!?](?:[\"')\]]+)?\s+")
FINAL_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"[.!?](?:[\"')\]]+)?\s*$")
LOWER_UPPER_RE: Final[re.Pattern[str]] = re.compile(r"[a-z][A-Z]")
URL_RE: Final[re.Pattern[str]] = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
    re.IGNORECASE,
)
LEADIN_RE: Final[re.Pattern[str]] = re.compile(
    r"(?:responsibilities include|requirements include|including|such as)\s*:?$",
    re.IGNORECASE,
)
REPLACEMENT_CHARACTER: Final[str] = "\ufffd"
NUL_CHARACTER: Final[str] = "\x00"


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Prepare for data transformation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-1. Record the start of execution
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


time_start = util.record_time()
print("=" * 100)
print(f"Start running {util.relative_path(Path(__file__))} at {time_start}")
print("=" * 100)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-2. Prepare the temporary output and reconciliation table
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


temp_output: Path = util.prepare_output(outpath=OUTPUT, overwrite=IF_OVERWRITE)
writer = pq.ParquetWriter(temp_output, SCHEMA, compression=COMPRESSION)

sql_create_processed_table = R"""
CREATE TABLE tab_processed (
    normalized_hash TEXT PRIMARY KEY
);
"""

con = util.create_sqlite_table(
    tabpath=TEMP_TABLE,
    sql_command=sql_create_processed_table,
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Compute diagnostics for unique normalized descriptions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Define helper functions for diagnostic construction
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def load_normalized_blocks(value: str) -> list[T_NormalizedBlock]:
    """
    Decode the canonical JSON representation of one normalized description.
    """
    blocks = json.loads(value)
    if not isinstance(blocks, list):
        raise TypeError("normalized_in_json must decode to a list")
    return cast(list[T_NormalizedBlock], blocks)


def calculate_character_diagnostics(
    blocks: list[T_NormalizedBlock],
) -> tuple[int, int, int, float]:
    """
    Calculate visible-character, alphanumeric, informative-unit, and script diagnostics.

    These measures share one character scan because repeating Unicode tests is expensive for long
    descriptions. ASCII alphabetic characters are Latin by construction, so Unicode-name lookup is
    needed only for non-ASCII letters.
    """
    count_character = 0
    count_alphanumeric = 0
    count_informative = 0
    count_letter = 0
    count_latin = 0
    for block in blocks:
        text = block["text"]
        count_character += len(text)
        count_block_alphanumeric = 0
        for character in text:
            if character.isalpha():
                count_alphanumeric += 1
                count_block_alphanumeric += 1
                count_letter += 1
                count_latin += character.isascii() or ("LATIN" in unicodedata.name(character, ""))
            elif character.isalnum():
                count_alphanumeric += 1
                count_block_alphanumeric += 1
        if count_block_alphanumeric < 20:
            continue
        if block["block_type"] == "PARAGRAPH" and len(text) > 300:
            count_informative += max(1, len(BOUNDARY_RE.findall(text)))
        else:
            count_informative += 1
    share_non_latin = 1.0 - count_latin / count_letter if count_letter else 0.0
    return count_character, count_alphanumeric, count_informative, share_non_latin


def has_unmatched_open_delimiter(text: str) -> bool:
    """
    Return whether an opening parenthesis, bracket, or brace remains unmatched.
    """
    cleaned = URL_RE.sub("", text)
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in cleaned:
        if character in "([{":
            stack.append(character)
        elif character in pairs and stack and stack[-1] == pairs[character]:
            stack.pop()
    return bool(stack)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Read normalized descriptions and write diagnostics in batches
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_insert_processed = R"""
INSERT INTO tab_processed (normalized_hash)
VALUES (?);
"""

input_file = pq.ParquetFile(INPUT)
n_diagnostics: int = 0
n_nul_characters: int = 0

for batch in input_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["normalized_hash", "normalized", "normalized_in_json"],
):
    output_rows: list[T_Diagnostics] = []
    processed_rows: list[tuple[str]] = []

    for record in cast(list[T_NormalizedDescription], batch.to_pylist()):
        normalized_hash = record["normalized_hash"]
        normalized = record["normalized"]
        blocks = load_normalized_blocks(record["normalized_in_json"])
        if not blocks:
            raise ValueError(f"Normalized description has no blocks: {normalized_hash}")
        if normalized != "\n\n".join(block["text"] for block in blocks):
            raise RuntimeError(f"Readable text and canonical blocks disagree: {normalized_hash}")

        (
            count_character,
            count_alphanumeric,
            count_informative_unit,
            share_non_latin_letter,
        ) = calculate_character_diagnostics(blocks)
        count_sentence_boundary = len(BOUNDARY_RE.findall(normalized)) + int(
            bool(FINAL_BOUNDARY_RE.search(normalized))
        )
        count_replacement_character = normalized.count(REPLACEMENT_CHARACTER)
        count_nul_character = normalized.count(NUL_CHARACTER)

        final_block = blocks[-1]
        first_block = blocks[0]
        first_alphabetic = re.search(r"[A-Za-z]", first_block["text"])
        flag_final_block_is_heading = final_block["block_type"] == "HEADING"
        flag_final_text_ends_with_leadin = bool(
            final_block["text"].rstrip().endswith(":") or LEADIN_RE.search(final_block["text"])
        )
        flag_unmatched_open_delimiter = has_unmatched_open_delimiter(normalized)
        flag_first_block_starts_lowercase = bool(
            first_alphabetic
            and first_alphabetic.group().islower()
            and first_block["block_type"] not in {"LIST_ITEM", "SOURCE_LINE"}
        )
        flag_possible_truncation = (
            flag_final_block_is_heading
            or flag_final_text_ends_with_leadin
            or flag_unmatched_open_delimiter
        )
        flag_run_on_text = (
            count_character >= 600
            and len(blocks) <= 2
            and count_sentence_boundary <= 2
            and len(LOWER_UPPER_RE.findall(normalized)) >= 2
        )

        output_rows.append(
            {
                "normalized_hash": normalized_hash,
                "count_character": count_character,
                "count_alphanumeric": count_alphanumeric,
                "count_word": len(WORD_RE.findall(normalized)),
                "count_line": len(normalized.splitlines()),
                "count_block": len(blocks),
                "count_heading": sum(b["block_type"] == "HEADING" for b in blocks),
                "count_paragraph": sum(b["block_type"] == "PARAGRAPH" for b in blocks),
                "count_list_item": sum(b["block_type"] == "LIST_ITEM" for b in blocks),
                "count_source_line": sum(b["block_type"] == "SOURCE_LINE" for b in blocks),
                "count_table_row": sum(b["block_type"] == "TABLE_ROW" for b in blocks),
                "count_informative_unit": count_informative_unit,
                "count_url": len(URL_RE.findall(normalized)),
                "count_email": len(EMAIL_RE.findall(normalized)),
                "count_replacement_character": count_replacement_character,
                "share_replacement_character": (count_replacement_character / count_character),
                "count_nul_character": count_nul_character,
                "share_non_latin_letter": share_non_latin_letter,
                "count_sentence_boundary": count_sentence_boundary,
                "flag_run_on_text": flag_run_on_text,
                "flag_possible_truncation": flag_possible_truncation,
                "flag_final_block_is_heading": flag_final_block_is_heading,
                "flag_final_text_ends_with_leadin": flag_final_text_ends_with_leadin,
                "flag_unmatched_open_delimiter": flag_unmatched_open_delimiter,
                "flag_first_block_starts_lowercase": flag_first_block_starts_lowercase,
            }
        )
        processed_rows.append((normalized_hash,))
        n_nul_characters += count_nul_character

    con.executemany(sql_insert_processed, processed_rows)
    con.commit()
    writer.write_table(pa.Table.from_pylist(output_rows, schema=SCHEMA))
    n_diagnostics += len(output_rows)
    print(f"Processed unique normalized descriptions: {n_diagnostics:,}")

writer.close()
print(f"Finished writing to the temp dataset: {util.relative_path(temp_output)}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Do simple diagnostics and save the final dataset
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-1. Do simple diagnostics
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


n_unique_hashes: int = con.execute("SELECT COUNT(*) FROM tab_processed;").fetchone()[0]
con.close()
if IF_DELETETABLE:
    # The database file will be removed from the disk if ``IF_DELETETABLE``.
    TEMP_TABLE.unlink()

assert n_diagnostics == input_file.metadata.num_rows
assert n_unique_hashes == n_diagnostics
assert pq.ParquetFile(temp_output).metadata.num_rows == n_diagnostics
assert n_nul_characters == 0


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-2. Save the final dataset
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


util.publish_output(temp_output=temp_output, final_output=OUTPUT)
print(f"Saved the final dataset: {util.relative_path(OUTPUT)}.")
print(f"Variables in the dataset: {LABEL}.")
print(f"In total, there are diagnostics for {n_diagnostics:,} normalized descriptions.")

time_end = util.record_time()
print("=" * 100)
print(f"Finished running {util.relative_path(Path(__file__))} at {time_end}")
print(f"Time used: {time_end - time_start}")
print("=" * 100)

"""
Task:
    Parse raw description texts (with HTML tags) into normalized document representations.

Inputs:
(a) data/b_temp_data/C01_PreProcessPostings/A_UniqueDescriptions.parquet
    <== Constructed by "A_HashDescriptions.py"

Outputs:
(a) data/b_temp_data/C01_PreProcessPostings/B_MapFromDescriptionsToNormalized.parquet
(b) data/b_temp_data/C01_PreProcessPostings/B_UniqueNormalized.parquet
(c) data/b_temp_data/C01_PreProcessPostings/B_ParsedBlocks.parquet

Description of the outputs:
(1) Output (a) is at description level, i.e., each row is one unique job description.
(2) Output (b) is at normalized text level, i.e., each row is one unique normalized job description.
(3) Output (c) is at normalized-block level, i.e., it decomposes one normalized job description into
    multiple blocks and store each normalized-block in one row.

Run:
    conda run -s -n Talent python codes/C01_PreProcessPostings/B_NormalizeDescriptions.py

Wang Wenzhi, with the help of Codex
Time: 2026-08-18
"""

import hashlib
import html
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, TypeAlias, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _Utilities_C01 as util
from B01_Util_ParsePlainTexts import (
    T_BlockType,
    T_ParsedBlocks,
    normalize_block_text,
    parse_plain_text_blocks,
)
from B02_Util_ParseHTMLContents import (
    detect_recognized_html,
    fallback_if_lxml_empty,
    remove_noncontent_elements,
    walk_html_blocks,
)


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
INPUT: Final[Path] = FOLDER_DATA / "A_UniqueDescriptions.parquet"
OUT_MAP: Final[Path] = FOLDER_DATA / "B_MapFromDescriptionsToNormalized.parquet"
OUT_NORMALIZED: Final[Path] = FOLDER_DATA / "B_UniqueNormalized.parquet"
OUT_BLOCKS: Final[Path] = FOLDER_DATA / "B_ParsedBlocks.parquet"
TEMP_TABLE: Final[Path] = FOLDER_DATA / "B_Table.db"

# Output (a): A map from descriptions to normalized descriptions
LABEL_MAP: Final[dict[str, str]] = {
    "description_hash": "ID for raw job descriptions",
    "normalized_hash": "ID for normalized job descriptions",
    "description": "Raw job descriptions",
    "normalized": "Readable normalized job descriptions",
    "normalized_in_json": "Deterministic JSON defining each normalized description",
    "parse_status": (
        "Outcome of parsing each raw description "
        "(5 categories: 'NO_VISIBLE_TEXT', 'PARSED_HTML_LXML', 'PARSED_HTML_STDLIB_FALLBACK', 'PARSED_PLAIN_TEXT', 'PARSER_FAILURE')"
    ),
    "parser_used": (
        "Parser used for each raw description (3 categories: 'html.parser', 'lxml', 'plain_text')"
    ),
    "parser_warning_codes": "Nonfatal warnings recorded while parsing",
}
SCHEMA_MAP: Final[pa.Schema] = pa.schema(
    [
        ("description_hash", pa.string(), False),
        ("normalized_hash", pa.string()),
        ("description", pa.large_string(), False),
        ("normalized", pa.large_string()),
        ("normalized_in_json", pa.large_string()),
        ("parse_status", pa.string(), False),
        ("parser_used", pa.string()),
        ("parser_warning_codes", pa.list_(pa.string())),
    ]
)

# Output (b): Normalized descriptions
LABEL_NORMALIZED: Final[dict[str, str]] = {
    "normalized_hash": "ID for normalized job descriptions",
    "normalized": "Readable normalized job descriptions",
    "normalized_in_json": "Deterministic JSON defining each normalized description",
    "normalized_multiplicity": "Number of postings represented by each normalized description",
    "normalized_multiplicity_in_description": (
        "Number of distinct raw descriptions sharing each normalized description"
    ),
}
SCHEMA_NORMALIZED: Final[pa.Schema] = pa.schema(
    [
        ("normalized_hash", pa.string(), False),
        ("normalized", pa.large_string(), False),
        ("normalized_in_json", pa.large_string(), False),
        ("normalized_multiplicity", pa.int64(), False),
        ("normalized_multiplicity_in_description", pa.int64(), False),
    ]
)

# Output (c): Blocks in normalized descriptions
LABEL_BLOCKS: Final[dict[str, str]] = {
    "normalized_hash": "ID for normalized job descriptions",
    "block_id": "ID for blocks within each normalized description",
    "block_order": "Position of each block within its normalized description",
    "block_type": (
        "Semantic category assigned to each block "
        "(5 types: 'HEADING', 'LIST_ITEM', 'PARAGRAPH', 'SOURCE_LINE', 'TABLE_ROW')"
    ),
    "source_tag": "HTML tag supplying each block, if applicable",
    "heading_level": "HTML heading level, if applicable",
    "text": "Normalized visible text in each block",
    "description_hash_example": "Raw description supplying the example block metadata",
}
SCHEMA_BLOCKS: Final[pa.Schema] = pa.schema(
    [
        ("normalized_hash", pa.string(), False),
        ("block_id", pa.string(), False),
        ("block_order", pa.int32(), False),
        ("block_type", pa.string(), False),
        ("source_tag", pa.string()),
        ("heading_level", pa.int8()),
        ("text", pa.large_string(), False),
        ("description_hash_example", pa.string(), False),
    ]
)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-3. Useful settings for type hinting
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


T_ParserUsed: TypeAlias = Literal["html.parser", "lxml", "plain_text"]
T_ParseStatus: TypeAlias = Literal[
    "NO_VISIBLE_TEXT",
    "PARSED_HTML_LXML",
    "PARSED_HTML_STDLIB_FALLBACK",
    "PARSED_PLAIN_TEXT",
    "PARSER_FAILURE",
]


class T_Descriptions(TypedDict):
    """Fields read from one unique raw description."""

    description_hash: str
    description: str
    description_multiplicity: int


class T_CanonicalBlocks(TypedDict):
    """Block fields that define a normalized-description identity."""

    block_type: T_BlockType
    heading_level: int | None
    text: str


class T_Maps(TypedDict):
    """Fields written to the raw-to-normalized description map."""

    description_hash: str
    normalized_hash: str | None
    description: str
    normalized: str | None
    normalized_in_json: str | None
    parse_status: T_ParseStatus
    parser_used: T_ParserUsed | None
    parser_warning_codes: list[str]


class T_Normalized(TypedDict):
    """Fields written for one unique normalized description."""

    normalized_hash: str
    normalized: str
    normalized_in_json: str
    normalized_multiplicity: int
    normalized_multiplicity_in_description: int


class T_Blocks(TypedDict):
    """Fields written for one parsed block."""

    normalized_hash: str
    block_id: str
    block_order: int
    block_type: T_BlockType
    source_tag: str | None
    heading_level: int | None
    text: str
    description_hash_example: str


T_ParseResult: TypeAlias = tuple[
    list[T_ParsedBlocks],
    T_ParseStatus,
    T_ParserUsed | None,
    list[str],
]
T_NormalizedSqlRow: TypeAlias = tuple[str, str, str, int, int, str, str]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 1. Prepare for data transformation
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


time_start = util.record_time()
util.report_status(text=f"Start running {Path(__file__).name} at {time_start}")

temp_map: Path = util.prepare_output(outpath=OUT_MAP, overwrite=IF_OVERWRITE)
temp_normalized: Path = util.prepare_output(outpath=OUT_NORMALIZED, overwrite=IF_OVERWRITE)
temp_blocks: Path = util.prepare_output(outpath=OUT_BLOCKS, overwrite=IF_OVERWRITE)

writer_map = pq.ParquetWriter(temp_map, SCHEMA_MAP, compression=COMPRESSION)
writer_normalized = pq.ParquetWriter(
    temp_normalized,
    SCHEMA_NORMALIZED,
    compression=COMPRESSION,
)
writer_blocks = pq.ParquetWriter(temp_blocks, SCHEMA_BLOCKS, compression=COMPRESSION)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 2. Define helper functions to parse job descriptions
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. The final parser for descriptions (plain texts and HTML contents)
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

for function in (
    normalize_block_text,
    parse_plain_text_blocks,
    detect_recognized_html,
    remove_noncontent_elements,
    walk_html_blocks,
    fallback_if_lxml_empty,
):
    util.richprint_docstring(function)


def parse_description(raw: str) -> T_ParseResult:
    """
    Select the HTML or plain-text route and return blocks and parser diagnostics.
    """
    warnings: list[str] = ["NUL_REMOVED"] if "\x00" in raw else []
    try:
        if detect_recognized_html(raw):
            soup = BeautifulSoup(raw, "lxml")
            remove_noncontent_elements(soup)
            blocks = walk_html_blocks(soup)
            parser_used: T_ParserUsed = "lxml"
            status: T_ParseStatus = "PARSED_HTML_LXML"
            if not blocks and normalize_block_text(html.unescape(raw)):
                blocks = fallback_if_lxml_empty(raw, blocks)
                parser_used = "html.parser"
                status = "PARSED_HTML_STDLIB_FALLBACK"
                warnings.append("LXML_EMPTY_FALLBACK_USED")
        else:
            blocks = parse_plain_text_blocks(raw)
            parser_used = "plain_text"
            status = "PARSED_PLAIN_TEXT"
        if not blocks:
            status = "NO_VISIBLE_TEXT"
        return blocks, status, parser_used, warnings
    except (ValueError, TypeError, UnicodeError) as error:
        warning = f"PARSER_{type(error).__name__.upper()}"
        return [], "PARSER_FAILURE", None, [warning]


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Define helper functions for constructing normalized identities
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def dump_json(value: object) -> str:
    """
    Serialize a JSON-compatible value in a deterministic compact form.
    """
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def generate_hash_normalized_text(value: str) -> str:
    """
    Generate a hash ID for a normalized description.
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_normalized_text(blocks: Sequence[T_ParsedBlocks]) -> str:
    """
    Join ordered block texts into a human-readable representation.
    """
    return "\n\n".join(block["text"] for block in blocks)


def build_canonical_blocks(
    blocks: Sequence[T_ParsedBlocks],
) -> list[T_CanonicalBlocks]:
    """
    Retain only the block fields that define a normalized identity.
    """
    return [
        {
            "block_type": block["block_type"],
            "heading_level": block["heading_level"],
            "text": block["text"],
        }
        for block in blocks
    ]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 3. Parse descriptions and construct normalized identities
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-1. Initialize the temporary table
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_create_normalized_table = R"""
CREATE TABLE tab_normalized (
    normalized_hash TEXT PRIMARY KEY,
    normalized TEXT NOT NULL,
    normalized_in_json TEXT NOT NULL,
    normalized_multiplicity INTEGER NOT NULL,
    normalized_multiplicity_in_description INTEGER NOT NULL,
    description_hash_example TEXT NOT NULL,
    example_document_json TEXT NOT NULL
);
"""

con = util.create_sqlite_table(
    tabpath=TEMP_TABLE,
    sql_command=sql_create_normalized_table,
)

input_file = pq.ParquetFile(INPUT)

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-3-2. Read descriptions in batches and construct normalized identities
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>

n_descriptions: int = 0
n_descriptions_with_normalized: int = 0
n_postings_with_normalized: int = 0
n_warnings: int = 0

for batch in input_file.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["description_hash", "description", "description_multiplicity"],
):
    out_map_rows: list[T_Maps] = []
    for record in cast(list[T_Descriptions], batch.to_pylist()):
        description_hash = record["description_hash"]
        description = record["description"]
        description_multiplicity = record["description_multiplicity"]

        blocks, status, parser_used, warnings = parse_description(description)
        n_warnings += len(warnings)

        normalized_hash: str | None = None
        normalized_in_json: str | None = None
        normalized: str | None = None

        if blocks:
            canonical_blocks = build_canonical_blocks(blocks)
            normalized_in_json = dump_json(canonical_blocks)
            normalized_hash = generate_hash_normalized_text(normalized_in_json)
            normalized = build_normalized_text(blocks)

            sql_select_normalized_by_hash = R"""
            SELECT normalized_in_json
            FROM tab_normalized
            WHERE normalized_hash = ?;
            """
            existing: tuple[str] | None = con.execute(
                sql_select_normalized_by_hash,
                (normalized_hash,),
            ).fetchone()
            if existing is None:
                sql_insert_normalized = R"""
                INSERT INTO tab_normalized (
                    normalized_hash,
                    normalized,
                    normalized_in_json,
                    normalized_multiplicity,
                    normalized_multiplicity_in_description,
                    description_hash_example,
                    example_document_json
                )
                VALUES (?, ?, ?, ?, 1, ?, ?);
                """
                con.execute(
                    sql_insert_normalized,
                    (
                        normalized_hash,
                        normalized,
                        normalized_in_json,
                        description_multiplicity,
                        description_hash,
                        dump_json(blocks),
                    ),
                )
            else:
                if existing[0] != normalized_in_json:
                    raise RuntimeError("Normalized SHA-256 collision detected")

                sql_update_normalized_multiplicity = R"""
                UPDATE tab_normalized
                SET
                    normalized_multiplicity = normalized_multiplicity + ?,
                    normalized_multiplicity_in_description = normalized_multiplicity_in_description + 1
                WHERE normalized_hash = ?;
                """
                con.execute(
                    sql_update_normalized_multiplicity,
                    (description_multiplicity, normalized_hash),
                )
            n_descriptions_with_normalized += 1
            n_postings_with_normalized += description_multiplicity

        out_map_rows.append(
            {
                "description_hash": description_hash,
                "normalized_hash": normalized_hash,
                "description": description,
                "normalized": normalized,
                "normalized_in_json": normalized_in_json,
                "parse_status": status,
                "parser_used": parser_used,
                "parser_warning_codes": warnings,
            }
        )

    writer_map.write_table(pa.Table.from_pylist(out_map_rows, schema=SCHEMA_MAP))
    n_descriptions += len(out_map_rows)
    con.commit()
    print(f"Parsed unique descriptions: {n_descriptions:,}")

writer_map.close()
print(f"Finished writing to the temp dataset: {util.relative_path(temp_map)}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 4. Write unique normalized descriptions and parsed blocks
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-4-1. Read normalized identities from the temporary table
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_select_normalized = R"""
SELECT
    normalized_hash,
    normalized,
    normalized_in_json,
    normalized_multiplicity,
    normalized_multiplicity_in_description,
    description_hash_example,
    example_document_json
FROM tab_normalized
ORDER BY normalized_hash;
"""

cursor = con.execute(sql_select_normalized)

n_normalized: int = 0
n_blocks: int = 0


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-4-2. Write normalized identities and their example blocks in batches
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


while rows := cursor.fetchmany(BATCH_SIZE):
    normalized_rows: list[T_Normalized] = []
    block_rows: list[T_Blocks] = []

    for row in cast(list[T_NormalizedSqlRow], rows):
        normalized_record: T_Normalized = {
            "normalized_hash": row[0],
            "normalized": row[1],
            "normalized_in_json": row[2],
            "normalized_multiplicity": row[3],
            "normalized_multiplicity_in_description": row[4],
        }
        normalized_rows.append(normalized_record)

        example_blocks = cast(list[T_ParsedBlocks], json.loads(row[6]))
        for block_order, block in enumerate(example_blocks, start=1):
            block_rows.append(
                {
                    "normalized_hash": normalized_record["normalized_hash"],
                    "block_id": f"b{block_order:04d}",
                    "block_order": block_order,
                    "block_type": block["block_type"],
                    "source_tag": block["source_tag"],
                    "heading_level": block["heading_level"],
                    "text": block["text"],
                    "description_hash_example": row[5],
                }
            )

    writer_normalized.write_table(pa.Table.from_pylist(normalized_rows, schema=SCHEMA_NORMALIZED))
    if block_rows:
        writer_blocks.write_table(pa.Table.from_pylist(block_rows, schema=SCHEMA_BLOCKS))
    n_normalized += len(normalized_rows)
    n_blocks += len(block_rows)

writer_normalized.close()
writer_blocks.close()
print(f"\nFinished writing to the temp dataset: {util.relative_path(temp_normalized)}.")
print(f"Finished writing to the temp dataset: {util.relative_path(temp_blocks)}.")


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 5. Do simple diagnostics and save the final datasets
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-5-1. Do simple diagnostics
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


sql_summarize_normalized = R"""
SELECT
    COUNT(*),
    COALESCE(SUM(normalized_multiplicity), 0),
    COALESCE(SUM(normalized_multiplicity_in_description), 0)
FROM tab_normalized;
"""

table_summary = cast(
    tuple[int, int, int],
    con.execute(sql_summarize_normalized).fetchone(),
)
con.close()
if IF_DELETETABLE:
    # The database file will be removed from the disk if ``IF_DELETETABLE``.
    TEMP_TABLE.unlink()

assert n_descriptions == input_file.metadata.num_rows
assert pq.ParquetFile(temp_map).metadata.num_rows == n_descriptions
assert pq.ParquetFile(temp_normalized).metadata.num_rows == n_normalized
assert pq.ParquetFile(temp_blocks).metadata.num_rows == n_blocks
assert table_summary[0] == n_normalized
assert table_summary[1] == n_postings_with_normalized
assert table_summary[2] == n_descriptions_with_normalized


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-5-2. Save the final datasets
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


util.publish_output(temp_output=temp_map, final_output=OUT_MAP)
print(f"\nSaved the final dataset: {util.relative_path(OUT_MAP)}.")
print(f"Variables in the dataset: {LABEL_MAP}.")

util.publish_output(temp_output=temp_normalized, final_output=OUT_NORMALIZED)
print(f"\nSaved the final dataset: {util.relative_path(OUT_NORMALIZED)}.")
print(f"Variables in the dataset: {LABEL_NORMALIZED}.")

util.publish_output(temp_output=temp_blocks, final_output=OUT_BLOCKS)
print(f"\nSaved the final dataset: {util.relative_path(OUT_BLOCKS)}.")
print(f"Variables in the dataset: {LABEL_BLOCKS}.")

print(
    f"\nIn total, there are {n_descriptions:,} descriptions, {n_normalized:,} normalized "
    f"descriptions, and {n_blocks:,} blocks."
)
print(f"Parser warnings: {n_warnings:,}.")


time_end = util.record_time()
util.report_status(
    text=(
        f"Finished running {Path(__file__).name} at {time_end}\nTime used: {time_end - time_start}"
    )
)

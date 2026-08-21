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
import re
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Literal, NamedTuple, TypeAlias, TypedDict, cast

import pyarrow as pa
import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from bs4.element import CData, NavigableString, Tag

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _Utilities_C01 as util


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Step 0. Specify global parameters
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

# impt: The following are the important rules during the normalization process.

BULLET_PATTERNS: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:[•·▪◦*-]|\d+[.)]|\(\d+\)|[A-Za-z][.)]|\([A-Za-z]\)|[ivx]{2,}[.)]|\([ivx]{2,}\))\s+"
)
HTML_TAGS: Final[str] = (
    "html|body|h1|h2|h3|h4|h5|h6|p|div|section|article|ul|ol|li|table|thead|"
    "tbody|tfoot|tr|td|th|br|span|a|em|i|u|strong|b|blockquote|pre|dl|dt|dd|"
    "script|style|noscript|iframe"
)
HTML_RE: Final[re.Pattern[str]] = re.compile(
    rf"</?(?:{HTML_TAGS})\b[^>]*>",
    re.IGNORECASE,
)
REMOVAL_CONTENTS: Final[list[str]] = ["script", "style", "noscript", "iframe"]
BLOCK_TAGS: Final[frozenset[str]] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "tr"})
HTML_CONTAINER_TAGS: Final[frozenset[str]] = frozenset(
    {
        "html",
        "head",
        "body",
        "div",
        "section",
        "article",
        "ul",
        "ol",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "blockquote",
        "pre",
        "dl",
        "dt",
        "dd",
    }
)
SHORT_TEXT_THRESHOLD = 120


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-0-1. General settings
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


BATCH_SIZE = 20_000
COMPRESSION = "zstd"
UNICODE_NORMALIZATION = "NFC"
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


T_BlockType: TypeAlias = Literal[
    "HEADING",
    "LIST_ITEM",
    "PARAGRAPH",
    "SOURCE_LINE",
    "TABLE_ROW",
]
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


class T_ParsedBlocks(TypedDict):
    """One ordered unit of visible text extracted from a job description."""

    block_type: T_BlockType
    heading_level: int | None
    text: str
    source_tag: str | None


class T_BlockOwner(NamedTuple):
    """Rules for collecting one continuous run of HTML text into a block."""

    block_type: T_BlockType
    heading_level: int | None
    source_tag: str | None
    locks_inline_markup: bool


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
# >> S-2-1. Define helper functions for parsing plain texts
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def normalize_block_text(value: str) -> str:
    """
    Standardize Unicode and whitespace while preserving meaningful punctuation.

    Notes:
    (1) "\x00" is the null character.
    (2) "\xa0" is the hexadecimal representation of a non-breaking space (NBSP).
    (3) "\r\n" represents a Windows newline sequence.
        (a) "\r" (Carriage Return or CR) moves the cursor back to the beginning of the current line.
        (b) "\n" (Line Feed or LF) moves the cursor down to the next line (ASCII 10).
    (4) "\t" (Horizontal Tab) moves the cursor forward to the next tab stop.
    (5) "\f" (Form Feed / Page Break) forces a printer or screen to advance to the next page or
        clear the screen.
    (6) "\v" (Vertical Tab) moves the cursor down to the next vertical tab stop without returning to
        the start of the line.
    """
    value = value.replace("\x00", "")
    value = unicodedata.normalize(UNICODE_NORMALIZATION, value)
    value = value.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(pattern=r"[\t\f\v ]+", repl=" ", string=value).strip()
    return value


def parse_plain_text_blocks(
    value: str,
    bullet_pattern: re.Pattern[str] = BULLET_PATTERNS,
) -> list[T_ParsedBlocks]:
    R"""
    Convert each nonempty plain-text line into a "SOURCE_LINE" or "LIST_ITEM" block.

    Notes:
    (1) For plain-text descriptions (i.e., those without HTML tags), I extract bullet items from
        them if a line starts with a pattern defined by the ``bullet_pattern``; otherwise, it is
        classified simply as a "SOURCE_LINE".
    (2) Current default pattern for "LIST_ITEM" is:
        r"^\s*(?:[•·▪◦*-]|\d+[.)]|\(\d+\)|[A-Za-z][.)]|\([A-Za-z]\)|[ivx]{2,}[.)]|\([ivx]{2,}\))\s+"

    Explanations of the default pattern:
    (A) ``^`` means "the start of the line."
        (a) A bullet-like marker found later in a sentence does not cause that sentence to be
            classified as a list item.
    (B) ``\s*`` allows zero or more whitespace characters before the marker.
        (a) ``\s`` means whitespace, while ``*`` means "zero or more of the preceding item."
        (b) This makes the pattern tolerate an indented list.
        (c) This pattern is for robustness purposes because, in the current function, leading
            whitespace is already removed by ``normalize_block_text``.
    (C) ``(?: ... )`` groups the possible marker formats together.
        (a) The symbols ``?:`` tell Python that the matched marker does not need to be saved as a
            separate result.
        (b) The vertical bars (``|``) inside the group mean "or," so any of the seven formats
            described next is accepted.
    (D) ``[•·▪◦*-]`` accepts exactly one of the characters inside the square brackets.
        (a) ``•``, ``·``, ``▪``, ``◦``, ``*``, and ``-`` can each serve as a bullet.
        (b) The hyphen is placed last so it means a literal hyphen rather than a range of
            characters.
    (E) ``\d+[.)]`` and ``\(\d+\)`` accept numbered markers.
        (a) In both formats, ``\d+`` means one or more decimal digits.
        (b) In the first format, ``[.)]`` means either a period or a closing parenthesis. Examples
            include ``1.`` and ``1)``.
        (c) In the second format, ``\(`` and ``\)`` mean a literal opening and closing parenthesis,
            respectively. Therefore, it accepts a fully parenthesized number such as ``(1)``.
        (d) When an opening parenthesis is used, the second format requires a closing parenthesis.
            Therefore, an unusual marker such as ``(1.`` is not accepted.
    (F) ``[A-Za-z][.)]`` and ``\([A-Za-z]\)`` accept lettered markers.
        (a) In both formats, ``[A-Za-z]`` means exactly one uppercase or lowercase English letter.
        (b) In the first format, the letter is followed by either a period or a closing parenthesis.
            Examples include ``a.``, ``a)``, ``B.``, and ``B)``.
        (c) In the second format, the letter is enclosed by literal opening and closing parentheses.
            Examples include ``(a)`` and ``(B)``.
        (d) These two formats do not accept multiple letters. General multiple-letter markers such
            as ``aa.`` are not accepted, while lowercase Roman numerals are handled separately by
            the formats described next.
        (e) When an opening parenthesis is used, the second format requires a closing parenthesis.
            Therefore, an unusual marker such as ``(a.`` is not accepted.
    (G) ``[ivx]{2,}[.)]`` and ``\([ivx]{2,}\)`` accept multi-character lowercase Roman-numeral
        markers.
        (a) ``[ivx]`` accepts one lowercase ``i``, ``v``, or ``x``.
        (b) ``{2,}`` requires two or more of those characters. Examples of strings satisfying this
            part include ``ii``, ``iii``, ``iv``, ``viii``, ``xix``, and ``xx``. This is a mandatory
            requirement in this rule, examples like ``(i)`` are covered by rule (F).
        (c) In the first format, the Roman numeral is followed by either a period or a closing
            parenthesis. Examples include ``ii.`` and ``iii)``.
        (d) In the second format, the Roman numeral is enclosed by literal opening and closing
            parentheses. An example is ``(ii)``.
        (e) Multi-character uppercase or mixed-case forms such as ``II.``, ``Ii.``, and ``(IV)``
            are deliberately not accepted.
        (f) Single-letter forms such as ``I.``, ``I)``, and ``(I)`` remain accepted by the general
            lettered-marker formats in rule (F), because they are indistinguishable from other one-
            letter uppercase Roman markers.
        (g) Restricting the allowed characters to ``i``, ``v``, and ``x`` recognizes all lowercase
            Roman numerals ranging up to ``xx``.
    (H) The final ``\s+`` requires one or more whitespace characters after the marker.
        (a) The ``+`` means "one or more of the preceding item."
        (b) This helps prevent text such as ``1.5`` or ``a.com`` from being mistakenly classified as
            a list item.
    (I) Putting the pieces together, the whole pattern means:
        (a) start at the beginning of a line,
        (b) allow optional indentation,
        (c) recognize one of the seven marker formats, and
        (d) require whitespace before the item's actual words.
    """
    # Convert all named and numeric character references (e.g. &gt;, &#62;, &#x3e;) in ``value`` to
    # the corresponding Unicode characters.
    decoded = html.unescape(value)
    # Iterate over individual lines in decoded description texts.
    blocks: list[T_ParsedBlocks] = []
    for line in decoded.splitlines():
        text = normalize_block_text(line)
        # Empty lines are ignored.
        if text:
            block_type: T_BlockType = "LIST_ITEM" if bullet_pattern.match(text) else "SOURCE_LINE"
            blocks.append(
                {
                    "block_type": block_type,
                    "heading_level": None,
                    "text": text,
                    "source_tag": None,
                }
            )
    normalized_text = normalize_block_text(decoded)
    # If nothing is collected by iterating lines but there are contents in ``normalized_text``, put
    # everything as one "PARAGRAPH". This is mainly for robustness purpose as a strong fallback.
    if not blocks and normalized_text:
        blocks.append(
            {
                "block_type": "PARAGRAPH",
                "heading_level": None,
                "text": normalized_text,
                "source_tag": None,
            }
        )
    return blocks


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Define helper functions for parsing HTML contents
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


def detect_recognized_html(
    value: str,
    bullet_pattern: re.Pattern[str] = HTML_RE,
) -> bool:
    """
    Return whether the text contains at least one recognized HTML tag.
    """
    return bool(bullet_pattern.search(value))


def remove_noncontent_elements(soup: BeautifulSoup, removal: list[str] = REMOVAL_CONTENTS) -> None:
    """
    Remove scripts, styles, embedded pages, and other invisible elements.
    """
    for element in soup.find_all(removal):
        element.decompose()


def walk_html_blocks(soup: BeautifulSoup) -> list[T_ParsedBlocks]:
    R"""
    Convert the remaining HTML text into ordered blocks without loss or duplication.

    Notes:
    (1) ``BeautifulSoup`` represents HTML as a tree.
        (a) An HTML element such as ``<p>...</p>`` is represented by a ``Tag`` object.
        (b) The actual words inside a tag are represented by ``NavigableString`` objects, which I
            refer to below as "text nodes."
        (c) Tags can contain other tags and text nodes. These nested objects are called children.
    (2) The lossless invariant is that every remaining text node is visited once, in document order,
        and assigned to exactly one output block.
        (a) No selected tag extracts all descendant text with ``get_text`` and then also visits
            those descendants. This prevents duplicated text.
        (b) No tag is ignored merely because it lacks a recognized semantic role. The traversal
            still visits its children, which prevents loose text from being lost.
    (3) "Lossless" applies after the intended preprocessing and normalization.
        (a) ``remove_noncontent_elements`` normally removes ``script``, ``style``, ``noscript``, and
            ``iframe`` elements before this function is called.
        (b) HTML markup, attributes, and the original whitespace are not retained as text.
        (c) ``normalize_block_text`` standardizes Unicode and whitespace in every retained text
            node.
    (4) HTML supplies the primary block type when it uses an explicit ``h1``--``h6``, ``p``, ``li``,
        or ``tr`` tag. Short bold text outside such a tag is treated as an inferred heading.

    Explanations of the workflow:
    (A) Initialize the output and the pending-text buffer.
        (a) ``blocks`` stores completed output blocks in document order.
        (b) ``pending_fragments`` stores consecutive normalized text nodes that belong to the same
            owner and can therefore be combined into one block.
        (c) ``pending_owner`` records the type, heading level, source tag, and inline-markup rule
            for those pending fragments.
        (d) Here, an "owner" is simply the rule and metadata responsible for one text run. It is not
            another object supplied by BeautifulSoup.
    (B) Define ``flush_pending`` to finish one accumulated block.
        (a) The pending fragments are joined with one space and normalized again.
        (b) A nonempty result is appended with the metadata supplied by ``pending_owner``.
        (c) The buffer and owner are then cleared before later text is collected.
    (C) Define ``append_text`` to assign one text node to one owner.
        (a) Empty text after normalization is ignored.
        (b) Text continues in the same pending block when its owner has not changed.
        (c) When the owner changes, the existing pending block is flushed before the new text is
            collected. This preserves the order of differently structured content.
    (D) Traverse the BeautifulSoup tree recursively with ``visit_node``.
        (a) Recursion means that the function visits a tag and then visits each of its children from
            first to last before returning to the tag's later siblings.
        (b) Ordinary ``NavigableString`` and ``CData`` nodes are sent to ``append_text`` exactly
            once.
        (c) Comments, declarations, and other special string subclasses are ignored, matching the
            main-content behavior of BeautifulSoup's ``get_text`` method.
        (d) A tag normally passes its children to ``visit_node`` even when the tag itself has no
            recognized semantic meaning. Therefore, text inside ``span``, ``a``, ``em``, and custom
            tags is retained.
    (E) Ignore intentionally removed content and handle visual boundaries.
        (a) A tag named in ``REMOVAL_CONTENTS`` and all of its children are ignored defensively.
        (b) ``hr`` always ends the current pending block.
        (c) ``br`` ends a fallback text run, but it does not split text already owned by an explicit
            paragraph, heading, list item, table row, or bold-text block.
    (F) Let explicit HTML block tags determine semantic ownership.
        (a) ``h1`` through ``h6`` create ``HEADING`` owners with numbered heading levels.
        (b) ``p`` creates a ``PARAGRAPH`` owner, ``li`` creates a ``LIST_ITEM`` owner, and ``tr``
            creates a ``TABLE_ROW`` owner.
        (c) Pending text is flushed before and after each explicit tag, making that tag a structural
            boundary.
        (d) Inline markup nested inside an explicit owner cannot create another owner. For example,
            ``<p>Hello <strong>world</strong></p>`` remains one paragraph containing
            ``Hello world``.
        (e) A nested explicit block can still create its own owner. Direct text before and after it
            is preserved as separate blocks belonging to the surrounding explicit tag.
    (G) Infer headings from ``strong`` and ``b`` only when no explicit owner has already claimed
        them.
        (a) Bold text at most ``SHORT_TEXT_THRESHOLD`` characters long becomes an unnumbered
            ``HEADING``. Longer bold text becomes a ``PARAGRAPH``.
        (b) The bold tag becomes a locked owner while its children are traversed. Nested bold tags
            therefore cannot extract the same text again.
        (c) If a bold tag contains an explicit block such as ``li``, the explicit block temporarily
            takes ownership of its own text. This retains the structure without duplicating it.
    (H) Use container tags to delimit otherwise unstructured text.
        (a) ``HTML_CONTAINER_TAGS`` includes common containers such as ``html``, ``body``, ``div``,
            ``section``, ``article``, lists, tables, and description-list containers.
        (b) Outside an explicit owner, entering or leaving one of these tags flushes pending text
            and assigns its loose text to a fallback ``PARAGRAPH`` owned by that container.
        (c) Inside an explicit owner, a nested container keeps the existing owner. This prevents a
            wrapper ``div`` inside an ``li`` from changing list text into a paragraph.
    (I) Start at the document root and finish the final pending block.
        (a) Root-level loose text initially receives a fallback ``PARAGRAPH`` owner with no source
            tag.
        (b) Each top-level child is visited in document order.
        (c) A final call to ``flush_pending`` stores any text remaining after the last child.
        (d) If the tree contains no remaining main-content text, the function returns an empty list.
    (J) Why the workflow is lossless with respect to normalized text.
        (a) Text is collected only from individual text nodes, never from both a tag and its
            descendants.
        (b) Every ordinary text node is reached because unrecognized tags are traversed rather than
            discarded.
        (c) Each text node enters one pending buffer under one owner, and every nonempty buffer is
            eventually flushed once.
        (d) Consequently, joining the returned block texts in order reconstructs the same normalized
            text-node sequence found in the preprocessed BeautifulSoup tree.
    """
    blocks: list[T_ParsedBlocks] = []
    pending_fragments: list[str] = []
    pending_owner: T_BlockOwner | None = None

    def flush_pending() -> None:
        """
        Store the current run of text, if any, and reset its owner.
        """
        nonlocal pending_owner
        if pending_fragments and pending_owner is not None:
            text = normalize_block_text(" ".join(pending_fragments))
            if text:
                blocks.append(
                    {
                        "block_type": pending_owner.block_type,
                        "heading_level": pending_owner.heading_level,
                        "text": text,
                        "source_tag": pending_owner.source_tag,
                    }
                )
        pending_fragments.clear()
        pending_owner = None

    def append_text(value: str, owner: T_BlockOwner) -> None:
        """
        Append one normalized text node to the pending block owned by ``owner``.
        """
        nonlocal pending_owner
        text = normalize_block_text(value)
        if not text:
            return
        if pending_owner != owner:
            flush_pending()
            pending_owner = owner
        pending_fragments.append(text)

    def visit_node(node: object, owner: T_BlockOwner) -> None:
        """
        Visit one BeautifulSoup node and then its children in document order.
        """
        # Check exact types because comments also inherit from ``NavigableString`` but should not
        # become job-description text.
        if type(node) in {NavigableString, CData}:
            append_text(str(node), owner)
            return
        if not isinstance(node, Tag):
            return

        if not isinstance(node.name, str):
            for child in node.children:
                visit_node(child, owner)
            return

        tag_name = node.name.lower()
        if tag_name in REMOVAL_CONTENTS:
            return
        if tag_name == "hr" or (tag_name == "br" and not owner.locks_inline_markup):
            flush_pending()
            return
        if tag_name == "br":
            return

        if tag_name in BLOCK_TAGS:
            flush_pending()
            if tag_name.startswith("h") and len(tag_name) == 2:
                tag_owner = T_BlockOwner("HEADING", int(tag_name[1]), tag_name, True)
            elif tag_name == "li":
                tag_owner = T_BlockOwner("LIST_ITEM", None, tag_name, True)
            elif tag_name == "tr":
                tag_owner = T_BlockOwner("TABLE_ROW", None, tag_name, True)
            else:
                tag_owner = T_BlockOwner("PARAGRAPH", None, tag_name, True)
            for child in node.children:
                visit_node(child, tag_owner)
            flush_pending()
            return

        if tag_name in {"strong", "b"} and not owner.locks_inline_markup:
            flush_pending()
            # This combined text chooses metadata only; individual child nodes supply the output.
            bold_text = normalize_block_text(node.get_text(" ", strip=True))
            bold_type: T_BlockType = (
                "HEADING" if len(bold_text) <= SHORT_TEXT_THRESHOLD else "PARAGRAPH"
            )
            bold_owner = T_BlockOwner(bold_type, None, tag_name, True)
            for child in node.children:
                visit_node(child, bold_owner)
            flush_pending()
            return

        if tag_name in HTML_CONTAINER_TAGS and not owner.locks_inline_markup:
            flush_pending()
            container_owner = T_BlockOwner("PARAGRAPH", None, tag_name, False)
            for child in node.children:
                visit_node(child, container_owner)
            flush_pending()
            return

        for child in node.children:
            visit_node(child, owner)

    root_owner = T_BlockOwner("PARAGRAPH", None, None, False)
    for child in soup.children:
        visit_node(child, root_owner)
    flush_pending()
    return blocks


def fallback_if_lxml_empty(
    raw: str,
    lxml_blocks: list[T_ParsedBlocks],
) -> list[T_ParsedBlocks]:
    """
    Return lxml blocks, or reparse the raw HTML with Python's parser when they are empty.
    """
    if lxml_blocks:
        return lxml_blocks
    fallback = BeautifulSoup(raw, "html.parser")
    remove_noncontent_elements(fallback)
    return walk_html_blocks(fallback)


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-3. The final parser for descriptions (plain texts and HTML contents)
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


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
# >> S-2-4. Define helper functions for constructing normalized identities
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

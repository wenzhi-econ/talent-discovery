"""
Descriptions:
    Type definitions for normalized job descriptions and their parsed blocks.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    None.

To be imported by:
    codes/C01_PreProcessPostings/B_NormalizeDescriptions.py
    codes/C01_PreProcessPostings/B03_Util_ParsePlainTexts.py
    codes/C01_PreProcessPostings/B04_Util_ParseHTMLContents.py

Notes:
(1) This file contains type-only definitions; it does not parse text or set normalization policy.
(2) Runtime classes that hold parsing state remain in the utility module that uses them.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-23
"""

from typing import Literal, TypeAlias, TypedDict

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Type group 1. Parsed job-description blocks
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


T_BlockType: TypeAlias = Literal[
    "HEADING",
    "LIST_ITEM",
    "PARAGRAPH",
    "SOURCE_LINE",
    "TABLE_ROW",
]


class T_ParsedBlocks(TypedDict):
    """
    Describe one ordered unit of visible text extracted from a job description.

    Attributes:
        block_type:
            Semantic role assigned by the plain-text or HTML parser. ``SOURCE_LINE`` is reserved
            for non-list lines from plain-text descriptions.
        heading_level:
            Integer from 1 through 6 for headings created by ``h1`` through ``h6``; otherwise
            ``None``. Inferred unnumbered headings also have ``None``.
        text:
            Visible block text after Unicode and whitespace normalization.
        source_tag:
            Lowercase HTML tag that supplied or inferred the block boundary; ``None`` for plain
            text and loose HTML text outside a structural tag.

    Notes:
    (1) Blocks are ordered by their position in the source description, but order is stored by the
        surrounding list rather than as a field in this dictionary.
    (2) ``source_tag`` is diagnostic metadata. It is excluded from the canonical representation
        used to identify unique normalized descriptions.
    """

    block_type: T_BlockType
    heading_level: int | None
    text: str
    source_tag: str | None


class T_CanonicalBlocks(TypedDict):
    """
    Describe the block fields that define a normalized-description identity.

    Attributes:
        block_type:
            Semantic role retained from the parsed block.
        heading_level:
            Numbered HTML heading level when applicable; otherwise ``None``.
        text:
            Normalized visible text in the block.

    Notes:
    (1) ``source_tag`` is deliberately omitted so equivalent visible structures can share a
        normalized identity even when their HTML tags differ.
    (2) The ordered list of these dictionaries is serialized deterministically before hashing.
    """

    block_type: T_BlockType
    heading_level: int | None
    text: str


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Type group 2. Parser outcomes
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


T_ParserUsed: TypeAlias = Literal["html.parser", "lxml", "plain_text"]
T_ParseStatus: TypeAlias = Literal[
    "NO_VISIBLE_TEXT",
    "PARSED_HTML_LXML",
    "PARSED_HTML_STDLIB_FALLBACK",
    "PARSED_PLAIN_TEXT",
    "PARSER_FAILURE",
]
T_ParseResult: TypeAlias = tuple[
    list[T_ParsedBlocks],
    T_ParseStatus,
    T_ParserUsed | None,
    list[str],
]


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Type group 3. Input and output records
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


class T_Descriptions(TypedDict):
    """
    Describe one unique raw job description read from the input dataset.

    Attributes:
        description_hash:
            Stable identifier inherited from the preceding description-hashing step.
        description:
            Raw description text, which may contain HTML, plain text, or no visible content.
        description_multiplicity:
            Number of original postings represented by this unique raw description.
    """

    description_hash: str
    description: str
    description_multiplicity: int


class T_Maps(TypedDict):
    """
    Describe one row in the raw-to-normalized description map.

    Attributes:
        description_hash:
            Identifier for the raw description.
        normalized_hash:
            Identifier for the normalized description, or ``None`` when no normalized identity is
            available.
        description:
            Original raw description.
        normalized:
            Human-readable normalized text, or ``None`` when parsing yields no blocks.
        normalized_in_json:
            Deterministic canonical block representation, or ``None`` when parsing yields no
            blocks.
        parse_status:
            Final route outcome, including the no-visible-text and parser-failure cases.
        parser_used:
            Parser backend that produced the result, or ``None`` after a handled parser failure.
        parser_warning_codes:
            Ordered nonfatal diagnostic codes recorded while processing the description.
    """

    description_hash: str
    normalized_hash: str | None
    description: str
    normalized: str | None
    normalized_in_json: str | None
    parse_status: T_ParseStatus
    parser_used: T_ParserUsed | None
    parser_warning_codes: list[str]


class T_Normalized(TypedDict):
    """
    Describe one unique normalized job description written to the output dataset.

    Attributes:
        normalized_hash:
            SHA-256 identifier generated from ``normalized_in_json``.
        normalized:
            Human-readable text formed by joining the ordered blocks.
        normalized_in_json:
            Deterministic JSON representation of the canonical ordered blocks.
        normalized_multiplicity:
            Number of original postings represented by this normalized description.
        normalized_multiplicity_in_description:
            Number of distinct raw descriptions sharing this normalized identity.
    """

    normalized_hash: str
    normalized: str
    normalized_in_json: str
    normalized_multiplicity: int
    normalized_multiplicity_in_description: int


class T_Blocks(TypedDict):
    """
    Describe one parsed block written to the block-level output dataset.

    Attributes:
        normalized_hash:
            Identifier of the normalized description containing the block.
        block_id:
            One-based identifier formatted within the normalized description.
        block_order:
            One-based position of the block within the normalized description.
        block_type:
            Semantic role assigned by the parser.
        source_tag:
            HTML tag supplying the example block metadata, when applicable.
        heading_level:
            Numbered HTML heading level, when applicable.
        text:
            Normalized visible text in the block.
        description_hash_example:
            Raw description from which the retained example metadata was obtained.
    """

    normalized_hash: str
    block_id: str
    block_order: int
    block_type: T_BlockType
    source_tag: str | None
    heading_level: int | None
    text: str
    description_hash_example: str


R"""
Notes:
(1) ``T_NormalizedSqlRow`` follows the column order in ``sql_select_normalized`` in the main file.
(2) The last two fields retain the example raw-description hash and its full parsed-block JSON.
"""

T_NormalizedSqlRow: TypeAlias = tuple[str, str, str, int, int, str, str]

"""
Descriptions:
    Central rules for parsing and normalizing job descriptions.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    None.

To be imported by:
    codes/C01_PreProcessPostings/B01_Util_ParsePlainTexts.py
    codes/C01_PreProcessPostings/B02_Util_ParseHTMLContents.py

Notes:
(1) Edit this file when normalization policy changes.
(2) The utility modules that should import this file to implement these rules -- they shouldn't
    define independent constants to add or modify the rules.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-21
"""

import re
from typing import Final, Literal, TypeAlias, TypedDict


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Rule set 1. Plain-text normalization and list-item recognition
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


R"""
Notes:
(1) ``UNICODE_NORMALIZATION`` is used in function ``normalize_block_text`` (file B01) to return the 
    normal form string.
(2) ``BULLET_PATTERNS`` is used in function ``parse_plain_text_blocks`` (file B01) to extract list
    items in plain-text descriptions.
(3) The detailed explanations for the usage of ``BULLET_PATTERNS`` are in the docstring of the 
    function ``parse_plain_text_blocks``.
"""

UNICODE_NORMALIZATION: Final[str] = "NFC"
BULLET_PATTERNS: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?:[•·▪◦*-]|\d+[.)]|\(\d+\)|[A-Za-z][.)]|\([A-Za-z]\)|[ivx]{2,}[.)]|\([ivx]{2,}\))\s+"
)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Rule set 2. HTML recognition and content removal
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


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


# HTML block construction
HTML_HEADING_LEVELS: Final[dict[str, int]] = {
    "h1": 1,
    "h2": 2,
    "h3": 3,
    "h4": 4,
    "h5": 5,
    "h6": 6,
}
HTML_LIST_ITEM_TAGS: Final[frozenset[str]] = frozenset({"li"})
HTML_TABLE_ROW_TAGS: Final[frozenset[str]] = frozenset({"tr"})
HTML_PARAGRAPH_TAGS: Final[frozenset[str]] = frozenset({"p"})
BLOCK_TAGS: Final[frozenset[str]] = (
    frozenset(HTML_HEADING_LEVELS) | HTML_LIST_ITEM_TAGS | HTML_TABLE_ROW_TAGS | HTML_PARAGRAPH_TAGS
)
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
HTML_TEXT_SEPARATOR_TAGS: Final[frozenset[str]] = frozenset({"td", "th"})
HTML_HARD_BOUNDARY_TAGS: Final[frozenset[str]] = frozenset({"hr"})
HTML_LINE_BREAK_TAGS: Final[frozenset[str]] = frozenset({"br"})
HTML_UNNUMBERED_HEADING_TAGS: Final[frozenset[str]] = frozenset({"strong", "b"})
UNNUMBERED_HEADING_MAX_LENGTH: Final[int] = 120


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Other common utilities
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
    One ordered unit of visible text extracted from a job description.
    """

    block_type: T_BlockType
    heading_level: int | None
    text: str
    source_tag: str | None

"""
Descriptions:
    Policies and rules for parsing and normalizing job descriptions.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    None.

To be imported by:
    codes/C01_PreProcessPostings/B03_Util_ParsePlainTexts.py
    codes/C01_PreProcessPostings/B04_Util_ParseHTMLContents.py

Notes:
(1) Edit this file when normalization policy changes.
(2) Utility modules should import the rules defined here instead of defining independent policy
    constants.
(3) Type-only definitions are stored separately in ``B02_Util_TypeHinting.py``.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-23
"""

import re
from typing import Final

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Rule set 1. Plain-text normalization and list-item recognition
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-1-1. Normalize Unicode and recognize plain-text list items
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


R"""
Notes:
(1) ``UNICODE_NORMALIZATION`` is used in function ``normalize_block_text`` (file B03) to return the
    normal form string.
(2) ``BULLET_PATTERNS`` is used in function ``parse_plain_text_blocks`` (file B03) to extract list
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

# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-1. Recognize HTML and remove noncontent elements
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


R"""
Notes:
(1) ``HTML_TAGS`` is the complete list of tags that cause a description to use the HTML parsing
    route. Matching is case-insensitive and accepts both opening and closing tags.
(2) ``REMOVAL_CONTENTS`` identifies elements whose full subtrees are excluded because they do not
    represent visible job-description text.
(3) The structural sets determine how retained HTML is divided into blocks. A recognized tag does
    not necessarily create a block: for example, inline tags help select the HTML route but normally
    remain within their surrounding block.
"""

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


# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>
# >> S-2-2. Construct blocks from HTML tags
# >>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>#>>


R"""
Notes:
(1) Tags ``h1`` through ``h6`` create numbered ``HEADING`` blocks. Tags ``li``, ``tr``, and ``p``
    create ``LIST_ITEM``, ``TABLE_ROW``, and ``PARAGRAPH`` blocks, respectively.
(2) Container tags can provide inferred paragraph boundaries when no explicit block tag already
    owns their text.
(3) Tags ``td`` and ``th`` separate neighboring table-cell text without creating independent
    blocks. Tag ``hr`` creates a hard boundary, and ``br`` creates a boundary only within inferred
    content.
(4) Tags ``strong`` and ``b`` outside an explicit block create an unnumbered ``HEADING`` when their
    normalized text contains at most ``UNNUMBERED_HEADING_MAX_LENGTH`` characters. The current
    threshold is 120 characters. Longer bold text becomes a ``PARAGRAPH``.
(5) Inside an explicit block, inferred rules do not split the block. For example, ``strong`` inside
    ``p`` remains part of the paragraph.
"""

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

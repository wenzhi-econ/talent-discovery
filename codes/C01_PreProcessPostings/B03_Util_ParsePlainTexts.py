"""
Descriptions:
    Utilities for parsing plain-text job descriptions into normalized blocks.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    codes/C01_PreProcessPostings/B01_RulesForNormalization.py
    codes/C01_PreProcessPostings/B02_Util_TypeHinting.py

To be imported by:
    codes/C01_PreProcessPostings/B_NormalizeDescriptions.py
    codes/C01_PreProcessPostings/B04_Util_ParseHTMLContents.py

Notes:
(1) This script defines 2 functions for processing plain texts.
(2) Function ``normalize_block_text`` is used to remove meaningless whitespace and normalize unicode
    representation.
(3) Function ``parse_plain_text_blocks`` is used to extract list items from plain-text descriptions.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-23
"""

import html
import re
import unicodedata

from B01_RulesForNormalization import BULLET_PATTERNS, UNICODE_NORMALIZATION
from B02_Util_TypeHinting import T_BlockType, T_ParsedBlocks

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 1. Normalize texts
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def normalize_block_text(value: str) -> str:
    R"""
    Standardize Unicode and whitespace while preserving meaningful punctuation.

    Parameters:
        value:
            One block or line of job-description text to normalize. The function expects a string
            and does not decode HTML character references.

    Returns:
        Text in the configured Unicode normal form with null characters removed, nonbreaking spaces
        replaced by ordinary spaces, Windows line endings standardized, repeated horizontal
        whitespace collapsed, and leading or trailing whitespace removed.

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
    (7) Newline characters are standardized but not collapsed. A caller can therefore decide whether
        line boundaries define separate blocks.
    (8) Meaningful punctuation, capitalization, and ordinary internal newlines are preserved.
    """
    value = value.replace("\x00", "")
    value = unicodedata.normalize(UNICODE_NORMALIZATION, value)
    value = value.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(pattern=r"[\t\f\v ]+", repl=" ", string=value).strip()
    return value


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 2. Parse plain text
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def parse_plain_text_blocks(
    value: str,
    bullet_pattern: re.Pattern[str] = BULLET_PATTERNS,
) -> list[T_ParsedBlocks]:
    R"""
    Convert each nonempty plain-text line into a "SOURCE_LINE" or "LIST_ITEM" block.

    Parameters:
        value:
            Plain-text job description to parse.
        bullet_pattern:
            Compiled regular expression used to identify list-item markers at the start of a
            normalized line.

    Returns:
        Parsed blocks in source order. Each block records its type, heading level, normalized text,
        and source tag.

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

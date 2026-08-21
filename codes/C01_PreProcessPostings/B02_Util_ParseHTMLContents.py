"""
Descriptions:
    Utilities for parsing HTML contents of job descriptions into normalized blocks.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    codes/C01_PreProcessPostings/B00_RulesForNormalization.py
    codes/C01_PreProcessPostings/B01_Util_ParsePlainTexts.py

To be imported by:
    codes/C01_PreProcessPostings/B_NormalizeDescriptions.py

Notes:
(1) This script defines functions for processing HTML contents.
(2) Function ``normalize_block_text`` is used to remove meaningless whitespace and normalize unicode
    representation.
(3) Function ``parse_plain_text_blocks`` is used to extract list items from plain-text descriptions.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-21
"""

import re
from collections.abc import Iterator, Sequence
from typing import Final, NamedTuple, TypeAlias
from bs4 import BeautifulSoup
from bs4.element import CData, NavigableString, Tag

from B00_RulesForNormalization import (
    BLOCK_TAGS,
    HTML_CONTAINER_TAGS,
    HTML_HARD_BOUNDARY_TAGS,
    HTML_HEADING_LEVELS,
    HTML_LINE_BREAK_TAGS,
    HTML_LIST_ITEM_TAGS,
    HTML_RE,
    HTML_TABLE_ROW_TAGS,
    HTML_TEXT_SEPARATOR_TAGS,
    HTML_UNNUMBERED_HEADING_TAGS,
    REMOVAL_CONTENTS,
    UNNUMBERED_HEADING_MAX_LENGTH,
    T_BlockType,
    T_ParsedBlocks,
)
from B01_Util_ParsePlainTexts import normalize_block_text


class T_BlockOwner(NamedTuple):
    """Rules for collecting one continuous run of HTML text into a block."""

    block_type: T_BlockType
    heading_level: int | None
    source_tag: str | None
    allows_inferred_blocks: bool


class T_HtmlTextEvent(NamedTuple):
    """One retained HTML text fragment and the block metadata that owns it."""

    owner: T_BlockOwner
    text: str


class T_HtmlBlockBoundary:
    """Marker separating text that must be stored in different blocks."""


T_HtmlEvent: TypeAlias = T_HtmlTextEvent | T_HtmlBlockBoundary
HTML_BLOCK_BOUNDARY: Final[T_HtmlBlockBoundary] = T_HtmlBlockBoundary()


def detect_recognized_html(
    value: str,
    bullet_pattern: re.Pattern[str] = HTML_RE,
) -> bool:
    """Return whether the text contains at least one recognized HTML tag."""
    return bool(bullet_pattern.search(value))


def remove_noncontent_elements(
    soup: BeautifulSoup,
    removal: list[str] = REMOVAL_CONTENTS,
) -> None:
    """Remove scripts, styles, embedded pages, and other invisible elements."""
    for element in soup.find_all(removal):
        element.decompose()


def owner_for_explicit_tag(tag_name: str) -> T_BlockOwner:
    """Return the block metadata supplied by an explicit HTML block tag."""
    if tag_name in HTML_HEADING_LEVELS:
        return T_BlockOwner("HEADING", HTML_HEADING_LEVELS[tag_name], tag_name, False)
    if tag_name in HTML_LIST_ITEM_TAGS:
        return T_BlockOwner("LIST_ITEM", None, tag_name, False)
    if tag_name in HTML_TABLE_ROW_TAGS:
        return T_BlockOwner("TABLE_ROW", None, tag_name, False)
    return T_BlockOwner("PARAGRAPH", None, tag_name, False)


def iter_html_block_events(
    node: object,
    owner: T_BlockOwner,
) -> Iterator[T_HtmlEvent]:
    """Yield each retained text node once, with explicit block-boundary markers."""
    # Check exact types because comments inherit from NavigableString but are not visible text.
    if type(node) in {NavigableString, CData}:
        yield T_HtmlTextEvent(owner, str(node))
        return
    if not isinstance(node, Tag):
        return

    tag_name = node.name.lower() if isinstance(node.name, str) else None
    if tag_name in REMOVAL_CONTENTS:
        return

    is_inferred_line_break = tag_name in HTML_LINE_BREAK_TAGS and owner.allows_inferred_blocks
    if tag_name in HTML_HARD_BOUNDARY_TAGS or is_inferred_line_break:
        yield HTML_BLOCK_BOUNDARY
        return
    if tag_name in HTML_LINE_BREAK_TAGS:
        yield T_HtmlTextEvent(owner, " ")
        return

    if tag_name in BLOCK_TAGS:
        yield HTML_BLOCK_BOUNDARY
        explicit_owner = owner_for_explicit_tag(tag_name)
        for child in node.children:
            yield from iter_html_block_events(child, explicit_owner)
        yield HTML_BLOCK_BOUNDARY
        return

    if tag_name in HTML_UNNUMBERED_HEADING_TAGS and owner.allows_inferred_blocks:
        yield HTML_BLOCK_BOUNDARY
        # get_text() chooses metadata only. Descendant nodes still supply all output text.
        bold_text = normalize_block_text(node.get_text(" ", strip=True))
        bold_type: T_BlockType = (
            "HEADING" if len(bold_text) <= UNNUMBERED_HEADING_MAX_LENGTH else "PARAGRAPH"
        )
        bold_owner = T_BlockOwner(bold_type, None, tag_name, False)
        for child in node.children:
            yield from iter_html_block_events(child, bold_owner)
        yield HTML_BLOCK_BOUNDARY
        return

    if tag_name in HTML_CONTAINER_TAGS and owner.allows_inferred_blocks:
        yield HTML_BLOCK_BOUNDARY
        container_owner = T_BlockOwner("PARAGRAPH", None, tag_name, True)
        for child in node.children:
            yield from iter_html_block_events(child, container_owner)
        yield HTML_BLOCK_BOUNDARY
        return

    if tag_name in HTML_TEXT_SEPARATOR_TAGS:
        for child in node.children:
            yield from iter_html_block_events(child, owner)
        yield T_HtmlTextEvent(owner, " ")
        return

    for child in node.children:
        yield from iter_html_block_events(child, owner)


def build_html_block(
    owner: T_BlockOwner | None,
    fragments: Sequence[str],
) -> T_ParsedBlocks | None:
    """Build one output block from adjacent text events sharing the same owner."""
    if owner is None or not fragments:
        return None
    text = normalize_block_text("".join(fragments))
    if not text:
        return None
    return {
        "block_type": owner.block_type,
        "heading_level": owner.heading_level,
        "text": text,
        "source_tag": owner.source_tag,
    }


def walk_html_blocks(soup: BeautifulSoup) -> list[T_ParsedBlocks]:
    """
    Convert retained HTML text into ordered, nonoverlapping blocks.

    The work has two explicit stages. ``iter_html_block_events`` walks the tree in document order
    and emits each retained text node exactly once, plus visual separators and markers where one
    block must end. This function combines adjacent fragments only when they share the same owner
    and no boundary separates them, then normalizes the completed block once. Deferring
    normalization preserves meaningful spacing across inline tags.

    The preservation guarantee applies to normalized retained text nodes after the documented
    removals; it does not preserve markup, attributes, comments, exact whitespace, or removed
    elements.
    """
    root_owner = T_BlockOwner("PARAGRAPH", None, None, True)
    blocks: list[T_ParsedBlocks] = []
    pending_owner: T_BlockOwner | None = None
    pending_fragments: list[str] = []

    for event in iter_html_block_events(soup, root_owner):
        is_boundary = isinstance(event, T_HtmlBlockBoundary)
        owner_changed = (
            not is_boundary and pending_owner is not None and pending_owner != event.owner
        )
        if is_boundary or owner_changed:
            if block := build_html_block(pending_owner, pending_fragments):
                blocks.append(block)
            pending_owner = None
            pending_fragments.clear()

        if isinstance(event, T_HtmlTextEvent):
            pending_owner = event.owner
            pending_fragments.append(event.text)

    if block := build_html_block(pending_owner, pending_fragments):
        blocks.append(block)
    return blocks


def fallback_if_lxml_empty(
    raw: str,
    lxml_blocks: list[T_ParsedBlocks],
) -> list[T_ParsedBlocks]:
    """Return lxml blocks, or reparse the raw HTML with Python's parser when they are empty."""
    if lxml_blocks:
        return lxml_blocks
    fallback = BeautifulSoup(raw, "html.parser")
    remove_noncontent_elements(fallback)
    return walk_html_blocks(fallback)

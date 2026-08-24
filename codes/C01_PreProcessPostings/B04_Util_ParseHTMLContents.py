"""
Descriptions:
    Utilities for parsing HTML job descriptions into normalized blocks.

Inputs:
    Not applicable.

Outputs:
    Not applicable.

Run:
    Not applicable.

Dependencies:
    codes/C01_PreProcessPostings/B01_RulesForNormalization.py
    codes/C01_PreProcessPostings/B02_Util_TypeHinting.py
    codes/C01_PreProcessPostings/B03_Util_ParsePlainTexts.py

To be imported by:
    codes/C01_PreProcessPostings/B_NormalizeDescriptions.py

Notes:
(1) This script defines functions and classes to parse HTML contents in job descriptions.
    (a) One goal is to preserve all visible text in document order, including text not enclosed by
        an HTML block tag. I will call this the lossless requirement.
    (b) The other goal is to utilize the HTML tags to generate meaningful structure for normalized
        texts and construct blocks. I will call this the structured requirement.
(2) Class ``HtmlBlockCollector`` holds the block currently being assembled and the completed
    blocks.
(3) Function ``collect_html_node`` visits each visible text node once and sends its text directly
    to the collector. It does not use a generator or construct an intermediate event stream.
(4) Function ``walk_html_blocks`` initializes the traversal and returns the completed blocks.
(5) Exact whitespace, markup, attributes, comments, and elements listed in ``REMOVAL_CONTENTS``
    are deliberately not preserved.

Wang Wenzhi, with the help of CODEX
Time: 2026-08-23
"""

import re
from typing import NamedTuple

from B01_RulesForNormalization import (
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
)
from B02_Util_TypeHinting import T_BlockType, T_ParsedBlocks
from B03_Util_ParsePlainTexts import normalize_block_text
from bs4 import BeautifulSoup
from bs4.element import CData, NavigableString, Tag

# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Class 1. Store the ownership rules for one block
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


class T_BlockOwner(NamedTuple):
    """
    Store the structural metadata assigned to one continuous run of HTML text.

    Attributes:
        block_type:
            Semantic type assigned to the resulting block.
        heading_level:
            Integer level for an ``h1`` through ``h6`` block; otherwise ``None``.
        source_tag:
            Lowercase HTML tag that supplied the block structure; ``None`` for loose text outside
            a recognized structural tag.
        allows_inferred_blocks:
            Whether descendants such as ``div``, ``br``, and ``strong`` may introduce blocks that
            are not already defined by an explicit ``p``, ``li``, ``tr``, or heading tag.

    Notes:
    (1) An owner is metadata, not a reference to a BeautifulSoup node.
    (2) Equal owners can be compared directly because ``NamedTuple`` values are immutable.
    (3) ``allows_inferred_blocks`` is ``False`` inside explicit blocks. For example, a ``strong``
        tag inside a ``p`` remains part of that paragraph instead of becoming a second heading.
    """

    block_type: T_BlockType
    heading_level: int | None
    source_tag: str | None
    allows_inferred_blocks: bool


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Class 2. Assemble blocks during the HTML-tree traversal
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


class HtmlBlockCollector:
    """
    Assemble normalized blocks while visible HTML text is visited in document order.

    Attributes:
        blocks:
            Completed output blocks in source order.

    Notes:
    (1) ``_pending_fragments`` stores adjacent raw text fragments until a structural boundary is
        reached. Normalizing only after joining the fragments preserves meaningful spaces around
        inline tags. For example, ``Hello <em>new</em> hire`` becomes ``Hello new hire``.
    (2) ``add_text`` automatically finishes the current block if the new fragment has a different
        owner. Therefore, text assigned to different structural tags cannot be merged.
    (3) ``finish_block`` is also called at explicit boundaries even when the owner is unchanged.
        Therefore, two adjacent ``p`` tags remain two blocks.
    (4) Empty or whitespace-only completed blocks are discarded after normalization.
    """

    def __init__(self) -> None:
        """
        Initialize an empty list of completed blocks and an empty pending block.

        Notes:
        (1) ``blocks`` is public because it is the final ordered result returned after traversal.
        (2) The pending owner and fragments are private implementation state and always refer to at
            most one unfinished block.
        """
        self.blocks: list[T_ParsedBlocks] = []
        self._pending_owner: T_BlockOwner | None = None
        self._pending_fragments: list[str] = []

    def add_text(self, owner: T_BlockOwner, text: str) -> None:
        """
        Add one visible text fragment to the pending block.

        Parameters:
            owner:
                Structural metadata assigned to the text fragment.
            text:
                Raw text from one BeautifulSoup text node, or one separator space introduced by
                an HTML tag such as ``br``, ``td``, or ``th``.

        Returns:
            ``None``. The pending state and, when ownership changes, ``blocks`` are updated in
            place.

        Notes:
        (1) Text is appended without immediate normalization because spaces may be split across
            neighboring text nodes and inline tags.
        (2) If ``owner`` differs from the pending owner, the old block is finished before the new
            fragment is stored. No fragment is read or appended more than once.
        """
        if self._pending_owner is not None and self._pending_owner != owner:
            self.finish_block()
        self._pending_owner = owner
        self._pending_fragments.append(text)

    def finish_block(self) -> None:
        """
        Normalize and store the pending block, then reset the pending state.

        Returns:
            ``None``. A nonempty normalized pending block is appended to ``blocks`` in place.

        Notes:
        (1) Joining precedes normalization so that inline markup cannot accidentally concatenate
            words or manufacture extra blocks.
        (2) Resetting the pending state after copying its contents ensures the same fragments
            cannot be included in a later block.
        (3) Calling this method when no text is pending is safe and has no effect on the output.
        """
        owner = self._pending_owner
        combined_text = "".join(self._pending_fragments)
        self._pending_owner = None
        self._pending_fragments.clear()

        if owner is None:
            return
        text = normalize_block_text(combined_text)
        if not text:
            return
        self.blocks.append(
            {
                "block_type": owner.block_type,
                "heading_level": owner.heading_level,
                "text": text,
                "source_tag": owner.source_tag,
            }
        )


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 1. Detect recognized HTML
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def detect_recognized_html(
    value: str,
    html_pattern: re.Pattern[str] = HTML_RE,
) -> bool:
    R"""
    Return whether a description contains at least one recognized HTML tag.

    Parameters:
        value:
            Raw job description to inspect.
        html_pattern:
            Compiled, case-insensitive expression defining the recognized HTML tags.

    Returns:
        ``True`` when a recognized opening or closing tag occurs; otherwise ``False``.

    Notes:
    (1) This function only chooses between the HTML and plain-text parsing routes.
    (2) It does not require the whole description to be a complete or valid HTML document.
    (3) For example, a mixed description such as ``Scientist<p>Requirements</p>PhD in biology`` will
        also invoke the HTML route because this function returns ``True``.
    (4) HTML-looking text involving an unrecognized tag alone uses the plain-text route.
    (5) The current default is constructed from ``HTML_TAGS`` and ``HTML_RE`` in file
        "B01_RulesForNormalization.py":

        HTML_TAGS = (
            "html|body|h1|h2|h3|h4|h5|h6|p|div|section|article|ul|ol|li|table|thead|"
            "tbody|tfoot|tr|td|th|br|span|a|em|i|u|strong|b|blockquote|pre|dl|dt|dd|"
            "script|style|noscript|iframe"
        )
        HTML_RE = re.compile(
            rf"</?(?:{HTML_TAGS})\b[^>]*>",
            re.IGNORECASE,
        )

    Explanations of the default pattern:
    (A) The prefixes ``r`` and ``f`` in ``rf"..."`` determine how Python constructs the pattern.
        (a) ``r`` makes it a raw string, so a backslash such as the one in ``\b`` is passed to the
            regular-expression engine instead of being interpreted as a Python string escape.
        (b) ``f`` makes it a formatted string, so ``{HTML_TAGS}`` is replaced by the value stored
            in ``HTML_TAGS`` before the regular expression is compiled.
    (B) ``<`` matches the literal opening angle bracket at the start of an HTML tag.
        (a) There is no ``^`` before it, so the tag does not need to occur at the beginning of the
            job description.
        (b) Function ``search`` scans the whole description and can find ``<`` after preceding
            plain text.
    (C) ``/?`` accepts either an opening tag or a closing tag.
        (a) ``/`` represents the literal forward slash used at the start of a closing tag.
        (b) ``?`` means "zero or one of the preceding item," so the slash is optional.
        (c) For example, both ``<p>`` and ``</p>`` satisfy this part of the pattern.
    (D) ``(?:{HTML_TAGS})`` accepts one of the recognized tag names.
        (a) ``(?: ... )`` is a noncapturing group. It groups the alternatives without storing the
            matched tag name as a separate result.
        (b) The vertical bars (``|``) inserted from ``HTML_TAGS`` mean "or." For example, the group
            can match ``html``, ``p``, ``div``, ``li``, ``strong``, or any other listed name.
        (c) A tag absent from ``HTML_TAGS``, such as ``custom``, is not recognized by this default
            pattern.
    (E) ``\b`` requires a word boundary immediately after the recognized tag name.
        (a) Every current tag name ends with a letter or digit, which is a regex "word character."
        (b) The next character must therefore be a nonword character, such as whitespace, ``/``,
            or ``>``.
        (c) This prevents a shorter listed name from matching only the beginning of a longer name.
            For example, ``p`` does not cause ``<paragraph>`` to be recognized, and ``h1`` does not
            cause ``<h10>`` to be recognized.
    (F) ``[^>]*`` accepts the remainder of the tag up to its closing angle bracket.
        (a) Square brackets define a character class, and the leading ``^`` inside the brackets
            negates it. Therefore, ``[^>]`` means "any character except ``>``."
        (b) ``*`` means "zero or more of the preceding item." A tag can consequently have no text
            after its name, or it can contain whitespace, attributes, and a self-closing slash.
        (c) For example, this part accommodates ``<p>``, ``<p class="summary">``, and ``<br/>``.
        (d) This is a recognition rule rather than a complete HTML validator. It does not verify
            attribute syntax and stops at the first closing angle bracket.
    (G) The final ``>`` matches the literal closing angle bracket of the tag.
        (a) A fragment such as ``<p class="summary"`` is not recognized because it lacks ``>``.
    (H) The flag ``re.IGNORECASE`` makes tag-name matching case-insensitive.
        (a) For example, ``<p>``, ``<P>``, and ``<Div>`` can all be recognized.
    (I) Method ``search`` returns the first match found anywhere in ``value``, or ``None`` if there
        is no match. Function ``bool`` then converts that result to ``True`` or ``False``.
    (J) Putting the pieces together, the whole expression means:
        (a) find an opening angle bracket anywhere in the description,
        (b) allow an optional closing-tag slash,
        (c) require one of the centrally listed tag names,
        (d) require the tag name to end at a word boundary,
        (e) allow zero or more non-``>`` characters for attributes or a self-closing slash, and
        (f) require a closing angle bracket.
    """
    return bool(html_pattern.search(value))


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 2. Remove elements without visible content
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def remove_noncontent_elements(
    soup: BeautifulSoup,
    removal: list[str] = REMOVAL_CONTENTS,
) -> None:
    """
    Remove elements whose contents should not enter the normalized description.

    Parameters:
        soup:
            Parsed BeautifulSoup tree to modify in place.
        removal:
            Tag names whose entire subtrees should be removed.

    Returns:
        ``None``. The supplied tree is modified in place.

    Notes:
    (1) ``decompose`` removes both the selected tag and all of its descendants. Current examples
        include scripts, styles, embedded pages, and ``noscript`` alternatives.
    (2) ``collect_html_node`` independently ignores these tags as a defensive check. Calling this
        function first also releases their parsed nodes and makes the retained tree easier to
        inspect.
    """
    for element in soup.find_all(removal):
        element.decompose()


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 3. Assign metadata from an explicit block tag
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def owner_for_explicit_tag(tag_name: str) -> T_BlockOwner:
    """
    Convert an explicit HTML block tag into the metadata for its output block.

    Parameters:
        tag_name:
            Lowercase tag name known to belong to ``BLOCK_TAGS``.

    Returns:
        Immutable block metadata for all visible text inside the tag.

    Notes:
    (1) Tags ``h1`` through ``h6`` become ``HEADING`` blocks and retain their numeric levels.
    (2) A ``li`` becomes a ``LIST_ITEM``, and a ``tr`` becomes a ``TABLE_ROW``.
    (3) Other explicit block tags currently consist of ``p`` and become ``PARAGRAPH`` blocks.
    (4) Explicit owners set ``allows_inferred_blocks`` to ``False``. Nested inline or container
        tags therefore do not split text that the explicit outer tag has already organized.
    """
    if tag_name in HTML_HEADING_LEVELS:
        return T_BlockOwner("HEADING", HTML_HEADING_LEVELS[tag_name], tag_name, False)
    if tag_name in HTML_LIST_ITEM_TAGS:
        return T_BlockOwner("LIST_ITEM", None, tag_name, False)
    if tag_name in HTML_TABLE_ROW_TAGS:
        return T_BlockOwner("TABLE_ROW", None, tag_name, False)
    return T_BlockOwner("PARAGRAPH", None, tag_name, False)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 4. Visit one HTML node and its descendants
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def collect_html_node(
    node: object,
    owner: T_BlockOwner,
    collector: HtmlBlockCollector,
) -> None:
    """
    Visit one parsed HTML node and add all retained text to ``collector`` in document order.

    Parameters:
        node:
            BeautifulSoup tag, visible text node, or another parsed node.
        owner:
            Structural metadata inherited from the closest tag that defines the current block.
        collector:
            Stateful block assembler shared by every recursive call for one description.

    Returns:
        ``None``. Retained text and completed blocks are added to ``collector`` in place.

    Notes:
    (1) Recursion mirrors the HTML tree. Children are visited from first to last, so source order
        is preserved even when plain text and HTML tags are mixed.
    (2) Exact-type checks retain ordinary text and CDATA while excluding comments, which inherit
        from BeautifulSoup's ``NavigableString`` class but are not visible description text.
    (3) Each text node reaches exactly one call to ``collector.add_text``. Tag-level branches then
        return, preventing their descendants from being visited again by the generic final loop.
    (4) Explicit block tags create boundaries before and after their descendants. This prevents
        their contents from merging with loose text or adjacent blocks.
    (5) Containers such as ``div`` and ``section`` infer paragraph ownership only when no explicit
        outer block already owns their text.
    (6) A ``strong`` or ``b`` tag outside an explicit block is treated as an unnumbered heading if
        its normalized text is short enough. Longer bold text becomes a paragraph.
    (7) Inside inferred content, ``br`` finishes the current block. Inside an explicit block, it
        contributes one space so the explicit block remains intact.
    (8) Cells ``td`` and ``th`` contribute a trailing separator space. Their enclosing ``tr``
        therefore becomes one table-row block without concatenating neighboring cell values.
    """
    # Check exact types because comments inherit from NavigableString but are not visible text.
    if type(node) in {NavigableString, CData}:
        collector.add_text(owner, str(node))
        return
    if not isinstance(node, Tag):
        return

    tag_name = node.name.lower() if isinstance(node.name, str) else None
    if tag_name in REMOVAL_CONTENTS:
        return

    is_inferred_line_break = tag_name in HTML_LINE_BREAK_TAGS and owner.allows_inferred_blocks
    if tag_name in HTML_HARD_BOUNDARY_TAGS or is_inferred_line_break:
        collector.finish_block()
        return
    if tag_name in HTML_LINE_BREAK_TAGS:
        collector.add_text(owner, " ")
        return

    if tag_name in BLOCK_TAGS:
        collector.finish_block()
        explicit_owner = owner_for_explicit_tag(tag_name)
        for child in node.children:
            collect_html_node(child, explicit_owner, collector)
        collector.finish_block()
        return

    if tag_name in HTML_UNNUMBERED_HEADING_TAGS and owner.allows_inferred_blocks:
        collector.finish_block()
        # get_text() chooses metadata only. Recursive calls still supply all output text once.
        bold_text = normalize_block_text(node.get_text(" ", strip=True))
        bold_type: T_BlockType = (
            "HEADING" if len(bold_text) <= UNNUMBERED_HEADING_MAX_LENGTH else "PARAGRAPH"
        )
        bold_owner = T_BlockOwner(bold_type, None, tag_name, False)
        for child in node.children:
            collect_html_node(child, bold_owner, collector)
        collector.finish_block()
        return

    if tag_name in HTML_CONTAINER_TAGS and owner.allows_inferred_blocks:
        collector.finish_block()
        container_owner = T_BlockOwner("PARAGRAPH", None, tag_name, True)
        for child in node.children:
            collect_html_node(child, container_owner, collector)
        collector.finish_block()
        return

    if tag_name in HTML_TEXT_SEPARATOR_TAGS:
        for child in node.children:
            collect_html_node(child, owner, collector)
        collector.add_text(owner, " ")
        return

    for child in node.children:
        collect_html_node(child, owner, collector)


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 5. Parse a prepared HTML tree
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def walk_html_blocks(soup: BeautifulSoup) -> list[T_ParsedBlocks]:
    """
    Convert retained HTML text into ordered, nonoverlapping normalized blocks.

    Parameters:
        soup:
            BeautifulSoup tree from which noncontent elements have normally already been removed.

    Returns:
        Normalized blocks in source order. Each block records its semantic type, heading level,
        normalized text, and structural source tag.

    Notes:
    (1) The root owner is an inferred paragraph with no source tag. It ensures that visible text
        outside all HTML tags is retained instead of silently discarded.
    (2) ``collect_html_node`` traverses the tree once and updates a single ``HtmlBlockCollector``.
        There is no intermediate event list and no generator that pauses and resumes execution.
    (3) The final call to ``finish_block`` stores loose text that occurs after the last HTML tag.
    (4) The preservation guarantee applies to normalized retained text. Markup, attributes,
        comments, exact whitespace, and elements listed in ``REMOVAL_CONTENTS`` are excluded by
        design.
    """
    root_owner = T_BlockOwner("PARAGRAPH", None, None, True)
    collector = HtmlBlockCollector()
    collect_html_node(soup, root_owner, collector)
    collector.finish_block()
    return collector.blocks


# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>
# <> Function 6. Reparse descriptions for which lxml returned no blocks
# <>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>#<>


def fallback_if_lxml_empty(
    raw: str,
    lxml_blocks: list[T_ParsedBlocks],
) -> list[T_ParsedBlocks]:
    """
    Keep successful lxml output or reparse empty output with Python's standard HTML parser.

    Parameters:
        raw:
            Original job description, including its HTML markup.
        lxml_blocks:
            Blocks returned after parsing ``raw`` with lxml.

    Returns:
        The unchanged lxml blocks when they are nonempty; otherwise, blocks obtained with
        BeautifulSoup's ``html.parser`` backend.

    Notes:
    (1) The fallback is deliberately narrow: it does not compare or combine two nonempty parser
        results, which could duplicate content.
    (2) Noncontent elements are removed from the fallback tree under the same policy used for the
        primary tree.
    (3) Parser status and warning codes are assigned by ``parse_description`` in the calling file.
    """
    if lxml_blocks:
        return lxml_blocks
    fallback = BeautifulSoup(raw, "html.parser")
    remove_noncontent_elements(fallback)
    return walk_html_blocks(fallback)

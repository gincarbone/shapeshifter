# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""Structural/static tests for alfa1_ui.py's generated HTML/CSS/JS.

No JS engine is available in this environment (no Node/npm — the project has
zero frontend build tooling by convention), so these tests validate
STRUCTURE, not runtime behavior: balanced braces, every DOM element id the
JS references actually exists in the markup, every function wired to a
click handler is actually defined, and every tool name the backend can emit
has a matching UI color — the kind of cross-file mismatch that's easy to
introduce silently (add a new action in alfa1_agent.py, forget the matching
UI entry) and that a live browser session won't reliably surface either
(a missing color just falls back to gray, a missing id/function silently
throws in the console, neither breaks the page outright).
"""
from __future__ import annotations

import re

import alfa1_agent
from alfa1_ui import ALFA1_HTML, _ALFA1_CSS, _ALFA1_JS


def test_html_shell_is_well_formed():
    assert ALFA1_HTML.strip().startswith("<!DOCTYPE html>")
    assert ALFA1_HTML.count("<script>") == ALFA1_HTML.count("</script>") == 1
    assert ALFA1_HTML.count("<style>") == ALFA1_HTML.count("</style>") == 1


def test_js_has_balanced_parens():
    # A cheap but real syntax-error tripwire: a stray unmatched paren
    # anywhere in the embedded script means the whole page's JS fails to
    # parse and nothing on it works. Braces aren't checked the same way:
    # regex literals legitimately contain unbalanced escaped braces (e.g.
    # `\{` matching a literal CSS `{` via lookahead, with no corresponding
    # `\}` needed), so a naive global brace count produces false positives
    # on correct code.
    assert _ALFA1_JS.count("(") == _ALFA1_JS.count(")")


def test_css_has_balanced_braces():
    assert _ALFA1_CSS.count("{") == _ALFA1_CSS.count("}")


def test_no_leftover_highlighter_placeholder_scheme():
    """Regression guard: the original syntax highlighter used a \\u0000<id>\\u0001
    placeholder-and-restore scheme that had a real bug (placeholder digits
    got re-matched by the number-token regex, corrupting output — see
    alfa1_ui.py's highlight() docstring). It was replaced with a single-pass
    tokenizer. Make sure the old scheme's telltale marker never comes back."""
    assert "\\u0000" not in _ALFA1_JS
    assert "\\u0001" not in _ALFA1_JS


def test_every_referenced_element_id_exists_in_html():
    referenced_ids = set(re.findall(r"getElementById\('([\w-]+)'\)", _ALFA1_JS))
    html_ids = set(re.findall(r'id="([\w-]+)"', ALFA1_HTML))
    missing = referenced_ids - html_ids
    assert not missing, f"JS references DOM ids with no matching element in the HTML: {missing}"


def test_every_wired_click_handler_function_is_defined():
    wired = set(re.findall(r"\.onclick\s*=\s*(\w+);", _ALFA1_JS))
    defined = set(re.findall(r"(?:async )?function (\w+)\(", _ALFA1_JS))
    missing = wired - defined
    assert not missing, f"onclick handlers reference undefined functions: {missing}"


def test_tool_colors_cover_every_action_the_backend_can_emit():
    """Every tool name alfa1_agent.py can send in a tool_call/tool_result
    event should have a distinct badge color in TOOL_COLORS — otherwise it
    silently falls back to generic gray. Derives the expected name set from
    the actual backend source (the <alfa1:...> action tag alternation, plus
    the two file-mutation tool names emitted directly) rather than
    hardcoding it, so this fails loudly if a new action is added on one
    side and forgotten on the other."""
    action_tag_names = set(
        alfa1_agent._ACTION_TAG_RE.pattern.split("(", 1)[1].split(")", 1)[0].split("|")
    )
    known_tool_names = action_tag_names | {"write_file", "apply_patch"}

    js_colors = dict(re.findall(r"(\w+):\s*'(#[0-9a-fA-F]{6})'", _ALFA1_JS))
    missing = known_tool_names - js_colors.keys()
    assert not missing, f"Tool names with no TOOL_COLORS entry in alfa1_ui.py: {missing}"


def test_lang_rules_cover_the_languages_ext_to_lang_maps_to():
    """extToLang() maps file extensions to language keys; LANG_RULES must
    have an entry for every key it can produce, or highlight() silently
    falls through to plain escaped text for that language."""
    lang_keys = set(re.findall(r"return '(\w+)';", _ALFA1_JS))
    lang_rules_block = re.search(r"const LANG_RULES = \{([\s\S]*?)\n\};", _ALFA1_JS)
    assert lang_rules_block, "LANG_RULES block not found in emitted JS"
    defined_langs = set(re.findall(r"^\s*(\w+):\s*\{", lang_rules_block.group(1), re.MULTILINE))
    missing = lang_keys - defined_langs
    assert not missing, f"extToLang() can return languages with no LANG_RULES entry: {missing}"

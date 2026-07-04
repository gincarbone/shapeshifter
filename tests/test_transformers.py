"""Tests for transformers.py — mode registry, coding-session detection, and
selective artifact retention (the "keep latest version per file" behavior)."""
from __future__ import annotations

from transformers import (
    TRANSFORMERS,
    VALID_MODES,
    _artifact_key,
    _collapse_unchanged_blocks,
    _extract_artifact_versions,
    _extract_latest_artifacts,
    _extract_latest_artifacts_collapsed,
    _extract_user_requirements,
    _format_artifacts_block,
    _is_coding_session,
    _split_definition_blocks,
    apply_transform,
    transform_hybrid,
    transform_incremental,
    transform_raw,
    transform_yaml,
)


def test_registry_matches_valid_modes():
    assert VALID_MODES == set(TRANSFORMERS.keys())
    assert "raw" in VALID_MODES
    assert "hybrid" in VALID_MODES


def test_transform_raw_is_identity():
    assert transform_raw("anything goes here") == "anything goes here"


def test_apply_transform_rejects_unknown_mode():
    try:
        apply_transform("not_a_real_mode", [{"role": "user", "content": "hi"}])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_apply_transform_builds_role_tagged_raw_context():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    raw, _ = apply_transform("raw", messages)
    assert "[USER]\nhello" in raw
    assert "[ASSISTANT]\nhi there" in raw


def test_is_coding_session_false_without_assistant_turn():
    assert _is_coding_session("[USER]\nplease write a function") is False


def test_is_coding_session_false_when_assistant_has_no_code():
    ctx = "[USER]\nhi\n\n[ASSISTANT]\nSure, happy to help with anything!"
    assert _is_coding_session(ctx) is False


def test_is_coding_session_true_when_assistant_wrote_code():
    ctx = "[USER]\nwrite add()\n\n[ASSISTANT]\n```python\ndef add(a, b):\n    return a + b\n```"
    assert _is_coding_session(ctx) is True


def test_extract_user_requirements_keeps_user_blocks_verbatim():
    ctx = (
        "[USER]\nCreate add(a,b)\n\n"
        "[ASSISTANT]\n```python\ndef add(a,b): return a+b\n```\n\n"
        "[USER]\nNow add sub(a,b)"
    )
    reqs = _extract_user_requirements(ctx)
    assert reqs == ["Create add(a,b)", "Now add sub(a,b)"]


# ---------------------------------------------------------------------------
# Selective artifact retention
# ---------------------------------------------------------------------------

def test_artifact_key_uses_filename_comment_when_present():
    block = "```python\n# app.py\ndef add(a, b):\n    return a + b\n```"
    assert _artifact_key(block, "") == "app.py"


def test_artifact_key_falls_back_to_fence_language():
    block = "```python\ndef add(a, b):\n    return a + b\n```"
    assert _artifact_key(block, "") == "__lang__:python"


def test_extract_latest_artifacts_keeps_only_last_version_of_same_file():
    ctx = (
        "[ASSISTANT]\n```python\n# app.py\ndef add(a, b):\n    return a + b\n```\n\n"
        "[USER]\nadd sub too\n\n"
        "[ASSISTANT]\n```python\n# app.py\ndef add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n```"
    )
    artifacts = _extract_latest_artifacts(ctx)
    assert list(artifacts.keys()) == ["app.py"]
    assert "def sub" in artifacts["app.py"]
    # only one version kept — not both turns concatenated
    assert artifacts["app.py"].count("def add") == 1


def test_extract_latest_artifacts_user_paste_supersedes_assistant_version():
    # User manually edited the file and pastes the current state back —
    # that must win over the assistant's earlier (now stale) draft.
    ctx = (
        "[ASSISTANT]\n```python\n# app.py\ndef add(a, b):\n    return a + b\n```\n\n"
        "[USER]\nI tweaked it myself, here's the current app.py:\n"
        "```python\n# app.py\ndef add(a, b):\n    return a + b  # rounded\n```"
    )
    artifacts = _extract_latest_artifacts(ctx)
    assert list(artifacts.keys()) == ["app.py"]
    assert "# rounded" in artifacts["app.py"]


def test_extract_latest_artifacts_assistant_edit_supersedes_user_paste():
    # Symmetric case: assistant's later turn must win over an earlier user paste.
    ctx = (
        "[USER]\nhere's my current app.py:\n```python\n# app.py\ndef add(a, b):\n    return a + b\n```\n\n"
        "[ASSISTANT]\n```python\n# app.py\ndef add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n```"
    )
    artifacts = _extract_latest_artifacts(ctx)
    assert list(artifacts.keys()) == ["app.py"]
    assert "def sub" in artifacts["app.py"]


def test_extract_latest_artifacts_tracks_multiple_files_independently():
    ctx = (
        "[ASSISTANT]\n```python\n# models.py\nclass User: pass\n```\n\n"
        "[USER]\nnow main.py\n\n"
        "[ASSISTANT]\n```python\n# main.py\nfrom models import User\n```\n\n"
        "[USER]\nadd email field to User in models.py\n\n"
        "[ASSISTANT]\n```python\n# models.py\nclass User:\n    email: str\n```"
    )
    artifacts = _extract_latest_artifacts(ctx)
    assert set(artifacts.keys()) == {"models.py", "main.py"}
    assert "email" in artifacts["models.py"]
    assert "from models import User" in artifacts["main.py"]


# ---------------------------------------------------------------------------
# Attention-aware artifact staleness (Feature 2 — stub collapsing)
# ---------------------------------------------------------------------------

def _two_file_artifacts():
    # Realistic sizes — the size guard only collapses when the stub is
    # actually smaller than what it replaces (same principle as Feature 1's
    # tool-read dedup guard).
    return {
        "models.py": "class User:\n    id: int\n    name: str\n" * 10,
        "main.py": "from models import User\nprint(User(1, 'a'))\n" * 10,
    }


def test_artifact_mentioned_in_current_turn_stays_expanded():
    lines = _format_artifacts_block(_two_file_artifacts(), current_text="add an email field to models.py")
    block = "\n".join(lines)
    assert "class User" in block           # models.py expanded in full
    assert "id: int" in block


def test_artifact_not_mentioned_in_current_turn_collapses_to_stub():
    lines = _format_artifacts_block(_two_file_artifacts(), current_text="add an email field to models.py")
    block = "\n".join(lines)
    assert "from models import User" not in block  # main.py body collapsed
    assert "main.py" in block and "unchanged since last shown" in block


def test_artifact_stays_expanded_when_only_its_class_name_is_mentioned():
    # "the User model" never spells out the filename "models.py" — the file's
    # own class name must be enough to recognize it as relevant. Missing this
    # would collapse the exact file the turn is about to edit, which is
    # worse than not compressing at all.
    lines = _format_artifacts_block(_two_file_artifacts(), current_text="add an email field to the User model")
    block = "\n".join(lines)
    assert "id: int" in block  # models.py (declares `class User`) stays expanded
    assert "from models import User" not in block  # main.py still collapses — not mentioned at all


def test_single_artifact_never_collapses_even_if_unmentioned():
    single = {"app.py": "def add(a, b):\n    return a + b\n" * 10}
    lines = _format_artifacts_block(single, current_text="totally unrelated question about pricing")
    block = "\n".join(lines)
    assert "def add" in block  # only artifact tracked — nothing to gain by collapsing


def test_unnamed_fence_language_artifact_never_collapses():
    # No filename could ever be resolved for a __lang__ key — there's no
    # name the model could use to ask for it again, so it always stays full.
    artifacts = {
        "models.py": "class User: pass\n" * 10,
        "__lang__:python": "def helper():\n    pass\n" * 10,
    }
    lines = _format_artifacts_block(artifacts, current_text="totally unrelated question")
    block = "\n".join(lines)
    assert "def helper" in block  # __lang__ key expanded despite not being mentioned
    assert "class User" not in block  # models.py collapsed as expected


def test_collapsed_stub_line_count_matches_actual_content():
    artifacts = {
        "models.py": "class User: pass\n" * 10,
        "main.py": ("line1\nline2\nline3\n" * 10),
    }
    lines = _format_artifacts_block(artifacts, current_text="touch models.py only")
    block = "\n".join(lines)
    expected_lines = artifacts["main.py"].count("\n") + 1
    assert f"{expected_lines} lines" in block


def test_size_guard_skips_collapsing_a_file_smaller_than_its_stub():
    # For a genuinely tiny file, the stub text itself would be bigger than
    # what it replaces — never collapse in that case (mirrors Feature 1's
    # tool-read dedup size guard).
    artifacts = {
        "models.py": "class User: pass\n" * 10,   # big enough to be "the point"
        "x.py": "x = 1\n",                        # tiny — collapsing would cost more than it saves
    }
    lines = _format_artifacts_block(artifacts, current_text="touch models.py only")
    block = "\n".join(lines)
    assert "x = 1" in block  # x.py stays expanded despite being unmentioned and un-named-relevant


def test_hybrid_coding_session_collapses_unmentioned_file():
    raw, transformed = apply_transform(
        "hybrid",
        [
            {"role": "user", "content": "Create models.py with a User class."},
            {"role": "assistant", "content": "```python\n# models.py\n" + "class User:\n    pass\n" * 10 + "```"},
            {"role": "user", "content": "Now create main.py that imports User."},
            {"role": "assistant", "content": "```python\n# main.py\n" + "from models import User\n" * 10 + "```"},
        ],
        current_text="Add an email field to the User model in models.py.",
    )
    assert "class User" in transformed          # models.py mentioned — expanded
    assert "from models import User" not in transformed  # main.py not mentioned — collapsed
    assert "unchanged since last shown" in transformed


def _coding_session_messages():
    return [
        {"role": "user", "content": "Create app.py with add(a,b)."},
        {"role": "assistant", "content": "```python\n# app.py\ndef add(a, b):\n    return a + b\n```"},
        {"role": "user", "content": "Add sub(a,b) to app.py too."},
        {"role": "assistant", "content": (
            "```python\n# app.py\ndef add(a, b):\n    return a + b\n\n"
            "def sub(a, b):\n    return a - b\n```"
        )},
    ]


def test_hybrid_coding_session_includes_latest_artifact_only():
    raw, transformed = apply_transform("hybrid", _coding_session_messages())
    assert "cumulative_requirements" in transformed
    assert "current_artifacts" in transformed
    assert transformed.count("def add") == 1
    assert "def sub" in transformed


def test_yaml_coding_session_includes_latest_artifact_only():
    raw, transformed = apply_transform("yaml", _coding_session_messages())
    assert "current_artifacts" in transformed
    assert transformed.count("def add") == 1


def test_incremental_coding_session_includes_latest_artifact_only():
    raw, transformed = apply_transform("incremental", _coding_session_messages())
    assert "current_artifacts" in transformed
    assert transformed.count("def add") == 1


def test_first_turn_has_no_artifacts_block_without_prior_assistant_code():
    messages = [{"role": "user", "content": "Create app.py with add(a,b)."}]
    raw, transformed = apply_transform("hybrid", messages)
    # No assistant turn yet — not a coding session, falls through to generic branch
    assert "current_artifacts" not in transformed


# ---------------------------------------------------------------------------
# Touched-region expansion / boilerplate collapse (Feature 3)
# ---------------------------------------------------------------------------

_CALC_V1 = (
    "class Calculator:\n"
    "    def add(self, a, b):\n"
    "        return a + b\n"
    "\n"
    "    def subtract(self, a, b):\n"
    "        return a - b\n"
    "\n"
    "    def multiply(self, a, b):\n"
    "        return a * b\n"
)


def test_split_definition_blocks_returns_empty_for_unrecognized_syntax():
    # Bash-style `foo() { ... }` (no "function" keyword) and arrow functions
    # aren't in the recognized keyword set — must not guess at boundaries.
    assert _split_definition_blocks("x=1\nfoo() {\n  echo hi\n}\n") == []
    assert _split_definition_blocks("const add = (a, b) => a + b;\n") == []


def test_split_definition_blocks_recognizes_javascript_functions():
    js = "function add(a, b) {\n    return a + b;\n}\n\nfunction sub(a, b) {\n    return a - b;\n}\n"
    blocks = _split_definition_blocks(js)
    headers = [h for h, _ in blocks if h is not None]
    assert headers == ["function add(a, b) {", "function sub(a, b) {"]
    add_block = next(b for h, b in blocks if h == "function add(a, b) {")
    assert "return a + b" in add_block
    assert "return a - b" not in add_block


def test_split_definition_blocks_recognizes_go_rust_class_family():
    assert _split_definition_blocks("func Add(a, b int) int {\n\treturn a + b\n}\n")
    assert _split_definition_blocks("fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n")
    assert _split_definition_blocks("struct Point {\n    x: i32,\n    y: i32,\n}\n")


def test_split_definition_blocks_recognizes_common_modifiers():
    # async def is the FastAPI-endpoint-handler shape — missing this would
    # mean route handlers never get per-function collapsing at all.
    assert _split_definition_blocks("    async def get_items():\n        return []\n")
    assert _split_definition_blocks("export function foo() {\n    return 1;\n}\n")
    assert _split_definition_blocks("export default function App() {\n    return null;\n}\n")
    assert _split_definition_blocks("pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n")


def test_collapse_unchanged_blocks_works_for_async_def():
    v1 = (
        "async def get_items():\n    return items_db\n\n"
        "async def get_item(id: int):\n    return items_db[id]\n"
    )
    v2 = v1.replace("return items_db[id]", "return items_db[id]  # fixed lookup")
    result = _collapse_unchanged_blocks(v1, v2)
    assert "return items_db\n" not in result  # get_items() unchanged -> collapsed
    assert "fixed lookup" in result           # get_item() changed -> stays in full


def test_split_definition_blocks_captures_full_body_per_def():
    blocks = _split_definition_blocks(_CALC_V1)
    headers = [h for h, _ in blocks if h is not None]
    assert headers == ["class Calculator:", "    def add(self, a, b):",
                        "    def subtract(self, a, b):", "    def multiply(self, a, b):"]
    add_block = next(b for h, b in blocks if h == "    def add(self, a, b):")
    assert "return a + b" in add_block
    assert "return a - b" not in add_block  # doesn't bleed into the next block


def test_split_definition_blocks_keeps_decorator_attached_to_its_def():
    code = "class C:\n    @staticmethod\n    def helper():\n        return 1\n"
    blocks = _split_definition_blocks(code)
    helper_block = next(b for h, b in blocks if h == "    def helper():")
    assert "@staticmethod" in helper_block  # decorator stays part of the block it decorates


# ---------------------------------------------------------------------------
# Trailing top-level code split off the last declaration (Feature 5)
# ---------------------------------------------------------------------------

def test_split_definition_blocks_splits_off_trailing_main_block():
    code = (
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n\n"
        "if __name__ == '__main__':\n    print(add(1, 2))\n"
    )
    blocks = _split_definition_blocks(code)
    sub_block = next(b for h, b in blocks if h == "def sub(a, b):")
    assert "return a - b" in sub_block
    assert "__main__" not in sub_block  # trailing demo code isn't absorbed into sub()
    trailing = next(b for h, b in blocks if h is None and "__main__" in b)
    assert "print(add(1, 2))" in trailing


def test_collapse_unchanged_blocks_collapses_last_function_despite_trailing_code_changing():
    # The real point of Feature 5: only the trailing demo code changes here,
    # sub() itself is untouched — it must still collapse.
    v1 = (
        "def add(a, b):\n    return a + b\n\n"
        "def sub(a, b):\n    return a - b\n\n"
        "if __name__ == '__main__':\n    print(add(1, 2))\n"
    )
    v2 = v1.replace("print(add(1, 2))", "print(add(1, 2))\n    print(sub(5, 2))")
    result = _collapse_unchanged_blocks(v1, v2)
    assert "return a - b" not in result       # sub() unchanged -> collapsed
    assert "print(sub(5, 2))" in result       # trailing code change preserved in full


def test_split_definition_blocks_no_trailing_code_is_unaffected():
    code = "def add(a, b):\n    return a + b\n"
    blocks = _split_definition_blocks(code)
    assert [h for h, _ in blocks] == ["def add(a, b):"]  # no spurious trailing block


def test_split_definition_blocks_handles_brace_language_closing_brace_correctly():
    # The closing "}" sits at the SAME indent as the function header in
    # brace languages — it must stay part of the block, not be misread as
    # the start of trailing code.
    code = (
        "function add(a, b) {\n    return a + b;\n}\n\n"
        "function sub(a, b) {\n    return a - b;\n}\n\n"
        "console.log('loaded');\n"
    )
    blocks = _split_definition_blocks(code)
    sub_block = next(b for h, b in blocks if h == "function sub(a, b) {")
    assert sub_block.rstrip().endswith("}")   # closing brace included, not cut off
    assert "console.log" not in sub_block
    trailing = next(b for h, b in blocks if h is None and "console.log" in b)
    assert "console.log('loaded')" in trailing


def test_collapse_unchanged_blocks_keeps_only_the_changed_method_in_full():
    v2 = _CALC_V1.replace("return a - b", "return a - b  # FIXED sign bug")
    result = _collapse_unchanged_blocks(_CALC_V1, v2)
    assert "return a + b" not in result       # add() unchanged -> collapsed
    assert "return a * b" not in result       # multiply() unchanged -> collapsed
    assert "FIXED sign bug" in result         # subtract() changed -> stays in full
    assert result.count("unchanged since previous version") == 2  # add + multiply


def test_collapse_unchanged_blocks_indents_stub_deeper_than_a_nested_method_header():
    # Regression: a method header at indent 4 needs its "..." stub at indent
    # 8+ (a valid method body), not a fixed indent that could be <= the
    # header's own — that would render as syntactically-wrong Python.
    v2 = _CALC_V1.replace("return a - b", "return a - b  # FIXED")
    result = _collapse_unchanged_blocks(_CALC_V1, v2)
    assert "    def add(self, a, b):\n        ...  # unchanged since previous version\n" in result


def test_collapse_unchanged_blocks_catches_a_change_on_the_very_last_line():
    # Edge case explicitly called out as risky in docs/token_savings_roadmap.md:
    # a change at the boundary of block detection must still be caught.
    v2 = _CALC_V1[:-1] + "  # tweaked\n"  # change on the last line of the last block
    result = _collapse_unchanged_blocks(_CALC_V1, v2)
    assert "tweaked" in result
    assert "return a * b  # tweaked" in result


def test_collapse_unchanged_blocks_catches_a_change_on_the_very_first_block():
    v2 = _CALC_V1.replace("class Calculator:", "class Calculator:  # renamed intent")
    result = _collapse_unchanged_blocks(_CALC_V1, v2)
    assert "renamed intent" in result


def test_collapse_unchanged_blocks_works_for_javascript_too():
    js = "function add(a, b) {\n    return a + b;\n}\n\nfunction sub(a, b) {\n    return a - b;\n}\n"
    js_v2 = js.replace("a - b;", "a - b; // fixed")
    result = _collapse_unchanged_blocks(js, js_v2)
    assert "return a + b" not in result   # add() unchanged -> collapsed
    assert "fixed" in result             # sub() changed -> stays in full


def test_collapse_unchanged_blocks_falls_back_to_full_for_unrecognized_syntax():
    bash = "foo() {\n  echo one\n}\n\nbar() {\n  echo two\n}\n"
    bash_v2 = bash.replace("echo two", "echo two fixed")
    result = _collapse_unchanged_blocks(bash, bash_v2)
    assert result == bash_v2  # no recognized declaration keyword — kept fully in full, no guessing


def test_collapse_unchanged_blocks_falls_back_for_single_block_file():
    single = "def add(a, b):\n    return a + b\n"
    result = _collapse_unchanged_blocks(single, single)
    assert result == single  # only one block total — nothing meaningful to collapse


def test_extract_artifact_versions_keeps_last_two_only():
    ctx = (
        "[ASSISTANT]\n```python\n# app.py\nv1 = 1\n```\n\n"
        "[USER]\nv2\n\n"
        "[ASSISTANT]\n```python\n# app.py\nv2 = 1\n```\n\n"
        "[USER]\nv3\n\n"
        "[ASSISTANT]\n```python\n# app.py\nv3 = 1\n```"
    )
    versions = _extract_artifact_versions(ctx)
    assert versions["app.py"] == ["```python\n# app.py\nv2 = 1\n```", "```python\n# app.py\nv3 = 1\n```"]


def test_extract_latest_artifacts_collapsed_no_previous_version_falls_back_to_full():
    ctx = f"[ASSISTANT]\n```python\n{_CALC_V1}```"
    artifacts = _extract_latest_artifacts_collapsed(ctx)
    assert "return a + b" in artifacts["__lang__:python"]  # first time seeing it — nothing to diff, full content


def test_extract_latest_artifacts_collapsed_second_version_collapses_unchanged_methods():
    v2 = _CALC_V1.replace("return a - b", "return a - b  # FIXED")
    ctx = (
        f"[ASSISTANT]\n```python\n{_CALC_V1}```\n\n"
        "[USER]\nfix the bug\n\n"
        f"[ASSISTANT]\n```python\n{v2}```"
    )
    artifacts = _extract_latest_artifacts_collapsed(ctx)
    result = artifacts["__lang__:python"]
    assert "FIXED" in result
    assert "return a + b" not in result

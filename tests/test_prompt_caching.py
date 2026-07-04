"""Tests for prompt-cache-friendly payload ordering (Feature 4 in
docs/token_savings_roadmap.md). None of this changes what information the
model receives — it only verifies that the parts of the payload that are
supposed to stay byte-identical across turns actually do, which is what lets
a provider's prompt cache recognize a repeated prefix and discount it."""
from __future__ import annotations

from output_contracts import build_system_prompt, detect_contract_type
from wrapper_server import _build_compressed_messages


def test_build_system_prompt_is_deterministic():
    a = build_system_prompt("hybrid", "code")
    b = build_system_prompt("hybrid", "code")
    assert a == b


def test_contract_type_is_frozen_to_the_first_user_turn():
    # Turn 2 introduces a new keyword ("error") that would flip
    # detect_contract_type's heuristic if it re-scanned the whole history —
    # the system message must stay identical regardless.
    messages_turn1 = [
        {"role": "user", "content": "write a function to add two numbers"},
    ]
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"},
        {"role": "user", "content": "there's an error in production, please fix it"},
    ]

    new_messages_1, _, _ = _build_compressed_messages(messages_turn1, "hybrid")
    new_messages_2, _, _ = _build_compressed_messages(messages_turn2, "hybrid")

    system_1 = next(m["content"] for m in new_messages_1 if m["role"] == "system")
    system_2 = next(m["content"] for m in new_messages_2 if m["role"] == "system")
    assert system_1 == system_2


def test_contract_type_matches_first_user_message_only():
    # Sanity check the freeze actually reflects turn 1, not a hardcoded value.
    generic_first = _build_compressed_messages(
        [{"role": "user", "content": "write a function to add two numbers"}], "hybrid",
    )[0]
    code_first = _build_compressed_messages(
        [{"role": "user", "content": "there is a bug in my code, please fix it"}], "hybrid",
    )[0]
    generic_system = next(m["content"] for m in generic_first if m["role"] == "system")
    code_system = next(m["content"] for m in code_first if m["role"] == "system")
    assert generic_system != code_system
    assert detect_contract_type([{"role": "user", "content": "there is a bug in my code, please fix it"}]) == "code"


def test_growing_history_prefix_stays_stable_across_turns():
    # The compressed "history so far" block for turn N (everything before
    # the CURRENT user message) should remain intact, byte-for-byte, once it
    # becomes part of turn N+1's history too — this is what lets a provider
    # cache the shared prefix across consecutive turns. It must not be
    # silently reworded or reordered just because more was appended after it.
    turn2 = [
        {"role": "user", "content": "Create app.py with add(a,b)."},
        {"role": "assistant", "content": "```python\n# app.py\ndef add(a, b):\n    return a + b\n```"},
        {"role": "user", "content": "Now add sub(a,b) too."},
    ]
    turn3 = turn2 + [
        {"role": "assistant", "content": (
            "```python\n# app.py\ndef add(a, b):\n    return a + b\n\n"
            "def sub(a, b):\n    return a - b\n```"
        )},
        {"role": "user", "content": "Now add mul(a,b) too."},
    ]

    _, _, transformed_2 = _build_compressed_messages(turn2, "hybrid")
    _, _, transformed_3 = _build_compressed_messages(turn3, "hybrid")

    # turn2's OWN compressed history (as sent for that turn) covers only the
    # first requirement — confirm turn3's compressed history for the SAME
    # turn2 messages reproduces that requirement text verbatim.
    assert "Create app.py with add(a,b)." in transformed_2
    assert "Create app.py with add(a,b)." in transformed_3
    # turn3 additionally carries forward the requirement that was turn2's
    # CURRENT (uncompressed) message — now that it's history, it must appear
    # verbatim in the compressed block, unaltered by having been rephrased.
    assert "Now add sub(a,b) too." in transformed_3

# Token-savings roadmap

Design notes for token-saving mechanisms, written down before implementation
so the work can resume across sessions. Status is tracked per feature at the
top of its section — update it as work progresses.

## Status at a glance / where to resume

**Done, tested, verified end-to-end against real OpenRouter:** Features 1–8,
all of them. All merged into the default pipeline. 93/93 tests pass
(`pytest -q` from repo root). All 3 benchmark scenarios (HTML, FastAPI,
edit_debug) were re-run after Features 7 and 8 landed, and README.md's
Benchmark Results / Cost sections now reflect those numbers. **Confirmed:
Feature 7 had zero measurable impact** on any of the 3 scenarios, as
expected — none of them are agentic/tool-call-heavy sessions, so its code
path never triggers. **Feature 8's impact is confirmed negligible**: HTML
reduction moved from 52.5–54.0% to 53.3–53.4%, FastAPI from 59.6–61.9% to
59.6–61.0%, edit_debug from 43.5–45.5% to 41.1–45.3% — all within normal
run-to-run model stochasticity for these scenarios, not a measurable step
change. Neither feature needs a dedicated re-run going forward. Feature 6
doesn't change compression ratios at all — it's a correctness/quality
mechanism for edit turns that need collapsed content back, not a
token-savings one.

**Nothing in this file is open right now.** The only explicitly-deferred
work is noted under Feature 6's "Left for a future iteration": extending
the retrieval tool to agentic requests, to streaming requests, and building
it a dedicated repeatable benchmark scenario (today it has real end-to-end
proof of correctness — see Feature 6's section — but not an automated
benchmark the way retention has Scenario 3). If picking that up, start
there rather than re-deriving the design from scratch.

If you're resuming and unsure where things stand, run `pytest -q` and check
`git diff --stat transformers.py wrapper_server.py` against the last commit
— everything described as DONE below should already be reflected there.

Context: this follows the selective-retention work (transformers.py's
`_extract_latest_artifacts`, keeps latest version per file for USER+ASSISTANT
blocks) and the tool-read dedup in wrapper_server.py
(`_dedupe_repeated_tool_file_reads`, currently exact-match-only). See
README.md's "Benchmark Results" and "Cost, not just percentage" sections for
the measured baseline this roadmap builds on (38–62% of raw tokens sent,
depending on scenario).

---

## Feature 1 — Generalize tool-read dedup from exact-match to latest-wins

**Status: DONE.** Implemented in `_dedupe_repeated_tool_file_reads`
(wrapper_server.py). Tests added (6 dedup tests in
tests/test_wrapper_pipeline.py, including the 3-read case and both size
guards). Verified end-to-end against real OpenRouter: a differing re-read
now gets replaced with a `"...since superseded..."` marker, the model still
answered correctly using the full last read, 40.3% reduction measured on
that exchange. README's "Important implementation note" updated to describe
the generalized behavior.

### What
`_dedupe_repeated_tool_file_reads` in `wrapper_server.py` currently only
replaces a `tool`-role file-read result with a marker if its content is
**byte-identical** to a later read of the same file path. Extend this so
**any** non-last occurrence of a given path is replaced with a marker,
whether identical or different — mirroring exactly what
`_extract_latest_artifacts` already does for assistant/user code (keep only
the latest version; earlier ones are superseded). This was originally
pitched to the user as "diff-based compression," but on reflection a plain
latest-wins marker is simpler, safer, and sufficient: the model only needs
the *current* state of a file to act on it, and that's always preserved in
full in the last occurrence. A real diff is deferred to Feature 3 where it's
actually load-bearing (to know which regions changed).

### Where
`wrapper_server.py`, functions `_build_tool_call_paths` (unchanged) and
`_dedupe_repeated_tool_file_reads` (extend).

### Design
1. Keep `_build_tool_call_paths` as-is (maps tool_call_id -> file path for
   read-like tool calls).
2. Compute, per path, the index of its **last** occurrence among `tool`
   messages (not keyed by content anymore — keyed by path only).
3. For every `tool` message whose path has a later occurrence (i.e. it is
   not the last read of that path):
   - If content is byte-identical to the path's last occurrence's content →
     marker: `f"[{path} unchanged since earlier read — content omitted]"`
     (existing wording).
   - If content differs → marker:
     `f"[{path} read here — since superseded by a later version shown further below]"`
     (new wording, so the model isn't led to believe nothing changed).
4. Keep the existing size guard: only replace if the marker is strictly
   shorter than the original content.
5. The **last** occurrence of every path is always left completely
   untouched, in both content and structure — this invariant must never
   change.

### Safety invariants to preserve
- Never touch non-`tool` messages.
- Never touch a `tool` message whose path can't be resolved (no read-like
  tool call matched).
- Never touch the last occurrence of a given path.
- Never produce a marker longer than what it replaces.

### Tests to add (tests/test_wrapper_pipeline.py)
- Differing (non-identical) earlier read gets replaced with the new
  "superseded" marker, later one stays full.
- Identical earlier read still gets the "unchanged" marker (regression, not
  a behavior change for that case).
- Three-or-more reads of the same path: only the last stays full, both
  earlier ones (whether identical to each other or not) get replaced.
- Size guard still applies to the differing case too.

---

## Feature 2 — Attention-aware artifact staleness (stub collapsing)

**Status: DONE.**

Implemented as designed, with two real bugs found and fixed during real
end-to-end verification against OpenRouter:

1. **Filename-only matching was too narrow.** A turn saying "add an email
   field to the User model" never spells out "models.py" — the literal
   substring check collapsed the exact file the turn was about to edit,
   which is worse than not compressing at all (this is precisely the
   failure mode the whole retention feature exists to prevent). Fixed by
   also matching any top-level `class`/`def`/`struct`/etc. identifier the
   artifact declares (`_artifact_identifiers`) against the current turn's
   text — "the User model" now matches via `class User`.
2. **Same size-guard gap as Feature 1.** For a small file, the stub text
   ("N lines, unchanged since last shown...") can be longer than the
   content it replaces. Added the same guard: never collapse unless the
   stub is actually smaller. Verified with a realistic-sized file
   end-to-end: 8.9% reduction on a single exchange, correct edit still
   applied to the mentioned file.

Tests added in `tests/test_transformers.py` (9 new tests): mentioned-by-name
stays expanded, unmentioned collapses, class-name-only mention still counts,
single-artifact never collapses, `__lang__` fallback keys never collapse,
stub line count is accurate, size guard skips tiny files, and a full
`transform_hybrid` integration test. 58/58 suite passes.

Threading note: `apply_transform` and all 9 `transform_*` functions gained
an optional `current_text: str = ""` parameter (default keeps existing
callers working unchanged); `_build_compressed_messages` in
wrapper_server.py now computes it from `current` and passes it through.

### What
In the coding-session branches of `transform_hybrid` / `transform_yaml` /
`transform_incremental` (transformers.py), `_extract_latest_artifacts`
currently shows the **full** latest version of every artifact the session
has touched, every single turn — even files that haven't been mentioned in
several turns and aren't relevant to the current request. Collapse artifacts
that aren't referenced by the current turn (or recent history) to a one-line
stub instead of full content; only fully expand artifacts the current turn
actually seems to be about.

### Where
`transformers.py`: `_format_artifacts_block` (needs a "what's relevant now"
signal) and `apply_transform`/the three `transform_*` coding-session
branches (need to pass through the *current* user message separately from
history, which they don't receive today — currently they only see
`history`, not `current`). This likely requires a signature change:
`apply_transform(mode, messages, current_text="")` or similar, threaded from
`_build_compressed_messages` in wrapper_server.py, which already separates
`history` from `current`.

### Design sketch
1. Relevance heuristic: an artifact key (filename or `__lang__:x`) is
   "active" for this turn if:
   - its filename literally appears in the current user message text, OR
   - no filename could ever be resolved for it (fall back to fence-language
     keys always being shown in full — collapsing those is riskier since
     there's no name to mention to "re-request" it), OR
   - it's the *only* artifact tracked (nothing to gain from collapsing).
2. For inactive artifacts, replace the full body with a stub:
   `f"{key} — {line_count} lines, unchanged since it was last shown — ask to see it again if you need the current content"`.
3. Active artifacts are shown in full, exactly as today.

### Open questions to resolve before implementing
- Should "recent" turns (not just the current one) count toward relevance,
  to avoid collapsing a file the user just asked about two turns ago and is
  still implicitly discussing? Start with current-turn-only; widen later if
  benchmarks show it collapsing things too eagerly.
- The explicit "ask to see it again" instruction in the stub matters: it
  gives the model an escape hatch instead of silently guessing at collapsed
  content. Keep it in the wording.

### Tests to add (tests/test_transformers.py)
- Artifact mentioned by name in the current turn stays fully expanded.
- Artifact not mentioned collapses to a stub.
- Single-artifact session never collapses (nothing to gain).
- Stub text never claims a specific unverified fact (line count must be
  computed from the actual retained content, not guessed).

---

## Feature 3 — Touched-region expansion, boilerplate collapse

**Status: DONE — wired into the default `hybrid`/`yaml`/`incremental` pipeline.**

Built as designed, but the actual collapse condition ended up stricter and
safer than the original "diff-based" framing suggested: a block only
collapses when it is **exactly byte-identical** to the same-named block in
the previous version — not "no changed line nearby in a diff." That removes
essentially all of the correctness risk flagged below; the only remaining
risk is block-*splitting* being correct, which is now covered by targeted
tests plus a real end-to-end check.

Two real bugs found and fixed during implementation/verification:
1. **Granularity bug**: the first version only split at column-0
   (module-level) `def`/`class` lines, so an entire class with several
   methods was treated as ONE indivisible block — no method-level
   collapsing happened at all for the most common real-world shape (a class
   with methods). Fixed `_split_python_blocks` to split at `def`/`class`
   lines at ANY indentation, which gives per-method granularity inside a
   class while a bare `class Foo:` header becomes its own trivial
   (never-collapsed) block.
2. **Indentation bug**: the collapsed stub's `...` line used a hardcoded
   4-space indent regardless of the header's own nesting depth, so a method
   at indent 4 got a stub at indent 4 too (same level as its own `def` line
   — invalid Python shape, though the model tolerated it in testing). Fixed
   to indent the stub 4 spaces deeper than the header's own indentation.

Real end-to-end verification against OpenRouter (3-turn session: build a
Calculator class, fix a bug in one method, then ask for a new method in a
4th turn): confirmed (a) unchanged methods correctly collapsed to indented
`... # unchanged` stubs, (b) the model reconstructed a fully correct,
complete file — all methods present, the earlier fix preserved, the new
method correctly added — with **zero literal `...` leakage** into its own
output, and (c) real positive token savings (9.4% reduction on that turn).

Tests added in `tests/test_transformers.py` (13 new tests): block-splitting
correctness (full body capture, decorator attachment, non-Python fallback),
collapse-condition correctness (only-changed-block stays full, first-line
and last-line edge cases, non-Python fallback, single-block fallback),
stub indentation depth, and version-history tracking (`_extract_artifact_versions`
keeps exactly the last two). 70/70 suite passes.

**Known limitation, by design, not a bug**: only Python-style (`def`/`class`
keyword) code gets block-level collapsing. Brace-languages (JS/Java/C#/Go)
and anything else fall back to full-content retention (Feature 3 becomes a
no-op for them, not a risk) — extending block-splitting to brace-matching
is a reasonable future addition but out of scope here.

### What
When an artifact's latest version differs from its previous version, most of
a large file is often untouched — only one function or region actually
changed. Instead of showing the full file every time it's retained, diff the
current version against the version before it, fully expand only the
regions that changed (plus a little surrounding context), and collapse
untouched top-level functions/classes to a one-line signature stub.

### Where
`transformers.py`: needs a new helper, e.g. `_extract_artifact_history` that
(unlike `_extract_latest_artifacts`, which discards everything but the
latest) keeps the **last two** versions per key instead of just one, so
there's something to diff against. Then a new formatting step that:
1. Runs `difflib` (stdlib, no new dependency) between the previous and
   current version.
2. Identifies which top-level blocks (heuristically: lines starting at
   column 0 with `def `/`class `/`function `/etc., reusing the language
   signal already in `_CODE_SIGNALS`) contain a changed line.
3. Renders changed blocks in full, unchanged top-level blocks as a
   signature-only stub (e.g. `def divide(self, a, b): ...  # unchanged`).

### Why this is the highest-risk feature of the four
Correctly detecting "block boundaries" language-agnostically is the hard
part — indentation-based (Python) vs brace-based (C-like) block detection
need different logic, and getting it wrong risks truncating a block the
model actually needs to see in full (e.g. a changed line deep inside an
unchanged-looking function signature). **Do not ship this without the
adversarial benchmark treatment Scenario 3 got** — build a scenario where a
change is deliberately non-adjacent to the diff (e.g. a one-line tweak deep
in a large function) and confirm the collapsed stub doesn't discard it.

### Recommendation
Build and merge Features 1, 2, and 4 first; come back to this one with a
dedicated benchmark scenario proving the block-boundary detection doesn't
eat a real change, before trusting it in the default path.

### Tests to add (tests/test_transformers.py)
- Two versions differing only inside one function: only that function is
  expanded, others collapse to signature stubs.
- A change on the very first or very last line of the file (edge of block
  detection) is still captured in full.
- No previous version exists (first time seeing the file) → falls back to
  full content, no collapsing (nothing to diff against).

---

## Feature 4 — Prompt-cache-aware payload ordering

**Status: DONE.**

Findings from doing the verification pass:
1. `build_system_prompt` itself was already deterministic (pure function of
   `mode` + `contract_type`, no per-request data). Confirmed with a test.
2. **Real bug found and fixed**: `contract_type` was derived from
   `detect_contract_type(original_messages)` — the *entire, growing*
   history — every turn. A later turn introducing a new keyword (e.g. the
   word "error") could flip the `OUTPUT_CONTRACT` section of the system
   message mid-session, breaking the byte-stable prefix a cache needs.
   Fixed in `_build_compressed_messages` (wrapper_server.py) to derive
   `contract_type` from the FIRST user turn only, frozen for the whole
   session — the opening ask defines the task type, later incidental
   keywords shouldn't retroactively change it. This is a genuine
   correctness improvement independent of caching, not just a cache
   optimization.
3. Artifact/requirement ordering was already stable (dict insertion order
   pinned by first-occurrence position, history built by literal prefix
   slicing) — added a regression test rather than needing a code change.
4. `cache_control` passthrough: `_extra_params` already forwards arbitrary
   top-level body fields untouched, so a client using OpenAI/DeepSeek-style
   automatic caching needs nothing extra. **Caveat found, not fixed**:
   Anthropic-style `cache_control` markers are embedded *inside individual
   message content blocks*, not as a top-level field. In the agentic path
   (tool-calling), original message structure is preserved, so such markers
   survive. In the **non-agentic/compressed path**, history gets rebuilt
   into a single compressed text blob — any client-supplied `cache_control`
   markers on historical messages do NOT survive that rebuild. Documented as
   a known limitation rather than solved; revisit only if a real client
   depending on Anthropic-style explicit cache breakpoints surfaces.

Tests added in `tests/test_prompt_caching.py`: system-prompt determinism,
contract-type freeze (with a deliberately keyword-flipping later turn),
contract-type still reflects the real first turn (not hardcoded), and a
growing-history-prefix stability check. 50/50 suite passes.

### What
Several providers (DeepSeek, Anthropic, OpenAI) discount input tokens that
match a previously-seen prefix of the same conversation (~90% off on a cache
hit for the providers that support it). This doesn't reduce the *token
count* ShapeShifter sends — it reduces what gets *billed*, by maximizing the
odds that consecutive requests in a session share a long, byte-identical
prefix a provider's cache recognizes.

### Where
`wrapper_server.py`'s `_build_compressed_messages`, and
`output_contracts.build_system_prompt`.

### Design
1. **Verify the system prompt is fully deterministic today.** Check
   `build_system_prompt(mode, contract_type)` — confirm it embeds no
   per-request data (timestamps, request IDs, random ordering). If it does,
   remove it: the system message must be byte-identical across turns for a
   cache hit to even be possible.
2. **Freeze ordering inside `_extract_latest_artifacts` / `_format_artifacts_block`.**
   Both already build a `dict`, and Python dicts preserve insertion order —
   confirm artifact iteration order is stable turn-to-turn (same key
   ordering) rather than changing based on dict-rebuild internals. Should
   already hold given current implementation, but add a regression test
   pinning it.
3. **Put growing/volatile content last.** The compressed-history message
   ShapeShifter builds today is: system → [compressed history block →
   "Understood." →] current message. Confirm cumulative_requirements (which
   only ever grows by appending, never rewrites earlier entries) and stable
   artifacts appear before anything that changes turn-to-turn, so the common
   prefix across turn N and turn N+1 is as long as possible.
4. **Consider `stream_options`/`cache_control` passthrough.** Some providers
   (Anthropic) require an explicit `cache_control` breakpoint marker in the
   request to opt into caching rather than doing it fully automatically.
   Since `_extra_params` in wrapper_server.py already forwards arbitrary
   body fields untouched, a client that wants explicit cache breakpoints can
   already pass them through today — verify this and document it, rather
   than inventing new config surface.

### What this does NOT require
No new compression logic, no risk to correctness — this is purely about
making sure output that's *already* deterministic stays byte-stable
turn-to-turn, and ordering volatile content last. Lowest-risk of the four.

### Tests to add
- `test_build_system_prompt_is_deterministic`: same `(mode, contract_type)`
  called twice returns byte-identical strings.
- `test_cumulative_requirements_prefix_is_stable_across_turns`: build
  compressed messages for turn N and turn N+1 of the same session, assert
  the turn-N output is a string-prefix of turn-(N+1)'s equivalent block
  (modulo the newly-added requirement/artifact at the end).

### README note to add once shipped
A short callout in "Cost, not just percentage" explaining that ordering is
now cache-friendly, with the caveat that the actual $ benefit depends on
whether the active provider supports prompt caching at all — this is a
provider-side discount ShapeShifter can position for but not guarantee.

---

## Feature 5 — Split off trailing top-level code from the last declaration block

**Status: DONE.** Implemented in `_split_definition_blocks` /
`_find_block_end_by_indent` (transformers.py). One real bug found and fixed
during implementation: indentation must be measured from the actual
`def`/`class` line, not from `start` — `start` can point at a decorator
line backed up to keep it attached to the block, and using its indentation
caused a `StopIteration` crash. Also handles brace languages correctly: a
lone closing `}`/`);`/etc. at the same indent as the header is the block's
own closing delimiter, not a boundary — `_LONE_CLOSER` regex skips past it
before looking for a real trailing-code boundary.

Tests added in `tests/test_transformers.py` (4 new): trailing
`if __name__ == "__main__":` split off correctly, the real point of the
feature (unchanged function still collapses even when only trailing demo
code changed), no-trailing-code case unaffected, and the brace-language
closing-brace edge case. 79/79 suite passes.

Verified end-to-end against real OpenRouter (3-turn Calculator session:
build v1, extend the demo block only, ask for a new method): `add`/`subtract`
correctly collapsed to stubs while the trailing demo block (which had
changed) stayed in full; model reconstructed a fully correct file with all
three methods and every demo call intact.

### What
`_split_definition_blocks` bounds an interior block by "up to the next
declaration's start line," which is correct — but the LAST declaration's
block currently runs all the way to end-of-file, silently absorbing any
top-level code that follows it (a classic `if __name__ == "__main__":` demo
block, module-level constants, etc.). This means: (a) the block's header
label is misleading (it's not really "just `divide()`", it's `divide()` plus
whatever trailing script code exists), and (b) if only the trailing code
changes between versions while the function itself doesn't, the whole merged
blob won't match and the function-level collapse opportunity is missed.

### Where
`transformers.py`: `_split_definition_blocks`, only the handling of the last
entry in `starts`.

### Design
Add a helper that, starting just after the last declaration's header line,
scans forward for the first non-blank line whose leading-whitespace count is
`<=` the header's own — that's the boundary between "this declaration's own
body" and "whatever comes after it at the same or shallower level." Works
for both indentation-delimited (Python) and brace-delimited (JS/Go/Rust/etc.)
code, since well-formatted generated code de-indents when a block ends
regardless of language. If a boundary is found before end-of-file, split the
remainder off as a `(None, ...)` trailing block — treated exactly like the
preamble: always kept in full, never collapsed.

```python
def _find_block_end_by_indent(lines, start, header_indent):
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if indent <= header_indent:
            return j
    return len(lines)
```

Applied only to the last entry in `starts` (interior blocks already have a
correct, tighter bound from the next declaration).

### Safety
Never reduces information — the trailing block is always rendered in full,
same as the preamble is today. This only improves label accuracy and unlocks
collapse opportunities that were previously blocked by an unrelated trailing
change; it cannot cause a function that DID change to look unchanged, since
the function's own text still has to match exactly for a collapse to happen.

### Tests to add (tests/test_transformers.py)
- A file with a trailing `if __name__ == "__main__":` block: the block
  before it collapses correctly when only its own body is unchanged, even
  though the trailing demo code differs between versions.
- No trailing code after the last declaration (existing behavior): no
  spurious empty trailing block appended.
- Trailing code containing blank lines and comments: still detected
  correctly (blank lines are skipped when scanning for the boundary).

---

## Feature 6 — Retrieval tool: let the model ask for collapsed content back

**Status: DONE.** Implemented exactly as scoped (non-agentic, non-streaming
only, bounded to `_MAX_RETRIEVAL_ROUNDS = 2`).

- `transformers.py`: `_extract_retrievable_pieces` (whole-file + per-function
  keys) and `_clean_declaration_name` (strips modifiers/params from a raw
  header down to a clean token) — kept fully separate from
  `apply_transform`, as planned, so no existing caller's return arity
  changed.
- `wrapper_server.py`: `_build_compressed_messages` now returns a 4th value
  (`retrieval_map`) — this DID change that function's arity, but it has only
  two callers (the main handler and this repo's own tests), both updated.
  `_SHAPESHIFTER_EXPAND_TOOL` (the synthetic tool schema),
  `_resolve_with_retrieval` (the bounded loop), and `_finalize_stats` now
  reports `retrieval_rounds` honestly in `_shapeshifter` stats rather than
  hiding the extra cost when it's spent.

Tests: 4 unit tests in `tests/test_retrieval_tool.py` mocking
`call_upstream` (resolves a tool call correctly, handles an unknown key
gracefully, caps out and forces a final answer, and — the safety-critical
one — costs exactly one upstream call with zero behavior change when the
model never calls the tool). Plus 3 unit tests for the retrieval-map
construction in `test_transformers.py`. 87/87 suite passes.

**Real end-to-end verification (the critical one)**: built a 3-turn session
where turn 3 requires reusing an exact, non-guessable class-attribute name
(`_call_tally`) that only exists in a collapsed method from turn 1 — not
paraphrased anywhere in the cumulative requirements text. Real call against
OpenRouter: `_shapeshifter.retrieval_rounds` came back as `1` (the model
genuinely called `shapeshifter_expand`), and the final generated code used
the exact real attribute name rather than inventing a plausible-sounding
one. This is the strongest possible confirmation that the mechanism works
as intended rather than just not crashing.

**Left for a future iteration** (explicitly out of scope for this version,
not forgotten): agentic requests, streaming requests, and a dedicated
benchmark scenario (this feature so far has real end-to-end proof of
correctness but not a repeatable automated benchmark the way Scenario 3 has
for retention).

### What
Today, when Feature 2 or Feature 3 collapses something, the stub text says
"ask to see it again if needed" — but in a **non-agentic** (no `tools`)
completion, the model has no actual way to act on that mid-generation: it
can only mention it in its final answer, which then requires a whole extra
round-trip initiated by whoever's driving the conversation. This feature
closes that gap: inject a small synthetic tool into the upstream request
that lets the model retrieve the full, uncollapsed content of anything it
was shown as a stub, have ShapeShifter answer that tool call *itself*
(instantly, from data already in hand — no second network round-trip to
anything external), and continue the model's turn transparently. The
original caller never sees the intermediate exchange; they just get a
normal completion, possibly a little slower and with a few more tokens
spent, only on the turns where the model actually needed something back.

This is strictly additive risk-wise: if the model never calls the tool,
behavior is identical to today (plus the small fixed cost of the tool
definition's tokens). It only spends more when the model actively decided a
stub wasn't enough — which is a self-correcting signal, not a guess we have
to get right ourselves.

### Scope for a first version (deliberately narrower than the full idea)
- **Non-agentic requests only.** Agentic (tool-calling) requests already
  pass through structurally untouched except for Feature 1's tool-read
  dedup; layering a second, ShapeShifter-owned tool into a request that
  already has the client's own `tools` is a sharper edge (harder to
  guarantee we don't confuse the model about which tool is "real") — left
  for a later iteration once this is proven on the simpler case.
- **Non-streaming only.** The internal retrieve-and-continue loop has to
  finish before we know what to send the client; doing this transparently
  under a streaming response is a bigger lift (buffer the loop, then stream
  only the final answer, or find a way to stream through it) — noted as a
  follow-up, not blocking this version.
- **Bounded loop.** Cap at 2 extra round-trips per request so a model that
  keeps asking for more can't turn one request into an unbounded chain of
  upstream calls.

### Where
- `transformers.py`: `_format_artifacts_block` and `_collapse_unchanged_blocks`
  need to also produce a `retrieval_map: dict[str, str]` (key → full content)
  for whatever they collapsed, alongside the text they already return. Kept
  as a separate accompanying function rather than changing `apply_transform`'s
  return arity (which every caller, including benchmark_coding.py and every
  existing test, depends on) — e.g. `_extract_retrievable_pieces(context) ->
  dict[str, str]`, called independently alongside `apply_transform` from
  `_build_compressed_messages`.
- `wrapper_server.py`: `_build_compressed_messages` returns the retrieval map
  too; `chat_completions` — when the map is non-empty and the request isn't
  streaming/agentic — injects a synthetic tool definition, and if the model's
  response calls it, resolves the request internally and loops.

### Retrieval key scheme
- Whole-file stub (Feature 2): key = the artifact key itself (filename, or
  the `__lang__:x` fallback).
- Per-function stub (Feature 3): key = `f"{artifact_key}#{function_name}"`
  (e.g. `calc.py#divide`) — needs extracting a clean name from the header
  line (strip `async`/`export`/etc. modifiers, parameters, trailing `{`/`:`)
  rather than using the whole raw header string as the key, since that's
  not a clean token for the model to echo back.

### Synthetic tool shape (draft)
```json
{
  "type": "function",
  "function": {
    "name": "shapeshifter_expand",
    "description": "Retrieve the full, current content of a file or function shown abbreviated in this conversation as '... unchanged' or a collapsed stub. Call this if you need to see or edit something that was collapsed rather than guessing at its content.",
    "parameters": {
      "type": "object",
      "properties": {"key": {"type": "string", "description": "The stub's file name, or file#function for a collapsed function."}},
      "required": ["key"]
    }
  }
}
```

### Internal loop sketch
```python
async def _resolve_with_retrieval(messages, model, temp, max_tok, extra, retrieval_map):
    tools = (extra.get("tools") or []) + [_SYNTHETIC_TOOL]
    for _ in range(MAX_RETRIEVAL_ROUNDS):
        resp, latency = await call_upstream(..., messages=messages, extra_params={**extra, "tools": tools, "tool_choice": "auto"})
        msg = resp["choices"][0]["message"]
        calls = [c for c in (msg.get("tool_calls") or []) if c["function"]["name"] == "shapeshifter_expand"]
        if not calls:
            return resp, latency  # model is done — this is the real answer
        messages = messages + [msg]
        for c in calls:
            key = json.loads(c["function"]["arguments"]).get("key", "")
            content = retrieval_map.get(key, f"No collapsed content found for '{key}'.")
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": content})
    # cap hit — ask once more WITHOUT the tool so the model is forced to answer with what it has
    return await call_upstream(..., messages=messages, extra_params={**extra, "tools": tools, "tool_choice": "none"})
```

### Stats/honesty considerations
- `_shapeshifter` stats should reflect the TOTAL cost including retrieval
  round-trips (don't hide the extra tokens spent when the model asks for
  something back) — this is a real, visible cost/benefit trade the dashboard
  should show accurately, not paper over.
- Log (at least at debug level) when a retrieval happens and for which key —
  useful signal for tuning which things get collapsed too aggressively if
  the model keeps asking for the same ones back.

### Tests to add
- Unit: a fake `call_upstream` that returns a tool call for a known
  retrieval key, then a real answer on the second call — confirm the loop
  resolves it and returns the final answer, with the intermediate exchange
  never surfaced to the "client".
- Unit: retrieval key not found in the map — confirm a graceful message is
  returned via the tool result rather than crashing.
- Unit: loop cap — a fake upstream that always asks for more — confirm it
  stops after `MAX_RETRIEVAL_ROUNDS` and forces a final answer.
- Real end-to-end: collapse something deliberately (small context window,
  a file the model needs mentioned only as a stub), phrase the current turn
  so answering correctly requires the collapsed content, and confirm the
  model calls the tool and produces a correct final answer.

### Recommendation
Ship Feature 5 first (small, independent, no shared code). This feature is
the biggest architectural addition in this roadmap — build it behind the
existing safety habits (unit tests for the loop mechanics, then one real
adversarial end-to-end check before trusting it), same as every other
feature here.

---

## Suggested implementation order

1. **Feature 1** — smallest change, extends code that already exists and is
   tested, lowest risk.
2. **Feature 4** — no new algorithms, mostly verification + a couple of
   ordering guarantees + tests; high value (ties into the $ framing) for
   low effort.
3. **Feature 2** — needs threading `current` into `apply_transform`, a
   moderate but contained change.
4. **Feature 3** — highest complexity and risk (language-agnostic block
   detection); do this last, and only after building a dedicated adversarial
   benchmark scenario for it the way Scenario 3 was built for retention.

---

## Feature 7 — Generalize tool-call dedup beyond file reads

**Status: DONE.** Implemented as designed: `_build_tool_call_paths` →
`_build_tool_call_keys` (returns `{id: (dedup_key, human_label)}` — file
reads keep the clean filename-based key/label for readable markers, every
other tool call gets `f"call:{name}:{canonical_json_args}"`);
`_dedupe_repeated_tool_file_reads` → `_dedupe_repeated_tool_calls`. Same
size guard, same latest-wins logic, same "last occurrence always kept in
full" invariant — pure scope generalization, no new risk.

Tests updated/added in `tests/test_wrapper_pipeline.py`: the old
"ignores non-read tools" test flipped to "tracks non-read tools too";
added different-arguments-get-different-keys, and a full dedupe test for a
repeated `execute_command`. 93/93 suite passes.

Real end-to-end verification against OpenRouter: an agentic session running
the same `pytest` command twice with identical output — the first result
collapsed to `"[execute_command call repeated with identical arguments —
output unchanged...]"`, the model answered correctly from the last (full)
occurrence, 40% reduction on that exchange.

---

## Feature 8 — Whitespace-tolerant collapsing for touched-region blocks

**Status: DONE.** Implemented `_normalize_for_comparison` exactly as
designed (rstrip per line + collapse blank-line runs, used ONLY for the
equality check — never for what's actually stored/shown). Wired into
`_collapse_unchanged_blocks`'s collapse condition.

Tests added in `tests/test_transformers.py`: trailing-whitespace-only
difference still collapses, blank-line-count difference still collapses
(interesting real finding — the inserted blank line lands at the TAIL of
the *previous* block, since blocks are split by the next declaration's
position, and normalizing blank-line runs to one means that block's
single-vs-double trailing blank line also compares equal — worth knowing if
extending this further), real reindentation still blocks the collapse
(regression guard), real token/expression change still blocks the collapse
(regression guard). 93/93 suite passes.

Real end-to-end verification against OpenRouter: a version pair differing
only in trailing whitespace + an extra blank line on `add()` (no real
change) plus a genuine comment addition to `subtract()` — `add()` correctly
collapsed despite the formatting noise, `subtract()` correctly stayed in
full, and the model reconstructed a fully correct file. Bonus: the model
also exercised Feature 6's retrieval tool during this same test
(`retrieval_rounds: 1`) and still produced the correct answer — confirms
Features 6 and 8 compose correctly.

### What
`_dedupe_repeated_tool_file_reads` (wrapper_server.py, Feature 1) only
tracks tool calls that *look like* a file read (matched via
`_FILE_READ_TOOL_NAME` against the function name, extracting a `path`
argument). But the same latest-wins safety principle applies to ANY
repeated tool call, not just reads: if `execute_command("npm test")` or
`search("TODO")` is called twice with identical arguments, the earlier
result is exactly as safe to summarize as an earlier identical file read —
nothing is lost, the full result still exists later in the same request.

### Design
Replace the file-read-specific key extraction with a fully general one:
key a tool call by `f"{function_name}:{canonical_json_args}"` for EVERY
tool call, not just ones matching a read-like name pattern. Rename
`_build_tool_call_paths` → `_build_tool_call_keys` and
`_dedupe_repeated_tool_file_reads` → `_dedupe_repeated_tool_calls` to match
(honesty about scope, same pattern as `_split_python_blocks` →
`_split_definition_blocks` earlier). Marker text should reference the tool
name only (not the raw arguments, which could be long/ugly) — e.g.
`f"[{name} call repeated with identical arguments — output unchanged, see the repeated call's result later in this conversation]"`. Same size guard, same
"last occurrence always kept in full" invariant, same "latest wins" logic
(identical → "unchanged" marker; differing → "superseded" marker) — this is
a scope generalization, not a new mechanism.

### Safety
Identical to Feature 1's existing safety argument: a marker only ever
replaces a message whose (name, arguments) pair repeats later with a
resolvable last occurrence; nothing is discarded that isn't recoverable
from later in the same request.

### Tests to update/add
- Existing "ignores non-read tools" test needs to flip — `execute_command`
  should now be tracked and deduped like anything else.
- Two calls to the same tool with the SAME arguments and identical output →
  earlier one collapses.
- Two calls to the same tool with the SAME arguments but DIFFERENT output →
  earlier one collapses with the "superseded" wording.
- Two calls to the same tool with DIFFERENT arguments → both stay in full
  (different key, not a dupe).

---

## Feature 8 — Whitespace-tolerant collapsing for touched-region blocks

**Status: not started**

### What
`_collapse_unchanged_blocks` (Feature 3) requires EXACT string equality
between a block and its previous version to collapse it — a single
trailing space or one extra blank line anywhere in an otherwise-identical
function blocks the collapse entirely. Since trailing whitespace and
blank-line-run counts are never semantically significant in any mainstream
language (unlike leading/indentation whitespace, which stays untouched),
loosening the equality check to ignore just those two things catches more
real "functionally unchanged" reformatting without any correctness risk.

### Design
Add `_normalize_for_comparison(text) -> str`: rstrip every line, collapse
runs of consecutive blank lines to a single blank line. Use this ONLY for
the equality check in `_collapse_unchanged_blocks` — the actual
stored/displayed content when a block does NOT collapse is always the real,
untouched text; normalization never touches what the model actually sees.

```python
def _normalize_for_comparison(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    normalized, prev_blank = [], False
    for ln in lines:
        is_blank = not ln
        if is_blank and prev_blank:
            continue
        normalized.append(ln)
        prev_blank = is_blank
    return "\n".join(normalized)
```

### Safety
Deliberately narrow: only trailing whitespace and blank-line-run count are
normalized away — leading (indentation) whitespace is never touched, so a
real reindentation or structural change still correctly blocks the
collapse. Any other single-character difference anywhere still blocks it
too, same as today.

### Tests to add
- Two versions differing only in trailing whitespace on one line → still
  collapses.
- Two versions differing only in blank-line count between statements →
  still collapses.
- Two versions differing in indentation (a real reformat) → does NOT
  collapse (this must still be treated as a real change).
- Two versions differing in an actual token/expression → does NOT collapse
  (regression guard — normalization must never mask a real change).

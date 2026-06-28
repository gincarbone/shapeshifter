# Copyright (c) 2026 Gaetano Marcello Incarbone. MIT License — see LICENSE file.
"""
Multi-turn coding benchmark for ShapeShifter context modes.

Simulates a real coding agent session: each turn adds to the conversation
history, which is then compressed by each mode's transformer before being
sent upstream. Measures token efficiency AND output quality (functional checks).

Usage:
    # Full run, all modes
    python benchmark_coding.py --scenario benchmarks/scenarios/html_landing_page.json

    # Subset of modes
    python benchmark_coding.py --scenario benchmarks/scenarios/html_landing_page.json --modes raw,hybrid,minimal,yaml

    # Dry run (no API calls, compression metrics only)
    python benchmark_coding.py --scenario benchmarks/scenarios/html_landing_page.json --local-only

    # Override model
    python benchmark_coding.py --scenario benchmarks/scenarios/html_landing_page.json --model deepseek/deepseek-v4-flash
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import html as html_module
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from llm_client import call_upstream
from output_contracts import build_system_prompt
from token_counter import compression_stats, count_tokens
from transformers import VALID_MODES, apply_transform

load_dotenv()

UPSTREAM_URL  = os.getenv("UPSTREAM_BASE_URL", "")
UPSTREAM_KEY  = os.getenv("UPSTREAM_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat")
MAX_TOKENS    = int(os.getenv("BENCHMARK_MAX_TOKENS", os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "4096")))

# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_checks(text: str, checks: list[dict]) -> list[dict]:
    results = []
    for c in checks:
        found = bool(re.search(c["pattern"], text, re.IGNORECASE | re.DOTALL))
        results.append({"name": c["name"], "passed": found})
    return results


# ---------------------------------------------------------------------------
# Multi-turn runner (one mode)
# ---------------------------------------------------------------------------

async def run_mode(
    scenario: dict,
    mode: str,
    model: str,
    max_tokens: int,
    local_only: bool,
) -> dict:
    """Run all scenario turns for a single context mode. Return per-turn metrics + final output."""
    turns_data = scenario["turns"]
    task_type  = scenario.get("task_type", "generic")
    system_prompt = build_system_prompt(mode, task_type)

    conversation: list[dict] = []   # grows with each turn
    turn_metrics: list[dict] = []
    final_output = ""
    total_latency = 0.0

    for i, turn in enumerate(turns_data):
        user_msg = turn["content"]
        label    = turn.get("label", f"Turn {i+1}")

        # Compress only the conversation HISTORY (not the current instruction).
        # The current user message is always sent verbatim so the model receives
        # the full requirements for this turn. What gets compressed is what the
        # model built in previous turns — this is the meaningful quality trade-off.
        if conversation:
            raw_ctx, compressed_history = apply_transform(mode, conversation)
            stats = compression_stats(raw_ctx, compressed_history)
        else:
            # Turn 1: no history to compress; measure token count of instruction only
            raw_ctx = user_msg
            compressed_history = ""
            stats = compression_stats(raw_ctx, raw_ctx)

        if local_only:
            turn_metrics.append({
                "turn": i + 1,
                "label": label,
                **stats,
                "latency_ms": 0,
                "output_tokens": 0,
                "response": "",
            })
            conversation.append({"role": "user",      "content": user_msg})
            conversation.append({"role": "assistant",  "content": "(local-only, no API call)"})
            continue

        # Build upstream messages:
        #   system → compressed history block (if any) → current instruction
        upstream_messages = [{"role": "system", "content": system_prompt}]
        if compressed_history:
            upstream_messages.append({
                "role": "user",
                "content": f"[COMPRESSED CONTEXT — previous turns]\n{compressed_history}"
            })
            upstream_messages.append({
                "role": "assistant",
                "content": "Understood. I'll continue building on the previous work."
            })
        upstream_messages.append({"role": "user", "content": user_msg})

        try:
            response, latency_ms = await call_upstream(
                base_url=UPSTREAM_URL,
                api_key=UPSTREAM_KEY,
                model=model,
                messages=upstream_messages,
                temperature=0.2,
                max_tokens=max_tokens,
            )
            answer = response["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            answer     = f"[ERROR] {exc}"
            latency_ms = 0.0

        output_tokens = count_tokens(answer)
        total_latency += latency_ms

        turn_metrics.append({
            "turn": i + 1,
            "label": label,
            **stats,
            "latency_ms": round(latency_ms, 1),
            "output_tokens": output_tokens,
            "response": answer,
        })

        # Add real exchange to conversation history for next turn
        conversation.append({"role": "user",      "content": user_msg})
        conversation.append({"role": "assistant",  "content": answer})

        final_output = answer

    # Extract artifact (strip markdown code fences if present)
    artifact = _extract_code(final_output, scenario.get("artifact_extension", "txt"))

    checks_results = run_checks(artifact or final_output, scenario.get("checks", []))

    total_saved  = sum(t["tokens_saved"] for t in turn_metrics)
    total_before = sum(t["tokens_before"] for t in turn_metrics)
    total_after  = sum(t["tokens_after"]  for t in turn_metrics)

    return {
        "mode":          mode,
        "model":         model,
        "turns":         turn_metrics,
        "artifact":      artifact,
        "checks":        checks_results,
        "total_tokens_saved":  total_saved,
        "total_tokens_before": total_before,
        "total_tokens_after":  total_after,
        "total_latency_ms":    round(total_latency, 1),
        "checks_passed":       sum(1 for c in checks_results if c["passed"]),
        "checks_total":        len(checks_results),
    }


def _extract_code(text: str, ext: str) -> str:
    """Strip markdown code fences; return the inner content."""
    # Try fenced block with language hint first
    m = re.search(r"```(?:html|css|js|javascript|python)?\s*\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: strip any ``` wrapping
    stripped = re.sub(r"^```[^\n]*\n?", "", text.strip())
    stripped = re.sub(r"\n?```$", "", stripped)
    if stripped != text.strip():
        return stripped.strip()
    return text.strip()


# ---------------------------------------------------------------------------
# Local-only compression report (no API)
# ---------------------------------------------------------------------------

def run_local(scenario: dict, modes: list[str]) -> list[dict]:
    turns_data = scenario["turns"]
    results = []
    for mode in modes:
        conversation: list[dict] = []
        total_saved = total_before = total_after = 0
        turns_metrics = []
        for i, turn in enumerate(turns_data):
            user_msg = turn["content"]
            messages_for_turn = conversation + [{"role": "user", "content": user_msg}]
            raw_ctx, transformed_ctx = apply_transform(mode, messages_for_turn)
            stats = compression_stats(raw_ctx, transformed_ctx)
            total_saved  += stats["tokens_saved"]
            total_before += stats["tokens_before"]
            total_after  += stats["tokens_after"]
            turns_metrics.append({"turn": i+1, "label": turn.get("label",""), **stats})
            conversation.append({"role": "user",      "content": user_msg})
            conversation.append({"role": "assistant",  "content": "(placeholder)"})
        results.append({
            "mode": mode,
            "turns": turns_metrics,
            "total_tokens_before": total_before,
            "total_tokens_after":  total_after,
            "total_tokens_saved":  total_saved,
            "total_latency_ms":    0,
            "artifact": "",
            "checks": [],
            "checks_passed": 0,
            "checks_total": 0,
        })
    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict], local_only: bool) -> None:
    print()
    if local_only:
        hdr = f"{'MODE':<12} {'TOT_BEFORE':>11} {'TOT_AFTER':>10} {'SAVED':>8} {'AVG_REDUC%':>11}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            before = r["total_tokens_before"]
            after  = r["total_tokens_after"]
            saved  = r["total_tokens_saved"]
            pct    = round((saved / before * 100) if before else 0, 1)
            print(f"{r['mode']:<12} {before:>11,} {after:>10,} {saved:>8,} {pct:>10.1f}%")
    else:
        hdr = f"{'MODE':<12} {'TOT_SAVED':>10} {'REDUC%':>8} {'LATENCY':>10} {'CHECKS':>8}"
        print(hdr)
        print("-" * len(hdr))
        for r in results:
            before = r["total_tokens_before"]
            saved  = r["total_tokens_saved"]
            pct    = round((saved / before * 100) if before else 0, 1)
            lat    = f"{r['total_latency_ms']:.0f}ms"
            chk    = f"{r['checks_passed']}/{r['checks_total']}"
            print(f"{r['mode']:<12} {saved:>10,} {pct:>7.1f}% {lat:>10} {chk:>8}")
    print()


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_REPORT_CSS = """
:root {
  --bg:#0f1117; --panel:#1a1d27; --border:#2a2d3e;
  --accent:#6c63ff; --green:#22c55e; --yellow:#eab308;
  --red:#ef4444; --text:#e2e2e2; --muted:#64748b;
  --font:'JetBrains Mono','Cascadia Code',Consolas,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--font);font-size:13px;padding:24px}
h1{font-size:20px;color:var(--accent);margin-bottom:4px}
.sub{color:var(--muted);font-size:11px;margin-bottom:24px}
table{width:100%;border-collapse:collapse;margin-bottom:20px}
th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:1px;
   text-align:left;padding:8px 10px;border-bottom:1px solid var(--border)}
td{padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:middle}
tr:last-child td{border-bottom:none}
.pill{display:inline-block;padding:2px 8px;border-radius:4px;font-size:10px;
      background:rgba(108,99,255,.15);color:var(--accent)}
.badge-g{background:rgba(34,197,94,.15);color:var(--green);border-radius:4px;padding:2px 8px;font-size:10px}
.badge-y{background:rgba(234,179,8,.15);color:var(--yellow);border-radius:4px;padding:2px 8px;font-size:10px}
.badge-r{background:rgba(239,68,68,.15);color:var(--red);border-radius:4px;padding:2px 8px;font-size:10px}
.section{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}
.section h2{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:12px}
details>summary{cursor:pointer;color:var(--accent);font-size:12px;padding:6px 0;user-select:none}
details>summary:hover{opacity:.8}
.turn-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:10px 0}
.turn-card{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;font-size:11px}
.turn-card .lbl{color:var(--muted);font-size:9px;text-transform:uppercase;margin-bottom:4px}
.turn-card .val{font-size:14px;font-weight:bold}
.preview-wrap{position:relative;margin-top:12px}
.preview-wrap iframe{width:100%;height:520px;border:1px solid var(--border);border-radius:6px;background:#fff}
.checks-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chk{font-size:10px;padding:2px 8px;border-radius:4px}
.chk.ok{background:rgba(34,197,94,.15);color:var(--green)}
.chk.fail{background:rgba(239,68,68,.15);color:var(--red)}
.bar-wrap{width:100%;background:var(--border);border-radius:3px;height:5px;margin-top:4px}
.bar{height:5px;border-radius:3px;background:var(--green)}
"""

def _checks_badge(passed: int, total: int) -> str:
    if total == 0:
        return '<span class="badge-y">N/A</span>'
    pct = passed / total
    cls = "badge-g" if pct >= 0.8 else ("badge-y" if pct >= 0.5 else "badge-r")
    return f'<span class="{cls}">{passed}/{total}</span>'


def _pct_color(pct: float) -> str:
    return "var(--green)" if pct >= 40 else ("var(--yellow)" if pct >= 15 else "var(--red)")


def _render_mode_section(r: dict, local_only: bool) -> str:
    mode   = r["mode"]
    before = r["total_tokens_before"]
    after  = r["total_tokens_after"]
    saved  = r["total_tokens_saved"]
    pct    = round((saved / before * 100) if before else 0, 1)
    lat    = r["total_latency_ms"]

    # Turn cards
    turn_cards = ""
    for t in r["turns"]:
        t_pct = round(t["reduction_pct"], 1)
        turn_cards += f"""
        <div class="turn-card">
          <div class="lbl">Turn {t['turn']} — {html_module.escape(t['label'][:22])}</div>
          <div class="val" style="color:{_pct_color(t_pct)}">{t_pct}%</div>
          <div class="lbl" style="margin-top:6px">saved {t['tokens_saved']:,} tok</div>
          <div class="bar-wrap"><div class="bar" style="width:{min(100,max(0,t_pct))}%;background:{_pct_color(t_pct)}"></div></div>
          {f'<div class="lbl" style="margin-top:4px">{t["latency_ms"]:.0f}ms · {t.get("output_tokens",0)} out</div>' if not local_only else ''}
        </div>"""

    # Checks
    checks_html = ""
    for c in r.get("checks", []):
        cls  = "ok" if c["passed"] else "fail"
        icon = "✓" if c["passed"] else "✗"
        checks_html += f'<span class="chk {cls}">{icon} {html_module.escape(c["name"])}</span>'

    # Artifact preview
    artifact = r.get("artifact", "")
    preview_html = ""
    if artifact and not local_only:
        b64 = base64.b64encode(artifact.encode("utf-8")).decode()
        preview_html = f"""
        <div class="preview-wrap">
          <div class="lbl" style="color:var(--muted);font-size:10px;text-transform:uppercase;margin-bottom:6px">
            Final output preview
          </div>
          <iframe id="frame-{mode}" onload="injectFrame('{mode}')"></iframe>
          <script>
            window._artifacts = window._artifacts || {{}};
            window._artifacts['{mode}'] = atob('{b64}');
          </script>
        </div>"""

    lat_display = f"{lat:.0f}ms total" if not local_only else "—"

    return f"""
    <div class="section">
      <details {"open" if mode in ("raw","hybrid") else ""}>
        <summary>
          <span class="pill">{mode}</span>&nbsp;&nbsp;
          saved <strong style="color:var(--green)">{saved:,}</strong> tokens
          &nbsp;·&nbsp; <strong style="color:{_pct_color(pct)}">{pct}%</strong> reduction
          &nbsp;·&nbsp; {lat_display}
          {"&nbsp;·&nbsp; checks " + _checks_badge(r["checks_passed"], r["checks_total"]) if not local_only else ""}
        </summary>

        <div class="turn-grid">{turn_cards}</div>

        {('<div style="margin-top:8px"><div class="lbl" style="color:var(--muted);font-size:9px;text-transform:uppercase;margin-bottom:6px">Functionality checks</div><div class="checks-row">' + checks_html + '</div></div>') if checks_html else ""}

        {preview_html}
      </details>
    </div>"""


def generate_report(scenario: dict, results: list[dict], out_dir: Path, local_only: bool) -> Path:
    name = scenario.get("name", "benchmark")
    desc = scenario.get("description", "")

    # Summary table
    rows = ""
    for r in results:
        before = r["total_tokens_before"]
        saved  = r["total_tokens_saved"]
        pct    = round((saved / before * 100) if before else 0, 1)
        lat    = f"{r['total_latency_ms']:.0f}ms" if not local_only else "—"
        rows += f"""<tr>
          <td><span class="pill">{r['mode']}</span></td>
          <td>{r['total_tokens_before']:,}</td>
          <td>{r['total_tokens_after']:,}</td>
          <td style="color:{_pct_color(pct)}"><strong>{pct}%</strong></td>
          <td style="color:var(--green)">{saved:,}</td>
          <td>{lat}</td>
          <td>{_checks_badge(r['checks_passed'], r['checks_total']) if not local_only else "—"}</td>
        </tr>"""

    mode_sections = "".join(_render_mode_section(r, local_only) for r in results)

    inject_script = """
<script>
function injectFrame(mode) {
  var f = document.getElementById('frame-' + mode);
  if (!f || !window._artifacts || !window._artifacts[mode]) return;
  var doc = f.contentDocument || f.contentWindow.document;
  doc.open(); doc.write(window._artifacts[mode]); doc.close();
}
window.addEventListener('DOMContentLoaded', function() {
  Object.keys(window._artifacts || {}).forEach(injectFrame);
});
</script>""" if not local_only else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ShapeShifter Coding Benchmark — {html_module.escape(name)}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<h1>⚡ ShapeShifter Coding Benchmark</h1>
<p class="sub">{html_module.escape(desc)} &nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="section">
  <h2>Summary</h2>
  <table>
    <thead><tr>
      <th>Mode</th><th>Tok Before</th><th>Tok After</th>
      <th>Reduction</th><th>Saved</th><th>Latency</th><th>Checks</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>
</div>

{mode_sections}

{inject_script}
</body>
</html>"""

    report_path = out_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(scenario: dict, results: list[dict], output_base: str, local_only: bool) -> Path:
    name     = scenario.get("name", "benchmark")
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir  = Path(output_base) / f"{date_str}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics JSON
    safe = [{k: v for k, v in r.items() if k != "artifact"} for r in results]
    (out_dir / "metrics.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # artifacts (final HTML / code output per mode)
    ext = scenario.get("artifact_extension", "txt")
    for r in results:
        artifact = r.get("artifact", "")
        if artifact:
            (out_dir / f"{r['mode']}.{ext}").write_text(artifact, encoding="utf-8")

    # HTML report
    report_path = generate_report(scenario, results, out_dir, local_only)

    return out_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="ShapeShifter multi-turn coding benchmark")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument(
        "--modes", default=",".join(sorted(VALID_MODES)),
        help="Comma-separated modes to test (default: all)"
    )
    parser.add_argument("--model", default="", help="Override model (default: from .env)")
    parser.add_argument("--local-only", action="store_true",
                        help="Skip API calls; measure compression only")
    parser.add_argument("--output-dir", default="benchmark_results",
                        help="Base directory for results")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help=f"Max output tokens per turn (default: {MAX_TOKENS})")
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    model    = args.model.strip() or DEFAULT_MODEL
    modes    = [m.strip() for m in args.modes.split(",") if m.strip() in VALID_MODES]

    if not modes:
        print(f"No valid modes. Valid: {sorted(VALID_MODES)}")
        return

    print(f"\n  Scenario : {scenario['name']}")
    print(f"  Turns    : {len(scenario['turns'])}")
    print(f"  Modes    : {', '.join(modes)}")
    print(f"  Model    : {model}")
    print(f"  API      : {'disabled (local-only)' if args.local_only else UPSTREAM_URL or '(not set)'}")
    print()

    if args.local_only:
        results = run_local(scenario, modes)
    else:
        if not UPSTREAM_URL or not UPSTREAM_KEY:
            print("WARNING: upstream not configured — switching to local-only")
            results = run_local(scenario, modes)
            args.local_only = True
        else:
            tasks = [
                run_mode(scenario, mode, model, args.max_tokens, local_only=False)
                for mode in modes
            ]
            print(f"  Running {len(modes)} modes in parallel...\n")
            results = await asyncio.gather(*tasks)

    print_summary(results, args.local_only)

    out_dir = save_results(scenario, list(results), args.output_dir, args.local_only)
    report  = out_dir / "report.html"
    print(f"  Results  : {out_dir}")
    print(f"  Report   : {report}\n")


if __name__ == "__main__":
    asyncio.run(main())

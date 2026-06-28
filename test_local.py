"""
Local compression test - no API key required.

Measures real token compression across all modes for all sample prompts.
Run with:
    python test_local.py

Or with verbose transformer output:
    python test_local.py --verbose
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from token_counter import compression_stats, count_tokens
from transformers import VALID_MODES, apply_transform
from output_contracts import build_system_prompt, detect_contract_type

SAMPLE_DIR = Path("prompts/samples")


def _load_samples() -> dict[str, str]:
    samples = {}
    if SAMPLE_DIR.exists():
        for f in sorted(SAMPLE_DIR.glob("*.txt")):
            samples[f.stem] = f.read_text(encoding="utf-8")
    if not samples:
        # fallback inline sample
        samples["inline_debug"] = (
            "I have a Tomcat 9 error on a JSP page:\n"
            "org.apache.jasper.JasperException: variable response is already defined\n"
            "at ev2jira_upd.jsp line 47:\n"
            "  JSONObject response = new JSONObject();\n"
            "  response.put('action_type_id', 19);\n"
            "Java 8, Tomcat 9, JSP 2.3. Fix the file."
        )
    return samples


def run_all(verbose: bool = False) -> dict:
    samples = _load_samples()
    modes = sorted(VALID_MODES)
    all_results: dict[str, list[dict]] = {}

    for sample_name, prompt_text in samples.items():
        print(f"\n{'='*70}")
        print(f"SAMPLE: {sample_name}  ({count_tokens(prompt_text)} tokens, {len(prompt_text)} chars)")
        print(f"{'='*70}")

        messages = [{"role": "user", "content": prompt_text}]
        contract_type = detect_contract_type(messages)

        col_w = 10
        print(f"{'MODE':<{col_w}} {'BEFORE':>8} {'AFTER':>8} {'RATIO':>7} {'REDUCT%':>8} {'SAVED':>7}")
        print("-" * 58)

        rows = []
        for mode in modes:
            raw_ctx, transformed_ctx = apply_transform(mode, messages)
            system_prompt = build_system_prompt(mode, contract_type)
            final_payload = system_prompt + "\n\n" + transformed_ctx
            stats = compression_stats(raw_ctx, final_payload)
            rows.append({"mode": mode, **stats, "contract_type": contract_type})

            print(
                f"{mode:<{col_w}} "
                f"{stats['tokens_before']:>8} "
                f"{stats['tokens_after']:>8} "
                f"{stats['compression_ratio']:>7.3f} "
                f"{stats['reduction_pct']:>7.1f}% "
                f"{stats['tokens_saved']:>7}"
            )

            if verbose:
                print("\n--- TRANSFORMED CONTEXT ---")
                print(transformed_ctx[:2500])
                print("--- END ---\n")

        all_results[sample_name] = rows

    return all_results


def summarize(results: dict[str, list[dict]]) -> None:
    print("\n" + "="*70)
    print("AVERAGE BY MODE")
    print("="*70)
    modes = sorted({row["mode"] for rows in results.values() for row in rows})

    print("  NOTE: tokens_after includes system prompt overhead for all modes")
    print("  RAW is passthrough plus the output contract; it is the cloud baseline")
    print()

    raw_costs = {}
    for sample_name, rows in results.items():
        raw_row = next((r for r in rows if r["mode"] == "raw"), None)
        if raw_row:
            raw_costs[sample_name] = raw_row["tokens_after"]

    print(f"{'MODE':<10} {'AVG_BEFORE':>11} {'AVG_AFTER':>10} {'AVG_RATIO':>10} {'AVG_REDUCT%':>11} {'VS_RAW%':>9}")
    print("-" * 70)
    for mode in modes:
        mode_rows = [r for rows in results.values() for r in rows if r["mode"] == mode]
        avg_before = sum(r["tokens_before"] for r in mode_rows) / len(mode_rows)
        avg_after  = sum(r["tokens_after"] for r in mode_rows) / len(mode_rows)
        avg_ratio  = sum(r["compression_ratio"] for r in mode_rows) / len(mode_rows)
        avg_red    = sum(r["reduction_pct"] for r in mode_rows) / len(mode_rows)

        vs_raw_parts = []
        for sample_name, rows in results.items():
            mode_row = next((r for r in rows if r["mode"] == mode), None)
            raw_cost = raw_costs.get(sample_name)
            if mode_row and raw_cost:
                vs_raw_parts.append((mode_row["tokens_after"] - raw_cost) / raw_cost * 100)
        vs_raw = sum(vs_raw_parts) / len(vs_raw_parts) if vs_raw_parts else 0

        print(
            f"{mode:<10} {avg_before:>11.0f} {avg_after:>10.0f} "
            f"{avg_ratio:>10.3f} {avg_red:>10.1f}% {vs_raw:>8.1f}%"
        )

    print()
    print("  VS_RAW%: negative = tokens saved vs sending raw context to the cloud")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Print transformed context for each mode")
    parser.add_argument("--json", action="store_true", help="Print JSON results")
    args = parser.parse_args()

    results = run_all(verbose=args.verbose)
    summarize(results)

    if args.json:
        print("\nJSON:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

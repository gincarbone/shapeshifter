"""
Benchmark batch — runs the same prompt through multiple context modes.

Usage (no API calls, local metrics only):
    python benchmark.py --input prompts/samples/debug_jsp.txt --local-only

Usage (real API calls):
    python benchmark.py --input prompts/samples/debug_jsp.txt --modes raw,minimal,yaml,json,table,hybrid,symbolic,matrix
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import os

from token_counter import compression_stats, count_tokens
from transformers import TRANSFORMERS, VALID_MODES, apply_transform
from output_contracts import build_system_prompt, detect_contract_type
from llm_client import call_upstream

load_dotenv()

UPSTREAM_URL  = os.getenv("UPSTREAM_BASE_URL", "")
UPSTREAM_KEY  = os.getenv("UPSTREAM_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat")
MAX_TOKENS    = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "1200"))


def _load_prompt(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def local_benchmark(prompt_text: str, modes: list[str]) -> list[dict]:
    """Run transformations locally and return compression metrics (no API calls)."""
    messages = [{"role": "user", "content": prompt_text}]
    results = []
    for mode in modes:
        raw_ctx, transformed_ctx = apply_transform(mode, messages)
        stats = compression_stats(raw_ctx, transformed_ctx)
        results.append({"mode": mode, **stats})
    return results


async def api_benchmark(prompt_text: str, modes: list[str]) -> list[dict]:
    """Run each mode against the real LLM and return full metrics."""
    messages = [{"role": "user", "content": prompt_text}]
    results = []
    for mode in modes:
        raw_ctx, transformed_ctx = apply_transform(mode, messages)
        stats = compression_stats(raw_ctx, transformed_ctx)

        contract_type = detect_contract_type(messages)
        system_prompt = build_system_prompt(mode, contract_type)
        new_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transformed_ctx},
        ]

        try:
            t0 = time.monotonic()
            response, latency_ms = await call_upstream(
                base_url=UPSTREAM_URL,
                api_key=UPSTREAM_KEY,
                model=DEFAULT_MODEL,
                messages=new_messages,
                max_tokens=MAX_TOKENS,
            )
            answer_text = response["choices"][0]["message"]["content"]
            output_tokens = count_tokens(answer_text)
        except Exception as exc:
            answer_text = f"[ERROR] {exc}"
            output_tokens = 0
            latency_ms = 0.0

        results.append({
            "mode": mode,
            **stats,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "response_preview": answer_text[:300],
        })
    return results


def _print_table(results: list[dict], include_latency: bool = False) -> None:
    print()
    if include_latency:
        header = f"{'MODE':<10} {'IN_TOKENS':>10} {'OUT_TOKENS':>11} {'RATIO':>7} {'REDUCTION':>10} {'LATENCY':>9}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['mode']:<10} "
                f"{r['tokens_before']:>10} "
                f"{r.get('output_tokens', '-'):>11} "
                f"{r['compression_ratio']:>7.3f} "
                f"{r['reduction_pct']:>9.1f}% "
                f"{r.get('latency_ms', '-'):>8.0f}ms"
            )
    else:
        header = f"{'MODE':<10} {'IN_TOKENS_BEFORE':>17} {'IN_TOKENS_AFTER':>16} {'RATIO':>7} {'REDUCTION':>10} {'TOKENS_SAVED':>13}"
        print(header)
        print("-" * len(header))
        for r in results:
            print(
                f"{r['mode']:<10} "
                f"{r['tokens_before']:>17} "
                f"{r['tokens_after']:>16} "
                f"{r['compression_ratio']:>7.3f} "
                f"{r['reduction_pct']:>9.1f}% "
                f"{r['tokens_saved']:>13}"
            )
    print()


def _save_results(results: list[dict], input_path: str, output_dir: str) -> None:
    stem = Path(input_path).stem
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(output_dir) / f"{date_str}_{stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    for r in results:
        if "response_preview" in r:
            (out_dir / f"{r['mode']}.md").write_text(
                r.get("response_preview", ""), encoding="utf-8"
            )
    print(f"Results saved to: {out_dir}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="ShapeShifter benchmark")
    parser.add_argument("--input", required=True, help="Path to prompt file")
    parser.add_argument(
        "--modes",
        default=",".join(sorted(VALID_MODES)),
        help="Comma-separated list of modes to benchmark",
    )
    parser.add_argument("--local-only", action="store_true", help="Skip API calls, measure compression only")
    parser.add_argument("--output-dir", default="benchmark_results", help="Where to save results")
    args = parser.parse_args()

    prompt_text = _load_prompt(args.input)
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in VALID_MODES]
    if not modes:
        print(f"No valid modes specified. Valid: {sorted(VALID_MODES)}")
        return

    print(f"Benchmarking {len(modes)} modes on: {args.input}")
    print(f"Original prompt: {count_tokens(prompt_text)} tokens, {len(prompt_text)} chars")

    if args.local_only:
        results = local_benchmark(prompt_text, modes)
        _print_table(results, include_latency=False)
    else:
        if not UPSTREAM_URL or not UPSTREAM_KEY or "xxxx" in UPSTREAM_KEY:
            print("WARNING: No upstream API configured — falling back to local-only mode")
            results = local_benchmark(prompt_text, modes)
            _print_table(results, include_latency=False)
        else:
            results = await api_benchmark(prompt_text, modes)
            _print_table(results, include_latency=True)

    _save_results(results, args.input, args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())

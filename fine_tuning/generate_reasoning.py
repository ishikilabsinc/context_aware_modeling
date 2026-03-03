#!/usr/bin/env python3
"""Generate CoT reasoning via Gemini; reads train_samples.jsonl, writes train_samples_with_reasoning.jsonl."""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.data_utils import filter_samples_with_context, load_samples

from fine_tuning.sampling_utils import (
    EQUAL_SAMPLING_SEED,
    EQUAL_SAMPLING_SPGI_TARGET,
    stratified_subsample_spgi,
)

TEACHER_SYSTEM_SUMMARY = """You are helping generate training data for a turn-taking prediction model.
The model uses this framework: (1) ACTIVE PARTICIPANT = target speaker has been speaking, was addressed, or is part of the current exchange. (2) BYSTANDER = target speaker has not been involved and is passively listening. The model predicts whether the target speaker should SPEAK or STAY SILENT after a pause."""

TEACHER_USER_TEMPLATE = """Given the conversation context below, the target speaker is {target_speaker}.
The ground truth decision is: {decision}

Write a 1-sentence reasoning that explains WHY {target_speaker} should {decision} based on the conversation dynamics.

Rules:
- Focus on WHO is being addressed, what was said, and the target speaker's role (active participant vs bystander)
- Reference specific conversational cues (e.g., "Monica asked Joey directly about the audition" or "Ross and Rachel are having a private exchange that doesn't involve Joey")
- Do NOT restate the label (don't say "joey spoke" or "joey stays silent")
- Do NOT mention "ground truth" or that you were told the answer
- Keep it to exactly 1 sentence
- Start with whether the target speaker is an ACTIVE PARTICIPANT or BYSTANDER

Conversation context:
{context}

Most recent utterance:
{recent_utterance}

Target speaker: {target_speaker}

Write the reasoning:"""


def format_context(turns: List[Dict]) -> str:
    if not turns:
        return "(No previous context)"
    return "\n".join(
        f"Speaker {t.get('speaker', '?')}: {t.get('text', '')}"
        for t in turns
    )


def build_teacher_prompt(sample: Dict) -> str:
    context_turns = sample.get("context_turns", [])
    current_turn = sample.get("current_turn", {})
    target_speaker = sample.get("target_speaker", "?")
    decision = sample.get("decision", "SILENT")
    if decision not in ("SPEAK", "SILENT"):
        decision = "SILENT" if str(decision).upper() == "SILENT" else "SPEAK"
    context_str = format_context(context_turns)
    recent = "(No current utterance)"
    if current_turn:
        recent = f"Speaker {current_turn.get('speaker', '?')}: {current_turn.get('text', '')}"
    return TEACHER_USER_TEMPLATE.format(
        target_speaker=target_speaker,
        decision=decision,
        context=context_str,
        recent_utterance=recent,
    )


_MIN_SENTENCE_PREFIX_LEN = 50


def extract_reasoning_from_response(text: str) -> str:
    """Parse Gemini output to one sentence; strip <reasoning> tags; cap at 500 chars."""
    text = (text or "").strip()
    m = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    for sentence_end in re.finditer(r"\.\s+(?=[A-Z])", text):
        if sentence_end.start() >= _MIN_SENTENCE_PREFIX_LEN:
            text = text[: sentence_end.end()].strip()
            break
    else:
        end_match = re.search(r"\.\s*$", text)
        if end_match:
            text = text[: end_match.end()].strip()
    if text and not text.endswith("."):
        text = text + "."
    return text[:500] if text else "No reasoning provided."


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def init_gemini(api_key: Optional[str] = None, model_name: Optional[str] = None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        print(
            "ERROR: Set GEMINI_API_KEY (env or --api-key). "
            "Get a key at https://aistudio.google.com/app/apikey"
        )
        sys.exit(1)
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Install the new SDK: pip install google-genai")
        sys.exit(1)
    client = genai.Client(api_key=api_key)
    name = model_name or DEFAULT_GEMINI_MODEL
    config = types.GenerateContentConfig(
        system_instruction=TEACHER_SYSTEM_SUMMARY,
        temperature=0.3,
        max_output_tokens=1024,
    )
    return client, name, config


def call_teacher(
    client,
    model_name: str,
    config,
    prompt: str,
    max_retries: int = 3,
    delay_seconds: float = 4.0,
    debug: bool = False,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            text = (getattr(response, "text", None) or "").strip()
            finish_reason = None
            if getattr(response, "candidates", None) and len(response.candidates) > 0:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
            if debug:
                print(f"RAW: {text[:200]}")
                print(f"FINISH: {finish_reason}")
            if text:
                return extract_reasoning_from_response(text)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay_seconds * (attempt + 1))
            else:
                raise e
        time.sleep(delay_seconds)
    return "No reasoning provided."


def get_completed_ids(path: Path) -> set:
    out = set()
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                sid = obj.get("decision_point_id") or obj.get("sequence_id")
                if sid:
                    out.add(sid)
            except json.JSONDecodeError:
                continue
    return out


class RateLimiter:
    def __init__(self, delay: float):
        self.delay = delay
        self._lock = threading.Lock()
        self._last_acquire = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_time = self._last_acquire + self.delay - now
            if wait_time > 0:
                time.sleep(wait_time)
            self._last_acquire = time.monotonic()


def _process_one(
    rate_limiter: RateLimiter,
    client,
    model_name: str,
    config,
    sample: Dict,
    delay: float,
    debug: bool,
) -> Tuple[Dict, str]:
    rate_limiter.wait()
    prompt = build_teacher_prompt(sample)
    try:
        reasoning = call_teacher(
            client, model_name, config, prompt,
            delay_seconds=delay, debug=debug,
        )
    except Exception:
        reasoning = "No reasoning provided."
    return (sample, reasoning)


def run_dataset(
    dataset: str,
    dry_run: Optional[int] = None,
    resume: bool = True,
    api_key: Optional[str] = None,
    delay: float = 5.0,
    model_name: Optional[str] = None,
    debug: bool = False,
    context_only: bool = False,
    workers: int = 1,
    equal_sampling: bool = False,
) -> None:
    data_dir = REPO_ROOT / "data" / dataset
    train_file = data_dir / "train" / "train_samples.jsonl"
    out_file = data_dir / "train" / "train_samples_with_reasoning.jsonl"
    if not train_file.exists():
        print(f"  Skip {dataset}: {train_file} not found")
        return
    samples = load_samples(train_file)
    if not samples:
        print(f"  Skip {dataset}: no samples")
        return
    if context_only:
        samples = filter_samples_with_context(samples)
        if not samples:
            print(f"  Skip {dataset}: no samples with context")
            return
        print(f"  {dataset}: filtering to {len(samples)} samples with context")
    if dataset == "spgi" and equal_sampling and len(samples) > EQUAL_SAMPLING_SPGI_TARGET:
        samples = stratified_subsample_spgi(
            samples,
            target_total=EQUAL_SAMPLING_SPGI_TARGET,
            seed=EQUAL_SAMPLING_SEED,
        )
        print(f"  {dataset}: equal-sampling to {len(samples):,} samples (stratified 50/50 SPEAK/SILENT, category-proportional)")
    if dry_run is not None and dry_run > 0:
        samples = samples[: dry_run]
        print(f"  Dry run: processing first {len(samples)} samples only")
    client, gemini_model, gen_config = init_gemini(api_key, model_name=model_name)
    # When equal_sampling for SPGI, write only the 11K subset so output file matches training; do not resume from a possibly full file.
    use_fresh_output = dataset == "spgi" and equal_sampling
    completed = set() if use_fresh_output else (get_completed_ids(out_file) if resume and out_file.exists() else set())
    to_process = [s for s in samples if (s.get("decision_point_id") or s.get("sequence_id") or str(id(s))) not in completed]
    if not to_process:
        print(f"  {dataset}: all {len(samples)} samples already in output (resume).")
        return
    out_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if use_fresh_output else ("a" if resume and out_file.exists() else "w")
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    def write_result(sample: Dict, reasoning: str, f) -> None:
        out = dict(sample)
        out["reasoning"] = reasoning
        confidence = sample.get("confidence", "medium")
        if confidence not in ("high", "medium", "low"):
            confidence = "medium"
        out["confidence"] = confidence
        f.write(json.dumps(out, ensure_ascii=False) + "\n")
        f.flush()

    if workers <= 1:
        with open(out_file, mode, encoding="utf-8") as f:
            for sample in tqdm(to_process, desc=f"{dataset}"):
                sid = sample.get("decision_point_id") or sample.get("sequence_id") or ""
                if sid in completed:
                    continue
                prompt = build_teacher_prompt(sample)
                try:
                    reasoning = call_teacher(
                        client, gemini_model, gen_config, prompt,
                        delay_seconds=delay, debug=debug,
                    )
                except Exception as e:
                    reasoning = "No reasoning provided."
                    if "tqdm" in sys.modules:
                        tqdm.write(f"  Error for {sid}: {e}")
                write_result(sample, reasoning, f)
                time.sleep(delay)
    else:
        rate_limiter = RateLimiter(delay)
        with open(out_file, mode, encoding="utf-8") as f:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _process_one,
                        rate_limiter,
                        client,
                        gemini_model,
                        gen_config,
                        sample,
                        delay,
                        debug,
                    ): sample
                    for sample in to_process
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"{dataset}"):
                    try:
                        sample, reasoning = future.result()
                        write_result(sample, reasoning, f)
                    except Exception as e:
                        sample = futures[future]
                        sid = sample.get("decision_point_id") or sample.get("sequence_id") or ""
                        if "tqdm" in sys.modules:
                            tqdm.write(f"  Error for {sid}: {e}")
                        write_result(sample, "No reasoning provided.", f)
    print(f"  {dataset}: wrote {len(to_process)} samples to {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning for CoT training using Gemini (teacher LLM)."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ami", "friends", "spgi"],
        help="Datasets to process (default: ami friends spgi)",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N",
        help="Process only N samples per dataset for manual review",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume; overwrite or start fresh (default: resume by appending new samples)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (default: GEMINI_API_KEY env)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        metavar="SECS",
        help="Seconds between API call starts (default: 5.0). Throughput ~1/delay req/s (e.g. 5->12/min, 1->60/min). Use 1.0 for 60 RPM paid quota to cut SPGI 79K from ~100h to ~22h.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent API workers (default: 1). Keep high (e.g. 64); global rate is still 1/delay req/s.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print raw Gemini response (first 200 chars) before extraction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="NAME",
        help=f"Gemini model ID (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="Process only train samples that have context_turns (non-empty conversation history).",
    )
    parser.add_argument(
        "--equal-sampling",
        action="store_true",
        help="For SPGI only: subsample to 11K (stratified 50/50 SPEAK/SILENT). Same as train_lora --equal_sampling. Use with --context-only to cut 79K to 11K and run ~7x faster.",
    )
    args = parser.parse_args()
    for dataset in args.datasets:
        run_dataset(
            dataset,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            api_key=args.api_key,
            delay=args.delay,
            model_name=args.model,
            debug=args.debug,
            context_only=args.context_only,
            workers=args.workers,
            equal_sampling=args.equal_sampling,
        )


if __name__ == "__main__":
    main()

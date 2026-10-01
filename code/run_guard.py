#!/usr/bin/env python3
"""Apply the frozen guard independently to JSONL text segments or one string.

JSONL output preserves all input fields, replaces text, and adds no scoring
fields. Optional audit JSONL contains counts, edit coordinates, and hashes.
Input and output transcripts are local derived corpus material, not public data.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from realmeetsep.asr_repetition_guard_v035 import guard_repetitions
from realmeetsep.asr_delooping_baselines_v046_postprimary import (
    collapse_ltr_character_repetitions, collapse_whisper_token_repetitions,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def transform(text: str, method: str = "canonical", tokenizer=None):
    if method == "canonical":
        result = guard_repetitions(text)
        events = []
        for event in result.events:
            row = asdict(event)
            row["unit_sha256"] = digest(row.pop("unit"))
            events.append(row)
    elif method == "ltr":
        result = collapse_ltr_character_repetitions(text)
        events = [event.audit_dict() for event in result.events]
    elif method == "token":
        if tokenizer is None:
            raise ValueError("token method requires a local frozen Whisper tokenizer")
        result = collapse_whisper_token_repetitions(text, tokenizer=tokenizer)
        events = [event.audit_dict() for event in result.events]
    else:
        raise ValueError(f"unknown method: {method}")
    return result.text, {
        "method": method, "input_sha256": digest(text),
        "output_sha256": digest(result.text), "input_characters": len(text),
        "output_characters": len(result.text), "removed_characters": result.removed_characters,
        "events": events,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--method", choices=("canonical", "ltr", "token"), default="canonical")
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    tokenizer = None
    if args.method == "token":
        if args.tokenizer_dir is None:
            parser.error("--method token requires --tokenizer-dir")
        from transformers import WhisperTokenizerFast
        tokenizer = WhisperTokenizerFast.from_pretrained(str(args.tokenizer_dir), local_files_only=True)
    if args.text is not None:
        output, audit = transform(args.text, args.method, tokenizer)
        print(json.dumps({"text": output, "audit": audit}, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--input requires --output")
    # Refuse overwrite before processing; input rows retain their original order.
    for path in (args.output, args.audit):
        if path is not None and path.exists():
            raise FileExistsError(path)
    outputs, audits = [], []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise ValueError(f"line {line_number}: expected an object with string text")
        text, audit = transform(row["text"], args.method, tokenizer)
        outputs.append(dict(row, text=text))
        audits.append(dict(audit, input_line=line_number))
    for path, rows in ((args.output, outputs), (args.audit, audits)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "segments": len(outputs), "modified": sum(bool(x["events"]) for x in audits)}))


if __name__ == "__main__":
    main()

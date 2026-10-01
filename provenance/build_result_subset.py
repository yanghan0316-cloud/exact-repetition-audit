"""Rebuild the public numeric subset from the authors' local research archive.

The archive is not distributed. No audio, transcripts, model weights, reviewer
worksheets, per-item listening labels, or private blinding maps are copied.
Run with Python 3.10+: python provenance/build_result_subset.py --source-root PATH
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.output.resolve()
    research = "code/ali_overlap_sep/"
    old = research + "paper/artifacts/v043/anonymous_review_artifact_v043/"
    records = []
    # Remove location metadata while retaining numeric values and integrity hashes.
    location_keys = {"path", "path_after_publish", "model_dir", "v5_protocol"}
    forbidden_content_keys = {
        "text", "raw_text", "guarded_text", "fallback_text", "fallback_guard_text",
        "removed_text", "removed_string", "transcript", "reference_text",
        "hypothesis_text", "comments", "comment", "notes", "audio_path",
    }
    absolute_path = re.compile(r"(?:[A-Za-z]:[\\/]|/root/|/home/)")
    cjk = re.compile("[\u3400-\u9fff]")

    def inspect(value, pointer=""):
        if isinstance(value, dict):
            for k, v in value.items():
                if k.lower() in forbidden_content_keys and isinstance(v, str):
                    raise ValueError(f"Content field at {pointer}/{k}")
                inspect(v, pointer + "/" + k)
        elif isinstance(value, list):
            for v in value:
                inspect(v, pointer + "[]")
        elif isinstance(value, str):
            if absolute_path.search(value) or cjk.search(value):
                raise ValueError(f"Unexpected path or speech-like content at {pointer}")

    def sanitize(value, removed, pointer=""):
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                p = pointer + "/" + k
                if k in location_keys or (isinstance(v, str) and absolute_path.search(v)):
                    removed.append(p)
                    continue
                result[k] = sanitize(v, removed, p)
            return result
        if isinstance(value, list):
            return [sanitize(v, removed, pointer + "[]") for v in value]
        return value

    def publish(src, dst, kind="copy", exclude=()):
        raw = (source / src).read_bytes()
        payload = raw
        record = {"source": src, "source_sha256": sha(raw), "release_path": dst}
        if kind == "json":
            data = json.loads(raw.decode("utf-8-sig"))
            removed = []
            for key in exclude:
                if key in data:
                    del data[key]
                    removed.append("/" + key)
            data = sanitize(data, removed)
            inspect(data)
            payload = (json.dumps(data, indent=2, ensure_ascii=True, sort_keys=True) + "\n").encode()
            record["transformation"] = "JSON projection; numeric fields unchanged; sorted UTF-8 serialization"
            record["removed_fields"] = sorted(set(removed))
        elif kind == "csv":
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
            fields = [f for f in reader.fieldnames if f not in exclude]
            rows = [{f: row[f] for f in fields} for row in reader]
            inspect(rows)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            payload = stream.getvalue().encode()
            record.update(transformation="CSV projection; retained cell values unchanged", removed_fields=list(exclude), rows=len(rows))
        elif kind == "trace_gzip":
            allowed = {"corpus", "decoder_condition", "deleted_unicode_characters", "events", "exact_event_count", "generated_system", "input_text_sha256", "input_unicode_characters", "method", "modified", "output_text_sha256", "output_unicode_characters", "residual_v035_exact_event_count", "residual_v035_modified", "schema", "segment_index", "session", "slot", "source", "transform_runtime_ns"}
            event_allowed = {"kept_span_chars", "kept_span_units", "method", "original_span_chars", "original_span_units", "original_start_char", "period_units", "removed_characters", "removed_end_char", "removed_start_char", "repetitions", "start_unit", "unit_kind", "unit_sha256"}
            rows = 0
            for line in raw.decode("utf-8-sig").splitlines():
                data = json.loads(line)
                if set(data) != allowed or any(set(e) - event_allowed for e in data["events"]):
                    raise ValueError("Unexpected content-free trace schema")
                inspect(data)
                rows += 1
            stream = io.BytesIO()
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as zipped:
                zipped.write(raw)
            payload = stream.getvalue()
            record.update(transformation="Schema-audited content-free JSONL; lossless gzip with mtime=0", rows=rows, uncompressed_sha256=sha(raw))
        else:
            if src.endswith(".json"):
                inspect(json.loads(raw.decode("utf-8-sig")))
            elif src.endswith(".csv"):
                inspect(list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))))
            record["transformation"] = "Byte-identical copy of reviewed content-free artifact"
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        record.update(release_sha256=sha(payload), release_bytes=len(payload))
        records.append(record)

    for name in ("eval36_summary.json", "eval36_recording_system_cpcer.csv", "eval36_paired_contrasts.csv", "empty_output_baseline.csv"):
        publish(old + "results/" + name, "results/" + name)

    decoder = research + "experiments/whisper_control_comparison_postprimary_v5/"
    publish(decoder + "analysis/RESULTS.json", "results/decoder_controls/RESULTS.json", "json")
    publish(decoder + "analysis/RESULTS.csv", "results/decoder_controls/RESULTS.csv", "csv")
    for corpus in ("aishell5", "misp2022"):
        publish(decoder + corpus + "/control_cpcer_fourteen_systems.csv", f"results/decoder_controls/{corpus}_recording_scores.csv", "csv", ("cp_assignment",))
        publish(decoder + corpus + "/control_transform_summary.json", f"results/decoder_controls/{corpus}_transform_summary.json", "json")

    misp = research + "experiments/streaming_v035_misp2022_eval8_v1/"
    publish(misp + "meeting_full_cpcer_four_systems.csv", "results/misp2022/recording_scores.csv", "csv", ("cp_assignment",))
    publish(misp + "misp2022_eval8_paired_analysis_v1.json", "results/misp2022/paired_analysis.json", "json")

    matched = research + "experiments/streaming_v035_eval36_matched_deletion_baselines_v1/"
    for name in ("meeting_scores_content_free.csv", "paired_comparisons.csv", "SUMMARY.json", "GO_NO_GO.json"):
        publish(matched + name, "results/matched_deletion/" + name, "json" if name.endswith("json") else "csv")

    sensitivity = research + "experiments/eval36_posthoc_oat_parameter_sensitivity_v1/"
    for name in ("oat_curves.csv", "oat_recording_scores.csv", "oat_split_aggregates.csv", "oat_variants.csv", "oat_robustness_summary.json"):
        publish(sensitivity + name, "results/parameter_sensitivity/" + name, "json" if name.endswith("json") else "csv")

    auxiliary = research + "experiments/streaming_v035_train12_four_system_stress_v1/"
    publish(auxiliary + "meeting_full_cpcer_four_systems.csv", "results/auxiliary/train12_recording_scores.csv", "csv", ("cp_assignment",))
    publish(auxiliary + "train12_four_system_stress_analysis_v1.json", "results/auxiliary/train12_analysis.json", "json")
    publish(research + "experiments/streaming_v039_fixed_screen12_900_v1/SCREEN12_CANONICAL_RESULT_v2.json", "results/auxiliary/screen12_negative_ablation.json", "json")

    direct = research + "experiments/delooping_direct_baselines_v046_postprimary_v1/"
    publish(direct + "aggregate_metrics.csv", "results/direct_baselines/aggregate_metrics.csv", "csv")
    publish(direct + "meeting_metrics.csv", "results/direct_baselines/meeting_metrics.csv", "csv", ("cp_assignment",))
    publish(direct + "SUMMARY.json", "results/direct_baselines/SUMMARY.json", "json")
    publish(direct + "content_free_event_audit.jsonl", "results/direct_baselines/content_free_event_audit.jsonl.gz", "trace_gzip")

    alignment = research + "paper/generated/v046/postprimary_alignment_retention/"
    for stem in ("absolute", "recording", "paired_recording", "paired_summary"):
        name = f"alignment_retention_{stem}_postprimary_v046.csv"
        publish(alignment + name, "results/alignment_retention/" + name, "csv", ("cp_assignment",) if stem == "recording" else ())
    publish(alignment + "alignment_retention_postprimary_v046.json", "results/alignment_retention/summary.json", "json")

    publish(old + "review/listening_review_aggregate.json", "results/listening/initial_aggregate.json", "json")
    probability = research + "paper/review/eval36_probability_listening_v1/analysis_v1/"
    publish(probability + "publication_summary.json", "results/listening/probability_summary.json", "json")
    publish(probability + "listening_probability_review_aggregate.json", "results/listening/probability_aggregate.json", "json", ("source_bindings", "private_evidence_receipt_status"))
    residual = research + "paper/review/postprimary_residual_listening_v046/analysis_postprimary_v047_corrected_unadjudicated/"
    publish(residual + "SUMMARY_POSTPRIMARY_V046.json", "results/listening/residual_aggregate.json", "json", ("notice",))

    for name in ("figure1_method_clocks.svg", "figure2_exact_period_guard.svg", "figure3_eval36_split_first.svg"):
        publish(old + "figures/" + name, "figures/historical_v043/" + name)
    publish("output/tmm_revision_v003/source/Figures/asr_pipeline_anonymous.pdf", "figures/asr_pipeline.pdf")
    publish("output/tmm_revision_v003/source/figure1_eval36_posthoc_action_map_anonymous.pdf", "figures/deletion_action_map.pdf")
    publish(research + "paper/figures/v044/figure1_eval36_posthoc_action_map_data.csv", "figures/deletion_action_map_data.csv", "csv")

    manuscript = []
    for name in ("main.tex", "supplement.tex"):
        p = "output/tmm_revision_v003/source/" + name
        manuscript.append({"source": p, "sha256": sha((source / p).read_bytes())})
    manifest = {
        "schema": "exact_repetition_public_result_provenance_v1",
        "manuscript": manuscript,
        "scope": "Public numeric/result-only projection of existing frozen evidence; no new experiments",
        "path_base": "Source paths are relative to the authors' research archive; they are not download links.",
        "listening_boundary": "Only aggregate listening results are released. Worksheets, per-item labels, comments, identities, audio, and blinding maps remain excluded. The residual summary is a public aggregate projection of a private package; its original package notice is omitted, not a grant to redistribute the private package.",
        "files": sorted(records, key=lambda r: r["release_path"]),
    }
    (out / "provenance").mkdir(parents=True, exist_ok=True)
    (out / "provenance/results_sources.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result_and_figure_files": len(records), "bytes": sum(r["release_bytes"] for r in records), "content_free_trace_rows": 80256}))


if __name__ == "__main__":
    main()

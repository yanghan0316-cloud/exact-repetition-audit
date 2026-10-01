# Executable scope and reproduction

Run commands from the repository root with Python 3.11 or newer. The canonical
guard and character baseline use only the standard library. Install
`python -m pip install -r requirements.txt` for the numeric verification and
the RapidFuzz scoring backend used for the recorded results.

```sh
python -m unittest discover -s tests -v
python code/verify_public_tables.py
python code/verify_extended_results.py
python code/run_guard.py --text abcdabcdabcdabcdabcdabcd
```

The last command returns `abcdabcd`, removes 16 characters, and reports the
selected span. Every fixture string is synthetic. The public verification
recomputes primary meeting scores and the frozen 50,000-draw bootstrap
contrasts; extended verification recomputes decoder/direct-baseline aggregates
from meeting counts and checks matched-budget accounting and method traces.
These count-based checks do not rerun audio recognition or listening judgments.

## Apply to locally held transcripts

```sh
python code/run_guard.py --input code/examples/synthetic_hypotheses.jsonl --output local/guarded.jsonl --audit local/guard_audit.jsonl
python code/scripts/score_meeting_cpcer.py --references code/examples/synthetic_references.jsonl --hypotheses local/guarded.jsonl --output local/guarded_score.json
```

The synthetic example scores 0 cpCER after the guard. The raw example scores
1.0. Each JSONL input must contain a `text` string; other fields are retained
unchanged. One row is processed independently, with no cross-row context and
no reference access. Scoring needs the stricter schema documented in
`scripts/score_meeting_cpcer.py`; it concatenates each anonymous stream in
chronological order and assigns speakers once per full meeting. The exact
permutation scorer supports up to eight streams.

Choose `--method ltr` for the direct left-to-right baseline. Choose
`--method token --tokenizer-dir PATH_TO_LOCAL_WHISPER_BASE` for the frozen
token baseline, after installing the decoder requirements. Tokenizer bytes
must be acquired separately from the pinned Whisper revision listed by
`python code/verify_external_assets.py`. The synthetic tokenizer tests exercise
offset-boundary behavior; they do not substitute for the real Whisper tokenizer.

Guard defaults remain `(max_period_chars, min_repetitions, min_span_chars,
keep_repetitions) = (12, 6, 24, 2)`. Candidate priority is largest removable
span, then shorter period, then earlier start. Candidates are recomputed after
each edit. The production implementation requires `min_repetitions >= 3` and
`0 < keep_repetitions < min_repetitions`. It operates on Python Unicode code
points before scoring normalization. Event coordinates refer to the current
string at that step, not necessarily the original string. The original module
returns the repeated unit and supports reverse reconstruction from its ordered
edit trace. The command-line sidecar replaces units with hashes; keep the
original input privately if reconstruction is required.

## Decoder and acoustic dependencies

`scripts/decode_whisper_meeting.py` and
`scripts/decode_whisper_cap224_fallback_postprimary_v4.py` are byte-identical
frozen research implementations. They expose `--help` after installing
`requirements-decoder.txt`. The latter retains original-batch membership,
cardinality checks, sealed-prefix checks, temperature seed derivation, and
OpenAI-style average-logprob accounting. The audited v5 result assembly
reused v4 decoder output; it did not perform another decode.

The frozen decoder command intentionally requires the original local audio
manifests, prepared-input receipts, raw hypotheses, decode-batch map, and
matching execution seal. The sanitized protocol in `protocol/` documents the
contract but has a different hash from the original seal. This repository
therefore provides runnable text intervention, scoring, aggregate checks,
decoder source, and synthetic decoder tests; it is not a turnkey rebuild of
the complete acoustic experiment. Licensed corpora, packet-MVDR waveform
exports, privately trained frontend weights, raw transcripts, and private
execution receipts are not bundled. Reconstructing an acoustic run requires
acquiring/creating those inputs and recording a new execution seal; do not
disable the original checks or describe a new run as the historical one.

`src/realmeetsep/postprimary_alignment_retention_v046.py` and its bootstrap
module expose the original retention estimators. The decoder helper is an
independent fixed-window compression branch, not the complete OpenAI
long-form/no-speech procedure. GPU, batching, and library changes may change
sampled outputs. The pure-Python edit-distance fallback preserves total
distance, but S/D/I alignment ties can differ from RapidFuzz; use the pinned
backend when comparing decomposed counts.

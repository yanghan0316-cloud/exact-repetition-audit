# Auditing Exact-Repetition Containment in Multi-Speaker Transcription

[中文说明](README.zh-CN.md)

Companion code and numeric artifacts for the manuscript **Auditing Exact-Repetition
Containment in Multi-Speaker Transcription**, corresponding to the saved
`tmm_revision_v003` manuscript snapshot. This release supports computational
checks of the reported results and reuse of the exact-repetition operator on
locally held transcripts.

The study connects decoder controls, deterministic text editing, and
deletion-sensitive evaluation across 36 AISHELL-5 recordings and eight MISP2022
meetings. The text operator contracts eligible exact periodic runs with frozen
parameters `(12, 6, 24, 2)`: maximum period, minimum repetitions, minimum span,
and retained copies. It records ordered edits and uses no reference transcript.
The release also contains direct character/token baselines, meeting-level
cpCER scoring, frozen decoder source, and alignment-retention estimators.

## Run the checks

Use Python 3.11 or newer. From the repository root:

```sh
python code/verify_manifest.py
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python code/verify_public_tables.py
python code/verify_extended_results.py
python code/run_guard.py --text abcdabcdabcdabcdabcdabcd
```

The first command checks the release SHA-256 manifest; the final command returns
`abcdabcd`. The numeric checks recompute 144 original
Eval36 score rows and 12 paired-bootstrap contrasts, and verify 616 decoder
meeting rows, 48 direct-baseline aggregate cells, 864 matched-deletion rows,
and 80,256 method traces. The standard-library test runner exercises the frozen
guard, direct baselines, normalization, speaker assignment, and reconstruction.
Three optional tensor tests are skipped when PyTorch is absent. GitHub Actions
runs the core tests and both numeric checks without downloading a corpus or
model.

For the optional decoder/tokenizer dependencies:

```sh
python -m pip install -r requirements-decoder.txt
python -m unittest discover -s tests -v
```

The recorded local validation passed all 23 tests, including those three tensor
tests. See [validation](environment/VALIDATION.json) and the
[recorded environment](environment/reproduction_environment.json).

## Try the synthetic transcript example

```sh
python code/run_guard.py --input code/examples/synthetic_hypotheses.jsonl --output local/guarded.jsonl --audit local/guard_audit.jsonl
python code/scripts/score_meeting_cpcer.py --references code/examples/synthetic_references.jsonl --hypotheses local/guarded.jsonl --output local/guarded_score.json
```

The synthetic raw transcript scores cpCER 1.0; the guarded transcript scores
0.0. Each input row is processed independently. Output files are local, and the
guard command refuses to overwrite an existing file. The sidecar contains edit
geometry and hashes, with repeated text omitted. See [code usage and input
contracts](code/README.md) for the left-to-right and Whisper-token alternatives,
scoring schema, trace reconstruction, and decoder prerequisites.

## Find the evidence

| Paper evidence | Released files |
| --- | --- |
| Original AISHELL-5 Primary / Replication contrasts | [Summary](results/eval36_summary.json), [meeting scores](results/eval36_recording_system_cpcer.csv), [paired contrasts](results/eval36_paired_contrasts.csv), [empty anchor](results/empty_output_baseline.csv) |
| Token cap and compression-triggered fallback | [Decoder controls](results/decoder_controls/) |
| Direct character and Whisper-token baselines | [Direct baselines and compressed method traces](results/direct_baselines/) |
| Equal-length deletion and retention diagnostics | [Matched deletion](results/matched_deletion/), [alignment retention](results/alignment_retention/) |
| Listening audits | [Aggregate judgments](results/listening/) |
| Supportive MISP2022 results | [MISP2022](results/misp2022/) |
| Parameter sensitivity and auxiliary outcomes | [Sensitivity](results/parameter_sensitivity/), [auxiliary results](results/auxiliary/) |
| Figures and plotted data | [Figures](figures/) |

The [paper evidence map](provenance/paper_evidence_map.json) connects manuscript
tables and claims to individual files. [Result provenance](provenance/results_sources.json)
and [source provenance](environment/SOURCE_PROVENANCE.json) record original
relative paths, transformations, and SHA-256 bindings. The
[public evaluation protocol](protocol/eval36_public_protocol.json) and
[decoder protocol](protocol/whisper_control_comparison_postprimary_v4.json)
preserve thresholds, seeds, grouping rules, and interpretation boundaries.
Historical version identifiers in filenames identify frozen research artifacts;
they are not separate software release versions.

## Reproducibility scope

The released code runs the text operator, direct baselines, synthetic contracts,
and meeting scorer. The released counts support independent aggregation and
primary paired-bootstrap verification. Content-free method traces support
output-hash and edit-accounting checks. These are computational reproduction
and consistency checks; they do not independently validate the audio or the
listeners' judgments.

A complete acoustic rerun requires separately obtained corpus audio and
annotations, public model assets, prepared packet-MVDR waveforms, original
batch maps, and matching execution receipts. The private S-FULLTAC frontend
weights and complete frontend reconstruction are not released. The original
decoder source retains its integrity checks and will require its bound local
inputs. The sanitized decoder protocol is a public description with a different
hash from the original execution seal. This repository alone therefore cannot
recreate every waveform or every historical ASR output.

Corpus media, verbatim corpus transcripts, individual reviewer worksheets,
reviewer identities, and model weights are excluded. Listening results are
aggregates. The probability-listening extension lacks a retained
playback-alias-to-audio checksum mapping; the separate residual census covers
43 actions in 41 windows. See the [third-party notices](THIRD_PARTY_NOTICES.md)
for official data/model sources and asset pins.

## Interpret the scores

cpCER is a fraction and can exceed 1; an empty hypothesis scores 1.0. The
study-specific Primary and Replication labels map to `Eval1`/`A5E1` and
`Eval2`/`A5E2` in the frozen files and should not be treated as reproductions of
official corpus benchmark tasks. The original four-channel guard comparison
is confirmatory; pooled, decoder-control, baseline, and MISP2022 analyses have
the descriptive/supportive roles recorded in their protocols. Reduced cpCER
and elimination of eligible exact runs do not by themselves establish
whole-transcript utility or semantic safety.

## License and attribution

Original software and authored documentation in this repository are released
under the [MIT License](LICENSE).
External datasets, models, and dependencies retain their own terms; the MIT
license grants no rights to those assets. Numerical research artifacts are
provided for verification with provenance retained. Please identify this
repository and the manuscript title when reusing the implementation or results.

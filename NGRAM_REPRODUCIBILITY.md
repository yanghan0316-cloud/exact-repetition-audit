# Reproduce the token 4-gram comparison

This addendum contains the post-primary n=4 experiment incorporated in the
`tmm_revision_v013` manuscript. It extends the original release at commit
`08258bb3a533cb39ee15c90b832fdfd0ce1b5b43` without changing its frozen results.
The evaluated subset is AISHELL-5 Primary/public: 18 complete recordings,
1,744 stream windows and 222 original batches.

## Results and interpretation

| Arm | Setting | Micro cpCER | Residual canonical events |
| --- | --- | --- | --- |
| A | 224-token cap and compression-triggered fallback | 1.073020 | 4 |
| B | A plus canonical character guard | 1.068336 | 0 |
| C | A plus token 4-gram blocking | 1.110009 | 0 |
| D | C plus canonical character guard | 1.110009 | 0 |

C minus A is +3.6989 percentage points, with a 95% recording-paired bootstrap
interval of [+0.5292, +7.5640] and recording wins/ties/losses of 5/0/13.
C minus B is +4.1673 points ([+1.0436, +7.9110], 5/0/13). D minus C is zero
([0, 0], 0/18/0); the recorded analysis finds no changed text or edits in D.
Relative to A, C has 711 fewer substitutions, 10 more deletions and 2,517 more
insertions, for 1,816 additional errors. All arms use 49,096 reference characters.

All 1,744 windows participate and no arm has an empty output. Each arm receives
one optimal speaker permutation per complete recording. The comparison uses
50,000 paired recording resamples with seed 20261002. Zero residual repetition
does not establish semantic preservation. No Replication, n=8 or additional
activity-source experiment was run. The previous 43-action listening census
concerns unblocked fallback outputs and supplies no C/D safety assessment.

## Verify without audio or model weights

Use Python 3.11 from the repository root:

```sh
python -B code/verify_manifest.py
python -B code/verify_ngram_manifest.py
python -m pip install -r requirements-ngram-numeric.txt
python -B code/verify_ngram_results.py
python -B -m unittest discover -s code/tests -p test_analyze_ngram_postprimary_v1.py -v
```

The result verifier recalculates all 72 recording-arm rows, arm totals,
micro/macro cpCER, four paired contrasts, bootstrap intervals, W/T/L and figure
CSV values using the original bootstrap implementation. It does not rescore
withheld transcripts or infer text identity from matching counts.

To run the seven tensor/decoder contracts and inspect the decoder interface:

```sh
python -m pip install -r requirements-ngram-decoder.txt
python -B code/tests/test_whisper_ngram_postprimary_v1.py
python -B code/scripts/decode_whisper_ngram_postprimary_v1.py --help
python -B code/scripts/analyze_ngram_postprimary_v1.py --help
```

Together with the eight analysis contracts, these are 15 additional synthetic
tests. They download no corpus or model. GitHub Actions runs the numeric checks,
analysis contracts and a separate CPU tensor-contract job.

## Source, configuration and figures

- [Frozen protocol](code/configs/whisper_ngram_blocking_postprimary_v1.json): n=4
  in the initial generation and every fallback attempt, original batch members,
  seed indices and guard settings. Only its machine-local source-plan path was
  replaced by a document title; this changes the public file's hash.
- [Decoder](code/scripts/decode_whisper_ngram_postprimary_v1.py),
  [constraint checks](code/src/realmeetsep/asr_whisper_ngram_postprimary_v1.py),
  [analysis](code/scripts/analyze_ngram_postprimary_v1.py) and
  [execution audit](code/scripts/audit_ngram_execution_v1.py).
- [Results](results/ngram_blocking/): aggregate analysis, four-arm summaries,
  comparisons and count-only meeting rows. Speaker-assignment IDs are omitted.
- [Figure data](figures/ngram_blocking/): the error decomposition and paired
  effects used in the manuscript figure.
- [Recorded environment](environment/recorded_environment.json),
  [source hashes and transformations](provenance/source_files.json), and
  [historical full execution audit](validation/recorded_full_execution_audit.json).

The historical `validation/package_validation.json` and
`provenance/base_repository.json` describe the local package before publication;
their `NOT_PUSHED` fields are historical records. Current integration is recorded
in [ngram integration provenance](provenance/ngram_integration.json), and the
repository's Actions results show the checks for each published commit.

Regenerate the figure locally with the optional plotting requirements:

```sh
python -m pip install -r requirements-ngram-figure.txt
python -B code/scripts/build_tmm_ngram_figure_v011.py
```

Outputs go to `local/ngram_figures/`, which is excluded from Git. The figure
requests Times New Roman; installed fonts can affect appearance. The plotted
numbers are independently checked against the released count data.

The [released manuscript figure](figures/ngram_blocking/ngram_control_tradeoff.svg)
is included as a vector image.

## Acoustic rerun prerequisites

The frozen scripts treat `code/` as their project root. A full decode requires
separately authorized local inputs and the same prepared acoustic frontend;
this release does not supply audio recovery or the complete frontend rebuild.
The following historical inputs are required, or must be newly generated and
documented as a separate execution:

1. `code/models/whisper-base/` from `openai/whisper-base` at revision
   `e37978b90ca9030d5170a5c07aadb050351a65bb`, including the download metadata
   expected by the frozen model verifier. Asset hashes are in the
   [third-party notices](THIRD_PARTY_NOTICES.md).
2. The original split manifest with `analysis_role=primary` and session IDs,
   public Sortformer posteriors, and matching packet-MVDR 16 kHz mono windows
   and boundaries. The decoder needs the full 1,744-row audio manifest.
3. The complete original V4 AISHELL-5 `decode_batch_map.jsonl`. Do not prefilter
   it: global batch indices determine the sampling seeds. Fallback calls retain
   every original member, while only active members adopt the generated output.
4. For four-arm scoring, local V4/V5 A/B hypotheses, diagnostics and audio
   manifest, full-meeting references, and original split membership. The
   analyzer exposes path overrides and checks A/B against the earlier V5 run.

After preparing those inputs, a full decoder invocation has this form:

```sh
python code/scripts/decode_whisper_ngram_postprimary_v1.py --audio-manifest code/experiments/whisper_ngram_blocking_postprimary_v1/audio_recovery/all_audio_manifest.jsonl --decode-batch-map code/experiments/whisper_control_comparison_postprimary_v4/aishell5/decode_batch_map.jsonl --model-dir code/models/whisper-base --protocol code/configs/whisper_ngram_blocking_postprimary_v1.json --output-dir code/experiments/whisper_ngram_blocking_postprimary_v1/decode --device cuda
python code/scripts/audit_ngram_execution_v1.py --run-dir code/experiments/whisper_ngram_blocking_postprimary_v1/decode --output code/experiments/whisper_ngram_blocking_postprimary_v1/validation/full_execution_audit.json --require-complete
python code/scripts/analyze_ngram_postprimary_v1.py --c-dir code/experiments/whisper_ngram_blocking_postprimary_v1/decode --output-dir code/experiments/whisper_ngram_blocking_postprimary_v1/analysis
```

For smoke testing, use a separate output directory with `--select-batches 7,411`;
that run is partial. `--resume` checks bound inputs and environment before
reusing completed batch checkpoints. The four Mandarin prompt tokens count in
n-gram history but not in the 224 generated-token budget. Fallback decisions
use each new output, and the average-logprob statistic reflects the processed,
n-gram-constrained distribution after undoing temperature scaling. Changed
hardware, library versions or batching can change sampled outputs.

The recorded C generation/validation time is 152.1 seconds; A's matching subset
time is unavailable, so no speedup is inferred. Original audio, hypotheses,
references, token histories, text-bearing traces, private manifests and model
weights are excluded. The recorded audit verifies the original execution;
independently rerunning that audit requires locally generated token/text logs.

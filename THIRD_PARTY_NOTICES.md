# Third-party sources and redistribution boundary

This release contains original research software and content-free numeric
artifacts. Third-party corpus media, annotations, verbatim speech, model
archives, and third-party source trees are not bundled. The repository's MIT
license applies to its original software and authored documentation and does
not replace external terms. Released figures are study-authored diagrams and
numeric plots; the software license does not relicense the underlying datasets
or any external material. Numeric tables retain their source attribution and
are supplied for research verification.

## Corpora

| Resource | Official source and terms | Role in this study |
| --- | --- | --- |
| AISHELL-5 | [OpenSLR 159](https://www.openslr.org/159/), listed as CC BY-SA 4.0 | 36-recording Primary / Replication audit |
| AliMeeting | [OpenSLR 119](https://www.openslr.org/119/), listed as CC BY-SA 4.0 | Training-overlapped auxiliary diagnostics |
| MISP2022 | [Official MISP Challenge 2022](https://mispchallenge.github.io/mispchallenge2022/); acquire data through the organizers and follow the applicable provider terms | Eight-meeting supportive audio-only subset |

The official MISP2022 site documents registration-based data distribution. No
general redistribution license for its corpus is asserted here. MISP reference
construction follows Track-2 content-tier filtering; the reported internal
character scores are not challenge-leaderboard equivalents. Obtain the corpus
and its annotations separately and preserve the provider's attribution and
access terms.

## Frozen public models

- NVIDIA [Streaming Sortformer 4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/tree/59620264e008f2b06a0e969688e0af3e8705478b).
  The research artifact records CC BY 4.0, revision
  `59620264e008f2b06a0e969688e0af3e8705478b`, and local model-archive SHA-256
  `b371afce2c4958186469df33d939936b9746c89f38b10a69cfd2c61254e83329`.
- [openai/whisper-base](https://huggingface.co/openai/whisper-base/blob/e37978b90ca9030d5170a5c07aadb050351a65bb/README.md),
  revision `e37978b90ca9030d5170a5c07aadb050351a65bb`. The pinned model-card
  metadata reports Apache-2.0. The expected `model.safetensors` SHA-256 is
  `07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b`;
  `config.json` SHA-256 is
  `a153c53883a6799b6f056b4a8d1a515c9926d03994682ba88a7616618d7da0c1`.

The original [OpenAI Whisper repository license](https://github.com/openai/whisper/blob/main/LICENSE)
is MIT. Its upstream source/weights and the Hugging Face model-card metadata
are distinct provenance records; this repository does not redistribute either
asset. The NVIDIA NeMo Speech source dependency is recorded as Apache-2.0 in
the preceding artifact; consult its [upstream license](https://github.com/NVIDIA-NeMo/Speech/blob/main/LICENSE).

`python code/verify_external_assets.py` prints the frozen pins and can check
authorized local Sortformer and Whisper files. It does not fetch models.
The private S-FULLTAC frontend weights are not included.

## Software dependencies and generated local outputs

NumPy, RapidFuzz, PyTorch, Transformers, and SoundFile are installed as external
dependencies using the requirements files. Their packages retain their own
licenses and notices; no dependency source is vendored. Decoder helper modules
in this repository are the study's original adapters, with their source hashes
recorded in [source provenance](environment/SOURCE_PROVENANCE.json).

Locally generated transcripts, waveform exports, and review materials may
contain corpus content and are excluded from Git by the supplied ignore rules.
Public result tables contain counts and hashes, not a grant to redistribute
the underlying recordings or individual annotations.

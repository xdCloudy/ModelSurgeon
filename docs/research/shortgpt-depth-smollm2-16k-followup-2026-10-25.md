# ShortGPT layer ranking: 16k-token follow-up

This follow-up extends the physical SmolLM2-135M depth-pruning comparison from
2,032 to 16,256 held-out WikiText-2 tokens and expands the predeclared random
controls from three to five. It uses the same 2,048 WikiText-2 train tokens for
layer ranking. The full result, hashes, and command are in the
[JSON evidence record](shortgpt-depth-smollm2-16k-followup-2026-10-25.json).

| Selector | Removed layer | Candidate PPL | Delta vs source |
| --- | ---: | ---: | ---: |
| Source | — | 33.5943 | — |
| ShortGPT block influence | 14 | 36.9562 | +3.3619 |
| Random, seed 11 | 14 | 36.9562 | +3.3619 |
| Random, seed 23 | 29 | 225.9937 | +192.3994 |
| Random, seed 47 | 11 | 805.6696 | +772.0753 |
| Random, seed 101 | 18 | 40.8468 | +7.2525 |
| Random, seed 202 | 24 | 65.7356 | +32.1413 |
| Weight magnitude | 0 | 61,886.2490 | +61,852.6547 |

Every candidate regressed on the measured held-out slice. ShortGPT again chose
layer 14, which exactly matches random seed 11, so this run does not show that
the block-influence ranking beats random selection. The random results also
show that a single random draw can vary sharply across removed layers. The
ShortGPT artifact reloaded with finite logits and reproduced the same PPL.

The candidate removed 3,540,096 parameters (2.63%) and reduced the saved model
directory from 272,437,465 to 265,504,056 bytes (2.55%). No inference-speed
benefit was measured. This is one model, one layer removal per candidate, one
calibration prefix, and the first 16,256 held-out tokens; it does not establish
quality retention, cross-model generalization, or a stable method ranking.

The source benchmark had a hard-coded limitation string saying “Three
predeclared random seeds” even when the run used five. The checked-in report
corrects that descriptive text without changing measured fields; the untouched
raw report digest and exact benchmark source revision are retained in the JSON
provenance. The benchmark now derives this limitation from the actual seed
count, with a regression test.

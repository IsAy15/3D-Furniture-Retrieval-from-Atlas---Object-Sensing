# Parameters and defaults

[Documentation index](README.md) | [Pipeline](PIPELINE.md)

## Configuration precedence

Scientific defaults live in [config.py](../config.py). Run Console builds its own
explicit configuration in `run_console.default_config()` and then applies saved
preferences and user edits. Debug uses per-stage defaults and its own saved state.
The CLI starts from its parser defaults and the scientific configuration; it does
not automatically read Run Console preferences. Consequently identical-looking
launches from different entry points need not use identical numeric settings.

This repository contains **no** `execution_parameters.json` or Debug `settings.json`.
It starts clean. Since 17 September 2026, Run Console defaults include the saved
experiment settings: neighborhood radius 0.06 m, curvature threshold 0.105,
ground exclusion height 0 m and a 350-point scan keypoint cap. Their superiority
over the previous defaults has not been established. Debug and direct CLI defaults
remain independent. Effective run configurations
are the authoritative record for reproducing an experiment.

## Main controls

| Control | Code/console reference | Effect |
| --- | --- | --- |
| Voxel size | 0.015 m | Smaller voxels preserve fine geometry and increase memory/work. |
| TSDF truncation | 0.060 m | Larger bands tolerate noise but mix nearby surfaces. |
| Frame stride / minimum | 6 / 50 | Stride is reduced when possible to preserve short-sequence viewpoints. |
| Maximum frames | 0 | No cap; positive values restrict observations. |
| Surface sampling / cap | 0.01 m / 175,000 | Controls extracted-cloud density and downstream cost. |
| Surface extraction | hybrid, raw field | Adds supported thin details to strict zero crossings. |
| Neighborhood / curvature | Console 0.06 m / 0.105; scientific defaults 0.05 m / 0.05 | Controls detector scale and planar rejection. The radius also affects geometric coverage tolerance. |
| Harris k / threshold | 0.04 / 0.008 | Balances planarity penalty and direct corner acceptance. |
| Reference neighbors | 6 | Normalizes Harris score amplitude across sampling densities. |
| Scan keypoint cap | Console 350; scientific default 400 | Upper budget after quality filtering; more points cost more. |
| Ground exclusion height | Console 0 m; scientific default 0.08 m | Removes keypoints below this height above the estimated ground. |
| Model keypoint cap | scientific default 120 | Bounds detailed correspondence work. |
| UDF / occupancy grid | 16³; half-extent 0.12 m | 4,096 scalar/categorical cells, separate from primitives. |
| Distance exponent / threshold | 4 / 128 | Fourth-power normalized UDF error; higher thresholds are more permissive. |
| Broad pool / nominal Top-k | Console 500 / 100 | Candidate recall versus reranking and matching cost. |
| Per-group final budget | 50 | Union of group shortlists can exceed nominal global Top-k. |
| Local candidate supplement | 64 | Adds proposals when the compact local index exists. |
| Correspondence cap | 8 per scan keypoint | Preserves local alternatives before geometric consensus. |
| RANSAC minimum support | 3 distinct, distributed pairs | Rejects weak or repeated-point constellations. |
| Constellation support required | true | Unconfirmed shape-only fallback seeds cannot become final objects. |
| Coverage threshold | Console 0.45; CLI parser 0.40 | Minimum forward coverage before acceptance. |
| Query mode | single | One query on final reconstruction; progressive is explicit. |

The table highlights entry-point differences instead of claiming universal parity.
Run Console additionally sets spatial balancing, verification and final-overlap
options. In particular, its `paper_verification` toggle is enabled in code defaults;
the direct CLI enables that option only when `--paper-verification` is supplied.
That switch disables selected added safeguards; it is not proof of exact fidelity
to the original paper. Inspect the recorded command to compare runs.

## Tuning without losing interpretability

Inspect surface quality before increasing retrieval budgets. Thin legs missing
from the input cannot be recovered by merely increasing Top-k. Check keypoint
coverage and object grouping next, then whether plausible models survive the
broad pool, and finally correspondence/verification rejection reasons.

Change one stage at a time and keep the same database, scenes and upstream
artifacts. Increasing a threshold can improve recall while admitting more false
positives. Larger pools do not correct an incorrect score or missing categories.
Debug help explains the higher/lower direction for individual controls. Display
point counts and browser styling affect only previews, not descriptor computation.

For the complete current command surface, use `python cli.py run-progressive -h`.
Programmatic settings are represented by the dataclasses in `config.py`; the
English per-control explanations are in [ui_help_en.py](../ui_help_en.py).

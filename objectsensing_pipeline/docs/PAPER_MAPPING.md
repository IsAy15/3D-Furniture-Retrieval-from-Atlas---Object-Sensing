# Original paper and this implementation

[Documentation index](README.md) | [Pipeline](PIPELINE.md)

Reference: Li, Dai, Guibas and Nießner, *Database-Assisted Object Retrieval for
Real-Time 3D Reconstruction*, Eurographics 2015.
[Publication and dataset](https://graphics.stanford.edu/projects/objectsensing/).

The internship project is an attempted reproduction and improvement of this
method. It does not demonstrate a completed novel retrieval method, exact
numerical reproduction or the original system performance.

| Component | Implemented approach and difference |
| --- | --- |
| Scanning | Offline RGB-D sequences with known camera poses, rather than an integrated live tracking application. |
| Fusion | Python dense NumPy/CuPy and sparse Open3D paths; visibility-aware surface recovery adds implementation-specific behavior. |
| CAD preprocessing | Triangle sampling, category height normalization and virtual ground; approximate preprocessing rather than identical original virtual scans. |
| Keypoints | Paper-inspired Harris/2D corners; added density normalization, wall/quality filters and adaptive budgets. |
| Descriptors | Local UDF versus partial occupancy and support primitives; explicit units and visibility handling documented in code. |
| Search | Added compact height/primitive index, grouped candidate pools, compact local UDF proposals and reranking before detailed matching. |
| Pose proposals | Ground-constrained correspondence transforms and 1-Point RANSAC, with bounded budgets and additional support gates. |
| Verification | Constrained ICP, reciprocal coverage/harmonic ranking and configurable visibility, structure and overlap checks. |
| Temporal behavior | Optional repeated offline prefix reconstruction and caches; default console query is on the final reconstruction. |
| Tools | Run Console, instrumented Debug stages and a standalone Final Viewer for inspection and reproducibility. |

The `paper_verification` option disables selected added safeguards. Other
implementation differences remain, so this option is not an exact-paper mode.

Known limitations include incomplete object groups, missing categories, ambiguity
between generic furniture parts, imperfect orientation/scale, sensitivity to thin
or poorly observed surfaces, and lack of a broad annotated retrieval benchmark.
Single Chair is a useful positive example, while complex scenes remain unreliable.
Atlas evaluation, more robust retrieval representations, systematic ablations and
broader quantitative assessment are future work. General pose refinement and
reconstruction are extensions rather than the primary completed objective.

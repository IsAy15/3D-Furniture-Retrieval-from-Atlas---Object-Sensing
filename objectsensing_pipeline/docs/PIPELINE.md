# Pipeline and evaluation metrics

[Documentation index](README.md) | [Paper comparison](PAPER_MAPPING.md)

```text
CAD meshes -> normalization -> surface sampling + virtual ground
           -> keypoints -> model UDFs + primitives -> database + indices
                                                        |
RGB-D + camera poses -> TSDF/visibility -> surface -> scan keypoints
    -> occupancy descriptors + groups -> broad/local candidate pool
    -> local reranking -> correspondences -> 1-Point RANSAC constellations
    -> constrained ICP + verification -> global selection -> ALN/JSON/HTML
```

## Database preparation

Normalize CAD geometry to Y-up, ground contact and a category-dependent physical
height. Sample triangles by area and add a virtual ground patch for descriptor
construction. This is a practical geometry-based approximation of the paper
preprocessing, not a complete rendered virtual-scanning reproduction. Store one
model per pickle to avoid loading the entire atlas in every worker.

## Fusion and keypoints

Integrate RGB-D observations into a TSDF with weights and visibility. Unknown
space must remain distinct from observed free space. Dense and sparse layouts
are available. Extract zero crossings with optional supported near-zero recovery,
estimate SDF-gradient normals and PCA surface variation, and align the scene to
gravity and ground. Accurate camera-to-world poses are assumed as input.

Keypoints combine curvature, density-normalized 3D Harris responses and 2D corner
recovery on planar boundaries. NMS, geometric adjustment and deduplication reduce
redundancy. Scan-specific wall, planar-interior, hole-boundary and repeatability
filters reject unreliable measurements. Adaptive minima and spatial/component
budgets protect recall without restoring hard visibility failures.

## Descriptor dimensions and comparison

Each model keypoint has a **16 × 16 × 16 unsigned distance field: 4,096 scalar
values**. Each scan keypoint has a corresponding **4,096-cell categorical
occupancy grid** distinguishing free, occupied and unknown space; seed-fill keeps
the occupied component supporting the keypoint. The half-extent is 0.12 m, so the
sampled cube spans 0.24 m. Endpoint grid spacing is 0.24 / 15 = 0.016 m; automatic
distance normalization uses 0.24 / 16 = 0.015 m. These are different conventions.

Position, height, response and primitive support are additional feature metadata.
The primitive size vector has **six components**: horizontal/vertical plane areas
and line lengths `(Ah, Av1, Av2, Lh, Lv1, Lv2)`. It is not a 4,096+6 learned embedding.

For a candidate yaw, sample the model UDF at rotated occupied scan cells, divide
distances by the normalization unit, raise to the fourth power and average.
Unknown cells inform confidence rather than being treated as free-space errors.
Primitive orientation and size filters constrain comparisons. When no primitive
direction is available, matching can test 36 horizontal rotations.

## Preselection and correspondence

The coarse candidate signature combines a **32-bin keypoint-height histogram**,
six primitive statistics and two scalar summaries: **40 values per model**,
stored as separate arrays rather than one learned embedding. The current index
can also store a **63-value shape signature**: 8 height bins, 8 radial bins,
8 pairwise-distance bins, a 6 × 6 top-view occupancy map and 3 normalized extents.
This optional signature adds coarse geometric evidence when present.
Grouped queries reserve coverage for several objects and combine their rankings.
A separate compact local proposal index downsamples UDFs to **6³ = 216 values**,
with up to 32 indexed features per model. This index supplements the broad pool;
it does not replace full descriptor verification.

Rerank candidate models using local descriptor support, coverage/spread and
optional extent compatibility. Full matching retains several alternatives per
scan keypoint. A local match gives an initial ground-constrained transform;
1-Point RANSAC collects geometrically consistent, spatially distributed pairs.
Shape-based group seeds are diagnostic fallbacks and require confirmed
constellation support for final acceptance in the current code.

## Verification, selection and metrics

Refine candidate placements with ICP constrained to yaw, translation and uniform
scale. Pose refinement supports retrieval verification here; a general pose
estimation or reconstruction system remains outside the completed contribution.
Reject inadmissible scale updates and test surface support, visibility and enabled
structural safeguards before global object selection.

Define metrics before interpreting a score:

- **Forward coverage C_f:** fraction of transformed model samples close to the scan.
- **Reverse coverage C_r:** fraction of non-ground scan points in the model footprint
  close to the transformed model. Some later checks restrict evidence to an object group.
- **Harmonic score H:** `2 C_f C_r / (C_f + C_r)`, with zero when the denominator is zero.
  Both coverages lie between zero and one. A high H indicates reciprocal geometric
  support, not necessarily the best CAD identity or visually correct furniture details.
- **Mean surface distance:** distance between supported surfaces, in meters; lower is
  better. Selection can penalize this separately from H.
- **Visibility support:** evidence from known occupied/free volume; unknown model
  regions are ignored. It is not interchangeable with geometric reverse coverage.
- **Recall@K:** fraction of annotated target objects with at least one acceptable
  CAD model in the first K candidates. Requires explicit acceptable-model annotations;
  the code does not invent ground truth from its own scores.

Global selection ranks admissible hypotheses against a no-object alternative and
suppresses redundant explained regions, same-category footprint overlap and
configured cross-category conflicts. An empty selection can be the correct outcome
when evidence is insufficient. Measure grouping quality, candidate recall, final
visual correctness and runtime separately to locate failures.

# Troubleshooting

[Documentation index](README.md)

| Symptom                                     | Check or action                                                                                                                            |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Missing Python module                       | Use the same virtual-environment interpreter for installation, server and CLI. Install `requirements-dev.txt`.                             |
| Open3D wheel unavailable                    | Check 64-bit Python 3.11 and the platform supported by the pinned wheel. Do not silently replace core versions when comparing experiments. |
| Open3D shared-library import error on Linux | Install the missing native runtime library named by the import error.                                                                      |
| Page loads but no scenes appear             | Data is excluded. Supply a compatible ZIP and select/register its server-visible path.                                                     |
| Debug has disabled scenes                   | Supply ZIPs in the configured scene directory, restart/reload, and start at Fusion.                                                        |
| Missing candidate index                     | Build the prepared database, then `build-candidate-index`. The local index has its own builder.                                            |
| Local proposals unavailable                 | Build `tools/build_local_candidate_index.py --db ...`; check descriptor compatibility and manifest fingerprint.                            |
| Model renders as a cloud or fails to load   | Source OBJ/ZIP paths stored during DB preparation are missing. Keep source files accessible or rebuild on the receiving machine.           |
| Port 8770 is busy                           | Stop the previous server or use `--port 8771`, then open that port.                                                                        |
| CUDA unavailable                            | Start with `auto` or `cpu`; verify the driver and optional GPU dependencies separately.                                                    |
| Excess memory use                           | Reduce concurrent scenes/workers, prefer sparse layout for large rooms, or increase voxel size with awareness of detail loss.              |
| Empty retrieval                             | Check surface, keypoints, object groups and broad-pool candidates before relaxing final verification. Inspect rejection reasons.           |
| Good score but wrong furniture              | Scores measure observed geometric support, not CAD identity. Compare shape zones and annotated acceptable candidates.                      |
| Resume signature mismatch                   | Use a new resume directory when data, parameters or scientific code changes.                                                               |
| Debug differs from Run Console              | Compare effective settings and upstream artifacts; parameters are synchronized only by explicit action.                                    |
| Standalone viewer has no detailed timeline  | The summary lacks a detailed progress manifest; generate from a current complete run.                                                      |

The repository contains synthetic tests but no scene benchmark. A passing test suite
does not establish accuracy on Office, Chairs or IKEA Table. Preserve run
configurations, logs and manifests when reporting a scientific failure.

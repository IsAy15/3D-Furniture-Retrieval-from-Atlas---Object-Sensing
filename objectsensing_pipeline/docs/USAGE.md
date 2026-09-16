# Run, debug and inspect

[Documentation index](README.md) | [Parameters](PARAMETERS.md)

## Run Console

Start `python run_console.py --host 127.0.0.1 --port 8770` from the program directory.
On Windows, the optional `run_console_terminal.ps1` launcher automatically uses
the `.venv` at the repository root, or Python on PATH. Its `-PythonPath`
argument selects a different environment explicitly.
Open `/`. Select the RGB-D scene ZIP, prepared database, candidate index and output
directory. The plus button registers an external scene path without copying it.
Build the local proposal index separately as described in [Setup](SETUP.md).

The playlist contains input validation, optional focused tests, coarse-index
preparation and the scientific stages. Running a scientific target inserts its
prerequisites automatically. **Selection** generates the final result and viewer.
The multi-scene option creates separate runs and processes checked scenes sequentially.

The center shows live geometry and the event timeline. Layers can show RGB-D
measurements, the fused surface, detected/retained keypoints and candidates.
The inspector provides parameters, candidate diagnostics and saved-run history.
Use a completed run to replay its trace, resume compatible work or open Final Viewer.
Rename and favorite operations alter display metadata, not technical run identifiers.

Terminal shortcuts are `R` restart HTTP service, `D` detailed logs, `S` status,
`C` clear display and `Q` quit. `--no-interactive` disables shortcuts.
`--debug` controls HTTP logging; it is not a retrieval algorithm preset.

## Debug Visualizer

Open `/debug`. Stages are Fusion → Keypoints → Descriptors → Top-k → Matching →
Verification → Selection. For a fresh source-only installation, start at **Fusion**
with an existing ZIP. Disabled example entries are placeholders, not bundled data.
Debug discovers ZIPs from the configured scene directory on initialization.

Each stage saves a complete artifact for the next stage. Select an upstream run
explicitly, or use **Run through** to chain stages while keeping the page open.
Existing matching/verification/selection artifacts can be inspected without
rebuilding fusion. A bare PLY can support some geometry diagnostics but lacks
the volumetric visibility needed for the full descriptor/retrieval chain.

Pin a baseline to compare changes. Inspect retained and rejected locations, not
only counts. Select a keypoint for rejection reasons or a Top-k candidate for
correspondences and a provisional placement. Preview placements are not accepted
retrieval results. The Selection view distinguishes the scan, retained models and
their overlay; model keypoints are optional.

Debug parameters are saved separately. **Apply to Run Console** explicitly copies
compatible settings into defaults for future console runs. Resetting a stage
returns to code defaults. Display density does not change scientific geometry.

For a headless stage chain, after data preparation:

```powershell
python dev/run_debug_chain.py --scene single-chair --from-stage fusion --to-stage selection
```

The scene name is the ZIP stem. Use `--upstream-run RUN_ID` when starting after
Fusion from an existing artifact. To compare with Run Console, use the same
settings explicitly; the two frontends do not share every default.

## CLI and output files

```powershell
python cli.py --help
python cli.py run-progressive --help
python cli.py run-progressive --zip scenes/single-chair.zip --db db_dir --out work/runs/single-chair.aln --query-mode single --candidate-index db_dir/candidate_index.npz --candidate-pool-k 500 --candidate-top-k 100 --candidate-top-per-group 50 --candidate-local-extra-k 64 --coverage 0.45 --jobs 6 --stage-cache-dir work/runs/single-chair-cache --resume-dir work/runs/single-chair-resume
python cli.py visualize --json work/runs/single-chair.json --scan work/runs/single-chair_scan.ply --out work/runs/single-chair_viz.ply --html work/runs/single-chair.html
```

Despite the command name, `run-progressive --query-mode single` queries the final
reconstruction once. `--query-mode progressive` uses temporal prefixes;
`--query-start-frames` and `--query-interval-frames` configure checkpoints.
`--stop-after fusion|keypoints|query|matching|verification|selection` stops at a
scientific boundary. Direct `run` is also supported, but differs in orchestration.

| Artifact | Purpose |
| --- | --- |
| `<name>.aln` | MeshLab model-to-scene transformations |
| `<name>.json` | Selected models, transforms and diagnostics |
| `<name>_scan.ply` | Observed scene surface for visualization |
| `<name>_progress.json` | Timeline and detailed retrieval decisions |
| `<name>_viz.ply` | Colored scan/model overlay |
| Generated `.html` | Standalone Final Viewer |
| Stage/resume directories | Reusable scientific and matching artifacts |

The exact console paths and command are recorded per run. Do not copy stale
caches into a new project blindly: signatures include data, parameters and
scientific-code fingerprints. An incompatible signature requires a fresh
resume directory. Stopping a run preserves completed atomic batches.

## Final Viewer

Open a generated HTML file directly or from Run Console history. Geometry, shared
UI code and Three.js/Lucide are embedded. The timeline exposes queries,
correspondences, constellations, verified poses and final decisions when recorded.
Older or minimal summaries provide fewer diagnostics. Use filtering and multiple
selection to compare hypotheses; **Show retained models** returns to the final set.
The back button targets the local console and is useful only while it is running.

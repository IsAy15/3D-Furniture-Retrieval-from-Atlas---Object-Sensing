# 3D Furniture Retrieval from Atlas

## Internship context

This repository contains the work I carried out during my internship at
**USTH (University of Science and Technology of Hanoi)**, under the supervision of
**Nguyen Hoang Ha**, as part of the **FISA Informatique programme at CESI Toulouse**.
The internship topic was **3D Furniture Retrieval from Atlas**.

The project's long-term objective is to design a novel retrieval method that,
given a partial observation of a furniture object (a partial point cloud or an
RGB-D scan), finds the best matching CAD model in a large atlas. **Atlas is the
target dataset**. Retrieval is the primary research objective; pose refinement
and reconstruction are potential future extensions.

Although a novel retrieval method was not fully developed and validated during
the internship, the work focused on **attempting to reproduce and exploring
improvements to** the method presented
by Li et al. in _Database-Assisted Object Retrieval for Real-Time 3D
Reconstruction_ (2015), together with tools for running experiments, inspecting
intermediate stages and comparing retrieved models. Geometric alignment in the
prototype supports candidate verification and visual assessment.

## Project scope

The prototype retrieves furniture CAD models from partial RGB-D observations using local
geometry, a candidate shortlist, constellation matching and geometric verification.
Current experiments use ObjectSensing RGB-D sequences and a ShapeNet CAD database;
an Atlas-specific adapter has not been completed.

**This is a source-only repository.** Supply the CAD models and scene ZIPs separately.
No database, scene, generated result, local preference or annotation is bundled.
The web application includes exactly three workflows: **Run Console**, **Debug
Visualizer**, and the generated **Final Viewer**. Data and generated state remain local and are excluded from version control.

## Start here

Use **64-bit Python 3.11** as the installation target.
The CPU installation is sufficient; CUDA is optional. Use a recent desktop browser
with WebGL support. Node.js is only needed for JavaScript checks, not to run the UI.

From the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r objectsensing_pipeline/requirements-dev.txt
cd objectsensing_pipeline
..\.venv\Scripts\python.exe run_console.py --host 127.0.0.1 --port 8770
```

Open [Run Console](http://127.0.0.1:8770/) or
[Debug Visualizer](http://127.0.0.1:8770/debug).
An empty source-only installation can open these pages, but cannot run retrieval
until a scene and prepared database are supplied. Stop the server with `Ctrl+C`.

On Linux, replace environment creation with `python3.11 -m venv .venv` and use
`.venv/bin/python` from the root or `../.venv/bin/python` from the program directory.
See [complete setup](objectsensing_pipeline/docs/SETUP.md) for dependencies,
external paths, optional GPU support and data preparation.

## Prepare external data

1. Obtain **ShapeNetCore v2 OBJ geometry** from the
   [official ShapeNet distribution](https://huggingface.co/ShapeNet).
   Follow the dataset access terms. This loader accepts extracted OBJ trees or
   one ZIP per synset; GLB/glTF distributions require conversion first.
2. Obtain RGB-D ZIP sequences from the
   [ObjectSensing project and dataset page](https://graphics.stanford.edu/projects/objectsensing/).
   Copy chosen archives to `objectsensing_pipeline/scenes/`, or configure an
   external scene directory. Use your own compatible RGB-D sequence if appropriate.
3. Build the database and both candidate indices. From `objectsensing_pipeline/`,
   with your virtual environment active:

```powershell
python cli.py build-db-dir --shapenet "C:\datasets\ShapeNetCore.v2" --out db_dir --n-points 3000 --jobs 6
python cli.py build-candidate-index --db db_dir
python tools/build_local_candidate_index.py --db db_dir
```

The default categories are chair, table and couch. The first command can take a
long time on a full dataset and resumes an existing compatible build by default.
For a quick installation check, start with `--max-per-synset 5` in a **separate**
database directory; such a small subset cannot reproduce full-database retrieval.
Keep the source meshes accessible: final model rendering reads their saved paths.

## Run and inspect

In Run Console, choose a scene, database directory and candidate index. The
**Validate inputs** step checks paths; **Selection** runs its prerequisites and
generates final artifacts. Open **Final Viewer** from the completed run history.
The HTML embeds its geometry and browser libraries and can be shared independently.

For a direct CLI example with explicit retrieval budgets:

```powershell
python cli.py run-progressive --zip scenes/single-chair.zip --db db_dir --out work/runs/single-chair.aln --query-mode single --candidate-index db_dir/candidate_index.npz --candidate-pool-k 500 --candidate-top-k 100 --candidate-top-per-group 50 --candidate-local-extra-k 64 --coverage 0.45 --jobs 6
python cli.py visualize --json work/runs/single-chair.json --scan work/runs/single-chair_scan.ply --out work/runs/single-chair_viz.ply --html work/runs/single-chair.html
```

This CLI example uses the scientific code defaults plus the stated options; it
does **not** silently import saved Run Console preferences. To reproduce a console
run exactly, reuse the command/configuration recorded with that run.

## Documentation

| Guide                                                             | What it explains                                            |
| ----------------------------------------------------------------- | ----------------------------------------------------------- |
| [Documentation index](objectsensing_pipeline/docs/README.md)      | Recommended reading order                                   |
| [Setup](objectsensing_pipeline/docs/SETUP.md)                     | Python, data layouts, DB/index construction, external paths |
| [Usage](objectsensing_pipeline/docs/USAGE.md)                     | Three interfaces, CLI, output files and resumption          |
| [Parameters](objectsensing_pipeline/docs/PARAMETERS.md)           | Defaults, overrides, units and effects                      |
| [Pipeline](objectsensing_pipeline/docs/PIPELINE.md)               | Descriptor dimensions, retrieval stages and metrics         |
| [Architecture](objectsensing_pipeline/docs/ARCHITECTURE.md)       | Module responsibilities and data flow                       |
| [Paper comparison](objectsensing_pipeline/docs/PAPER_MAPPING.md)  | Reproduced ideas, changes and unimplemented goals           |
| [Troubleshooting](objectsensing_pipeline/docs/TROUBLESHOOTING.md) | Installation, paths, caches, memory and poor retrieval      |
| [Repository contents](#repository-contents)                                   | Included files, exclusions and export-specific changes      |
| [Recorded-validation history](#recorded-validation-history)                                       | Checks performed and remaining portability limits           |

## Validation commands

From `objectsensing_pipeline/` in the installed environment:

```powershell
python -m pytest -m smoke -q
python -m pytest -q
python tools/check_docs.py
node --test tests/debug_chain.test.cjs tests/debug_selection_view.test.cjs
```

Synthetic tests do not establish retrieval accuracy on real scenes. Single Chair
has been the most convincing visual example; Office, Chairs and IKEA Table still
show substantial candidate, grouping and alignment errors. The implementation is
an offline research prototype, with no demonstrated real-time performance or
guarantee that defaults are optimal for every scene.

## Repository contents

This repository starts from the English source handoff prepared on 16 September
2026 and includes subsequent local edits. It uses native Python installation;
Docker configuration has been removed.

### Included

- Scientific Python pipeline, CLI and reusable database/index builders.
- Run Console, Debug Visualizer and generated Final Viewer, with English UI text.
- Shared browser code and locally vendored Three.js/Lucide with their licenses.
- Installation, usage, architecture, parameter and troubleshooting documentation.
- Synthetic/unit tests and native runtime configuration.

### Excluded

- CAD databases, ShapeNet geometry, candidate indices and RGB-D scene archives.
- Generated point clouds, meshes, run outputs, caches, logs and execution history.
- Local parameter preferences, annotations, credentials, virtual environments and prior Git history.
- Annotator interface, candidate comparison galleries, experiment notebooks,
  paper PDFs, report/poster files and obsolete dataset-dependent experiments.

### Source adaptation

In this repository, UI/help text and documentation
are in English, the annotator page and annotation write endpoint are disabled,
database paths default to `objectsensing_pipeline/db_dir`, and Debug discovers
ZIP files in the configured scene directory. Without data it displays disabled
example entries. The Final Viewer embeds its browser modules for offline use.
The terminal launcher resolves the project virtual environment instead of a personal
Python path. Missing viewer database statistics are shown as unrecorded instead of
using fixed example counts. The scientific algorithm is preserved. Run Console
now defaults to the saved experiment settings: neighborhood radius 0.06 m,
curvature threshold 0.105, ground exclusion height 0 m and 350 scan keypoints.
These settings are available without a local preference file; their superiority
over the previous defaults has not been established. Debug and direct CLI defaults
remain independent; see the [parameter reference](objectsensing_pipeline/docs/PARAMETERS.md).

Focused tests for the excluded annotator UI are omitted. Annotation storage
utilities remain available to offline evaluation code, with no saved labels.
## Version control

The repository tracks program source, documentation, tests and vendored browser
libraries. `.gitignore` excludes environments, caches, datasets, generated outputs,
local preferences and obsolete export metadata. Keep large CAD and RGB-D inputs
outside version control. The previous export manifest is not a record of the
current repository; Git commits are the source history.

After changing code or documentation, review and record your changes:

```bash
git status
git diff
git add <files>
git commit -m "Describe the change"
```

This is a local repository. No hosting provider or remote is configured by setup.

## Recorded-validation history

The following results describe the **16 September 2026 source handoff**, before
subsequent local edits. They are retained for traceability and are not a claim
that the complete suite has been rerun against every later commit. Use the
[validation commands](#validation-commands) above to validate new changes.

### Previous checks

- Full included Python suite: **340 passed**. After the final viewer and text edits,
  its focused viewer, Run Console and project-structure suite was rerun: **41 passed**.
- Browser behavior tests: **8 passed** (debug stage chaining and final-selection display).
- Run Console and Debug Visualizer opened in headless Microsoft Edge on a separate
  local test server, with working WebGL canvases and no JavaScript page errors.
- Generated Final Viewer opened from a local file with the browser network disabled.
  Its embedded synthetic chair, retained-model selection and camera framing worked;
  no external HTTP request or JavaScript page error was recorded.
- Removed annotator page routes returned HTTP 404.
- The original packaging checks compiled Python source, resolved local Markdown links, compared
  scientific dataclass defaults with the working source, rejected personal authoring
  paths, and excluded all runtime/data directories and binary scene/database formats.
- A separate extraction into a path containing spaces passed CLI, progressive CLI,
  server and local-index-builder help commands, documentation checks and the eight
  JavaScript behavior tests. No original database or scene path was required.

### Scope and limitations

Tests use synthetic fixtures and do not establish real-scene retrieval accuracy.
The Python test environment is the author's existing Windows Python 3.9 installation;
the setup guide targets Python 3.11. A fresh dependency installation, CUDA execution
and a new full-database retrieval run have not
been performed as part of the original source handoff. External CAD models, RGB-D sequences,
access permissions and database construction remain the recipient's responsibility.


## Attribution and redistribution

Scientific reference: Yangyan Li, Angela Dai, Leonidas Guibas and Matthias Nießner,
_Database-Assisted Object Retrieval for Real-Time 3D Reconstruction_, 2015.
[Paper and project](https://graphics.stanford.edu/projects/objectsensing/).

The repository preserves third-party browser licenses in
[vendor/VERSIONS.md](objectsensing_pipeline/vendor/VERSIONS.md). No new license is assigned
to the original research code by this packaging operation. Dataset and source
redistribution permissions remain separate.

Three.js 0.160.0 and its included OrbitControls/PLYLoader modules retain the Three.js license. Lucide 0.468.0 retains its license. Versions, upstream locations and recorded hashes are in [vendor/VERSIONS.md](objectsensing_pipeline/vendor/VERSIONS.md).

Python dependencies are installed separately using pinned requirement files and retain their own licenses. No third-party Python environment is bundled. ShapeNet, RGB-D datasets and publication PDFs are not redistributed in this repository. Consult each dataset provider for access and permitted use.

Packaging does not assign a new license to the original project source.

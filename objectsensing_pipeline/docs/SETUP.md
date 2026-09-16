# Setup and data preparation

[Documentation index](README.md) | [Usage](USAGE.md)

## Python and dependencies

Use 64-bit Python 3.11 for a new environment. CPU operation needs NumPy 2.0.2,
SciPy 1.13.1, scikit-learn 1.6.1, trimesh 4.12.2, Pillow 11.3.0 and Open3D 0.19.0.
Versions are pinned in [requirements.txt](../requirements.txt). The development
requirements add pytest 8.4.2 and are recommended because Run Console offers a
focused test step. Browser libraries are vendored; no npm build is required.

Windows, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r objectsensing_pipeline/requirements-dev.txt
```

You can use the explicit interpreter path instead of activating the environment.
For shorter commands in PowerShell, run `.\.venv\Scripts\Activate.ps1` from the
repository root, then `cd objectsensing_pipeline`. If script execution is disabled,
continue using `..\.venv\Scripts\python.exe` instead of `python` from the program
directory; changing the system execution policy is unnecessary. On Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r objectsensing_pipeline/requirements-dev.txt
```

On Linux, provide any shared libraries required by the Open3D wheel; use the
missing library name in the import error to diagnose this. See
[troubleshooting](TROUBLESHOOTING.md).

## Optional CUDA

CPU is the simplest first check. For compatible NVIDIA hardware and drivers:

```powershell
python -m pip install -r objectsensing_pipeline/requirements-gpu.txt
```

This adds CuPy for CUDA 12 and its configured runtime/compiler components.
`auto` falls back to CPU when CUDA is unavailable; explicitly requesting `cuda`
can fail. Dense CuPy acceleration and Open3D sparse-device availability are
separate capabilities. Do not assume installing CuPy makes every stage run on GPU.
Matching and most scientific operations remain CPU workloads.

## CAD input

Obtain OBJ data through the [official ShapeNet distribution](https://huggingface.co/ShapeNet)
and its access process. The loader supports:

```text
ShapeNetCore.v2/
  03001627/<model_id>/models/model_normalized.obj
  04379243/<model_id>/models/model_normalized.obj
  04256520/<model_id>/models/model_normalized.obj
```

Alternatively, provide `03001627.zip`, `04379243.zip` and `04256520.zip` in a
directory, containing the corresponding OBJ paths. The loader also searches
for OBJ files recursively under each synset as a fallback. GLB/glTF assets are
not directly supported. Categories are chair (`03001627`), table (`04379243`)
and couch (`04256520`). There is no dedicated cabinet category in this default atlas.

From `objectsensing_pipeline/`:

```powershell
python cli.py build-db-dir --shapenet "C:\datasets\ShapeNetCore.v2" --out db_dir --max-per-synset 0 --n-points 3000 --jobs 6
python cli.py build-candidate-index --db db_dir
python tools/build_local_candidate_index.py --db db_dir
```

Database layout: `index.json`, `cfg.pkl` and `model_*.pkl`; the coarse index is
`candidate_index.npz`. The optional local proposal index is a directory named
`local_candidate_index`, containing a manifest and compressed shards. It must
be built to activate the current local-candidate supplement. It expects 16³ UDFs
with half-extent 0.12 m. Changing descriptor geometry requires compatible
database/index rebuilding, not merely changing the query settings.

Builds resume completed model files. Start in a new directory when changing
preprocessing parameters; do not mix models prepared with different configurations.
Keep original model files/ZIPs accessible at the paths recorded during the build.
Moving an old database between machines does not rewrite its source mesh paths.
Only load pickle databases and caches from a trusted source.

## RGB-D input

The [ObjectSensing dataset page](https://graphics.stanford.edu/projects/objectsensing/)
provides Office, Single Chair, Chairs and IKEA Table sequences and documents its
CC BY-NC-SA 4.0 data license. Use RGB-D archives, not the authors’ retrieved-model
or reconstruction files, as pipeline inputs.

Each sequence contains numbered color/depth images and camera poses:

```text
frame-000000.color.png       RGB image
frame-000000.depth.png       16-bit depth, millimeters; zero is invalid
frame-000000.pose.txt        4x4 camera-to-world matrix
depthIntrinsics.txt         depth camera intrinsics
colorIntrinsics.txt         color camera intrinsics
```

Place ZIPs in `objectsensing_pipeline/scenes/`, or use an external path through
Run Console. Camera tracking is an input assumption; the program does not infer
camera trajectories from arbitrary images. For Atlas or another format, write
an adapter respecting depth scale, intrinsics and pose conventions.

## Runtime paths

Set environment variables **before** starting the server. Relative paths resolve
from the current working directory; absolute external paths are easier to share
between launch commands. Example from `objectsensing_pipeline/`:

```powershell
$env:OBJECTSENSING_SCENE_DIR = "D:\data\rgbd"
$env:OBJECTSENSING_DB_DIR = "D:\data\objectsensing_db"
$env:OBJECTSENSING_RUN_DIR = "D:\results\runs"
python run_console.py --host 127.0.0.1 --port 8770
```

| Variable                                    | Default relative to the program directory |
| ------------------------------------------- | ----------------------------------------- |
| `OBJECTSENSING_SCENE_DIR`                   | `scenes`                                  |
| `OBJECTSENSING_DB_DIR`                      | `db_dir`                                  |
| `OBJECTSENSING_RUN_DIR`                     | `work/cli_run`                            |
| `OBJECTSENSING_STATE_DIR`                   | `work/run_console`                        |
| `OBJECTSENSING_DEBUG_DIR`                   | `work/debug_visualizer`                   |
| `OBJECTSENSING_HEADLESS`                    | false                                     |
| `OBJECTSENSING_HOST` / `OBJECTSENSING_PORT` | `127.0.0.1` / `8770` for native use       |

`OBJECTSENSING_ALLOWED_ROOTS` optionally restricts file browsing; separate paths
with `;` on Windows and `:` on Linux. This local HTTP server has no
authentication. Keep the default loopback binding for desktop use.

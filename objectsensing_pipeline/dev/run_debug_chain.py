"""Execute consecutive instrumented Debug stages without the web server."""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_visualizer import (
    DebugVisualizerManager,
    default_parameters,
    default_sources,
    PIPELINE_STAGES,
)


STAGES = tuple(item["id"] for item in PIPELINE_STAGES if item["available"])


def wait_for_run(manager, run_id):
    printed = 0
    while True:
        snapshot = manager.snapshot()
        logs = snapshot.get("logs", [])
        for line in logs[printed:]:
            print(line, flush=True)
        printed = len(logs)
        if snapshot.get("run_id") == run_id and snapshot.get("status") != "running":
            return snapshot
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="office")
    parser.add_argument(
        "--all-scenes", action="store_true",
        help="Run all enabled scenes in one chain",
    )
    parser.add_argument("--from-stage", choices=STAGES, default="fusion")
    parser.add_argument("--to-stage", choices=STAGES, default="query")
    parser.add_argument("--upstream-run", help="Reuse the immediately preceding stage run")
    args = parser.parse_args()

    first = STAGES.index(args.from_stage)
    last = STAGES.index(args.to_stage)
    if first > last:
        parser.error("--from-stage must precede --to-stage")

    manager = DebugVisualizerManager()
    manager.watch_enabled = False
    available_sources = default_sources()
    if args.all_scenes:
        selected_sources = [
            item for item in available_sources if item.get("enabled", True)
        ]
    else:
        source = next(
            (item for item in available_sources if item["id"] == args.scene),
            None,
        )
        if source is None:
            parser.error(f'Unknown scene: {args.scene}')
        selected_sources = [source]
    scene_label = ", ".join(item["id"] for item in selected_sources)

    previous_run = args.upstream_run
    for stage in STAGES[first:last + 1]:
        if previous_run is None:
            if stage != "fusion":
                parser.error(
                    "A standalone chain without an upstream run must start at fusion"
                )
            sources = [{**source, "enabled": True} for source in selected_sources]
        else:
            sources = manager.chain_sources(previous_run, stage)["sources"]
            if not args.all_scenes:
                sources = [source for source in sources if source["id"] == args.scene]
                if not sources:
                    parser.error('Scene missing from upstream run')
        print(f"\n=== Debug {stage}: {scene_label} ===", flush=True)
        parameters = default_parameters(stage)
        parameters["parallel_scenes"] = 1
        snapshot = manager.start({
            "stage": stage,
            "parameters": parameters,
            "sources": sources,
        }, trigger="debug-chain")
        previous_run = snapshot["run_id"]
        completed = wait_for_run(manager, previous_run)
        if completed["status"] != "completed":
            raise SystemExit(
                f"Run {previous_run} finished with status "
                f"{completed['status']}"
            )
        print(f"Run completed: {previous_run}", flush=True)

    print(f"\nChain completed. Last run: {previous_run}", flush=True)


if __name__ == "__main__":
    main()

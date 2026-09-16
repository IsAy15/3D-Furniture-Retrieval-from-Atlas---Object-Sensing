'Generate a standalone retrieval timeline viewer. Embed the PLY in the HTML; detailed events come from the *_progress.json manifest produced by run-progressive.\n'

from __future__ import annotations

import base64
import json
from pathlib import Path

from web_assets import WEB_SHARED_ROOT, WEB_VIEWER_ROOT


_MODULE_DIR = Path(__file__).resolve().parent


def _script_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _inline_script(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("</script", "<\\/script")


def _default_progress_path(summary_path: Path) -> Path:
    return summary_path.with_name(summary_path.stem + "_progress.json")


def _scan_point_count(ply_path: Path) -> int:
    """Return the leading grey scan size written by ``viz.build_visualization``."""
    try:
        from viz import read_points_ply

        _, colors = read_points_ply(ply_path)
    except (OSError, ValueError, IndexError):
        return 0
    if not len(colors):
        return 0
    scan_mask = (
        (colors[:, 0] == 150)
        & (colors[:, 1] == 150)
        & (colors[:, 2] == 150)
    )
    return int(scan_mask.sum())


def build_timeline_html(json_path, viz_ply, out_html, progress_json=None):
    summary_path = Path(json_path)
    ply_path = Path(viz_ply)
    progress_path = Path(progress_json) if progress_json else _default_progress_path(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists() else {}
    )
    payload = {
        "run_name": summary_path.stem,
        "summary": summary,
        "progress": progress,
        "scan_point_count": _scan_point_count(ply_path),
        "ply_base64": base64.b64encode(ply_path.read_bytes()).decode("ascii"),
    }
    import_paths = {
        "three": "three/three.module.js",
        "three/addons/controls/OrbitControls.js": "three/addons/controls/OrbitControls.js",
        "three/addons/loaders/PLYLoader.js": "three/addons/loaders/PLYLoader.js",
    }
    imports = {
        name: "data:text/javascript;base64," + base64.b64encode(
            (_MODULE_DIR / "vendor" / relative).read_bytes()
        ).decode("ascii")
        for name, relative in import_paths.items()
    }
    document = (
        _DOCUMENT
        .replace("__OBJECTSENSING_IMPORTMAP__", _script_json({"imports": imports}))
        .replace("__OBJECTSENSING_PAYLOAD__", _script_json(payload))
        .replace(
            "__OBJECTSENSING_SHARED_CSS__",
            (WEB_SHARED_ROOT / "ui_shared.css").read_text(encoding="utf-8"),
        )
        .replace(
            "__OBJECTSENSING_VIEWER_CSS__",
            (WEB_VIEWER_ROOT / "timeline_viewer.css").read_text(encoding="utf-8"),
        )
        .replace(
            "__OBJECTSENSING_LUCIDE__",
            _inline_script(_MODULE_DIR / "vendor" / "lucide" / "lucide.min.js"),
        )
        .replace(
            "__OBJECTSENSING_TIMELINE_COMPONENTS__",
            (WEB_SHARED_ROOT / "ui_timeline.js").read_text(encoding="utf-8"),
        )
        .replace(
            "__OBJECTSENSING_SHARED_COMPONENTS__",
            (WEB_SHARED_ROOT / "ui_shared.js").read_text(encoding="utf-8"),
        )
        .replace(
            "__OBJECTSENSING_SCENE3D_COMPONENTS__",
            (WEB_SHARED_ROOT / "ui_scene3d.js").read_text(encoding="utf-8"),
        )
    )
    output = Path(out_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return str(output)


_DOCUMENT = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>ObjectSensing - execution timeline</title>
  <style>
    :root{--bg:#f4f6f7;--panel:#fff;--ink:#182027;--muted:#64717b;--line:#d4dbe0;--accent:#e35d35;--green:#23835c;--amber:#aa6d16;--red:#bd3f49;--violet:#675cab;--cyan:#137e9a;--scene:#15191c;font-family:Inter,"Segoe UI",Arial,sans-serif;color-scheme:light}
    *{box-sizing:border-box}html,body{height:100%}body{margin:0;background:var(--bg);color:var(--ink);letter-spacing:0}button,input{font:inherit}button:focus-visible,input:focus-visible{outline:3px solid color-mix(in srgb,var(--accent),transparent 45%);outline-offset:2px}
    .app{height:100%;min-height:680px;display:grid;grid-template-rows:auto minmax(0,1fr)}
    .topbar{min-height:64px;padding:12px 20px;display:flex;align-items:center;justify-content:space-between;gap:18px;border-bottom:1px solid var(--line);background:var(--panel)}
    .brand{display:flex;align-items:center;gap:12px;min-width:0}.brand-mark{width:27px;aspect-ratio:1;display:grid;grid-template-columns:1fr 1fr;gap:3px;transform:rotate(45deg)}.brand-mark i{background:var(--ink)}.brand-mark i:last-child{background:var(--accent)}
    .brand strong{display:block;font-size:17px}.brand span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted);font-size:12px}
    .run-state{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:12px}.run-state i{width:9px;height:9px;border-radius:50%;background:var(--green)}
    .stages{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));border-bottom:1px solid var(--line);background:#e9edf0}
    .stage-tab{min-width:0;padding:11px 12px;border:0;border-right:1px solid var(--line);border-bottom:4px solid transparent;background:transparent;color:var(--muted);text-align:left;cursor:pointer}.stage-tab:last-child{border-right:0}.stage-tab[aria-selected="true"]{border-bottom-color:var(--accent);background:var(--panel);color:var(--ink)}.stage-tab small{display:block;margin-bottom:3px;color:var(--accent);font-size:10px;font-weight:700}.stage-tab strong{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:12px}
    .workspace{min-height:0;display:grid;grid-template-columns:minmax(0,1.52fr) minmax(350px,.48fr)}
    .scene-column{min-width:0;min-height:0;display:grid;grid-template-rows:minmax(0,1fr) auto;background:var(--scene)}
    .scene{position:relative;min-height:360px;overflow:hidden}.scene canvas{display:block;width:100%;height:100%;cursor:grab}.scene canvas:active{cursor:grabbing}.scene-message{position:absolute;inset:0;display:grid;place-items:center;padding:30px;color:#dfe6ea;text-align:center}.scene-message[hidden]{display:none}
    .scene-legend{position:absolute;left:15px;bottom:14px;display:flex;flex-wrap:wrap;gap:9px;padding:8px 10px;background:rgba(21,25,28,.78);color:#dfe6ea;font-size:11px}.legend-item{display:flex;align-items:center;gap:6px}.legend-dot{width:8px;height:8px;border-radius:50%;background:var(--dot)}
    .scene-badge{position:absolute;top:14px;left:15px;max-width:calc(100% - 30px);padding:8px 10px;background:rgba(21,25,28,.78);color:#fff;font-size:12px}.scene-badge strong{color:#ffb39d}
    .controls{padding:12px 16px 14px;border-top:1px solid #30383d;background:#20262a;color:#eef3f5}.control-row{display:grid;grid-template-columns:40px 40px minmax(0,1fr) 40px 40px 40px;gap:8px;align-items:center}.control-button{width:40px;height:40px;border:1px solid #58646b;border-radius:5px;background:#2b3338;color:#fff;cursor:pointer}.control-button.major{border-color:#7d8b93;background:#343e44}.control-button:disabled{opacity:.35;cursor:not-allowed}.range-wrap{display:grid;grid-template-columns:auto minmax(0,1fr);gap:11px;align-items:center;font-size:11px}.range-wrap output{min-width:72px;color:#c9d2d7;font-variant-numeric:tabular-nums}.range-wrap input{width:100%;accent-color:var(--accent)}
    .event-caption{margin:9px 0 0;color:#c5ced3;font-size:12px}.event-caption strong{color:#fff}
    .inspector{min-width:0;min-height:0;display:grid;grid-template-rows:auto auto minmax(160px,1fr) auto;background:var(--panel);border-left:1px solid var(--line)}
    .event-head{padding:20px 20px 15px;border-bottom:1px solid var(--line)}.event-head .eyebrow{margin:0 0 7px;color:var(--accent);font-size:10px;font-weight:800;text-transform:uppercase}.event-head h1{margin:0;font-size:23px;line-height:1.15}.event-head p{margin:9px 0 0;color:var(--muted);font-size:13px;line-height:1.5}
    .metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-bottom:1px solid var(--line)}.metric{min-width:0;padding:12px 13px;border-right:1px solid var(--line)}.metric:last-child{border-right:0}.metric small{display:block;margin-bottom:4px;color:var(--muted);font-size:9px;text-transform:uppercase}.metric strong{display:block;overflow:hidden;text-overflow:ellipsis;font-size:16px;font-variant-numeric:tabular-nums}
    .list-panel{min-height:0;display:grid;grid-template-rows:auto minmax(0,1fr)}.list-head{padding:11px 14px;display:flex;align-items:center;justify-content:space-between;gap:10px;border-bottom:1px solid var(--line)}.list-head strong{font-size:12px}.list-head span{color:var(--muted);font-size:11px}.candidate-list{min-height:0;overflow:auto;padding:6px}
    .candidate{width:100%;min-height:48px;display:grid;grid-template-columns:30px minmax(0,1fr) auto;gap:9px;align-items:center;padding:7px 8px;border:0;border-bottom:1px solid var(--line);background:transparent;color:var(--ink);text-align:left;cursor:pointer}.candidate:hover,.candidate[aria-pressed="true"]{background:#edf1f3}.candidate-rank{color:var(--muted);font-size:10px;font-variant-numeric:tabular-nums}.candidate-name{min-width:0}.candidate-name strong,.candidate-name span{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.candidate-name strong{font-size:11px}.candidate-name span{margin-top:3px;color:var(--muted);font-size:10px}.status{padding:4px 6px;border-radius:3px;background:#e8ecef;color:var(--muted);font-size:9px;font-weight:800;text-transform:uppercase}.status.selected,.status.candidate,.status.kept{background:#ddefe6;color:#176344}.status.matching{background:#dcecf2;color:#0f6278}.status.pending{background:#edf0f2;color:#78858e}.status.rejected_min_symmetric_score,.status.rejected_coverage,.status.rejected_reverse,.status.rejected_verification,.status.rejected_no_correspondence,.status.rejected_no_constellation{background:#f5dfe1;color:#8d2630}.status.rejected_horizontal_iou,.status.rejected_cross_category_collision,.status.rejected_global_explained_overlap,.status.rejected_explained_overlap,.status.rejected_category_min_coverage{background:#f5ead6;color:#79500f}
    .detail{padding:14px 18px;border-top:1px solid var(--line);background:#f7f8f9}.detail h2{margin:0 0 8px;font-size:14px}.detail-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:7px}.detail-grid span{min-width:0;padding-left:8px;border-left:3px solid var(--line);font-size:10px;color:var(--muted)}.detail-grid b{display:block;margin-top:3px;overflow:hidden;text-overflow:ellipsis;color:var(--ink);font-size:12px}.detail p{margin:9px 0 0;color:var(--muted);font-size:11px;line-height:1.45}.trace-pills,.relation-buttons{display:flex;flex-wrap:wrap;gap:5px;margin-top:9px}.trace-pills span,.relation-button{max-width:100%;overflow:hidden;text-overflow:ellipsis;padding:5px 7px;border:0;border-radius:3px;background:#e9edf0;color:var(--ink);font-size:10px}.relation-button{cursor:pointer}.relation-button[aria-pressed="true"]{background:var(--ink);color:#fff}.relation-label{width:100%;margin-top:4px;color:var(--muted);font-size:10px;font-weight:700;text-transform:uppercase}
    @media(max-width:1000px){.app{height:auto;min-height:100%}.workspace{grid-template-columns:1fr}.scene-column{min-height:620px}.inspector{min-height:680px;border-left:0;border-top:1px solid var(--line)}}
    @media(max-width:640px){.topbar{padding:10px 12px}.run-state{display:none}.stages{grid-template-columns:repeat(3,minmax(0,1fr))}.stage-tab:nth-child(3n){border-right:0}.workspace{display:block}.scene-column{min-height:0}.scene{height:430px;min-height:0}.inspector{min-height:650px}.event-head{padding:17px 14px}.event-head h1{font-size:20px}.metrics{grid-template-columns:1fr 1fr}.metric:nth-child(3){grid-column:1/-1;border-top:1px solid var(--line)}.detail{padding:13px}.detail-grid{grid-template-columns:1fr 1fr}.control-row{grid-template-columns:36px 36px minmax(0,1fr) 36px 36px}.control-button{width:36px;height:36px}.control-row #play{grid-column:1/-1;width:100%}.range-wrap{grid-template-columns:1fr}.range-wrap output{text-align:center}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important}}
  </style>
  <style>__OBJECTSENSING_SHARED_CSS__</style>
  <style>__OBJECTSENSING_VIEWER_CSS__</style>
</head>
<body>
  <div class="app viewer-app" id="app">
    <header class="topbar ui-topbar">
      <div class="brand"><a class="icon-button viewer-back" href="/" aria-label="Back to Run Console" title="Back to Run Console" data-tooltip="Back to Run Console" data-tooltip-detail="Return to run controls in this tab."><i data-lucide="arrow-left"></i></a><span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span><div><strong>ObjectSensing</strong><span id="run-name"></span></div></div>
      <div class="viewer-topbar-center"><span class="viewer-chip"><strong>Final Viewer</strong> · saved timeline</span></div>
      <div class="viewer-topbar-actions"><button class="icon-button" id="reset-camera" type="button" aria-label="Reset camera" title="Reset camera"><i data-lucide="focus"></i></button><button class="icon-button" id="toggle-viewer-inspector" type="button" aria-label="Collapse inspector" data-tooltip="Inspector" data-tooltip-detail="Hover over the rail to preview; click to keep the panel open."><i data-lucide="panel-right"></i></button><div class="run-state"><i aria-hidden="true"></i>Run completed</div></div>
    </header>
    <main class="workspace" id="viewer-workspace">
      <section class="scene-column" aria-label="3D scene">
        <div class="scene ui-scene" id="scene"><div class="scene-message" id="scene-message">Loading 3D scene…</div><div class="scene-badge ui-scene-badge" id="scene-badge"></div><div class="scene-legend ui-scene-legend left" aria-label="3D layer legend"><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#77b8d6"></i>RGB-D fusion</span><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#969696"></i>TSDF scan</span><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#e6a03c"></i>candidate pose</span><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#65c7d6"></i>multiple selection</span><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#f4d35e"></i>keypoints</span><span class="legend-item ui-legend-item"><i class="legend-dot ui-legend-dot" style="--dot:#e45b64"></i>correspondences</span><button class="legend-item legend-toggle ui-legend-item" id="toggle-all-constellations" type="button" aria-pressed="false" data-tooltip="Show all constellations" data-tooltip-detail="Draw scene keypoint groups supporting saved poses at the current checkpoint."><i class="legend-dot ui-legend-dot" style="--dot:#9d92f2"></i><span>constellations</span><b id="constellation-layer-count"></b></button></div></div>
        <div class="viewer-timeline-resizer ui-resizer" id="viewer-timeline-resizer" role="separator" aria-label="Resize timeline" aria-orientation="horizontal" tabindex="0" data-tooltip="Timeline height" data-tooltip-detail="Drag vertically, use arrow keys, or double-click to reset."></div>
        <div class="controls ui-panel">
          <div class="viewer-timeline-heading">
            <span id="current-stage-title">Pipeline</span>
            <strong id="current-step-title">Initialization</strong>
          </div>
          <div class="viewer-chapter-track" id="timeline-chapters" aria-label="Timeline chapters"></div>
          <div class="viewer-event-track" id="viewer-event-track" aria-label="Detailed timeline stages"></div>
          <div class="control-row"><button class="control-button major" id="major-prev" type="button" aria-label="Previous milestone" title="Previous milestone"><i data-lucide="skip-back"></i></button><button class="control-button" id="prev" type="button" aria-label="Previous event" title="Previous event"><i data-lucide="chevron-left"></i></button><label class="range-wrap" for="timeline"><output id="timeline-output"></output><input id="timeline" type="range" min="0" value="0" step="1"></label><button class="control-button" id="next" type="button" aria-label="Next event" title="Next event"><i data-lucide="chevron-right"></i></button><button class="control-button major" id="major-next" type="button" aria-label="Next milestone" title="Next milestone"><i data-lucide="skip-forward"></i></button><button class="control-button" id="play" type="button" aria-label="Play timeline" title="Play timeline"><i data-lucide="play"></i></button></div>
          <p class="event-caption" id="event-caption"></p>
        </div>
      </section>
      <div class="viewer-inspector-resizer ui-resizer" id="viewer-inspector-resizer" role="separator" aria-label="Resize inspector" aria-orientation="vertical" tabindex="0" data-tooltip="Inspector width" data-tooltip-detail="Drag horizontally, use arrow keys, or double-click to reset."></div>
      <aside class="inspector ui-panel" id="viewer-inspector">
        <div class="inspector-scroll" id="inspector-scroll">
          <details class="ui-accordion" data-panel="event" data-height="108"><summary>Event <i data-lucide="chevron-right"></i></summary><div class="ui-accordion-body"><div class="event-head"><p class="eyebrow" id="event-eyebrow"></p><h1 id="event-title"></h1><p id="event-description"></p></div></div></details>
          <details class="ui-accordion" data-panel="metrics" data-height="84"><summary>Metrics <i data-lucide="chevron-right"></i></summary><div class="ui-accordion-body"><div class="metrics" id="metrics"></div></div></details>
          <details class="ui-accordion" data-panel="detail" data-height="220" open><summary>Candidate details <i data-lucide="chevron-right"></i></summary><div class="ui-accordion-body">
          <section class="detail" id="candidate-detail"><h2>Select a candidate</h2><p>Details show correspondences, constellations, verification scores and rejection reasons.</p></section>
          </div></details>
          <details class="ui-accordion" data-panel="candidates" data-height="360" open><summary>Candidates <i data-lucide="chevron-right"></i></summary><div class="ui-accordion-body">
          <section class="list-panel">
            <div class="list-head"><strong id="list-title"></strong><span id="list-count"></span></div>
            <div class="list-toolbar">
              <div class="list-toolbar-top">
                <div class="list-scope" role="tablist" aria-label="List content"><button class="active" id="scope-timeline" type="button" data-list-scope="timeline" role="tab" aria-selected="true">Timeline</button><button id="scope-final" type="button" data-list-scope="final" role="tab" aria-selected="false">Final candidates <span id="final-count"></span></button></div>
                <div class="list-toolbar-actions" id="final-list-actions" hidden><button class="icon-button compact" id="show-all-final" type="button" aria-label="Show all" data-tooltip="Show all" data-tooltip-detail="Overlay all visible final candidates on the RGB-D scan."><i data-lucide="eye"></i></button><button class="icon-button compact" id="show-selected-final" type="button" aria-label="Show retained candidates" title="Show retained candidates" data-tooltip="Show retained models" data-tooltip-detail="Enable all retained final decisions at once."><i data-lucide="badge-check"></i></button><button class="icon-button compact" id="focus-final-selection" type="button" aria-label="Frame candidates in the scene" title="Frame candidates in the scene" data-tooltip="Frame selection" data-tooltip-detail="Focus on displayed models while keeping the RGB-D scan as a reference." disabled><i data-lucide="focus"></i></button><button class="icon-button compact" id="clear-final-selection" type="button" aria-label="Hide all" title="Hide all" data-tooltip="Hide all" data-tooltip-detail="Remove all candidate previews from the scene."><i data-lucide="eye-off"></i></button></div>
              </div>
              <div class="candidate-filters">
                <label class="candidate-search"><i data-lucide="search"></i><input id="candidate-search" type="search" autocomplete="off" placeholder="Search by model or reason" aria-label="Search candidates"></label>
                <select id="candidate-category" aria-label="Filter by category"><option value="all">All categories</option></select>
                <select id="candidate-status" aria-label="Filter by decision"><option value="all">All decisions</option><option value="selected">Retained</option><option value="rejected">Rejected</option><option value="pending">Waiting</option></select>
                <button class="icon-button compact" id="reset-candidate-filters" type="button" aria-label="Reset filters" data-tooltip="Reset filters"><i data-lucide="list-filter"></i></button>
              </div>
            </div>
            <div class="candidate-list" id="candidate-list"></div>
          </section>
          </div></details>
        </div>
      </aside>
    </main>
    <div class="ui-tooltip viewer-tooltip" id="viewer-tooltip" role="tooltip" hidden><strong></strong><span></span></div>
  </div>
  <script>window.OBJECTSENSING_RUN=__OBJECTSENSING_PAYLOAD__;</script>
  <script>__OBJECTSENSING_LUCIDE__</script>
  <script type="importmap">
    __OBJECTSENSING_IMPORTMAP__
  </script>
  <script type="module">
    import * as THREE from "three";
    import {OrbitControls} from "three/addons/controls/OrbitControls.js";
    import {PLYLoader} from "three/addons/loaders/PLYLoader.js";
    __OBJECTSENSING_SHARED_COMPONENTS__
    __OBJECTSENSING_SCENE3D_COMPONENTS__
    __OBJECTSENSING_TIMELINE_COMPONENTS__

    const payload=window.OBJECTSENSING_RUN, progress=payload.progress||{}, summary=payload.summary||[];
    const stages=["ShapeNet database","RGB-D fusion","Query / Top-k","Matching","Verification","Selection"];
    const stageIds=["database","fusion","query","matching","verification","selection"];
    const reasonLabels={selected:"retained",kept:"validated",candidate:"candidate",matching:"matching",pending:"waiting",rejected_no_correspondence:"no correspondence",rejected_no_constellation:"no constellation",rejected_verification:"verification failed",rejected_coverage:"insufficient coverage",rejected_reverse:"insufficient reverse coverage",rejected_min_symmetric_score:"symmetric score below threshold",rejected_horizontal_iou:"spatial duplicate",rejected_cross_category_collision:"cross-category collision",rejected_global_explained_overlap:"surface already explained",rejected_explained_overlap:"surface already explained",rejected_category_min_coverage:"category minimum coverage",rejected_category_limit:"category limit",rejected_icp_or_scale:"invalid ICP or scale",rejected_reverse_gate:"insufficient reverse coverage",rejected_structure:"structure too flat",rejected_thickness:"insufficient thickness",accepted_pose:"pose accepted"};
    $("#run-name").textContent=payload.run_name;

    const viewerLayout={
      inspector:{property:"--viewer-inspector-width",storage:"objectsensing-viewer-inspector-width",defaultValue:390,min:300,max:620},
      timeline:{property:"--viewer-timeline-height",storage:"objectsensing-viewer-timeline-height",defaultValue:286,min:190,max:560},
    };
    function viewerLayoutLimit(kind){
      const config=viewerLayout[kind];
      if(kind==="inspector"){
        const width=$("#viewer-workspace")?.clientWidth||innerWidth;
        return Math.max(config.min,Math.min(config.max,width-480));
      }
      const height=$(".scene-column")?.clientHeight||innerHeight;
      return Math.max(config.min,Math.min(config.max,height-260));
    }
    const viewerResizers={};
    function setViewerLayout(kind,value,persist=true){return viewerResizers[kind]?.set(value,persist)}
    function bindViewerResizer(kind){
      const config=viewerLayout[kind];
      viewerResizers[kind]=bindResizableDimension({
        handle:`#viewer-${kind}-resizer`,
        container:kind==="inspector"?"#viewer-workspace":".scene-column",
        property:config.property,
        storageKey:config.storage,
        defaultValue:config.defaultValue,
        min:config.min,
        getMax:()=>viewerLayoutLimit(kind),
        axis:kind==="inspector"?"x":"y",
        edge:kind==="inspector"?"end":"bottom",
        disabledBelow:1000,
        stateTarget:$("#app"),
        stateClass:kind==="inspector"?"resizing-panels":"resizing-timeline",
      });
    }
    function bindViewerResizers(){
      bindResizableAccordions({container:$("#inspector-scroll"),storagePrefix:"objectsensing-viewer-sections"});
      bindViewerResizer("inspector");
      bindViewerResizer("timeline");
      window.addEventListener("resize",()=>Object.values(viewerResizers).forEach(resizer=>resizer?.refresh()));
    }

    function updateViewerPanelToggle(){
      const closed=$("#app").classList.contains("inspector-collapsed"),button=$("#toggle-viewer-inspector");
      button.setAttribute("aria-expanded",String(!closed));
      button.setAttribute("aria-label",closed?"Open inspector":"Collapse inspector");
      button.dataset.tooltip=closed?"Open inspector":"Collapse inspector";
    }

    const traceSteps=(progress.execution_trace||[]).filter(Boolean);
    const fallbackSteps=progress.steps||[];
    const fusionTrace=progress.fusion_trace||{};
    const fusionFrames=fusionTrace.frames||[];
    const finalRevalidation=progress.final_revalidation_trace||progress.final_revalidation||progress.final_candidates||[];
    const finalSelection=progress.final_selection_trace||progress.final_selection||progress.selected||summary;

    function buildEvents(){
      const events=[{stage:0,kind:"database",major:true,title:"Model database available",description:"ShapeNet models were normalized, sampled and described before the scene run."}];
      events.push({stage:1,kind:"fusion-start",major:true,visibleCount:0,title:"TSDF volume initialized",description:"The volume is empty. Subsequent RGB-D measurements add observed surfaces in common coordinates."});
      if(fusionFrames.length){
        fusionFrames.forEach((frame,index)=>events.push({stage:1,kind:"fusion-frame",major:index===fusionFrames.length-1||index===0||(index+1)%10===0,visibleCount:index+1,frame,index,title:`RGB-D fusion · frame ${frame.frame_index||index+1}/${fusionTrace.total_frames||fusionFrames.length}`,description:"Colored points come from this depth frame, transformed by its camera pose before TSDF integration."}));
      }else{
        events.push({stage:1,kind:"fusion",major:true,title:"Fused scene surface",description:"This export contains only the final isosurface, without intermediate RGB-D samples."});
      }
      if(traceSteps.length){
        traceSteps.forEach((step,stepIndex)=>{
          const full=step.full||{}, pre=full.preselection||{}, matching=full.matching||[];
          const addFillEvents=(kind,items,label,description)=>{
            const relatedModels=items.map(item=>typeof item==="string"?item:item?.model).filter(Boolean);
            events.push({stage:2,kind,step,stepIndex,major:true,visibleCount:0,relatedModels:[],title:`Checkpoint ${step.frame_count} · ${label} vide`,description:`List ${label} has been created. ${description}`});
            const batch=Math.max(1,Math.ceil(items.length/10));
            for(let count=batch;count<=items.length+batch;count+=batch){
              const visibleCount=Math.min(count,items.length);
              if(!visibleCount)break;
              events.push({stage:2,kind,step,stepIndex,major:true,visibleCount,relatedModels:relatedModels.slice(0,visibleCount),title:`Checkpoint ${step.frame_count} · ${label} ${visibleCount}/${items.length}`,description});
              if(visibleCount===items.length)break;
            }
          };
          const keypoints=full.scan?.keypoints||[];
          events.push({stage:2,kind:"keypoints",step,stepIndex,major:true,visibleCount:0,title:`Checkpoint ${step.frame_count} · keypoints hidden`,description:"3D Harris responses are ranked before spatial balancing."});
          const kpBatch=Math.max(1,Math.ceil(keypoints.length/10));
          for(let count=kpBatch;count<=keypoints.length+kpBatch;count+=kpBatch){
            const visibleCount=Math.min(count,keypoints.length);
            if(!visibleCount)break;
            events.push({stage:2,kind:"keypoints",step,stepIndex,major:true,visibleCount,title:`Retained keypoints · ${visibleCount}/${keypoints.length}`,description:"Each yellow point is a scene keypoint available for model correspondences."});
            if(visibleCount===keypoints.length)break;
          }
          addFillEvents("pool",pre.pool||[],"pool", "The first level reduces the database to a broad pool before local reranking.");
          addFillEvents("topk",pre.top_k||[],"top-k", "Only models in this list proceed to full matching.");
          matching.forEach((candidate,index)=>{
            const correspondences=candidate.correspondence_preview||[], constellations=candidate.constellation_preview||[], poses=candidate.verification?.poses||[];
            events.push({stage:3,kind:"candidate-start",step,stepIndex,index,candidate,major:true,title:`Candidate ${index+1}/${matching.length} · ${candidate.model}`,description:"The model is loaded. Its descriptors will be compared with scene keypoints."});
            if(correspondences.length)events.push({stage:3,kind:"correspondences",step,stepIndex,index,candidate,pose:correspondences[0],title:`${correspondences.length} correspondence(s) highlighted`,description:"Each line connects a transformed model keypoint to the scene keypoint proposed by its descriptor."});
            constellations.forEach((pose,poseIndex)=>events.push({stage:3,kind:"constellation",step,stepIndex,index,candidate,pose,poseIndex,title:`Constellation ${poseIndex+1}/${constellations.length} · Q=${number(pose.quality,1)}`,description:"1-Point RANSAC groups compatible correspondences into a shared pose hypothesis."}));
            poses.forEach((pose,poseIndex)=>events.push({stage:3,kind:"pose-verification",step,stepIndex,index,candidate,pose,previousPose:constellations[Math.max(0,(pose.rank||poseIndex+1)-1)],poseIndex,title:`Verified pose ${poseIndex+1}/${poses.length} · ${reasonLabels[pose.status]||pose.status}`,description:"Orange shows the post-ICP model. The blue ghost shows the constellation hypothesis before refinement."}));
            events.push({stage:3,kind:"candidate-result",step,stepIndex,index,candidate,pose:candidate.registration||(candidate.registrations||[])[0],major:false,title:`Result · ${reasonLabels[candidate.status]||candidate.status}`,description:"The candidate leaves the queue with its status and verification metrics."});
          });
          events.push({stage:3,kind:"checkpoint",step,stepIndex,major:true,title:`Checkpoint ${step.frame_count} · consolidation`,description:"Competing checkpoint poses are deduplicated before entering the progressive cache."});
        });
      }else{
        fallbackSteps.forEach((step,stepIndex)=>events.push({stage:2,kind:"fallback-query",step,stepIndex,major:true,title:`Checkpoint ${step.frame_count}`,description:"This legacy manifest stores proposed models and cache state but no detailed Top-k trace."}));
        (progress.final_candidates||[]).forEach((candidate,index)=>events.push({stage:3,kind:"fallback-matching",index,candidate,major:true,title:`Candidate ${index+1}`,description:"Pose hypothesis retained until final revalidation."}));
      }
      finalRevalidation.forEach((candidate,index)=>events.push({stage:4,kind:"verification",index,candidate,major:true,title:`Revalidation ${index+1}/${finalRevalidation.length}`,description:"The pose is measured again on the final scan using forward and reverse coverage."}));
      finalSelection.forEach((candidate,index)=>events.push({stage:5,kind:"selection",index,candidate,major:true,title:`Global decision ${index+1}/${finalSelection.length}`,description:"Selection accepts or rejects the pose using score, duplicates, collisions and already explained surfaces."}));
      events.push({stage:5,kind:"final",major:true,title:"Final result",description:`${summary.length} model(s) remain visible in the scene.`});
      return events;
    }
    const events=buildEvents(); let eventIndex=0,playTimer=null,selectedCandidate=null,selectedKeypointIndex=null,selectedConstellationContext=null,highlightRelation=null,listScope="timeline",renderedCandidateItems=[],candidateEventGroups=[],candidateRelatedEvents=new Set(),finalSelectionAnchor=null;
    const selectedFinalKeys=new Set();
    let candidateFilterController=null;
    $("#timeline").max=Math.max(0,events.length-1);
    function nameOf(item){return item?.model||item||"—"} function categoryOf(item){return item?.category||String(nameOf(item)).split("_",1)[0]||"—"}
    function statusOf(item){return item?.status||"pending"} function pct(value){return value==null?"—":`${Math.round(Number(value)*100)} %`} function number(value,digits=2){return value==null?"—":Number(value).toFixed(digits)}
    function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,char=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[char]))}
    function currentTrace(event){return event.step?.full||null}
    const modelTraceMap=new Map();traceSteps.forEach(step=>(step.full?.matching||[]).forEach(candidate=>{if(!modelTraceMap.has(candidate.model))modelTraceMap.set(candidate.model,candidate)}));
    function detailedCandidate(item){return item?.model_preview?item:(modelTraceMap.get(nameOf(item))||item)}
    const finalCandidateSource=finalSelection.length?finalSelection:finalRevalidation;
    const finalCandidates=finalCandidateSource.map((item,index)=>{
      const candidate=typeof item==="object"&&item?{...item}:{model:item,status:"pending"};
      candidate._finalKey=`${nameOf(candidate)}:${candidate.event_index??candidate.checkpoint_frames??index}:${index}`;
      return candidate;
    });
    candidateFilterController=createCandidateFilterController({search:"#candidate-search",category:"#candidate-category",status:"#candidate-status",reset:"#reset-candidate-filters",getCategory:categoryOf,getStatus:decisionGroup,getSearchText:item=>{const status=statusOf(item);return[nameOf(item),categoryOf(item),status,reasonLabels[status],item?.reason].filter(Boolean).join(" ")},seedCategories:[...Object.keys(progress.database_summary?.categories||{}),...finalCandidates.map(categoryOf)],onChange:()=>rerenderCandidateList()});
    $("#final-count").textContent=finalCandidates.length?`(${finalCandidates.length})`:"";
    function eventList(event){
      if(event.kind==="database"){const db=progress.database_summary||{};return Object.entries(db.categories||{}).map(([category,count])=>({model:category,category:"synset",count,status:"kept"}));}
      if(event.kind==="fusion-start")return [];
      if(event.kind==="fusion-frame")return fusionFrames.slice(0,event.visibleCount).map((frame,index)=>({model:`Frame ${frame.frame_index||index+1}`,category:`${frame.points?.length||0} points RGB-D`,status:"kept"}));
      if(event.kind==="fusion")return (traceSteps.length?traceSteps:fallbackSteps).map(step=>({model:`${step.frame_count} frames`,category:"checkpoint",status:"kept"}));
      if(event.kind==="keypoints")return (currentTrace(event)?.scan?.keypoints||[]).slice(0,event.visibleCount).map((keypoint,index)=>({model:`Keypoint #${index+1}`,category:`response ${number(keypoint.response,4)}`,status:"kept",keypointIndex:index,keypoint}));
      if(event.kind==="pool")return (currentTrace(event)?.preselection?.pool||[]).slice(0,event.visibleCount).map(model=>({model,category:categoryOf(model),status:"pending"}));
      if(event.kind==="topk")return (currentTrace(event)?.preselection?.top_k||[]).slice(0,event.visibleCount).map(model=>({model,category:categoryOf(model),status:"pending"}));
      if(["candidate-start","correspondences","constellation","pose-verification","candidate-result"].includes(event.kind)){
        const trace=currentTrace(event),top=trace?.preselection?.top_k||[],finished=event.kind==="candidate-result"?event.index+1:event.index,done=new Map((trace?.matching||[]).slice(0,finished).map(item=>[item.model,item]));
        if(event.kind!=="candidate-result"&&event.candidate)done.set(event.candidate.model,{...event.candidate,status:"matching"});
        return top.map(model=>done.get(model)||{model,category:categoryOf(model),status:"pending"});
      }
      if(event.kind==="checkpoint")return currentTrace(event)?.checkpoint_selection||[];
      if(event.kind==="fallback-query")return (event.step.models||[]).map(model=>({model,category:categoryOf(model),status:"candidate"}));
      if(event.kind==="fallback-matching")return (progress.final_candidates||[]).slice(0,event.index+1);
      if(event.kind==="verification")return finalRevalidation.slice(0,event.index+1);
      if(event.kind==="selection")return finalSelection.slice(0,event.index+1);
      if(event.kind==="final")return finalSelection.filter(item=>(item.status||"selected")==="selected");
      return [];
    }
    function constellationScanIndices(constellation){
      return new Set((constellation?.inlier_pairs||[]).map(pair=>Number(pair.scan_keypoint)).filter(Number.isFinite));
    }
    function candidateRelationForScanIndices(candidate,scanIndices){
      const constellations=candidate?.constellation_preview||[];
      let best=null;
      constellations.forEach((constellation,index)=>{
        const indices=constellationScanIndices(constellation),overlap=[...indices].filter(value=>scanIndices.has(value)).length;
        if(overlap&&(!best||overlap>best.overlap||(overlap===best.overlap&&Number(constellation.quality||0)>Number(best.constellation.quality||0))))best={type:"constellation",index,constellation,overlap};
      });
      if(best)return best;
      const correspondenceIndex=(candidate?.correspondence_preview||[]).findIndex(corr=>scanIndices.has(Number(corr.scan_keypoint)));
      return correspondenceIndex>=0?{type:"correspondence",index:correspondenceIndex,overlap:1}:null;
    }
    function buildKeypointConstellationContext(keypointIndex,event){
      const trace=currentTrace(event)||(traceSteps.at(-1)?.full),matching=trace?.matching||[],owners=[];
      matching.forEach(candidate=>(candidate.constellation_preview||[]).forEach((constellation,index)=>{
        const scanIndices=constellationScanIndices(constellation);
        if(scanIndices.has(keypointIndex))owners.push({candidate,constellation,index,scanIndices});
      }));
      owners.sort((left,right)=>Number(right.constellation.quality||0)-Number(left.constellation.quality||0));
      const primary=owners[0]||null,scanIndices=primary?.scanIndices||new Set([keypointIndex]);
      const candidates=matching.filter(candidate=>Boolean(candidateRelationForScanIndices(candidate,scanIndices)));
      return{sourceKeypointIndex:keypointIndex,primaryCandidate:primary?.candidate||candidates[0]||null,constellation:primary?.constellation||null,constellationIndex:primary?.index??null,scanIndices,candidates,ownerCount:owners.length};
    }
    function metricsFor(event,list){
      if(event.kind==="database"){const db=progress.database_summary||{};return [["Models",db.models??"Not recorded"],["Categories",db.categories?Object.keys(db.categories).length:"Not recorded"]];}
      if(event.stage===1){const integrated=event.visibleCount||0,points=fusionFrames.slice(0,integrated).reduce((sum,frame)=>sum+(frame.points?.length||0),0);return [["Frames",`${integrated}/${fusionTrace.total_frames||traceSteps.at(-1)?.frame_count||fallbackSteps.at(-1)?.frame_count||"—"}`],["Displayed measurements",points.toLocaleString("en-US")],["Backend",fusionTrace.backend||progress.run_config?.fusion_backend||"—"]];}
      if(event.kind==="keypoints"){const total=currentTrace(event)?.scan?.keypoints?.length||0;return [["Checkpoint",event.step?.frame_count||"—"],["Keypoints",`${list.length}/${total}`],["Detected",currentTrace(event)?.scan?.detected_keypoints||"—"]];}
      if(event.stage===2){const pre=currentTrace(event)?.preselection||{},poolTotal=pre.pool?.length||pre.pool_count||"—",topTotal=pre.top_k?.length||pre.top_k_count||"—";return [["Checkpoint",event.step?.frame_count||"—"],["Pool",event.kind==="pool"?`${list.length}/${poolTotal}`:poolTotal],["Top-k",event.kind==="topk"?`${list.length}/${topTotal}`:topTotal]];}
      if(event.stage===3){const done=list.filter(item=>!["pending","matching"].includes(statusOf(item)));return [["Tested",done.length],["Candidates",done.filter(item=>statusOf(item)==="candidate").length],["Remaining",list.filter(item=>statusOf(item)==="pending").length]];}
      if(event.stage===4)return [["Revalidated",list.length],["Retained",list.filter(item=>["kept","selected"].includes(statusOf(item))).length],["Coverage threshold",pct(progress.run_config?.coverage||.45)]];
      return [["Decided",list.length],["Retained",list.filter(item=>(item.status||"selected")==="selected").length],["Rejected",list.filter(item=>String(statusOf(item)).startsWith("rejected")).length]];
    }
    function listTitle(event){if(event.kind==="keypoints")return "Scene keypoints";return ["Indexed categories","Integrated RGB-D frames","Preselection list","Top-k being matched","Final revalidation","Selection decisions"][event.stage]}
    function candidateColor(item,index=0){
      const palette=["#65c7d6","#efbd5a","#ee746d","#78cf8c","#9d92f2","#e78bc4","#72a7ee","#e38d62"];
      return palette[index%palette.length];
    }
    function candidateVisualIndex(item,fallback=0){
      const index=finalCandidates.indexOf(item);
      return index>=0?index:fallback;
    }
    function drawCandidateThumbnail(canvas,item,index){
      const context=canvas.getContext("2d"),width=136,height=96,detail=detailedCandidate(item),points=detail?.model_preview||[];
      canvas.width=width;canvas.height=height;context.clearRect(0,0,width,height);
      context.fillStyle="#0f1214";context.fillRect(0,0,width,height);
      if(!points.length){
        context.fillStyle="#6f7880";context.font="600 17px Segoe UI";context.textAlign="center";context.fillText(categoryOf(item).slice(0,3).toUpperCase(),width/2,height/2+6);
        return;
      }
      const projected=points.map(point=>{const x=Number(point[0]),y=Number(point[1]),z=Number(point[2]),c=.82,s=.57,xr=c*x+s*z,zr=-s*x+c*z;return[xr+zr*.22,-y+zr*.28]});
      const xs=projected.map(point=>point[0]),ys=projected.map(point=>point[1]),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys),spanX=Math.max(maxX-minX,.001),spanY=Math.max(maxY-minY,.001),scale=Math.min((width-18)/spanX,(height-16)/spanY),offsetX=(width-spanX*scale)/2-minX*scale,offsetY=(height-spanY*scale)/2-minY*scale;
      context.fillStyle=candidateColor(item,candidateVisualIndex(item,index));context.globalAlpha=.88;
      projected.forEach(point=>context.fillRect(offsetX+point[0]*scale,offsetY+point[1]*scale,2,2));
      context.globalAlpha=1;
    }
    function renderCandidateThumbnails(list){
      document.querySelectorAll("canvas[data-candidate-thumb]").forEach(canvas=>{const index=Number(canvas.dataset.candidateThumb);if(list[index])drawCandidateThumbnail(canvas,list[index],index)});
    }
    function decisionGroup(item){
      const status=statusOf(item);
      if(["selected","kept"].includes(status))return"selected";
      if(String(status).startsWith("rejected"))return"rejected";
      return"pending";
    }
    function syncCategoryFilter(list){
      candidateFilterController.syncCategories(list);
    }
    function applyCandidateFilters(list){
      return candidateFilterController.filter(list);
    }
    function renderList(event,timelineList){
      const finalMode=listScope==="final",relationMode=!finalMode&&Boolean(selectedConstellationContext),sourceList=finalMode?finalCandidates:(relationMode?selectedConstellationContext.candidates:timelineList);
      syncCategoryFilter(sourceList);
      const list=applyCandidateFilters(sourceList);renderedCandidateItems=list;
      $("#list-title").textContent=finalMode?"Final selection candidates":relationMode?(selectedConstellationContext.constellation?`Candidates linked to C${selectedConstellationContext.constellationIndex+1}`:`Candidates linked to keypoint S${selectedConstellationContext.sourceKeypointIndex+1}`):listTitle(event);
      $("#list-count").textContent=finalMode?`${selectedFinalKeys.size} in scene · ${list.length}/${sourceList.length} visible`:relationMode?`${list.length}/${sourceList.length} linked candidate(s)`:sourceList.length===list.length?`${list.length} item(s)`:`${list.length}/${sourceList.length} visible(s)`;
      $("#candidate-list").innerHTML=list.map((item,index)=>{
        const status=statusOf(item),sub=item.count!=null?`${item.count.toLocaleString("en-US")} models`:`${categoryOf(item)} · ${reasonLabels[status]||status}`,active=finalMode?selectedFinalKeys.has(item._finalKey):Boolean(selectedCandidate&&nameOf(selectedCandidate)===nameOf(item)),detail=detailedCandidate(item),hasPreview=Array.isArray(detail?.model_preview)&&detail.model_preview.length>0;
        const leading=finalMode?`<span class="candidate-check" aria-hidden="true"><i data-lucide="check"></i></span>`:`<span class="candidate-rank">${String(index+1).padStart(3,"0")}</span>`;
        const preview=hasPreview?`<span class="candidate-thumb"><canvas data-candidate-thumb="${index}" aria-label="Preview of ${escapeHtml(nameOf(item))}"></canvas></span>`:`<span class="candidate-thumb"><span class="candidate-thumb-fallback">${escapeHtml(categoryOf(item).slice(0,3))}</span></span>`;
        return `<button class="candidate ${active?"filtered-selected":""}" type="button" data-candidate="${index}" data-position="${index}" data-candidate-tooltip="${index}" aria-pressed="${active}">${leading}${preview}<span class="candidate-name"><strong>${escapeHtml(nameOf(item))}</strong><span>${escapeHtml(sub)}</span></span><span class="status ${escapeHtml(status)}">${escapeHtml(reasonLabels[status]||status)}</span></button>`;
      }).join("")||`<p class="empty-list">${sourceList.length?"No candidate matches the current filters.":"The list is empty at this point in the timeline."}</p>`;
      renderCandidateThumbnails(list);
      window.lucide?.createIcons();
      if(!finalMode)document.querySelectorAll("[data-candidate]").forEach(button=>button.addEventListener("click",()=>{
        const item=list[Number(button.dataset.candidate)];
        if(item.keypointIndex!=null){selectKeypoint(item.keypointIndex,event);return}
        selectedCandidate=item;
        const relation=selectedConstellationContext?candidateRelationForScanIndices(item,selectedConstellationContext.scanIndices):null;
        highlightRelation=relation?{type:relation.type,index:relation.index}:null;
        renderCandidateDetail(item);renderList(event,timelineList);updateScene(event);
      }));
    }
    function applyFinalCandidateSelection(position,interaction={}){
      if(listScope!=="final")return;const list=renderedCandidateItems,item=list[position];if(!item)return;
      if(interaction.range&&Number.isInteger(finalSelectionAnchor)){
        if(!interaction.additive)selectedFinalKeys.clear();
        const start=Math.min(finalSelectionAnchor,position),end=Math.max(finalSelectionAnchor,position);for(const value of list.slice(start,end+1))selectedFinalKeys.add(value._finalKey);
      }else if(interaction.additive||interaction.touch){
        if(interaction.touch)selectedFinalKeys.add(item._finalKey);else if(selectedFinalKeys.has(item._finalKey))selectedFinalKeys.delete(item._finalKey);else selectedFinalKeys.add(item._finalKey);
        finalSelectionAnchor=position;
      }else{selectedFinalKeys.clear();selectedFinalKeys.add(item._finalKey);finalSelectionAnchor=position}
      selectedCandidate=item;highlightRelation=null;const event=events[eventIndex];renderCandidateDetail(item);renderList(event,eventList(event));updateScene(event);
    }
    function setListScope(scope,{preserveCandidate=null}={}){
      listScope=scope==="final"?"final":"timeline";
      document.querySelectorAll("[data-list-scope]").forEach(button=>{const active=button.dataset.listScope===listScope;button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active))});
      $("#final-list-actions").hidden=listScope!=="final";
      if(listScope==="final"){selectedConstellationContext=null;selectedKeypointIndex=null;highlightRelation=null}
      if(listScope==="timeline")selectedCandidate=preserveCandidate||events[eventIndex].candidate||null;
      const event=events[eventIndex],list=eventList(event);renderList(event,list);renderCandidateDetail(selectedCandidate);updateScene(event);
    }
    function candidateTooltip(item){
      const detail=detailedCandidate(item)||{},registration=finalCandidatePose(item)||detail.registration||{},status=statusOf(item),translation=registration.translation||item?.translation,distance=item?.mean_surface_dist_m??registration.mean_surface_dist_m;
      const lines=[
        `${categoryOf(item)} · ${reasonLabels[status]||status}`,
        `Coverage ${pct(item?.coverage??registration.coverage)} · reverse ${pct(item?.cov_reverse??registration.cov_reverse)}`,
        `Score harmonique ${number(item?.balanced_score??registration.balanced_score,3)} · distance ${distance!=null?`${number(distance,3)} m`:"—"}`,
        `Rotation ${number(item?.theta_deg??registration.theta_deg,1)}° · scale ${number(item?.scale??registration.scale,3)}`,
        translation?.length===3?`Position ${translation.map(value=>number(value,2)).join(", ")}`:null,
        item?.checkpoint_frames!=null?`Checkpoint ${item.checkpoint_frames} frames`:null,
        item?._finalKey?`In scene: ${selectedFinalKeys.has(item._finalKey)?"displayed":"hidden"}`:null,
        item?.reason?`Recorded reason: ${reasonLabels[status]||item.reason}`:null,
      ].filter(Boolean);
      return{title:nameOf(item),detail:lines.join("\n")};
    }
    installSharedPrimitives();
    const viewerTooltips=createTooltipController({selector:"#viewer-tooltip",delay:140,targetSelector:"[data-candidate-tooltip],[data-tooltip],button[aria-label],a[aria-label]",resolveContent:target=>{const index=target.dataset.candidateTooltip;return index!=null?candidateTooltip(renderedCandidateItems[Number(index)]):null}});
    function relationContextMarkup(){
      const context=selectedConstellationContext;
      if(!context)return"";
      const keypoints=[...context.scanIndices].sort((a,b)=>a-b).map(index=>`S${index+1}`).join(", "),title=context.constellation?`Constellation C${context.constellationIndex+1} selected`:`Keypoint S${context.sourceKeypointIndex+1} selected`,copy=context.constellation?`${keypoints} · ${context.candidates.length} candidate(s) share at least one of these keypoints.`:`No recorded constellation contains this point · ${context.candidates.length} candidate(s) have a direct correspondence.`;
      return`<div class="relation-context"><div class="relation-context-copy"><strong>${escapeHtml(title)}</strong><span>${escapeHtml(copy)}</span></div><button class="icon-button compact" id="clear-relation-filter" type="button" aria-label="Clear constellation filter" data-tooltip="Show full list"><i data-lucide="x"></i></button></div>`;
    }
    function clearKeypointContext(){
      selectedKeypointIndex=null;selectedConstellationContext=null;highlightRelation=null;selectedCandidate=events[eventIndex].candidate||null;
      const event=events[eventIndex];renderList(event,eventList(event));renderCandidateDetail(selectedCandidate);updateScene(event);
    }
    function bindRelationContextClear(){$("#clear-relation-filter")?.addEventListener("click",clearKeypointContext)}
    function candidateDetailHeading(item){
      const eventCandidate=events[eventIndex]?.candidate,current=Boolean(item&&eventCandidate&&nameOf(item)===nameOf(eventCandidate));
      const context=selectedConstellationContext?"constellation active":current?"stage candidate":item?(listScope==="final"?"final selection":"tracked context"):"no context";
      return`<div class="candidate-detail-heading"><span>Candidate details</span><b>${escapeHtml(context)}</b></div>`;
    }
    function renderCandidateDetail(item){
      if(!item){const context=selectedConstellationContext;$("#candidate-detail").innerHTML=`${candidateDetailHeading(null)}${relationContextMarkup()}<h2>${context?`Keypoint scene #${context.sourceKeypointIndex+1}`:"Select a candidate"}</h2><p>${context?"No saved candidate shares this keypoint.":"Details show correspondences, constellations, scores and rejection reasons."}</p>`;bindRelationContextClear();updateCandidateTimelineContext(null);window.lucide?.createIcons();return;}
      const detail=detailedCandidate(item)||{},verification=detail.verification||{},registration=finalCandidatePose(item)||detail.registration||{},rejects=verification.rejection_counts||{},pills=[detail.correspondences!=null&&`${detail.correspondences} correspondences`,detail.constellations!=null&&`${detail.constellations} constellations`,item.checkpoint_frames!=null&&`checkpoint ${item.checkpoint_frames}`,...Object.entries(rejects).map(([key,value])=>`${value} rejection(s) ${reasonLabels[`rejected_${key}`]||key}`)].filter(Boolean),correspondences=detail.correspondence_preview||[],constellations=detail.constellation_preview||[];
      const relationButtons=`${correspondences.length?`<span class="relation-label">Keypoint correspondences</span>${correspondences.map((corr,index)=>`<button class="relation-button" type="button" data-relation="correspondence" data-relation-index="${index}" aria-pressed="${highlightRelation?.type==="correspondence"&&highlightRelation.index===index}">M${corr.model_keypoint+1} ↔ S${corr.scan_keypoint+1}</button>`).join("")}`:""}${constellations.length?`<span class="relation-label">Pose constellations</span>${constellations.map((cons,index)=>`<button class="relation-button" type="button" data-relation="constellation" data-relation-index="${index}" aria-pressed="${highlightRelation?.type==="constellation"&&highlightRelation.index===index}">C${index+1} · ${cons.inliers} inliers</button>`).join("")}`:""}`;
      const status=statusOf(item),translation=item.translation||registration.translation,distance=item.mean_surface_dist_m??registration.mean_surface_dist_m,sceneState=item._finalKey?` It is currently ${selectedFinalKeys.has(item._finalKey)?"overlaid on the grey RGB-D scan":"hidden"} in the scene.`:"";
      $("#candidate-detail").innerHTML=`${candidateDetailHeading(item)}${relationContextMarkup()}<h2>${escapeHtml(nameOf(item))}</h2><div class="detail-grid"><span>Decision<b>${escapeHtml(reasonLabels[status]||status)}</b></span><span>Category<b>${escapeHtml(categoryOf(item))}</b></span><span>Checkpoint<b>${item.checkpoint_frames??"—"}</b></span><span>Coverage<b>${pct(item.coverage??registration.coverage)}</b></span><span>Reverse<b>${pct(item.cov_reverse??registration.cov_reverse)}</b></span><span>Score H<b>${number(item.balanced_score??registration.balanced_score,3)}</b></span><span>Surface distance<b>${distance!=null?`${number(distance,3)} m`:"—"}</b></span><span>Rotation<b>${number(item.theta_deg??registration.theta_deg,1)}°</b></span><span>Scale<b>${number(item.scale??registration.scale,3)}</b></span></div><div id="candidate-timeline-navigation" aria-label="Stages linked to this candidate"></div>${translation?.length===3?`<div class="trace-pills"><span>position ${translation.map(value=>number(value,2)).join(", ")}</span>${pills.map(value=>`<span>${escapeHtml(value)}</span>`).join("")}</div>`:pills.length?`<div class="trace-pills">${pills.map(value=>`<span>${escapeHtml(value)}</span>`).join("")}</div>`:""}${relationButtons?`<div class="relation-buttons">${relationButtons}</div>`:""}<p>${item.reason?`Recorded reason: ${escapeHtml(reasonLabels[status]||item.reason)}.`:"Click a correspondence or constellation to highlight it."}${sceneState}</p>`;
      updateCandidateTimelineContext(item);
      bindRelationContextClear();
      document.querySelectorAll("[data-relation]").forEach(button=>button.addEventListener("click",()=>{highlightRelation={type:button.dataset.relation,index:Number(button.dataset.relationIndex)};renderCandidateDetail(item);updateScene(events[eventIndex])}));
    }
    function selectKeypoint(index,event){
      selectedKeypointIndex=index;selectedConstellationContext=buildKeypointConstellationContext(index,event);selectedCandidate=selectedConstellationContext.primaryCandidate;
      const relation=selectedCandidate?candidateRelationForScanIndices(selectedCandidate,selectedConstellationContext.scanIndices):null;
      highlightRelation=relation?{type:relation.type,index:relation.index}:null;
      renderList(event,eventList(event));renderCandidateDetail(selectedCandidate);updateScene(event);
    }
    function previousMajor(from=eventIndex){for(let i=from-1;i>=0;i--)if(events[i].major)return i;return 0}function nextMajor(from=eventIndex){for(let i=from+1;i<events.length;i++)if(events[i].major)return i;return events.length-1}
    function renderViewerTimeline(){
      const chapterHost=$("#timeline-chapters"),eventHost=$("#viewer-event-track");
      chapterHost.innerHTML=stages.map((name,stage)=>{
        const indices=events.map((event,index)=>event.stage===stage?index:-1).filter(index=>index>=0);
        return `<button type="button" class="viewer-chapter-segment" data-stage="${stage}" data-tooltip="${escapeHtml(name)}" data-tooltip-detail="${indices.length} stage(s) in this chapter." style="flex:${Math.max(1,indices.length)}">${escapeHtml(name)}</button>`;
      }).join("");
      eventHost.innerHTML=events.map((event,index)=>`<button type="button" class="viewer-event-cell ${event.major?"major":""}" data-event-index="${index}" data-tooltip="${escapeHtml(event.title)}" data-tooltip-detail="${escapeHtml(event.description)}" aria-label="${escapeHtml(event.title)}"></button>`).join("");
      chapterHost.querySelectorAll("[data-stage]").forEach(button=>button.addEventListener("click",()=>{const target=events.findIndex(event=>event.stage===Number(button.dataset.stage));if(target>=0){stopPlay();setEvent(target)}}));
      eventHost.querySelectorAll("[data-event-index]").forEach(button=>button.addEventListener("click",()=>{stopPlay();setEvent(Number(button.dataset.eventIndex))}));
    }
    function syncViewerTimelineState(){
      const event=events[eventIndex];
      $("#current-stage-title").textContent=stages[event.stage];
      $("#current-step-title").textContent=event.title;
      $("#timeline-chapters").querySelectorAll("[data-stage]").forEach(button=>{
        const stage=Number(button.dataset.stage),related=candidateEventGroups.some(group=>group.indices.some(index=>events[index].stage===stage));
        button.classList.toggle("active",stage===event.stage);
        button.classList.toggle("candidate-related",related);
      });
      applyRelatedTimelineState($("#viewer-event-track"),candidateRelatedEvents);
      $("#viewer-event-track").querySelectorAll("[data-event-index]").forEach(button=>button.classList.toggle("active",Number(button.dataset.eventIndex)===eventIndex));
    }
    function updateCandidateTimelineContext(item){
      candidateEventGroups=item?candidateTimelineGroups(events,item,{phaseOf:event=>stageIds[event.stage]}):[];
      candidateRelatedEvents=relatedTimelineIndices(candidateEventGroups);
      const container=$("#candidate-timeline-navigation");
      if(container)renderCandidateTimelineNavigation(container,candidateEventGroups,{currentIndex:eventIndex,onNavigate:index=>{stopPlay();setListScope("timeline",{preserveCandidate:item});setEvent(index,{preserveCandidate:true})}});
      syncViewerTimelineState();
      window.lucide?.createIcons();
    }
    function activateTimelineScope(){
      if(listScope!=="final")return;
      listScope="timeline";
      document.querySelectorAll("[data-list-scope]").forEach(button=>{const active=button.dataset.listScope==="timeline";button.classList.toggle("active",active);button.setAttribute("aria-selected",String(active))});
      $("#final-list-actions").hidden=true;
    }
    function setEvent(value,{preserveCandidate=false}={}){activateTimelineScope();eventIndex=Math.max(0,Math.min(events.length-1,Number(value)));const event=events[eventIndex],list=eventList(event);if(!preserveCandidate)selectedCandidate=event.candidate||null;selectedKeypointIndex=null;selectedConstellationContext=null;highlightRelation=null;$("#timeline").value=eventIndex;$("#timeline-output").textContent=`${eventIndex+1} / ${events.length}`;$("#prev").disabled=eventIndex===0;$("#next").disabled=eventIndex===events.length-1;$("#major-prev").disabled=eventIndex===0;$("#major-next").disabled=eventIndex===events.length-1;$("#event-eyebrow").textContent=`Stage ${event.stage+1} · ${stages[event.stage]}`;$("#event-title").textContent=event.title;$("#event-description").textContent=event.description;$("#event-caption").innerHTML=`<strong>${event.title}</strong> · ${event.description}`;$("#scene-badge").innerHTML=`<strong>${stages[event.stage]}</strong> · event ${eventIndex+1}/${events.length}${event.major?" · milestone":""}`;$("#metrics").innerHTML=metricsFor(event,list).map(([label,value])=>`<div class="metric"><small>${label}</small><strong>${value}</strong></div>`).join("");renderList(event,list);renderCandidateDetail(selectedCandidate);syncViewerTimelineState();updateScene(event);}
    function setPlayIcon(name,label){const button=$("#play");button.innerHTML=`<i data-lucide="${name}"></i>`;button.setAttribute("aria-label",label);button.title=label;window.lucide?.createIcons()}
    function togglePlay(){if(playTimer)stopPlay();else{if(eventIndex===events.length-1)setEvent(0);playTimer=setInterval(()=>{if(eventIndex>=events.length-1)stopPlay();else setEvent(eventIndex+1)},900);setPlayIcon("pause","Pause timeline")}}function jumpStage(delta){const targetStage=Math.max(0,Math.min(stages.length-1,events[eventIndex].stage+delta)),target=events.findIndex(event=>event.stage===targetStage);if(target>=0)setEvent(target)}
    $("#timeline").addEventListener("input",event=>{stopPlay();setEvent(event.target.value)});$("#prev").addEventListener("click",()=>{stopPlay();setEvent(eventIndex-1)});$("#next").addEventListener("click",()=>{stopPlay();setEvent(eventIndex+1)});$("#major-prev").addEventListener("click",()=>{stopPlay();setEvent(previousMajor())});$("#major-next").addEventListener("click",()=>{stopPlay();setEvent(nextMajor())});$("#play").addEventListener("click",togglePlay);function stopPlay(){if(playTimer){clearInterval(playTimer);playTimer=null}setPlayIcon("play","Play timeline")}
    document.querySelectorAll("[data-list-scope]").forEach(button=>button.addEventListener("click",()=>setListScope(button.dataset.listScope)));
    $("#clear-final-selection").addEventListener("click",()=>{selectedFinalKeys.clear();renderList(events[eventIndex],eventList(events[eventIndex]));updateScene(events[eventIndex])});
    $("#show-all-final").addEventListener("click",()=>{selectedFinalKeys.clear();finalCandidates.forEach(item=>selectedFinalKeys.add(item._finalKey));renderList(events[eventIndex],eventList(events[eventIndex]));updateScene(events[eventIndex])});
    $("#show-selected-final").addEventListener("click",()=>{selectedFinalKeys.clear();finalCandidates.filter(item=>["selected","kept"].includes(statusOf(item))).forEach(item=>selectedFinalKeys.add(item._finalKey));renderList(events[eventIndex],eventList(events[eventIndex]));updateScene(events[eventIndex])});
    $("#focus-final-selection").addEventListener("click",()=>focusFinalSelection());
    function rerenderCandidateList(){const event=events[eventIndex];renderList(event,eventList(event))}
    document.addEventListener("keydown",event=>{if(["INPUT","BUTTON","SELECT","TEXTAREA"].includes(document.activeElement?.tagName))return;let handled=true;if(event.key==="ArrowLeft"){stopPlay();setEvent(event.shiftKey?previousMajor():eventIndex-1)}else if(event.key==="ArrowRight"){stopPlay();setEvent(event.shiftKey?nextMajor():eventIndex+1)}else if(event.key==="ArrowUp"){stopPlay();jumpStage(-1)}else if(event.key==="ArrowDown"){stopPlay();jumpStage(1)}else if(event.key==="Home"){stopPlay();setEvent(0)}else if(event.key==="End"){stopPlay();setEvent(events.length-1)}else if(event.key===" "){togglePlay()}else handled=false;if(handled)event.preventDefault()});

    renderViewerTimeline();
    const container=$("#scene"),viewport=createSceneViewport({THREE,OrbitControls,host:container,gridSize:12,gridDivisions:24}),{renderer,camera,controls,root}=viewport,relationGroup=new THREE.Group(),constellationGroup=new THREE.Group(),finalPreviewGroup=new THREE.Group();root.add(constellationGroup);root.add(relationGroup);root.add(finalPreviewGroup);let scanPoints=null,scanBounds=null,scenePointSize=SCENE3D_THEME.pointSize.surface,modelGroup=new THREE.Group(),keypointPoints=null,candidatePoints=null,fusionPoints=null,cameraPath=null,candidatePreview=null,previousCandidatePreview=null,candidateTarget=null,allConstellationsVisible=false;root.add(modelGroup);
    function subsetGeometry(source,indices){const pos=source.getAttribute("position"),col=source.getAttribute("color"),p=new Float32Array(indices.length*3),c=new Float32Array(indices.length*3);indices.forEach((index,out)=>{for(let axis=0;axis<3;axis++){p[out*3+axis]=pos.array[index*3+axis];c[out*3+axis]=col.array[index*3+axis]}});const geometry=new THREE.BufferGeometry();geometry.setAttribute("position",new THREE.BufferAttribute(p,3));geometry.setAttribute("color",new THREE.BufferAttribute(c,3));return geometry}
    function pointsFromPositions(items,color,size,opacity=.95){return createPointCloud({THREE,positions:items,color,size,opacity})}
    function disposeObject(object){disposeObject3D(object)}
    function buildFusionCloud(){if(!fusionFrames.length)return;const positions=[],colors=[],cameraPositions=[];fusionFrames.forEach(frame=>{(frame.points||[]).forEach((point,index)=>{positions.push(...point);const color=frame.colors?.[index]||[119,184,214];colors.push(color[0]/255,color[1]/255,color[2]/255)});if(frame.camera_position?.length===3)cameraPositions.push(...frame.camera_position);frame._pointEnd=positions.length/3;frame._cameraEnd=cameraPositions.length/3});fusionPoints=createPointCloud({THREE,positions,colors,color:0xffffff,size:SCENE3D_THEME.pointSize.fusion,opacity:.92});fusionPoints.geometry.setDrawRange(0,0);root.add(fusionPoints);if(cameraPositions.length>=6){cameraPath=createPolyline({THREE,positions:cameraPositions,color:SCENE3D_THEME.fusion,opacity:.8});cameraPath.geometry.setDrawRange(0,0);root.add(cameraPath)}}
    function validBounds(box){return validSceneBounds(box)}
    function fitCameraToBox(box,padding=1.25,{remember=false}={}){
      viewport.fitBox(box,padding,{remember});
    }
    function loadCloud(){
      try{
        const raw=atob(payload.ply_base64),bytes=new Uint8Array(raw.length);
        for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
        const geometry=new PLYLoader().parse(bytes.buffer),colors=geometry.getAttribute("color"),groups=new Map(),scanPointCount=Math.min(Math.max(Number(payload.scan_point_count)||0,0),colors.count);
        for(let i=scanPointCount;i<colors.count;i++){
          const key=`${Math.round(colors.getX(i)*255)},${Math.round(colors.getY(i)*255)},${Math.round(colors.getZ(i)*255)}`;
          (groups.get(key)||groups.set(key,[]).get(key)).push(i);
        }
        const fallbackScanKey=scanPointCount?null:[...groups.entries()].sort((left,right)=>right[1].length-left[1].length)[0]?.[0],scanIndices=scanPointCount?Array.from({length:scanPointCount},(_,index)=>index):(groups.get(fallbackScanKey)||[]),modelKeys=[...groups.keys()].filter(key=>key!==fallbackScanKey),scanGeometry=subsetGeometry(geometry,scanIndices);
        geometry.computeBoundingBox();scanGeometry.computeBoundingBox();
        scanBounds=validBounds(scanGeometry.boundingBox)?scanGeometry.boundingBox.clone():geometry.boundingBox.clone();
        const center=scanBounds.getCenter(new THREE.Vector3()),size=scanBounds.getSize(new THREE.Vector3());
        scenePointSize=SCENE3D_THEME.pointSize.surface;
        scanPoints=new THREE.Points(scanGeometry,new THREE.PointsMaterial({color:SCENE3D_THEME.surface,size:scenePointSize,sizeAttenuation:true,transparent:true,opacity:.62,depthWrite:false}));
        root.add(scanPoints);
        modelKeys.forEach(key=>{
          const points=new THREE.Points(subsetGeometry(geometry,groups.get(key)),new THREE.PointsMaterial({vertexColors:true,size:SCENE3D_THEME.pointSize.candidate,sizeAttenuation:true,transparent:true,opacity:1,depthWrite:false}));
          modelGroup.add(points);
        });
        buildFusionCloud();fitCameraToBox(scanBounds,1.12,{remember:true});
        viewport.replaceGrid({size:Math.max(size.x,size.z)*1.2,divisions:20,center:[center.x,0,center.z],y:scanBounds.min.y});
        const markerData=(progress.final_selection_trace||progress.final_selection||[]).filter(item=>Array.isArray(item.translation));
        if(markerData.length){
          const geometryMarkers=new THREE.BufferGeometry();
          geometryMarkers.setAttribute("position",new THREE.Float32BufferAttribute(markerData.flatMap(item=>[item.translation[0],item.translation[1]+.45,item.translation[2]]),3));
          geometryMarkers.setAttribute("color",new THREE.Float32BufferAttribute(markerData.flatMap(item=>String(item.status).startsWith("rejected")?[.75,.25,.29]:[.14,.52,.36]),3));
          candidatePoints=new THREE.Points(geometryMarkers,new THREE.PointsMaterial({vertexColors:true,size:.09,sizeAttenuation:true,transparent:true,opacity:.9,depthWrite:false}));root.add(candidatePoints);
        }
        $("#scene-message").hidden=true;updateScene(events[eventIndex]);
      }catch(error){
        $("#scene-message").textContent=`Cannot load 3D scene: ${error.message}`;
      }
    }
    function updateFusion(event){if(!fusionPoints)return;const count=event.stage===1?(event.visibleCount||0):fusionFrames.length,frame=count?fusionFrames[Math.min(count,fusionFrames.length)-1]:null;fusionPoints.geometry.setDrawRange(0,frame?frame._pointEnd:0);fusionPoints.visible=event.stage===1;if(cameraPath){cameraPath.geometry.setDrawRange(0,frame?frame._cameraEnd:0);cameraPath.visible=event.stage===1}}
    function constellationPositions(constellation,keypoints){
      const positions=[],seen=new Set();
      (constellation?.scan_keypoints||[]).forEach(index=>{const position=keypoints[Number(index)]?.position;if(!Array.isArray(position)||position.length!==3)return;const key=position.map(value=>Number(value).toFixed(4)).join(":");if(seen.has(key))return;seen.add(key);positions.push(position.map(Number))});
      (constellation?.inlier_pairs||[]).forEach(pair=>{const index=Number(pair.scan_keypoint),position=pair.scan_position||keypoints[index]?.position;if(!Array.isArray(position)||position.length!==3)return;const key=position.map(value=>Number(value).toFixed(4)).join(":");if(seen.has(key))return;seen.add(key);positions.push(position.map(Number))});
      return positions;
    }
    function syncAllConstellations(event){
      clearObjectGroup(constellationGroup);
      const button=$("#toggle-all-constellations"),countHost=$("#constellation-layer-count");
      button.setAttribute("aria-pressed",String(allConstellationsVisible));button.classList.toggle("active",allConstellationsVisible);
      if(!allConstellationsVisible){countHost.textContent="";return}
      const trace=currentTrace(event)||(traceSteps.at(-1)?.full),keypoints=trace?.scan?.keypoints||[];
      let count=0;
      (trace?.matching||[]).forEach(candidate=>(candidate.constellation_scene_support||candidate.constellation_preview||[]).forEach(constellation=>{
        const points=constellationPositions(constellation,keypoints);if(points.length<2)return;
        const positions=[];for(let index=0;index<points.length-1;index++)positions.push(...points[index],...points[index+1]);if(points.length>2)positions.push(...points.at(-1),...points[0]);
        const color=0x9d92f2,line=createPolyline({THREE,positions,color,opacity:.72,segments:true}),markers=pointsFromPositions(points,color,.10,.96);line.renderOrder=18;markers.renderOrder=19;line.material.depthTest=false;line.material.depthWrite=false;markers.material.depthTest=false;markers.material.depthWrite=false;constellationGroup.add(line);constellationGroup.add(markers);count++;
      }));
      countHost.textContent=count?String(count):"0";
    }
    function relationPairs(event,detail){if(highlightRelation?.type==="correspondence")return[detail.correspondence_preview?.[highlightRelation.index]].filter(Boolean);if(highlightRelation?.type==="constellation")return detail.constellation_preview?.[highlightRelation.index]?.inlier_pairs||[];if(selectedKeypointIndex!=null)return[];if(event.kind==="correspondences")return detail.correspondence_preview||[];if(event.kind==="constellation")return event.pose?.inlier_pairs||[];return[]}
    function highlightedScanIndices(event,detail){const out=new Set(selectedConstellationContext?.scanIndices||[]);if(selectedKeypointIndex!=null)out.add(selectedKeypointIndex);relationPairs(event,detail).forEach(pair=>out.add(pair.scan_keypoint));return out}
    function updateKeypoints(event){disposeObject(keypointPoints);keypointPoints=null;const source=event.step?.full?.scan?.keypoints||((event.stage>=2&&traceSteps.length)?traceSteps.at(-1)?.full?.scan?.keypoints:[])||[];let count=source.length;if(event.kind==="keypoints")count=Math.min(count,event.visibleCount||0);if(!count)return;const contextualCandidate=selectedConstellationContext?selectedCandidate:null,detail=detailedCandidate(contextualCandidate||event.candidate||selectedCandidate||{}),highlighted=highlightedScanIndices(event,detail),positions=[],colors=[];source.slice(0,count).forEach((keypoint,index)=>{positions.push(...keypoint.position);const color=highlighted.has(index)?[.95,.27,.32]:[.96,.83,.37];colors.push(...color)});keypointPoints=createPointCloud({THREE,positions,colors,color:SCENE3D_THEME.keypointRetained,size:SCENE3D_THEME.pointSize.keypointRetained,opacity:.98});keypointPoints.userData.source=source.slice(0,count);root.add(keypointPoints)}
    function poseValue(pose){if(!pose)return null;return{theta:Number(pose.theta_deg||0)*Math.PI/180,scale:Number(pose.scale||1),translation:(pose.translation||[0,0,0]).map(Number)}}
    function setObjectPose(object,pose){const value=poseValue(pose);if(!object||!value)return;object.position.set(...value.translation);object.rotation.set(0,value.theta,0);object.scale.setScalar(value.scale)}
    function transformPosition(position,pose){const value=poseValue(pose);if(!value)return position;const [x,y,z]=position,c=Math.cos(value.theta),s=Math.sin(value.theta),scale=value.scale;return[scale*(c*x+s*z)+value.translation[0],scale*y+value.translation[1],scale*(-s*x+c*z)+value.translation[2]]}
    function previewObject(points,color,opacity){const object=pointsFromPositions(points,color,SCENE3D_THEME.pointSize.candidate,opacity);object.material.depthWrite=false;root.add(object);return object}
    function clearCandidatePreview(){disposeObject(candidatePreview);disposeObject(previousCandidatePreview);candidatePreview=null;previousCandidatePreview=null;candidateTarget=null;while(relationGroup.children.length)disposeObject(relationGroup.children[0])}
    function finalCandidatePose(item){
      if(Array.isArray(item?.translation))return item;
      if(Array.isArray(item?.registration?.translation))return item.registration;
      const registration=(item?.registrations||[]).find(candidate=>Array.isArray(candidate?.translation));
      if(registration)return registration;
      const detail=detailedCandidate(item);
      if(Array.isArray(detail?.registration?.translation))return detail.registration;
      return (detail?.registrations||[]).find(candidate=>Array.isArray(candidate?.translation))||null;
    }
    function finalModelKeypoints(detail){
      const recorded=detail?.model_keypoint_preview;
      const positions=Array.isArray(recorded)?recorded:[...(detail?.correspondence_preview||[]),...(detail?.constellation_preview||[]).flatMap(value=>value.inlier_pairs||[])].map(pair=>pair.model_position);
      const unique=new Map();
      positions.forEach(point=>{if(Array.isArray(point)&&point.length===3&&point.every(Number.isFinite))unique.set(point.join(":"),point)});
      return [...unique.values()];
    }
    function syncFinalCandidatePreviews(){
      clearObjectGroup(finalPreviewGroup);
      if(listScope!=="final"){container.dataset.finalPreviewCount="0";return}
      finalCandidates.forEach((item,index)=>{
        if(!selectedFinalKeys.has(item._finalKey))return;
        const detail=detailedCandidate(item),points=detail?.model_preview||[],pose=finalCandidatePose(item);
        if(!points.length||!pose)return;
        const object=pointsFromPositions(points,candidateColor(item,index),SCENE3D_THEME.pointSize.candidate,1);
        object.renderOrder=10;
        object.userData.finalKey=item._finalKey;object.userData.candidate=item;
        const keypoints=finalModelKeypoints(detail);
        if(keypoints.length){
          const markers=pointsFromPositions(keypoints,SCENE3D_THEME.keypointRetained,SCENE3D_THEME.pointSize.keypointRetained,1);
          markers.material.sizeAttenuation=false;markers.material.size=SCENE3D_THEME.pointSize.keypointOverlayPixels;
          markers.material.depthTest=false;markers.material.depthWrite=false;markers.renderOrder=20;
          markers.userData.layer="final-model-keypoints";
          object.add(markers);
        }
        finalPreviewGroup.add(object);
        setObjectPose(object,pose);
      });
      container.dataset.finalPreviewCount=String(finalPreviewGroup.children.length);
      $("#focus-final-selection").disabled=finalPreviewGroup.children.length===0;
      $("#scene-badge").innerHTML=`<strong>Final candidates</strong> · ${finalPreviewGroup.children.length}/${finalCandidates.length} displayed on RGB-D scan · multiple selection`;
    }
    function focusFinalSelection(){
      if(!finalPreviewGroup.children.length){fitCameraToBox(scanBounds,1.12);return}
      finalPreviewGroup.updateMatrixWorld(true);
      const box=new THREE.Box3().setFromObject(finalPreviewGroup),size=box.getSize(new THREE.Vector3()),margin=Math.max(size.length()*.55,.65);
      box.expandByScalar(margin);fitCameraToBox(box,1.2);
    }
    function showRelations(event,detail,pose){while(relationGroup.children.length)disposeObject(relationGroup.children[0]);const pairs=relationPairs(event,detail);if(!pairs.length||!pose)return;const segments=[],targets=[];pairs.forEach(pair=>{if(!pair?.model_position||!pair?.scan_position)return;segments.push(...transformPosition(pair.model_position,pose),...pair.scan_position);targets.push(pair.scan_position)});if(segments.length)relationGroup.add(createPolyline({THREE,positions:segments,color:SCENE3D_THEME.rejected,opacity:.95,segments:true}));if(targets.length)relationGroup.add(pointsFromPositions(targets,SCENE3D_THEME.rejected,.085,1))}
    function showCandidate(event){clearCandidatePreview();let item=selectedConstellationContext?selectedCandidate:(event.candidate||selectedCandidate);if(!item)return;const detail=detailedCandidate(item),points=detail.model_preview||[];if(!points.length)return;let pose=(selectedConstellationContext&&highlightRelation?.type==="constellation"?detail.constellation_preview?.[highlightRelation.index]:null)||event.pose||item.registration||(item.registrations||[])[0]||item,previous=selectedConstellationContext?null:event.previousPose;if(!pose?.translation){pose=detail.constellation_preview?.[0]||detail.correspondence_preview?.[0]}if(!pose?.translation)return;if(previous?.translation){previousCandidatePreview=previewObject(points,0x77b8d6,.28);setObjectPose(previousCandidatePreview,previous)}const rejected=String(statusOf(item)).startsWith("rejected")||String(event.pose?.status||"").startsWith("rejected");candidatePreview=previewObject(points,rejected?0xe45b64:0xe6a03c,.95);setObjectPose(candidatePreview,previous?.translation?previous:pose);const target=poseValue(pose);candidateTarget=target?{position:new THREE.Vector3(...target.translation),theta:target.theta,scale:target.scale}:null;showRelations(event,detail,pose)}
    function updateScene(event){
      if(!scanPoints)return;
      if(listScope==="final"){
        updateFusion({stage:-1});
        scanPoints.visible=true;modelGroup.visible=false;finalPreviewGroup.visible=true;
        container.dataset.sceneReference="scan-rgbd";
        if(candidatePoints)candidatePoints.visible=false;
        disposeObject(keypointPoints);keypointPoints=null;clearCandidatePreview();syncFinalCandidatePreviews();syncAllConstellations(event);
        return;
      }
      delete container.dataset.sceneReference;$("#focus-final-selection").disabled=true;finalPreviewGroup.visible=false;updateFusion(event);scanPoints.visible=event.stage>=2||(event.stage===1&&!fusionFrames.length);modelGroup.visible=event.kind==="final";if(candidatePoints)candidatePoints.visible=event.kind==="selection"||event.kind==="verification";updateKeypoints(event);if(keypointPoints)keypointPoints.visible=event.stage===2||event.stage===3;showCandidate(event);syncAllConstellations(event)
    }
    $("#toggle-all-constellations").addEventListener("click",()=>{allConstellationsVisible=!allConstellationsVisible;syncAllConstellations(events[eventIndex])});
    const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();raycaster.params.Points.threshold=.09;renderer.domElement.addEventListener("click",event=>{if(!keypointPoints?.visible)return;const rect=renderer.domElement.getBoundingClientRect();mouse.x=((event.clientX-rect.left)/rect.width)*2-1;mouse.y=-((event.clientY-rect.top)/rect.height)*2+1;raycaster.setFromCamera(mouse,camera);const hit=raycaster.intersectObject(keypointPoints,false)[0];if(hit){selectKeypoint(hit.index,events[eventIndex]);updateKeypoints(events[eventIndex])}});
    $("#reset-camera").addEventListener("click",()=>viewport.resetCamera());
    viewport.addFrameListener(()=>{if(candidatePreview&&candidateTarget){candidatePreview.position.lerp(candidateTarget.position,.14);const delta=Math.atan2(Math.sin(candidateTarget.theta-candidatePreview.rotation.y),Math.cos(candidateTarget.theta-candidatePreview.rotation.y));candidatePreview.rotation.y+=delta*.14;const scale=candidatePreview.scale.x+(candidateTarget.scale-candidatePreview.scale.x)*.14;candidatePreview.scale.setScalar(scale)}});
    bindMultiSelectListInteractions({container:"#candidate-list",itemSelector:".candidate",getPosition:item=>Number(item.dataset.position),onActivate:applyFinalCandidateSelection});
    viewerTooltips.bind();bindViewerResizers();bindHoverPreviewPanels({app:"#app",panels:[{element:"#viewer-inspector",collapsedClass:"inspector-collapsed",peekClass:"inspector-peeking"}]});
    $("#toggle-viewer-inspector").addEventListener("click",()=>{$("#app").classList.toggle("inspector-collapsed");$("#app").classList.remove("inspector-peeking");updateViewerPanelToggle()});updateViewerPanelToggle();window.lucide?.createIcons();loadCloud();setEvent(0);
  </script>
</body>
</html>'''

import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {
  $,
  $$,
  bindHoverPreviewPanels,
  bindMultiSelectListInteractions,
  bindResizableDimension,
  createCandidateFilterController,
  createNotifier,
  createTooltipController,
  installSharedPrimitives,
  refreshIcons as refreshInterfaceIcons,
} from "./ui_shared.js";
import {
  SCENE3D_THEME,
  createPointCloud,
  createPolyline,
  createSceneViewport,
  disposeObject3D,
  validSceneBounds,
} from "./ui_scene3d.js";

const retrievalStages=["query","matching","verification","selection"];
const debugClientVersion="20260910-selection-view-2";
document.documentElement.dataset.debugClientVersion=debugClientVersion;
const state={
  defaults:null,current:null,baseline:null,step:0,eventId:0,viewers:new Map(),
  focusedScene:null,status:"idle",pendingAuto:false,autoTimer:null,syncing:false,
  renameRunId:null,stage:"keypoints",stageSteps:[],stageResults:new Map(),
  sourcePaths:new Map(),sourceEnabled:new Map(),sourceLineage:new Map(),
  upstreamRunId:null,logs:[],logRunId:null,logsCleared:false,
  runUntilStage:null,runUntilActive:false,runUntilSceneIds:null,
  querySelections:new Map(),queryDetails:new Map(),queryLoading:new Set(),
  queryActiveCandidate:null,querySelectionAnchors:new Map(),queryQueues:new Map(),
  queryGroupByScene:new Map(),
  featuredScene:null,sceneWeights:new Map(),
  sceneDisplaySettings:new Map(),
};
let queryFilterController=null;
const roleColors={surface:SCENE3D_THEME.surface,fusion:SCENE3D_THEME.fusion,unknown:0x737b82,free:0x4f89bd,occupied:0xef9b58,normal:0x58d3b2,candidate:SCENE3D_THEME.candidate,wall:0xc58cff,accepted:SCENE3D_THEME.accepted,rejected:SCENE3D_THEME.rejected,retained:SCENE3D_THEME.selected,vector:0x75a8e8,baseline:0xa798ef};
const queryCandidateColors=[0x69d2e7,0xf6b44b,0x8fd694,0xb48cf2,0xed7d76,0x7aa7ff];
const icons={fusion:"layers-3",keypoints:"crosshair",descriptors:"binary",query:"list-filter",matching:"waypoints",verification:"badge-check",selection:"check-check"};
const layerVisibilityStorage="objectsensing-debug-layer-visibility";
const stageStorage="objectsensing-debug-stage";
const sceneWeightsStorage="objectsensing-debug-scene-weights";
const sceneDisplayStorage="objectsensing-debug-scene-display-v1";
const defaultSceneDisplay=Object.freeze({colorMode:"auto",surfaceColor:"#91989d",backgroundColor:"#101315",pointScale:1,opacity:1,grid:true});
const layerRoles=["surface","fusion","unknown","free","occupied","normal","candidate","wall","accepted","retained","rejected","baseline","vector"];
const layoutSizes={
  left:{property:"--panel-left",storage:"objectsensing-debug-panel-left",defaultSize:286,min:220,max:520,collapsedSize:58,handle:"#left-panel-resizer"},
  right:{property:"--panel-right",storage:"objectsensing-debug-panel-right",defaultSize:304,min:250,max:560,collapsedSize:52,handle:"#right-panel-resizer"},
};

function activeStepDefinition(){return state.stageSteps[state.step]||null}
function isFusionFrameStep(step){return /^frame-\d+$/.test(String(step?.id||""))}
function resultStepForDefinition(result,definition=activeStepDefinition()){
  if(!result?.steps?.length)return null;
  if(definition?.sourceSelector==="last-fusion-frame")return[...result.steps].reverse().find(isFusionFrameStep)||null;
  const stepId=definition?.sourceStepId||definition?.id;
  return result.steps.find(step=>step.id===stepId)||result.steps[state.step]||result.steps.at(-1);
}
function resultStepIndex(result,definition=activeStepDefinition()){const step=resultStepForDefinition(result,definition);return step?result.steps.indexOf(step):-1}
function stepMetricLabels(step,stage=state.stage){
  if(step?.kept_label||step?.rejected_label)return{kept:step.kept_label,rejected:step.rejected_label};
  if(stage==="descriptors"&&step?.id==="distance")return{kept:"Correspondences",rejected:"No correspondence"};
  if(stage==="descriptors"&&step?.id==="ambiguity")return{kept:"Informative",rejected:"Others"};
  return{kept:null,rejected:null};
}
function renderVisibleLayers(){const definition=activeStepDefinition();state.viewers.forEach(viewer=>viewer.renderStep(definition))}
function layerToggle(role){return $(`[data-layer-role="${role}"]`)}
function layerVisible(role){return layerToggle(role)?.checked!==false}
function isFinalSelection(result,step){return result?.stage==="selection"&&step?.id==="final"}
function finalSelectionLayerVisible(layer,mode,showKeypoints){
  if(layer.id==="surface"||layer.role==="surface")return mode!=="models";
  if(layer.display_role==="model-keypoints")return mode!=="scene"&&showKeypoints;
  return mode!=="scene"&&(layer.display_role==="model"||/^pose-\d+$/.test(layer.id));
}
function savedLayerVisibility(){try{return JSON.parse(localStorage.getItem(layerVisibilityStorage)||"{}")}catch{return{}}}
function persistLayerVisibility(){const values={};for(const role of layerRoles)values[role]=layerVisible(role);localStorage.setItem(layerVisibilityStorage,JSON.stringify(values))}
function syncLegacyLayerToggle(role,visible){if(role==="rejected")$("#show-rejected").checked=visible;if(role==="baseline")$("#show-baseline").checked=visible}
function setLayerVisibility(role,visible,persist=true){const input=layerToggle(role);if(input)input.checked=visible;syncLegacyLayerToggle(role,visible);if(persist)persistLayerVisibility();renderVisibleLayers()}
function bindLayerVisibility(){
  const saved=savedLayerVisibility();
  for(const input of $$("[data-layer-role]")){const role=input.dataset.layerRole;if(typeof saved[role]==="boolean")input.checked=saved[role];syncLegacyLayerToggle(role,input.checked);input.addEventListener("change",()=>setLayerVisibility(role,input.checked))}
  $("#show-rejected").addEventListener("change",event=>setLayerVisibility("rejected",event.target.checked));
  $("#show-baseline").addEventListener("change",event=>setLayerVisibility("baseline",event.target.checked));
  $("#curvature-heatmap").addEventListener("change",renderVisibleLayers);
}

function cssPixels(property,fallback){const value=Number.parseFloat(getComputedStyle(document.documentElement).getPropertyValue(property));return Number.isFinite(value)?value:fallback}
function panelMaximum(side){const config=layoutSizes[side],otherSide=side==="left"?"right":"left",other=layoutSizes[otherSide],workspace=$(".workspace")?.clientWidth||innerWidth,otherCollapsed=$("#debug-app")?.classList.contains(`${otherSide}-collapsed`),otherSize=otherCollapsed?other.collapsedSize:cssPixels(other.property,other.defaultSize);return Math.max(config.min,Math.min(config.max,workspace-otherSize-420))}
const layoutResizers={};
function setPanelSize(side,size,persist=true){return layoutResizers[side]?.set(size,persist)}
function bindPanelResizer(side){const config=layoutSizes[side];layoutResizers[side]=bindResizableDimension({handle:config.handle,container:".workspace",property:config.property,storageKey:config.storage,defaultValue:config.defaultSize,min:config.min,getMax:()=>panelMaximum(side),axis:"x",edge:side==="left"?"start":"end",disabledBelow:1000,stateTarget:$("#debug-app"),stateClass:"resizing-panels"})}
function timelineMaximum(){const height=$(".visual-workspace")?.clientHeight||innerHeight;return Math.max(190,Math.min(520,height-190))}
function setTimelineSize(size,persist=true){return layoutResizers.timeline?.set(size,persist)}
function bindTimelineResizer(){layoutResizers.timeline=bindResizableDimension({handle:"#debug-timeline-resizer",container:".visual-workspace",property:"--timeline",storageKey:"objectsensing-debug-timeline",defaultValue:214,min:190,getMax:timelineMaximum,axis:"y",edge:"bottom",stateTarget:$("#debug-app"),stateClass:"resizing-timeline"})}
function bindLayoutResizers(){bindPanelResizer("left");bindPanelResizer("right");bindTimelineResizer();window.addEventListener("resize",()=>Object.values(layoutResizers).forEach(resizer=>resizer?.refresh()))}

installSharedPrimitives();
const tooltipController=createTooltipController({selector:"#tooltip"});
function hideTooltip(target=null){tooltipController.hide(target)}
function bindTooltips(){tooltipController.bind()}
function updatePanelToggles(){const app=$("#debug-app"),leftClosed=app.classList.contains("left-collapsed"),rightClosed=app.classList.contains("right-collapsed"),left=$("#toggle-left"),right=$("#toggle-right");left.setAttribute("aria-label",leftClosed?"Open controls":"Collapse controls");left.setAttribute("aria-expanded",String(!leftClosed));left.dataset.tooltip=leftClosed?"Open controls":"Collapse controls";right.setAttribute("aria-label",rightClosed?"Open inspector":"Collapse inspector");right.setAttribute("aria-expanded",String(!rightClosed));right.dataset.tooltip=rightClosed?"Open inspector":"Collapse inspector"}
function bindHoverPanels(){bindHoverPreviewPanels({app:"#debug-app",panels:[
  {element:"#control-panel",collapsedClass:"left-collapsed",peekClass:"left-peeking"},
  {element:"#inspector-panel",collapsedClass:"right-collapsed",peekClass:"right-peeking"},
]})}

function replaceNonFiniteJsonNumbers(text){
  let output="",inString=false,escaped=false,replaced=false;
  for(let index=0;index<text.length;){
    const character=text[index];
    if(inString){output+=character;if(escaped)escaped=false;else if(character==="\\")escaped=true;else if(character==='"')inString=false;index++;continue}
    if(character==='"'){inString=true;output+=character;index++;continue}
    const tokens=["-Infinity","Infinity","NaN"],token=tokens.find(candidate=>text.startsWith(candidate,index));
    if(token){const before=text[index-1]||"",after=text[index+token.length]||"",boundaryBefore=!/[A-Za-z0-9_.]/.test(before),boundaryAfter=!/[A-Za-z0-9_.]/.test(after);if(boundaryBefore&&boundaryAfter){output+="null";index+=token.length;replaced=true;continue}}
    output+=character;index++;
  }
  return replaced?output:text;
}
function parseApiPayload(text,path){
  if(!text)return{};
  try{return JSON.parse(text)}catch(firstError){
    const compatible=replaceNonFiniteJsonNumbers(text);
    if(compatible!==text){try{return JSON.parse(compatible)}catch{}}
    throw new Error(`Invalid JSON response for ${path}: ${firstError.message}`);
  }
}
async function api(path,options={}){
  const response=await fetch(path,{headers:{"Content-Type":"application/json"},...options});
  const payload=parseApiPayload(await response.text(),path);
  if(!response.ok)throw new Error(payload.error||`${response.status} ${response.statusText}`);
  return payload;
}
function formatInt(value){return Number(value||0).toLocaleString("en-US")}
function formatTime(seconds){if(seconds==null)return"—";if(seconds<60)return`${seconds.toFixed(1)} s`;return`${Math.floor(seconds/60)} min ${Math.round(seconds%60)} s`}
const notify=createNotifier({selector:"#toast",duration:3200});
function refreshIcons(){refreshInterfaceIcons()}
function reportLoadError(context,error){console.error(`[Debug Visualizer] ${context}`,error);notify(`${context} : ${error?.message||error}`,true)}
function defaultSources(){return Array.isArray(state.defaults?.sources)?state.defaults.sources:[]}
function normalizeDebugResult(payload){
  const value=payload&&typeof payload==="object"?payload:{},scenes=value.scenes&&typeof value.scenes==="object"?value.scenes:{},manifest={...(value.manifest||{})};
  manifest.stage=manifest.stage||Object.values(scenes).find(Boolean)?.stage||state.stage||"keypoints";
  manifest.sources=Array.isArray(manifest.sources)?manifest.sources:defaultSources();
  return{...value,manifest,scenes,logs:Array.isArray(value.logs)?value.logs:[]};
}

function sourcePathKey(stage,sourceId){return`${stage}:${sourceId}`}
function sourcePathFor(source,stage=state.stage){const remembered=state.sourcePaths.get(sourcePathKey(stage,source.id));if(remembered!=null)return remembered;const staged=source.stage_paths?.[stage];if(staged!=null)return staged;return ["descriptors",...retrievalStages].includes(stage)?"":(source.path||"")}
function rememberSourcePaths(){
  $$(".source-item").forEach(row=>{const key=sourcePathKey(state.stage,row.dataset.sourceId),input=$("input[type=text]",row),enabled=$("input[type=checkbox]",row);if(input)state.sourcePaths.set(key,input.value);if(enabled)state.sourceEnabled.set(key,enabled.checked)});
}
function stageDefinition(stage=state.stage){return(state.defaults?.pipeline_stages||[]).find(item=>item.id===stage)}
function stageLabel(stage=state.stage){return stageDefinition(stage)?.label||stage}
function stageFallbackSteps(stage=state.stage){return state.defaults?.stage_steps_by_stage?.[stage]||[]}
function stageResultSteps(){
  const result=Object.values(state.current?.scenes||{}).find(item=>item?.stage===state.stage||state.current?.manifest?.stage===state.stage);
  const rawSteps=result?.steps?.length?result.steps:stageFallbackSteps();
  const fallbackById=new Map(stageFallbackSteps().map(step=>[step.id,step]));
  const steps=rawSteps.map(step=>{const fallback=fallbackById.get(step?.id)||{};return{...fallback,...step,label:step?.label||fallback.label||step?.id||"Stage",description:step?.description||step?.explanation||fallback.description||"No explanation recorded.",assessment:step?.assessment||fallback.assessment||"",group:step?.group||fallback.group||stageLabel()}});
  if(state.stage!=="fusion")return steps;
  const frameSteps=steps.filter(isFusionFrameStep);
  if(!frameSteps.length)return steps;
  const lastFrame=frameSteps.at(-1),grouped=[],frameCount=frameSteps.length;
  for(const step of steps){
    if(!isFusionFrameStep(step)){grouped.push(step);continue}
    if(grouped.some(item=>item.id==="frames"))continue;
    grouped.push({
      ...lastFrame,id:"frames",label:"RGB-D integration",
      description:`Cumulative projection and integration of ${frameCount} RGB-D frame${frameCount>1?"s":""}. The display shows complete fusion after the last frame.`,
      sourceSelector:"last-fusion-frame",frameCount,
    });
  }
  return grouped;
}
function renderStageNavigation(){
  const track=$("#debug-stage-track");track.replaceChildren();
  for(const stage of state.defaults.pipeline_stages||[]){
    const button=document.createElement("button");button.type="button";button.className=`debug-stage-button ${stage.id===state.stage?"active":""}`;button.disabled=!stage.available;button.dataset.stage=stage.id;button.dataset.tooltip=stage.available?`Debug ${stage.label}`:`${stage.label} is not instrumented yet`;button.dataset.tooltipDetail=stage.available?"Load sources, parameters and results for this stage.":"This stage remains visible to show its position in the pipeline.";
    button.innerHTML=`<i data-lucide="${icons[stage.id]||"circle"}"></i><span>${stage.label}</span>`;
    if(stage.available)button.addEventListener("click",()=>selectStage(stage.id));
    track.append(button);
  }
  refreshIcons();
}
function updateStageUi(){
  const fusion=state.stage==="fusion",keypoints=state.stage==="keypoints",descriptors=state.stage==="descriptors",query=retrievalStages.includes(state.stage);
  $("#debug-app").dataset.stage=state.stage;
  $$(".keypoint-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!keypoints));
  $$(".fusion-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!fusion));
  $$(".fusion-descriptor-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!fusion&&!descriptors));
  $$(".descriptor-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!descriptors));
  $$(".query-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!query));
  $$(".feature-only").forEach(element=>element.classList.toggle("stage-specific-hidden",fusion));
  $$(".downstream-only").forEach(element=>element.classList.toggle("stage-specific-hidden",fusion));
  $$(".upstream-only").forEach(element=>element.classList.toggle("stage-specific-hidden",!nextStageByStage[state.stage]));
  const title={fusion:"RGB-D fusion / TSDF volume",keypoints:"Keypoint detection",descriptors:"Local descriptors",query:"Scene query / Top-k"}[state.stage]||stageLabel();
  const copy={fusion:"Settings are isolated to inspect scan construction before keypoint detection.",keypoints:"Values remain separate from Run Console settings until you explicitly apply them.",descriptors:"Build the local representation passed to ShapeNet retrieval.",query:"Global preselection followed by local reranking of ShapeNet models."}[state.stage]||"";
  $("#parameter-stage-title").textContent=title;$("#parameter-stage-copy").textContent=copy;
  $("#run-debug span").textContent=["matching","verification","selection"].includes(state.stage)?`Run ${stageLabel()}`:fusion?"Run fusion":(descriptors?"Compute descriptors":(query?"Compute Top-k":"Detect keypoints"));
  $("#pin-baseline").disabled=!keypoints||!state.current?.manifest?.id;
  $("#pin-baseline").closest(".panel-footer").classList.toggle("stage-specific-hidden",!keypoints);
  $("#apply-execution-parameters").classList.toggle("stage-specific-hidden",descriptors||query);$("#parameter-sync-status").classList.toggle("stage-specific-hidden",descriptors||query);
  const labels=$$(".timeline-summary span");
  if(labels.length>=3){labels[0].textContent="Input";labels[1].textContent=fusion?"Output":"Retained";labels[2].textContent=fusion?"Difference":"Rejected"}
  const vectorLabel=$("#layer-vector")?.closest(".legend-toggle");if(vectorLabel){vectorLabel.dataset.tooltip=fusion?"Camera trajectory":(descriptors?"Descriptor volume":"Adjustment vectors");vectorLabel.dataset.tooltipDetail=fusion?"Connect successive RGB-D camera positions.":(descriptors?"Outline the local grid centered on the inspected keypoint.":"Displacement from the initial keypoint position to its adjusted position.");const text=$("span",vectorLabel);if(text)text.textContent=fusion?"Camera":(descriptors?"Grille locale":"Adjustments")}
  $(".distribution-section").classList.toggle("stage-specific-hidden",fusion||query);$(".comparison-section").classList.toggle("stage-specific-hidden",!keypoints);
  const histogramOptions=descriptors?[["descriptor_distance","Distance"],["descriptor_margin","Margin"],["descriptor_entropy","Entropy"],["valid_models","Compatible models"]]:[["curvature","Curvature"],["harris","Harris"],["confidence","Confiance TSDF"],["wall_proximity","Wall proximity"],["wall_affinity","Wall affinity"],["wall_score","Wall score"],["object_score","Object evidence"],["object_anchor","Object support"]];const histogram=$("#histogram-kind"),previous=histogram.value;histogram.replaceChildren(...histogramOptions.map(([value,label])=>{const option=document.createElement("option");option.value=value;option.textContent=label;return option}));if(histogramOptions.some(([value])=>value===previous))histogram.value=previous;
  if(!state.current)$("#run-id").textContent=`${stageLabel()} · no computation`;
  renderChainControls();
}
async function selectStage(stage,{loadLatest=true}={}){
  if(state.runUntilActive&&loadLatest){notify("Stop the chain before switching stages.");return}
  const definition=stageDefinition(stage);
  if(!definition?.available)return;
  rememberSourcePaths();
  state.stage=stage;
  localStorage.setItem(stageStorage,stage);
  state.current=state.stageResults.get(stage)||null;
  state.step=0;
  renderStageNavigation();renderSources();renderParameters();updateStageUi();
  state.viewers.forEach(viewer=>{viewer.clearResult();viewer.hasFit=false});
  if(state.current){
    applyCurrentResults(state.current,{switchStage:false});
    return;
  }
  renderTimeline();
  if(loadLatest){
    const latest=(state.defaults.recent_runs||[]).find(run=>(run.stage||"keypoints")===stage);
    if(latest)await loadRun(latest.id);
  }
}
const previousStageByStage={keypoints:"fusion",descriptors:"keypoints",query:"descriptors",matching:"query",verification:"matching",selection:"verification"};
const nextStageByStage={fusion:"keypoints",keypoints:"descriptors",descriptors:"query",query:"matching",matching:"verification",verification:"selection"};
const executableStages=["fusion","keypoints","descriptors",...retrievalStages];
function upstreamRuns(stage=state.stage){const previous=previousStageByStage[stage];return(state.defaults?.recent_runs||[]).filter(run=>run.stage===previous)}
function chainedRunFromSources(sources=[]){
  const expected=previousStageByStage[state.stage],enabled=sources.filter(source=>source.enabled!==false),runIds=new Set(enabled.map(source=>source.upstream?.run_id).filter(Boolean));
  return expected&&enabled.length&&runIds.size===1&&enabled.every(source=>source.upstream?.stage===expected)?[...runIds][0]:null;
}
function currentRunIsChainable(){
  const stage=state.current?.manifest?.stage;if(stage!==state.stage||!nextStageByStage[stage])return false;
  const names={fusion:["scene_state","geometry"],keypoints:["keypoints"],descriptors:["descriptors"],query:["query"],matching:["matching"],verification:["verification"]}[stage]||[];
  return Object.values(state.current.scenes||{}).some(result=>names.some(name=>Boolean(result?.artifacts?.[name]?.path)));
}
function renderChainControls(preferredRunId=null){
  if(!state.defaults)return;
  const select=$("#upstream-fusion-run"),button=$("#chain-fusion-run"),status=$("#chain-status"),currentButton=$("#chain-current-fusion"),currentStatus=$("#chain-current-status"),targetSelect=$("#chain-target-stage");
  if(!select||!button||!status||!currentButton||!currentStatus||!targetSelect)return;
  const previous=previousStageByStage[state.stage],next=nextStageByStage[state.stage],runs=upstreamRuns(),inferred=chainedRunFromSources(state.current?.manifest?.stage===state.stage?(state.current?.manifest?.sources||[]):[]),selected=preferredRunId||state.upstreamRunId||inferred||select.value;
  $("#chain-direction").textContent=previous?`${stageLabel(previous)} → ${stageLabel()}`:"—";$("#upstream-stage-label").textContent=previous?`Run ${stageLabel(previous)}`:"Run amont";$("#chain-upstream-label").textContent=previous?`Use this ${stageLabel(previous)} run`:"Use this run";
  select.replaceChildren();
  const placeholder=document.createElement("option");placeholder.value="";placeholder.textContent=previous?`Choose a ${stageLabel(previous)} run…`:"No upstream stage";select.append(placeholder);
  for(const run of runs){
    const option=document.createElement("option");option.value=run.id;option.disabled=!run.chainable;option.textContent=`${run.label||run.id}${run.chainable?` · ${run.chainable_scene_count} scene${run.chainable_scene_count>1?"s":""}`:" · legacy run without artifact"}`;select.append(option);
  }
  select.value=[...select.options].some(option=>option.value===selected)?selected:"";
  const run=runs.find(item=>item.id===select.value);button.disabled=!run?.chainable;
  const configured=Boolean(previous&&(state.upstreamRunId||inferred));
  status.className=configured?"active":(run&&!run.chainable?"warning":"");
  status.textContent=configured?`Input explicitly linked to ${run?.label||configured}.`:(previous?`No ${stageLabel(previous)} run linked to ${stageLabel()} sources.`:"No upstream stage.");
  const currentIndex=executableStages.indexOf(state.stage),targets=executableStages.slice(currentIndex),previousTarget=state.runUntilStage||targetSelect.value||"selection";
  targetSelect.replaceChildren(...targets.map(stage=>{const option=document.createElement("option");option.value=stage;option.textContent=stageLabel(stage);return option}));
  targetSelect.value=targets.includes(previousTarget)?previousTarget:targets.at(-1);targetSelect.disabled=state.runUntilActive;
  const selectedSources=$$(".source-item").filter(row=>$("input[type=checkbox]",row).checked),busy=state.runUntilActive||state.status==="running";
  $("#stop-debug").disabled=!busy;
  const artifactNames={fusion:["scene_state","geometry"],keypoints:["keypoints"],descriptors:["descriptors"],query:["query"],matching:["matching"],verification:["verification"]}[state.stage]||[];
  const allInputsReady=selectedSources.every(row=>artifactNames.some(name=>state.current?.scenes?.[row.dataset.sourceId]?.artifacts?.[name]?.path));
  const canContinue=currentRunIsChainable()&&allInputsReady&&executableStages.indexOf(targetSelect.value)>currentIndex;
  currentButton.disabled=!canContinue||busy||!selectedSources.length;
  $("#chain-start").disabled=busy||!selectedSources.length||selectedSources.some(row=>!$("input[type=text]",row).value.trim());
  $("#chain-next-stage").textContent=stageLabel(targetSelect.value);
  $("#chain-start-label").textContent=`Start ${stageLabel()} → ${stageLabel(targetSelect.value)}`;
  $("#chain-current-label").textContent="Continue from loaded result";
  $("#chain-next-copy").textContent="Checked scenes pass through each stage to the destination. Subsequent stages reuse produced artifacts. Keep this page open during the chain.";
  currentStatus.className="";
  currentStatus.textContent=state.runUntilActive?`Pipeline running through ${stageLabel(state.runUntilStage)}…`:`${selectedSources.length} scene(s) checked. ${canContinue?"The loaded result can also be reused without recomputation.":"Start includes the displayed stage."}`;
}
async function chainStageRun(runId,targetStage=state.stage,sceneIds=null){
  if(!runId)return;
  const chained=await api("/api/debug/chain",{method:"POST",body:JSON.stringify({run_id:runId,target_stage:targetStage})});
  if(sceneIds){
    chained.sources=chained.sources.filter(source=>sceneIds.includes(source.id));
    if(sceneIds.some(id=>!chained.sources.some(source=>source.id===id)))throw new Error("A checked scene has no upstream artifact. Load a complete result or start a new chain.");
  }
  if(state.stage!==targetStage)await selectStage(targetStage,{loadLatest:false});
  state.stage=targetStage;localStorage.setItem(stageStorage,targetStage);state.current=null;state.stageResults.delete(targetStage);state.step=0;state.upstreamRunId=runId;
  for(const source of defaultSources()){const key=sourcePathKey(targetStage,source.id);state.sourceEnabled.set(key,false);state.sourceLineage.delete(key)}
  for(const source of chained.sources||[]){const key=sourcePathKey(targetStage,source.id);state.sourcePaths.set(key,source.path);state.sourceEnabled.set(key,true);state.sourceLineage.set(key,source.upstream)}
  renderStageNavigation();renderSources();renderParameters();updateStageUi();state.viewers.forEach(viewer=>{viewer.clearResult();viewer.hasFit=false});renderTimeline();
  $("#run-id").textContent=`${stageLabel()} · depuis ${chained.source_run?.label||runId}`;renderChainControls();notify(`${chained.sources.length} source${chained.sources.length>1?"s":""} linked to ${stageLabel()}`);
}
async function continueRunUntil(runId,completedStage){
  if(!state.runUntilActive)return;
  const target=state.runUntilStage,completedIndex=executableStages.indexOf(completedStage),targetIndex=executableStages.indexOf(target);
  if(targetIndex<0||completedIndex>=targetIndex){
    state.runUntilActive=false;
    notify(`Debug pipeline completed at ${stageLabel(target||completedStage)}`);
    renderChainControls();
    return;
  }
  const next=nextStageByStage[completedStage];
  if(!next){state.runUntilActive=false;renderChainControls();return}
  await chainStageRun(runId,next,state.runUntilSceneIds);
  if(!state.runUntilActive)return;
  await runDebug();
}
async function runToSelectedStage(){
  const target=$("#chain-target-stage").value,runId=state.current?.manifest?.id;
  if(!target||!runId||!currentRunIsChainable())return;
  state.runUntilStage=target;state.runUntilActive=true;
  state.runUntilSceneIds=collectPayload().sources.filter(source=>source.enabled).map(source=>source.id);
  renderChainControls();
  try{await continueRunUntil(runId,state.stage)}catch(error){state.runUntilActive=false;renderChainControls();throw error}
}
async function startToSelectedStage(){
  const sources=collectPayload().sources.filter(source=>source.enabled);
  if(state.status==="running"||state.runUntilActive)return;
  if(!sources.length||sources.some(source=>!source.path.trim()))throw new Error("Check at least one scene with a valid input.");
  state.runUntilStage=$("#chain-target-stage").value;
  state.runUntilSceneIds=sources.map(source=>source.id);state.runUntilActive=true;
  clearTimeout(state.autoTimer);state.pendingAuto=false;
  renderChainControls();
  try{await runDebug()}catch(error){state.runUntilActive=false;renderChainControls();throw error}
}
function chainFusionRun(runId){return chainStageRun(runId,"keypoints")}
function renderSources(){
  const list=$("#source-list");list.replaceChildren();
  defaultSources().forEach(source=>{
    const row=document.createElement("div");row.className="source-item";row.dataset.sourceId=source.id;
    const key=sourcePathKey(state.stage,source.id),checkbox=document.createElement("input");checkbox.type="checkbox";checkbox.checked=state.sourceEnabled.has(key)?state.sourceEnabled.get(key):source.enabled!==false;checkbox.setAttribute("aria-label",`Activer ${source.label}`);
    const copy=document.createElement("div");copy.className="source-copy";
    const label=document.createElement("label"),strong=document.createElement("strong"),status=document.createElement("i");strong.textContent=source.label;status.textContent="ready";label.append(strong,status);
    const path=document.createElement("input");path.type="text";path.value=sourcePathFor(source);path.setAttribute("aria-label",`Source ${source.label}`);
    status.textContent={fusion:"RGB-D archive",keypoints:"scene state",descriptors:"chained keypoints",query:"chained descriptors"}[state.stage]||"source";
    copy.append(label,path);row.append(checkbox,copy);list.append(row);
    checkbox.addEventListener("change",()=>{state.sourceEnabled.set(key,checkbox.checked);renderChainControls();parametersChanged()});path.addEventListener("input",()=>{state.sourcePaths.set(key,path.value);state.sourceLineage.delete(key);state.upstreamRunId=null;renderChainControls()});path.addEventListener("change",parametersChanged);
  });
  updateSourceCount();ensureSceneCards(defaultSources());
}
function updateSourceCount(){const active=$$(".source-item input[type=checkbox]").filter(input=>input.checked).length;$("#source-count").textContent=`${active} scene${active>1?"s":""}`}

const parameterGroups={
  fusion:[
    ["TSDF volume",["voxel_size","truncation","depth_trunc","volume_layout","backend"]],
    ["RGB-D sequence",["frame_stride","min_frames","max_frames"]],
    ["Silhouettes RGB-D",["depth_edge_threshold","depth_edge_background_weight","depth_edge_radius"]],
    ["Iso-surface",["iso_sampling","max_surface_points","surface_field","surface_min_support","gradient_smoothing_sigma"]],
    ["Volume sparse",["sparse_block_resolution","sparse_block_count"]],
    ["Visibility",["visibility_depth_stride","visibility_surface_band_factor"]],
    ["Execution",["parallel_scenes"]],
  ],
  keypoints:[
    ["Local detection",["neighbor_radius","curvature_threshold","harris_k","harris_threshold","harris_reference_neighbors"]],
    ["2D recovery",["convex_hull_ratio","corner_plane_radius_factor","corner_plane_min_area","corner_plane_normal_cos"]],
    ["Stabilisation",["nms_radius","adjust_iterations","jitter_reject","dedup_radius"]],
    ["Vertical plane detection",["wall_vertical_normal_cos","wall_plane_normal_cos","wall_plane_angle_degrees","wall_plane_distance","wall_plane_min_points","wall_plane_min_width","wall_plane_min_height","wall_plane_min_area","wall_plane_max_count","wall_plane_sample_points"]],
    ["Wall scoring and filtering",["wall_small_radius","wall_large_radius","wall_protrusion_radius","wall_protrusion_distance","wall_object_protection","wall_filter_enabled","wall_penalty_weight","wall_hard_reject","wall_reject_threshold","wall_reject_max_object_score"]],
    ["Wall budget",["wall_budget_enabled","wall_budget_affinity_threshold","wall_budget_proximity_threshold","wall_budget_max_fraction","wall_budget_cell_size","wall_budget_max_per_cell","wall_budget_min_features"]],
    ["Budget per region",["component_budget_enabled","component_budget_radius","component_budget_max_per_component","component_budget_min_features"]],
    ["Geometric quality",["planar_filter_enabled","planar_filter_radius","planar_filter_max_residual","planar_filter_normal_cos","planar_filter_angular_coverage","planar_filter_min_neighbors","hole_boundary_filter_enabled","hole_boundary_probe_radius","hole_boundary_max_fraction","repeatability_filter_enabled","repeatability_radius_factor","repeatability_response_ratio","quality_response_ratio","quality_score_radius","geometric_nms_radius","quality_filter_min_features"]],
    ["Descriptor quality",["descriptor_keypoint_filter_enabled","descriptor_min_occupied","descriptor_max_hole_fraction","descriptor_keypoint_min_features","quality_nms_radius","quality_min_score_ratio"]],
    ["Output to matching",["floor_height","max_scan_keypoints","spatial_keypoint_balance","spatial_keypoint_cell","min_2d_corner_fraction"]],
    ["Object groups",["group_keypoint_link_radius","group_part_merge_gap","group_component_merge_gap","group_wall_clearance","group_surface_connection","group_assignment_radius","group_orphan_attach_radius"]],
    ["Execution",["parallel_scenes"]],
  ],
  descriptors:[
    ["Grille d'occupation",["occ_grid_res","occ_extent","max_occupied","visibility_surface_band_factor"]],
    ["Local distance",["distance_unit","distance_exponent","utility_distance_threshold","utility_rotations"]],
    ["Diagnostic pool",["utility_candidate_models","utility_model_features"]],
    ["Discriminative power",["utility_margin_threshold","utility_max_valid_models","utility_entropy_threshold"]],
    ["Execution",["parallel_scenes"]],
  ],
  query:[
    ["Object preselection",["grouped_query","candidate_min_group_descriptors","candidate_group_weighting","candidate_pool_k","candidate_local_extra_k","candidate_global_pool_fraction","candidate_pool_min_per_group","candidate_diverse","candidate_diversity"]],
    ["Budget per model",["candidate_max_groups"]],
    ["Local reranking",["local_rerank","candidate_top_k","candidate_top_min_per_group","candidate_extent_quota_fraction"]],
    ["3D preview",["surface_preview_points"]],
    ["Execution",["parallel_scenes"]],
  ],
};
function stageParameters(stage=state.stage){return state.defaults.parameters_by_stage?.[stage]||state.defaults.parameters}
function stageParameterSchema(stage=state.stage){return state.defaults.parameter_schemas?.[stage]||state.defaults.parameter_schema}
function renderParameters(){
  const form=$("#parameter-form");form.replaceChildren();const defaults=stageParameters(),schema=new Map(stageParameterSchema().map(item=>[item.id,item]));
  (parameterGroups[state.stage]||[["Parameters",stageParameterSchema().map(item=>item.id)]]).forEach(([title,ids])=>{
    const group=document.createElement("section");group.className="parameter-group";group.dataset.group=title;
    ids.forEach(id=>{const spec=schema.get(id);if(!spec)return;const row=document.createElement("div");row.className=`parameter-row ${spec.type==="bool"?"boolean":""}`;
      const copy=document.createElement("div");copy.className="parameter-copy";
      const heading=document.createElement("div");heading.className="parameter-heading";
      const label=document.createElement("label");label.htmlFor=`param-${id}`;label.textContent=spec.label;
      const help=document.createElement("button");help.type="button";help.className="parameter-help";help.innerHTML='<i data-lucide="circle-help"></i>';help.setAttribute("aria-label",`Details for ${spec.label}`);help.dataset.tooltip=spec.label;help.dataset.tooltipDetail=spec.summary;help.addEventListener("click",()=>openParameterHelp(spec));
      const summary=document.createElement("small");summary.className="parameter-summary";summary.id=`param-${id}-summary`;summary.textContent=spec.summary;
      heading.append(label,help);copy.append(heading,summary);
      const control=document.createElement("div");control.className="parameter-control";let input;
      if(spec.type==="enum"){input=document.createElement("select");(spec.choices||[]).forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value;input.append(option)});input.value=String(defaults[id]??spec.choices?.[0]??"")}
      else{input=document.createElement("input");if(spec.type==="bool"){input.type="checkbox";input.checked=Boolean(defaults[id])}else{input.type="number";input.min=spec.min;input.max=spec.max;input.step=spec.step;const value=defaults[id];input.value=value==null?"":value}}
      input.id=`param-${id}`;input.dataset.parameter=id;
      input.setAttribute("aria-describedby",summary.id);control.append(input);if(spec.unit){const unit=document.createElement("small");unit.textContent=spec.unit;control.append(unit)}row.append(copy,control);group.append(row);
      input.addEventListener(["bool","enum"].includes(spec.type)?"change":"input",parametersChanged);
    });form.append(group);
  });
  refreshIcons();
}
function resetStageParameters(){
  renderParameters();
  const status=$("#parameter-sync-status");
  status.textContent="Current configuration defaults restored.";
  status.classList.remove("success");
  notify(`${stageLabel()} parameters reset`);
}
function openParameterHelp(spec){
  $("#parameter-help-key").textContent=spec.id;$("#parameter-help-title").textContent=spec.label;$("#parameter-help-summary").textContent=spec.summary;$("#parameter-help-detail").textContent=spec.detail;$("#parameter-help-higher").textContent=spec.higher;$("#parameter-help-lower").textContent=spec.lower;$("#parameter-help-cost").textContent=spec.cost;hideTooltip();$("#parameter-help-dialog").hidden=false;$("#close-parameter-help").focus();
}
function closeParameterHelp(){$("#parameter-help-dialog").hidden=true}
function collectPayload(){
  const parameters={...stageParameters()},schema=stageParameterSchema();
  $$('[data-parameter]').forEach(input=>{const spec=schema.find(item=>item.id===input.dataset.parameter);if(spec.type==="bool")parameters[spec.id]=input.checked;else if(spec.type==="enum")parameters[spec.id]=input.value;else if(spec.type==="int")parameters[spec.id]=Number.parseInt(input.value||"0",10);else parameters[spec.id]=input.value===""?null:Number.parseFloat(input.value)});
  const sourceDefinitions=new Map(defaultSources().map(source=>[source.id,source]));
  const sources=$$(".source-item").map(row=>{const id=row.dataset.sourceId,key=sourcePathKey(state.stage,id),source={id,label:sourceDefinitions.get(id)?.label||id,path:$("input[type=text]",row).value,enabled:$("input[type=checkbox]",row).checked},upstream=state.sourceLineage.get(key);if(upstream)source.upstream=upstream;return source});
  return{stage:state.stage,parameters,sources};
}
function applyManifestInputs(manifest){
  const parameters=manifest?.parameters||{};
  $$("[data-parameter]").forEach(input=>{const value=parameters[input.dataset.parameter];if(value===undefined)return;if(input.type==="checkbox")input.checked=Boolean(value);else input.value=value==null?"":String(value)});
  const sources=new Map((manifest?.sources||[]).map(source=>[source.id,source]));
  $$(".source-item").forEach(row=>{const source=sources.get(row.dataset.sourceId),key=sourcePathKey(state.stage,row.dataset.sourceId),checkbox=$("input[type=checkbox]",row),path=$("input[type=text]",row);if(!source){checkbox.checked=false;state.sourceEnabled.set(key,false);state.sourceLineage.delete(key);return}checkbox.checked=source.enabled!==false;path.value=source.path||"";state.sourceEnabled.set(key,checkbox.checked);if(source.upstream)state.sourceLineage.set(key,source.upstream);else state.sourceLineage.delete(key)});
  sources.forEach(source=>state.sourcePaths.set(sourcePathKey(state.stage,source.id),source.path||""));
  state.upstreamRunId=previousStageByStage[state.stage]?chainedRunFromSources([...sources.values()]):null;
  updateSourceCount();renderChainControls();
}
function parametersChanged(){updateSourceCount();if(state.runUntilActive||!$("#auto-run").checked)return;clearTimeout(state.autoTimer);state.autoTimer=setTimeout(()=>{if(state.status==="running")state.pendingAuto=true;else runDebug()},650)}

function renderTimeline({preserveStep=false}={}){
  const previousId=preserveStep?state.stageSteps[state.step]?.id:null;
  state.stageSteps=stageResultSteps();const track=$("#chapter-track");track.replaceChildren();track.style.setProperty("--debug-step-count",Math.max(1,state.stageSteps.length));track.classList.toggle("dense",state.stageSteps.length>20);
  state.stageSteps.forEach((step,index)=>{const button=document.createElement("button"),isFrame=String(step.id||"").startsWith("frame-")&&step.id!=="frame-selection";button.type="button";button.className=`chapter-node ${isFrame?"frame-step":"major-step"}`;button.dataset.index=index;button.dataset.stepId=step.id||String(index);button.dataset.tooltip=step.label;button.dataset.tooltipDetail=[step.description,step.assessment].filter(Boolean).join(" ");button.setAttribute("aria-label",`${index+1}. ${step.label}`);button.innerHTML=`<span>${step.label}</span><i></i>`;button.addEventListener("click",()=>setStep(index));track.append(button)});
  const preservedIndex=previousId?state.stageSteps.findIndex(step=>step.id===previousId):-1;
  setStep(preservedIndex>=0?preservedIndex:(preserveStep?state.step:0));
}
function revealTimelineStep(node){
  const track=$("#chapter-track");if(!node||!track)return;
  const margin=16,left=node.offsetLeft,right=left+node.offsetWidth,visibleLeft=track.scrollLeft,visibleRight=visibleLeft+track.clientWidth;
  if(left<visibleLeft+margin)track.scrollTo({left:Math.max(0,left-margin),behavior:"smooth"});
  else if(right>visibleRight-margin)track.scrollTo({left:Math.max(0,right-track.clientWidth+margin),behavior:"smooth"});
}
function setStep(index){
  const max=Math.max(0,state.stageSteps.length-1),parsed=Number(index);state.step=Math.max(0,Math.min(max,Number.isFinite(parsed)?Math.round(parsed):0));const definition=state.stageSteps[state.step]||{label:stageLabel(),description:"Run or load a result to inspect this stage."};
  $("#step-index").textContent=state.stageSteps.length?`${state.step+1} / ${max+1}`:"0 / 0";$("#step-title").textContent=definition.label;$("#step-description").textContent=definition.description;
  $$(".chapter-node").forEach((node,i)=>{node.classList.toggle("active",i===state.step);node.classList.toggle("visited",i<state.step);if(i===state.step)node.setAttribute("aria-current","step");else node.removeAttribute("aria-current")});
  $("#previous-step").disabled=state.step<=0;$("#next-step").disabled=state.step>=max;revealTimelineStep($(".chapter-node.active"));
  state.viewers.forEach(viewer=>viewer.renderStep(definition));updateTimelineMetrics();updateHistogram();updateComparison();renderQueryCandidates();
}

function activeStepTotals(){const definition=activeStepDefinition();let input=0,kept=0,rejected=0,count=0;for(const result of Object.values(state.current?.scenes||{})){const step=resultStepForDefinition(result,definition);if(!step)continue;input+=Number(step.input||0);kept+=Number(step.kept||0);rejected+=Number(step.rejected||0);count++}return{input,kept,rejected,count}}
function openStepHelp(){const step=activeStepDefinition();if(!step)return;const totals=activeStepTotals();$("#step-help-group").textContent=step.group||stageLabel();$("#step-help-title").textContent=step.label||step.id;$("#step-help-description").textContent=step.description||"No explanation recorded.";$("#step-help-assessment").textContent=step.assessment||"No assessment recorded for this stage yet.";$("#step-help-input").textContent=totals.count?formatInt(totals.input):"—";$("#step-help-kept").textContent=totals.count?formatInt(totals.kept):"—";$("#step-help-rejected").textContent=totals.count?formatInt(totals.rejected):"—";hideTooltip();$("#step-help-dialog").hidden=false;$("#close-step-help").focus()}
function closeStepHelp(){$("#step-help-dialog").hidden=true}

function loadSceneWeights(){
  try{return new Map(Object.entries(JSON.parse(localStorage.getItem(sceneWeightsStorage)||"{}")).map(([id,value])=>[id,Math.max(.2,Number(value)||1)]))}catch{return new Map()}
}
function persistSceneWeights(){localStorage.setItem(sceneWeightsStorage,JSON.stringify(Object.fromEntries(state.sceneWeights)))}
function loadSceneDisplaySettings(){
  try{return new Map(Object.entries(JSON.parse(localStorage.getItem(sceneDisplayStorage)||"{}")))}catch{return new Map()}
}
function sceneDisplaySettings(sceneId){return{...defaultSceneDisplay,...(state.sceneDisplaySettings.get(sceneId)||{})}}
function persistSceneDisplaySettings(){localStorage.setItem(sceneDisplayStorage,JSON.stringify(Object.fromEntries(state.sceneDisplaySettings)))}
function setSceneDisplaySettings(sceneId,patch,{persist=true}={}){
  const settings={...sceneDisplaySettings(sceneId),...patch};state.sceneDisplaySettings.set(sceneId,settings);if(persist)persistSceneDisplaySettings();state.viewers.get(sceneId)?.applyDisplaySettings(activeStepDefinition());return settings;
}
function updateSceneGridLayout(){
  const grid=$("#scene-grid"),cards=$$(".scene-card",grid),focused=state.featuredScene&&state.viewers.has(state.featuredScene)?state.featuredScene:null;
  state.featuredScene=focused;grid.classList.toggle("scene-focused",Boolean(focused));grid.dataset.focusScene=focused||"";
  cards.forEach((card,index)=>{const active=card.dataset.sceneId===focused;card.classList.toggle("featured",active);const handle=$(".scene-resizer",card);if(handle)handle.hidden=Boolean(focused)||index===cards.length-1});
  if(focused){grid.style.gridTemplateColumns="minmax(0,1fr)";return}
  const columns=cards.map(card=>`minmax(180px,${Math.max(.2,state.sceneWeights.get(card.dataset.sceneId)||1)}fr)`);
  grid.style.gridTemplateColumns=columns.join(" ");
}
function setFeaturedScene(sceneId=null){
  state.featuredScene=state.featuredScene===sceneId?null:sceneId;updateSceneGridLayout();
  state.viewers.forEach(viewer=>viewer.updateViewActions());requestAnimationFrame(()=>state.viewers.forEach(viewer=>viewer.resize()));
}
async function toggleSceneFullscreen(viewer){
  try{if(document.fullscreenElement===viewer.card)await document.exitFullscreen();else await viewer.card.requestFullscreen()}catch(error){notify(error.message||"Fullscreen is unavailable",true)}
}
function beginSceneResize(event,sceneId){
  if(state.featuredScene||event.button!==0)return;const cards=$$(".scene-card",$("#scene-grid")),index=cards.findIndex(card=>card.dataset.sceneId===sceneId),left=cards[index],right=cards[index+1];if(!left||!right)return;
  event.preventDefault();const startX=event.clientX,leftWidth=left.getBoundingClientRect().width,rightWidth=right.getBoundingClientRect().width,totalWeight=(state.sceneWeights.get(left.dataset.sceneId)||1)+(state.sceneWeights.get(right.dataset.sceneId)||1),totalWidth=leftWidth+rightWidth;
  const move=moveEvent=>{const nextLeft=Math.max(180,Math.min(totalWidth-180,leftWidth+moveEvent.clientX-startX)),leftWeight=totalWeight*nextLeft/totalWidth;state.sceneWeights.set(left.dataset.sceneId,leftWeight);state.sceneWeights.set(right.dataset.sceneId,totalWeight-leftWeight);updateSceneGridLayout();state.viewers.forEach(viewer=>viewer.resize())};
  const end=()=>{document.body.classList.remove("resizing-scenes");window.removeEventListener("pointermove",move);window.removeEventListener("pointerup",end);window.removeEventListener("pointercancel",end);persistSceneWeights()};
  document.body.classList.add("resizing-scenes");window.addEventListener("pointermove",move);window.addEventListener("pointerup",end);window.addEventListener("pointercancel",end);
}

function ensureSceneCards(sources){
  const grid=$("#scene-grid"),wanted=new Set(sources.map(source=>source.id));
  for(const[id,viewer]of state.viewers){if(!wanted.has(id)){viewer.dispose();state.viewers.delete(id);viewer.card.remove()}}
  sources.forEach(source=>{if(state.viewers.has(source.id))return;const card=document.createElement("article");card.className="scene-card ui-scene";card.dataset.sceneId=source.id;
    const header=document.createElement("header");header.innerHTML=`<div class="scene-title"><strong></strong><span>Waiting for a result</span></div><div class="scene-header-actions"><span class="scene-status idle"><i></i><b>Ready</b></span><button class="scene-view-action scene-display" type="button" aria-label="Adjust this scene display" aria-expanded="false" data-tooltip="Display" data-tooltip-detail="Scene-specific colors, point size, opacity and grid."><i data-lucide="palette"></i></button><button class="scene-view-action scene-feature" type="button" aria-label="Expand this viewer" data-tooltip="Expand viewer" data-tooltip-detail="Use the full central area while keeping panels and timeline visible."><i data-lucide="maximize-2"></i></button><button class="scene-view-action scene-fullscreen" type="button" aria-label="Show this viewer fullscreen" data-tooltip="Fullscreen" data-tooltip-detail="Display only this viewer across the screen. Press Escape to return."><i data-lucide="expand"></i></button></div>`;$(".scene-title strong",header).textContent=source.label;
    const displayPopover=document.createElement("section");displayPopover.className="scene-display-popover";displayPopover.hidden=true;displayPopover.innerHTML=`<header><div><strong>3D display</strong><span>Settings for this scene</span></div><button type="button" class="scene-display-close" aria-label="Close"><i data-lucide="x"></i></button></header><label class="scene-display-field"><span>Render mode</span><select data-display="colorMode"><option value="auto">Automatic</option><option value="neutral">Solid color</option><option value="rgb">RGB reconstruit</option><option value="curvature">Curvature</option></select></label><p class="scene-rgb-status"></p><div class="scene-display-colors"><label><span>Surface</span><input type="color" data-display="surfaceColor"></label><label><span>Background</span><input type="color" data-display="backgroundColor"></label></div><label class="scene-display-field scene-display-range"><span>Point size <output data-display-output="pointScale"></output></span><input type="range" min="0.5" max="3" step="0.1" data-display="pointScale"></label><label class="scene-display-field scene-display-range"><span>Opacity <output data-display-output="opacity"></output></span><input type="range" min="0.15" max="1" step="0.05" data-display="opacity"></label><label class="scene-display-check"><input type="checkbox" data-display="grid"><span>Show ground grid</span></label><footer><p>RGB mode projects frame colors onto TSDF points; it does not create a UV texture.</p><button type="button" class="scene-display-reset"><i data-lucide="rotate-ccw"></i><span>Reset</span></button></footer>`;
    const empty=document.createElement("div");empty.className="scene-empty";empty.innerHTML=`<i data-lucide="scan-line"></i><strong>Geometry not loaded</strong><span>The previous result remains visible during recomputation.</span>`;
    const canvas=document.createElement("canvas");canvas.setAttribute("aria-label",`3D view ${source.label}`);const metrics=document.createElement("div");metrics.className="scene-metrics";
    const finalControls=document.createElement("div");finalControls.className="scene-final-controls";finalControls.hidden=true;finalControls.setAttribute("aria-label","Final result display");finalControls.innerHTML=`<div class="scene-final-modes" role="group" aria-label="Displayed geometry"><button type="button" data-final-mode="scene" aria-pressed="false">Scene</button><button type="button" data-final-mode="models" aria-pressed="false">Models</button><button type="button" data-final-mode="both" aria-pressed="true">Both</button></div><button type="button" class="scene-final-focus" data-tooltip="Frame retained models" aria-label="Frame retained models"><i data-lucide="focus"></i></button><button type="button" class="scene-final-keypoints" aria-pressed="false" data-tooltip="Model keypoints" aria-label="Show retained model keypoints"><i data-lucide="crosshair"></i><span>Keypoints</span></button>`;card.append(finalControls);
    const progress=document.createElement("div");progress.className="scene-progress";progress.innerHTML="<i></i>";const resizer=document.createElement("div");resizer.className="scene-resizer ui-resizer";resizer.setAttribute("role","separator");resizer.setAttribute("aria-orientation","vertical");resizer.setAttribute("aria-label",`Resize ${source.label}`);resizer.dataset.tooltip="Resize viewer";resizer.dataset.tooltipDetail="Drag to divide space between this viewer and the next.";card.append(canvas,header,displayPopover,empty,metrics,progress,resizer);grid.append(card);
    const viewer=new SceneViewer(source,card,canvas);state.viewers.set(source.id,viewer);$(".scene-feature",card).addEventListener("click",event=>{event.stopPropagation();setFeaturedScene(source.id)});$(".scene-fullscreen",card).addEventListener("click",event=>{event.stopPropagation();toggleSceneFullscreen(viewer)});resizer.addEventListener("pointerdown",event=>beginSceneResize(event,source.id));resizer.addEventListener("dblclick",()=>{state.sceneWeights.set(source.id,1);const next=card.nextElementSibling;if(next?.dataset.sceneId)state.sceneWeights.set(next.dataset.sceneId,1);persistSceneWeights();updateSceneGridLayout();state.viewers.forEach(item=>item.resize())});resizer.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const next=card.nextElementSibling;if(!next?.dataset.sceneId)return;const delta=event.key==="ArrowRight"?0.1:-0.1,left=Math.max(.2,(state.sceneWeights.get(source.id)||1)+delta),right=Math.max(.2,(state.sceneWeights.get(next.dataset.sceneId)||1)-delta);state.sceneWeights.set(source.id,left);state.sceneWeights.set(next.dataset.sceneId,right);persistSceneWeights();updateSceneGridLayout();state.viewers.forEach(item=>item.resize())});
  });
  const sceneCount=Math.max(1,state.viewers.size);
  grid.dataset.sceneCount=String(sceneCount);
  grid.style.setProperty("--scene-count",String(Math.min(3,sceneCount)));updateSceneGridLayout();
  refreshIcons();bindTooltips();
}

class SceneViewer{
  constructor(source,card,canvas){
    this.source=source;this.card=card;this.canvas=canvas;this.result=null;this.baseline=null;this.objects=new Map();this.baselineObjects=new Map();this.queryObjects=new Map();this.hasFit=false;
    this.displaySettings=sceneDisplaySettings(source.id);this.finalMode="both";this.finalKeypoints=false;this.finalSelectionWasActive=false;
    this.viewport=createSceneViewport({THREE,OrbitControls,host:card,canvas,gridSize:10,gridDivisions:20,pixelRatio:2,cameraPosition:[3,2.2,3],cameraTarget:[0,.8,0]});
    this.scene=this.viewport.scene;this.camera=this.viewport.camera;this.renderer=this.viewport.renderer;this.controls=this.viewport.controls;this.root=this.viewport.root;this.grid=this.viewport.grid;
    this.controls.addEventListener("change",()=>syncCameraFrom(this));
    this.raycaster=new THREE.Raycaster();this.raycaster.params.Points.threshold=.055;this.pointer=new THREE.Vector2();canvas.addEventListener("pointerdown",event=>this.pick(event));this.bindDisplayControls();this.bindFinalControls();this.applyDisplaySettings();
  }
  bindFinalControls(){
    for(const button of $$("[data-final-mode]",this.card))button.addEventListener("click",()=>{this.finalMode=button.dataset.finalMode;this.renderStep(activeStepDefinition());if(this.finalMode==="scene")this.fitScene();else this.fitSelection()});
    $(".scene-final-focus",this.card).addEventListener("click",()=>{if(this.finalMode==="scene"){this.finalMode="both";this.renderStep(activeStepDefinition())}this.fitSelection()});
    $(".scene-final-keypoints",this.card).addEventListener("click",()=>{this.finalKeypoints=!this.finalKeypoints;this.renderStep(activeStepDefinition())});
  }
  syncFinalControls(final){
    $(".scene-final-controls",this.card).hidden=!final;
    this.card.dataset.finalSelection=String(final);this.card.dataset.finalMode=this.finalMode;
    for(const button of $$("[data-final-mode]",this.card))button.setAttribute("aria-pressed",String(button.dataset.finalMode===this.finalMode));
    const keypoints=$(".scene-final-keypoints",this.card);keypoints.setAttribute("aria-pressed",String(this.finalKeypoints));keypoints.disabled=this.finalMode==="scene"||!this.result?.layers?.some(layer=>layer.display_role==="model-keypoints");
    $(".scene-final-focus",this.card).disabled=!this.result?.metrics?.retained;
  }
  fitScene(){const surface=this.objects.get("surface");if(surface&&this.viewport.fitObjects([surface],1.12,{remember:true}))this.fitRadius=this.viewport.fitRadius}
  fitSelection(){
    const step=resultStepForDefinition(this.result),retained=new Set(step?.visible_layers||[]),models=[...this.objects.entries()].filter(([id,object])=>retained.has(id)&&finalSelectionLayerVisible(object.userData.layer||{},"models",false)).map(([,object])=>object);
    if(this.viewport.fitObjects(models,1.5,{remember:true}))this.fitRadius=this.viewport.fitRadius;else this.fitScene();
  }
  resize(){this.viewport.resize()}
  hasRgbColors(){return Boolean(this.result?.layers?.some(layer=>Array.isArray(layer.rgb_colors)&&layer.rgb_colors.length))}
  setDisplayPopoverOpen(open){const popover=$(".scene-display-popover",this.card),button=$(".scene-display",this.card);popover.hidden=!open;button.setAttribute("aria-expanded",String(open));button.classList.toggle("active",open);if(open)this.syncDisplayControls()}
  syncDisplayControls(){
    this.displaySettings=sceneDisplaySettings(this.source.id);const popover=$(".scene-display-popover",this.card),settings=this.displaySettings;
    for(const input of $$('[data-display]',popover)){const key=input.dataset.display;if(input.type==="checkbox")input.checked=Boolean(settings[key]);else input.value=String(settings[key])}
    $('[data-display-output="pointScale"]',popover).textContent=`${Number(settings.pointScale).toFixed(1)}×`;$('[data-display-output="opacity"]',popover).textContent=`${Math.round(Number(settings.opacity)*100)} %`;
    const available=this.hasRgbColors(),status=$(".scene-rgb-status",popover);status.classList.toggle("available",available);status.textContent=available?"Reconstructed RGB available for this run.":"RGB unavailable: rerun fusion, then chain subsequent stages.";
  }
  bindDisplayControls(){
    const button=$(".scene-display",this.card),popover=$(".scene-display-popover",this.card);button.addEventListener("click",event=>{event.stopPropagation();hideTooltip(button);const open=popover.hidden;state.viewers.forEach(viewer=>viewer.setDisplayPopoverOpen(false));this.setDisplayPopoverOpen(open)});popover.addEventListener("pointerdown",event=>event.stopPropagation());$(".scene-display-close",popover).addEventListener("click",()=>this.setDisplayPopoverOpen(false));$(".scene-display-reset",popover).addEventListener("click",()=>{state.sceneDisplaySettings.set(this.source.id,{...defaultSceneDisplay});persistSceneDisplaySettings();this.syncDisplayControls();this.applyDisplaySettings(activeStepDefinition())});
    for(const input of $$('[data-display]',popover)){const eventName=input.type==="range"||input.type==="color"?"input":"change";input.addEventListener(eventName,event=>{const key=event.target.dataset.display,value=event.target.type==="checkbox"?event.target.checked:(event.target.type==="range"?Number(event.target.value):event.target.value);this.displaySettings=setSceneDisplaySettings(this.source.id,{[key]:value});this.syncDisplayControls()})}
  }
  updateViewActions(){const featured=state.featuredScene===this.source.id,fullscreen=document.fullscreenElement===this.card,feature=$(".scene-feature",this.card),full=$(".scene-fullscreen",this.card);feature.dataset.tooltip=featured?"Restore grid":"Expand viewer";feature.setAttribute("aria-label",featured?"Restore all viewers":"Expand this viewer");feature.innerHTML=`<i data-lucide="${featured?"minimize-2":"maximize-2"}"></i>`;full.dataset.tooltip=fullscreen?"Exit fullscreen":"Fullscreen";full.setAttribute("aria-label",fullscreen?"Exit fullscreen":"Show this viewer fullscreen");full.innerHTML=`<i data-lucide="${fullscreen?"shrink":"expand"}"></i>`;refreshIcons()}
  clearObjects(collection=this.objects){collection.forEach(object=>disposeObject3D(object));collection.clear()}
  clearQueryCandidates(){this.clearObjects(this.queryObjects)}
  removeQueryCandidate(candidateIndex){const object=this.queryObjects.get(Number(candidateIndex));if(!object)return;disposeObject3D(object);this.queryObjects.delete(Number(candidateIndex));this.viewport.updateClipping(this.root)}
  setQueryCandidate(candidateIndex,detail,color){
    this.removeQueryCandidate(candidateIndex);
    if(!detail?.placed||!detail.model_points?.length)return;
    const group=new THREE.Group();group.userData.queryCandidate=Number(candidateIndex);
    const model=createPointCloud({THREE,positions:detail.model_points,color,size:.028,opacity:.82,depthWrite:true,renderOrder:4});model.userData.queryRole="model";model.userData.display={baseSize:.028,baseOpacity:.82,surface:false};group.add(model);
    if(detail.correspondence_segments?.length){const lines=createPolyline({THREE,positions:detail.correspondence_segments.flat(2),color,opacity:.9,segments:true});lines.userData.queryRole="correspondences";lines.renderOrder=5;group.add(lines)}
    group.visible=retrievalStages.includes(state.stage)&&!isFinalSelection(this.result,resultStepForDefinition(this.result));this.root.add(group);this.queryObjects.set(Number(candidateIndex),group);this.applyDisplaySettings(activeStepDefinition());this.viewport.updateClipping(this.root);
  }
  clearResult(){this.result=null;this.baseline=null;this.finalSelectionWasActive=false;this.syncFinalControls(false);this.clearObjects();this.clearObjects(this.baselineObjects);this.clearQueryCandidates();$(".scene-empty",this.card).hidden=false;$(".scene-title span",this.card).textContent="Waiting for a result";$(".scene-metrics",this.card).replaceChildren();const status=$(".scene-status",this.card);status.className="scene-status idle";$("b",status).textContent="Ready"}
  setResult(result,baseline){const wasSyncing=state.syncing;state.syncing=true;try{if(result!==this.result)this.finalSelectionWasActive=false;this.result=result;this.baseline=baseline||null;this.clearObjects();this.clearObjects(this.baselineObjects);this.clearQueryCandidates();$(".scene-empty",this.card).hidden=true;this.buildLayers();this.syncDisplayControls();this.renderStep(activeStepDefinition());if(!this.hasFit){this.fit();this.hasFit=true}this.updateHeader()}finally{state.syncing=wasSyncing}}
  buildLayers(){
    (this.result.layers||[]).forEach(layer=>{let object;if(layer.kind==="lines")object=this.makeLines(layer);else object=this.makePoints(layer);object.visible=false;object.userData.layer=layer;this.objects.set(layer.id,object);this.root.add(object)});
    if(this.baseline){(this.baseline.steps||[]).forEach((step,index)=>{const baselineLayerId=primaryLayerId(step,this.baseline),currentStep=this.result.steps?.[index],currentLayerId=primaryLayerId(currentStep,this.result);if(!baselineLayerId||!currentLayerId)return;const layer=this.baseline.layers.find(item=>item.id===baselineLayerId),currentLayer=this.result.layers.find(item=>item.id===currentLayerId);if(!layer||!currentLayer)return;const currentSources=new Set(currentLayer.source_index||[]),selection=(layer.source_index||[]).map((value,i)=>currentSources.has(value)?-1:i).filter(i=>i>=0);if(!selection.length)return;const filtered={...layer,id:`baseline-${index}`,role:"baseline",points:selection.map(i=>layer.points[i]),source_index:selection.map(i=>layer.source_index[i]),curvature:selection.map(i=>layer.curvature?.[i]),response:selection.map(i=>layer.response?.[i]),height:selection.map(i=>layer.height?.[i])};const object=this.makePoints(filtered,true);object.visible=false;object.userData.layer=filtered;this.baselineObjects.set(index,object);this.root.add(object)})}
  }
  makePoints(layer,baseline=false){const values=layer.points||[];
    const role=layer.role||"candidate",isSurface=layer.kind==="surface";
    if(layer.display_role==="model"||layer.display_role==="model-keypoints"){
      const keypoints=layer.display_role==="model-keypoints",size=keypoints?SCENE3D_THEME.pointSize.keypointOverlayPixels:SCENE3D_THEME.pointSize.candidate,color=keypoints?SCENE3D_THEME.keypointRetained:(role==="rejected"?SCENE3D_THEME.rejected:SCENE3D_THEME.selected),object=createPointCloud({THREE,positions:values,color,size,opacity:1,depthWrite:!keypoints,renderOrder:keypoints?20:4});
      if(keypoints){object.material.sizeAttenuation=false;object.material.depthTest=false}
      object.userData.display={baseSize:size,baseOpacity:1,surface:false};return object;
    }
    if(isSurface){const colors=new Float32Array(values.length*3),hot=new THREE.Color(roleColors.candidate),cold=new THREE.Color(0x40505a);(layer.curvature||[]).forEach((value,index)=>{const color=cold.clone().lerp(hot,Math.max(0,Math.min(1,Number(value)/.12)));colors[index*3]=color.r;colors[index*3+1]=color.g;colors[index*3+2]=color.b});const object=createPointCloud({THREE,positions:values,colors,color:roleColors.surface,size:SCENE3D_THEME.pointSize.surface,opacity:.48,renderOrder:0}),curvatureColors=object.geometry.getAttribute("color");object.geometry.setAttribute("curvatureColor",curvatureColors);if(Array.isArray(layer.rgb_colors)&&layer.rgb_colors.length===values.length)object.geometry.setAttribute("rgbColor",new THREE.Float32BufferAttribute(layer.rgb_colors.flat(),3));object.material.vertexColors=false;object.userData.display={baseSize:SCENE3D_THEME.pointSize.surface,baseOpacity:.48,surface:true,curvatureColors,rgbColors:object.geometry.getAttribute("rgbColor")||null};return object}
    const volumeSizes={unknown:.017,free:.02,occupied:.027},size=baseline ? .045 : (volumeSizes[role]??(role==="fusion" ? SCENE3D_THEME.pointSize.fusion : (role==="retained" ? SCENE3D_THEME.pointSize.keypointRetained : .042))),volumeOpacity={unknown:.14,free:.3,occupied:.86},opacity=baseline ? .72 : (volumeOpacity[role]??(role==="fusion" ? .82 : .95)),renderOrder=["unknown","free","occupied"].includes(role)?0:(role==="fusion"?1:3);
    const object=createPointCloud({THREE,positions:values,colors:layer.colors||null,color:roleColors[role]||roleColors.candidate,size,opacity,depthWrite:role==="occupied",renderOrder});object.userData.display={baseSize:size,baseOpacity:opacity,surface:false};return object}
  makeLines(layer){const role=layer.role||"vector";return createPolyline({THREE,positions:(layer.segments||[]).flat(),color:roleColors[role]||roleColors.vector,opacity:role==="normal"?.82:.65,segments:true})}
  applyDisplaySettings(definition=activeStepDefinition()){
    this.displaySettings=sceneDisplaySettings(this.source.id);const final=isFinalSelection(this.result,resultStepForDefinition(this.result,definition)),settings=this.displaySettings,background=new THREE.Color(settings.backgroundColor||defaultSceneDisplay.backgroundColor);this.scene.background=background;this.renderer.setClearColor(background,1);
    const applyObject=object=>object.traverse?.(child=>{if(!child.isPoints)return;const display=child.userData.display||{baseSize:child.material.size,baseOpacity:child.material.opacity,surface:false};child.material.size=Math.max(.001,Number(display.baseSize||.02)*Number(settings.pointScale||1));child.material.opacity=Math.max(.02,Math.min(1,Number(display.baseOpacity??1)*Number(settings.opacity||1)));child.material.transparent=child.material.opacity<1;if(display.surface){const layer=child.userData.layer||object.userData.layer||{},automatic=layer.color_mode==="curvature"||(layer.id==="surface"&&state.stage==="keypoints"&&$("#curvature-heatmap").checked),requested=settings.colorMode||"auto",mode=requested==="auto"?(final&&display.rgbColors?"rgb":(automatic?"curvature":"neutral")):requested,attribute=mode==="rgb"?display.rgbColors:(mode==="curvature"?display.curvatureColors:null);if(final){child.material.opacity=(this.finalMode==="scene"?.95:.62)*Number(settings.opacity||1);child.material.transparent=child.material.opacity<1}if(attribute){child.geometry.setAttribute("color",attribute);child.material.vertexColors=true;child.material.color.set(0xffffff)}else{child.material.vertexColors=false;child.material.color.set(settings.surfaceColor||defaultSceneDisplay.surfaceColor)}}child.material.needsUpdate=true});
    this.objects.forEach(applyObject);this.baselineObjects.forEach(applyObject);this.queryObjects.forEach(applyObject);const stageGridVisible=state.stage!=="fusion"||["ground","normals","curvature"].includes(definition?.id);this.grid.visible=Boolean(settings.grid)&&stageGridVisible;
  }
  renderStep(definition){
    if(!this.result)return;
    const followHome=this.hasFit&&this.viewport.isAtHome(),step=resultStepForDefinition(this.result,definition),index=resultStepIndex(this.result,definition),visible=new Set(step?.visible_layers||[]),final=isFinalSelection(this.result,step),enteringFinal=final&&!this.finalSelectionWasActive;
    this.objects.forEach((object,id)=>{
      const layer=object.userData.layer||{},role=layer.role||"candidate",total=object.geometry?.getAttribute("position")?.count||0,count=step?.draw_counts?.[id];
      object.geometry?.setDrawRange(0,Number.isFinite(count)?Math.max(0,Math.min(total,count)):total);
      object.visible=visible.has(id)&&(final?finalSelectionLayerVisible(layer,this.finalMode,this.finalKeypoints):layerVisible(role));
    });
    this.baselineObjects.forEach((object,i)=>object.visible=Boolean(state.stage==="keypoints"&&layerVisible("baseline")&&this.baseline&&i===index));
    this.queryObjects.forEach(object=>object.visible=retrievalStages.includes(state.stage)&&!final);
    this.finalSelectionWasActive=final;this.syncFinalControls(final);this.applyDisplaySettings(definition);
    if(enteringFinal||followHome)this.fit();else this.viewport.updateClipping(this.root);
    this.updateHeader(definition);
  }
  updateHeader(definition=activeStepDefinition()){if(!this.result)return;const step=resultStepForDefinition(this.result,definition),metrics=this.result.metrics||{},fusion=this.result.stage==="fusion",partial=fusion&&metrics.preview_only,metricLabels=stepMetricLabels(step,this.result.stage),keptLabel=metricLabels.kept?.toLowerCase()||"retained",rejectedLabel=metricLabels.rejected?.toLowerCase()||"rejections";$(".scene-title span",this.card).textContent=fusion?`${formatInt(metrics.integrated_frames??metrics.frames)} / ${formatInt(metrics.source_frames??metrics.frames)} frames · ${formatTime(this.result.duration_seconds)}`:`${formatInt(this.result.scene.points)} points · ${formatTime(this.result.duration_seconds)}`;const status=$(".scene-status",this.card);status.className=`scene-status ${partial?"partial":"completed"}`;$("b",status).textContent=partial?"Partial preview":"Computed";const box=$(".scene-metrics",this.card);box.innerHTML=fusion?`<span>input <b>${formatInt(step?.input)}</b></span><span>output <b>${formatInt(step?.kept)}</b></span><span>difference <b>${formatInt(step?.rejected)}</b></span><span>moteur <b>${metrics.backend||"—"}</b></span>`:`<span>in <b>${formatInt(step?.input)}</b></span><span>${keptLabel} <b>${formatInt(step?.kept)}</b></span><span>${rejectedLabel} <b>${formatInt(step?.rejected)}</b></span><span>transmis <b>${formatInt(metrics.retained??metrics.descriptors)}</b></span>`}
  fit(){if(isFinalSelection(this.result,resultStepForDefinition(this.result))){if(this.finalMode==="scene")this.fitScene();else this.fitSelection();return}const visible=[...this.objects.values(),...this.baselineObjects.values(),...this.queryObjects.values()].filter(object=>object.visible);let fitted=this.viewport.fitObjects(visible,1.12,{remember:true});if(!fitted&&this.result?.bounds?.min&&this.result?.bounds?.max){const min=new THREE.Vector3(...this.result.bounds.min),max=new THREE.Vector3(...this.result.bounds.max),box=new THREE.Box3(min,max);if(validSceneBounds(box))fitted=this.viewport.fitBox(box,1.12,{remember:true})}if(fitted)this.fitRadius=this.viewport.fitRadius}
  pick(event){if(!this.result)return;const rect=this.canvas.getBoundingClientRect();this.pointer.x=((event.clientX-rect.left)/rect.width)*2-1;this.pointer.y=-((event.clientY-rect.top)/rect.height)*2+1;this.raycaster.setFromCamera(this.pointer,this.camera);const targets=[...this.objects.values()].filter(object=>object.visible&&object.isPoints&&object.userData.layer?.kind!=="surface");const hit=this.raycaster.intersectObjects(targets,false)[0];if(!hit)return;const layer=hit.object.userData.layer,index=hit.index;showPointDetail(this.source.id,layer,index)}
  dispose(){this.clearObjects();this.clearObjects(this.baselineObjects);this.clearQueryCandidates();this.viewport.dispose()}
}

function primaryLayerId(step,result){if(!step)return null;const layers=new Map((result.layers||[]).map(layer=>[layer.id,layer]));const candidates=(step.visible_layers||[]).filter(id=>["accepted","retained"].includes(layers.get(id)?.role));return candidates.at(-1)||null}
function syncCameraFrom(source){
  if(state.syncing||!source?.camera||!source?.controls||!$("#sync-cameras")?.checked)return;
  state.syncing=true;
  try{
    const offset=source.camera.position.clone().sub(source.controls.target),distance=Math.max(.01,offset.length()),direction=offset.normalize();
    state.viewers.forEach(viewer=>{
      if(viewer===source||!viewer?.camera||!viewer?.controls)return;
      const ratio=(viewer.fitRadius||1)/(source.fitRadius||1);
      viewer.camera.position.copy(viewer.controls.target).add(direction.clone().multiplyScalar(distance*ratio));
      viewer.camera.zoom=source.camera.zoom;
      viewer.camera.updateProjectionMatrix();
      viewer.controls.update();
    });
  }finally{state.syncing=false}
}

function showPointDetail(sceneId,layer,index){
  state.focusedScene=sceneId;
  const value=name=>layer[name]?.[index],data={source:value("source_index"),frame:value("frame_index"),curvature:value("curvature"),response:value("response"),selectionScore:value("selection_score"),height:value("height"),wallAffinity:value("wall_affinity"),wallScore:value("wall_score"),objectScore:value("object_score"),objectAnchor:value("object_anchor_score"),smallSupport:value("small_support"),largeSupport:value("large_support"),protrusion:value("protrusion_score"),discontinuity:value("discontinuity_score"),wallPlane:value("plane_index"),occupied:value("occupied_cells"),unknown:value("unknown_cells"),bestDistance:value("best_distance"),secondDistance:value("second_distance"),margin:value("descriptor_margin"),entropy:value("descriptor_entropy"),validModels:value("valid_model_count"),compatibleFeatures:value("compatible_feature_count"),bestModel:value("best_model"),bestSynset:value("best_synset"),groupId:value("group_id"),groupSize:value("group_size"),groupScore:value("group_score"),reason:value("reason")};
  $("#point-empty").hidden=true;$("#point-detail").hidden=false;$("#point-scene").textContent=state.current?.scenes?.[sceneId]?.scene?.label||sceneId;$("#point-layer").textContent=layer.label;$("#point-index").textContent=data.frame?`frame ${data.frame}`:(data.source??"—");$("#point-curvature").textContent=Number.isFinite(data.curvature)?data.curvature.toFixed(6):"—";$("#point-response").textContent=Number.isFinite(data.response)?data.response.toFixed(6):"—";$("#point-selection-score").textContent=Number.isFinite(data.selectionScore)?data.selectionScore.toFixed(6):"—";$("#point-height").textContent=Number.isFinite(data.height)?`${data.height.toFixed(3)} m`:"—";$("#point-wall-affinity").textContent=Number.isFinite(data.wallAffinity)?data.wallAffinity.toFixed(3):"—";$("#point-wall-score").textContent=Number.isFinite(data.wallScore)?data.wallScore.toFixed(3):"—";$("#point-object-score").textContent=Number.isFinite(data.objectScore)?data.objectScore.toFixed(3):"—";$("#point-object-anchor").textContent=Number.isFinite(data.objectAnchor)?data.objectAnchor.toFixed(3):"—";$("#point-wall-support").textContent=Number.isFinite(data.smallSupport)&&Number.isFinite(data.largeSupport)?`${data.smallSupport.toFixed(2)} / ${data.largeSupport.toFixed(2)}`:"—";$("#point-wall-object-detail").textContent=Number.isFinite(data.protrusion)&&Number.isFinite(data.discontinuity)?`${data.protrusion.toFixed(2)} / ${data.discontinuity.toFixed(2)}`:"—";$("#point-wall-plane").textContent=Number.isFinite(data.wallPlane)?`#${data.wallPlane+1}`:"—";
  $("#point-descriptor-occupancy").textContent=Number.isFinite(data.occupied)?`${data.occupied} / ${data.unknown??0}`:"—";$("#point-descriptor-distance").textContent=Number.isFinite(data.bestDistance)?`${data.bestDistance.toFixed(3)} / ${Number.isFinite(data.secondDistance)?data.secondDistance.toFixed(3):"∞"}`:"—";$("#point-descriptor-ambiguity").textContent=Number.isFinite(data.margin)?`${data.margin.toFixed(3)} / ${Number(data.entropy||0).toFixed(3)}`:"—";$("#point-descriptor-model-count").textContent=Number.isFinite(data.validModels)?`${data.validModels} models · ${data.compatibleFeatures??0} features`:"—";$("#point-descriptor-model").textContent=data.bestModel?`${data.bestModel} · ${data.bestSynset||"—"}`:"—";
  $("#point-reason").textContent=data.reason||(Number.isFinite(data.groupId)?`Group ${data.groupId} · ${data.groupSize??"?" } descriptors · score ${Number(data.groupScore||0).toFixed(3)}`:(layer.role==="fusion"?"Depth measurement projected into world coordinates":(layer.role==="wall"?"Compatible with a large vertical plane":(["accepted","retained"].includes(layer.role)?"Retained at this substep":"Candidate"))));activateInspectorTab("point");updateHistogram();
}

function setSceneStatus(sceneId,status,message){const viewer=state.viewers.get(sceneId);if(!viewer)return;const partial=status==="completed"&&viewer.result?.stage==="fusion"&&viewer.result?.metrics?.preview_only,badge=$(".scene-status",viewer.card);badge.className=`scene-status ${partial?"partial":status}`;$("b",badge).textContent=partial?"Partial preview":(message||status);const bar=$(".scene-progress i",viewer.card);bar.style.width=status==="completed"?"100%":status==="running"?"45%":"0"}
function runDisplayName(manifest){return manifest?.label||manifest?.id||"no computation"}
function setDebugLogs(lines,runId=null){state.logs=(Array.isArray(lines)?lines:[]).slice(-1000);state.logRunId=runId;state.logsCleared=false;renderDebugLogs()}
function appendDebugLog(line){if(!line)return;state.logs.push(String(line));state.logs=state.logs.slice(-1000);renderDebugLogs(true)}
function renderDebugLogs(stick=false){
  const pre=$("#debug-logs"),wasBottom=pre.scrollTop+pre.clientHeight>=pre.scrollHeight-20,count=state.logs.length;
  pre.textContent=state.logs.join("\n");$("#debug-log-count").textContent=`${count} line${count>1?"s":""}`;
  const badge=$("#debug-log-badge");badge.hidden=count===0;badge.textContent=count>999?"999+":String(count);
  if(stick&&wasBottom)pre.scrollTop=pre.scrollHeight;
}
function toggleDebugLogs(force){
  const workspace=$(".visual-workspace"),open=force===undefined?!workspace.classList.contains("logs-open"):Boolean(force),button=$("#toggle-debug-logs");
  workspace.classList.toggle("logs-open",open);button.setAttribute("aria-pressed",String(open));$("#debug-log-drawer").setAttribute("aria-hidden",String(!open));
  if(open)requestAnimationFrame(()=>{$("#debug-logs").scrollTop=$("#debug-logs").scrollHeight});
}
function applyCurrentResults(rawPayload,{switchStage=true}={}){
  const payload=normalizeDebugResult(rawPayload);
  const stage=payload?.manifest?.stage||Object.values(payload?.scenes||{})[0]?.stage||"keypoints";
  const stageChanged=stage!==state.stage,runChanged=payload?.manifest?.id!==state.current?.manifest?.id;
  state.stageResults.set(stage,payload);
  if(switchStage&&stageChanged){state.stage=stage;localStorage.setItem(stageStorage,stage);renderStageNavigation();renderSources();renderParameters()}
  clearQueryCandidates({render:false});state.current=payload;setDebugLogs(payload.logs||[],payload.manifest?.id||null);updateStageUi();$("#run-id").textContent=`${stageLabel()} · ${runDisplayName(payload.manifest)}`;applyManifestInputs(payload.manifest);ensureSceneCards(payload.manifest.sources);
  state.viewers.forEach(viewer=>{viewer.clearResult();if(stageChanged)viewer.hasFit=false});
  renderTimeline({preserveStep:true});
  if(stage==="selection"&&(stageChanged||runChanged)){const finalIndex=state.stageSteps.findIndex(step=>step.id==="final");if(finalIndex>=0)setStep(finalIndex)}
  for(const[sceneId,result]of Object.entries(payload.scenes||{})){const baseline=stage==="keypoints"?state.baseline?.scenes?.[sceneId]:null;state.viewers.get(sceneId)?.setResult(result,baseline)}
  $("#pin-baseline").disabled=stage!=="keypoints";renderHistory();updateTimelineMetrics();updateHistogram();updateComparison();renderQueryCandidates();renderQueryCandidateDetail();
}
function applyBaseline(rawPayload){const payload=rawPayload?normalizeDebugResult(rawPayload):null;if(payload?.manifest?.stage&&payload.manifest.stage!=="keypoints")return;state.baseline=payload;$("#baseline-label").textContent=payload?runDisplayName(payload.manifest):"No baseline";if(state.stage==="keypoints"&&state.current)for(const[sceneId,result]of Object.entries(state.current.scenes||{}))state.viewers.get(sceneId)?.setResult(result,payload?.scenes?.[sceneId]);updateComparison()}

function updateTimelineMetrics(){const definition=state.stageSteps[state.step]||{description:"Run or load a result to inspect this stage."};let input=0,kept=0,rejected=0,count=0,warning=false,keptLabel=null,rejectedLabel=null;for(const result of Object.values(state.current?.scenes||{})){const step=resultStepForDefinition(result,definition);if(!step)continue;const metricLabels=stepMetricLabels(step,result.stage);input+=Number(step.input||0);kept+=Number(step.kept||0);rejected+=Number(step.rejected||0);warning=warning||Boolean(step.warning)||Boolean(result.stage==="descriptors"&&["distance","ambiguity"].includes(step.id));keptLabel=keptLabel||metricLabels.kept;rejectedLabel=rejectedLabel||metricLabels.rejected;count++}const labels=$$(".timeline-summary span");if(labels.length>=3){labels[1].textContent=keptLabel||(state.stage==="fusion"?"Output":"Retained");labels[2].textContent=rejectedLabel||(state.stage==="fusion"?"Difference":"Rejected")}$("#timeline-input").textContent=count?formatInt(input):"—";$("#timeline-kept").textContent=count?formatInt(kept):"—";$("#timeline-rejected").textContent=count?formatInt(rejected):"—";const explanation=$("#timeline-explanation");explanation.textContent=definition.description;explanation.classList.toggle("warning",warning)}
function updateComparison(){const box=$("#comparison-metrics");box.replaceChildren();let added=0,removed=0,common=0,valid=0;const definition=activeStepDefinition();if(state.current&&state.baseline){for(const[sceneId,result]of Object.entries(state.current.scenes||{})){const base=state.baseline.scenes?.[sceneId],step=resultStepForDefinition(result,definition),baseStep=resultStepForDefinition(base,definition);if(!base||!step||!baseStep)continue;const currentLayer=result.layers.find(layer=>layer.id===primaryLayerId(step,result)),baseLayer=base.layers.find(layer=>layer.id===primaryLayerId(baseStep,base));if(!currentLayer||!baseLayer)continue;const currentSet=new Set(currentLayer.source_index||[]),baseSet=new Set(baseLayer.source_index||[]);currentSet.forEach(value=>baseSet.has(value)?common++:added++);baseSet.forEach(value=>{if(!currentSet.has(value))removed++});valid++}}
  [["Communs",common,""],["Added",added,"positive"],["Removed",removed,"negative"]].forEach(([label,value,className])=>{const item=document.createElement("div");item.innerHTML=`<span>${label}</span><strong class="${className}">${valid?formatInt(value):"—"}</strong>`;box.append(item)})}

function querySceneId(){return state.focusedScene&&state.current?.scenes?.[state.focusedScene]?state.focusedScene:Object.keys(state.current?.scenes||{})[0]}
function querySelection(sceneId=querySceneId()){if(!state.querySelections.has(sceneId))state.querySelections.set(sceneId,new Set());return state.querySelections.get(sceneId)}
function queryDetailKey(runId,sceneId,candidateIndex,groupId=selectedQueryGroup()){return`${runId}:${sceneId}:${candidateIndex}:${groupId}`}
function queryCandidateColor(candidateIndex){return queryCandidateColors[Math.abs(Number(candidateIndex)||0)%queryCandidateColors.length]}
function queryCandidateColorCss(candidateIndex){return`#${queryCandidateColor(candidateIndex).toString(16).padStart(6,"0")}`}
function queryGroups(){return state.current?.scenes?.[querySceneId()]?.query_groups||[]}
function selectedQueryGroup(){return state.queryGroupByScene.get(querySceneId())??"all"}
function syncQueryGroupFilter(){
  const select=$("#query-candidate-group");if(!select)return;
  const groups=queryGroups(),previous=selectedQueryGroup();select.replaceChildren();
  const all=document.createElement("option");all.value="all";all.textContent=groups.length?`Merged ranking · ${groups.length} objects`:"Classement global";select.append(all);
  for(const group of groups){const option=document.createElement("option");option.value=String(group.id);option.textContent=`Object ${group.id} · ${formatInt(group.descriptor_count)} descriptors · weight ${Number(group.query_weight??1).toFixed(2)} · ${formatInt(group.top_k?.length)} models`;select.append(option)}
  select.value=[...select.options].some(option=>option.value===String(previous))?String(previous):"all";state.queryGroupByScene.set(querySceneId(),select.value);
}
function allQueryCandidates(){
  const sceneId=querySceneId(),result=state.current?.scenes?.[sceneId],step=activeStepDefinition()?.id;
  if(result?.candidate_details)return result.top_k||[];
  if(step==="database"||step==="object_groups")return[];
  const groupId=selectedQueryGroup();
  if(groupId!=="all")return result?.query_groups?.find(group=>String(group.id)===String(groupId))?.top_k||[];
  return step==="global_pool"?(result?.candidate_pool||[]):(result?.top_k||[]);
}
function queryCandidateCategory(candidate){
  const labels={"03001627":"Chaise","04379243":"Table / desk","04256520":"Sofa"};
  return labels[String(candidate?.synset||"")]||String(candidate?.synset||"Autre");
}
function queryCandidateStatus(candidate){
  const sceneId=querySceneId(),runId=state.current?.manifest?.id,index=Number(candidate?.index),key=queryDetailKey(runId,sceneId,index);
  if(querySelection(sceneId).has(index))return"selected";
  if(candidate.status)return candidate.status.startsWith("rejected")?"rejected":candidate.status==="selected"?"retained":"accepted";
  if(state.queryLoading.has(key))return"loading";
  const detail=state.queryDetails.get(key);
  if(detail)return detail.placed?"placed":"loaded";
  return"pending";
}
function visibleQueryCandidates(){return queryFilterController?.filter(allQueryCandidates())||allQueryCandidates()}
function clearSceneQueryCandidates(sceneId,{render=true}={}){
  querySelection(sceneId).clear();state.querySelectionAnchors.delete(sceneId);state.viewers.get(sceneId)?.clearQueryCandidates();
  if(state.queryActiveCandidate!==null)state.queryActiveCandidate=null;
  if(render){renderQueryCandidates();renderQueryCandidateDetail()}
}
function clearQueryCandidates({render=true}={}){
  state.querySelections.clear();state.querySelectionAnchors.clear();state.queryActiveCandidate=null;state.viewers.forEach(viewer=>viewer.clearQueryCandidates());
  if(render){renderQueryCandidates();renderQueryCandidateDetail()}
}
function renderQueryCandidateDetail(){
  const panel=$("#query-candidate-detail"),empty=$("#query-candidate-detail-empty");if(!panel||!empty)return;
  const sceneId=querySceneId(),runId=state.current?.manifest?.id,index=state.queryActiveCandidate,key=queryDetailKey(runId,sceneId,index),detail=state.queryDetails.get(key),record=allQueryCandidates().find(candidate=>Number(candidate.index)===Number(index));
  panel.replaceChildren();
  if(index===null||index===undefined){panel.hidden=true;empty.hidden=false;empty.textContent=state.stage==="query"?"Select a candidate to compute correspondences and a provisional pose.":"Select a hypothesis to inspect its saved pose and diagnostics without recomputing.";return}
  empty.hidden=true;panel.hidden=false;
  if(!detail){const loading=document.createElement("p");loading.className="query-candidate-loading";loading.textContent="Computing correspondences and pose…";panel.append(loading);return}
  const header=document.createElement("header"),heading=document.createElement("div"),eyebrow=document.createElement("span"),title=document.createElement("strong"),hide=document.createElement("button");eyebrow.textContent=`Candidate #${detail.candidate_index}`;title.textContent=detail.name;heading.append(eyebrow,title);hide.type="button";hide.className="query-hide-all";hide.textContent="Hide all";hide.addEventListener("click",()=>clearQueryCandidates());header.append(heading,hide);
  const summary=document.createElement("div");summary.className="query-detail-metrics";
  const evidence=record?.group_evidence||{};
  [["Rang Top-k",record?.rank?`#${record.rank}`:"—"],["Pool rank",record?.pool_rank?`#${record.pool_rank}`:"—"],["Object query",detail.group_id==null?"Globale":`Group ${detail.group_id}`],["Group weight",record?.group_weight==null?"—":Number(record.group_weight).toFixed(2)],["Group descriptors",detail.group_descriptor_count==null?"—":formatInt(detail.group_descriptor_count)],["Matches distincts",evidence.matches==null?"—":formatInt(evidence.matches)],["Coverage",evidence.coverage==null?"—":`${Math.round(Number(evidence.coverage)*100)} %`],["Spatial extent",evidence.spatial_spread==null?"—":`${Math.round(Number(evidence.spatial_spread)*100)} %`],["Local distance",detail.local_score===null?"—":Number(detail.local_score).toFixed(4)],["Correspondences",formatInt(detail.correspondences)],["Constellations",formatInt(detail.constellations)],["Computation",formatTime(detail.elapsed_seconds)]].forEach(([label,value])=>{const item=document.createElement("div"),small=document.createElement("span"),strong=document.createElement("b");small.textContent=label;strong.textContent=value;item.append(small,strong);summary.append(item)});
  const pose=document.createElement("section");pose.className=`query-pose ${detail.placed?"placed":"missing"}`;const poseTitle=document.createElement("strong"),poseText=document.createElement("p"),support=detail.pose?.support_count??detail.pose?.inliers??0,supportLabel=detail.pose?.support_kind==="best_correspondences"?"best correspondences displayed":"inliers";poseTitle.textContent=detail.placed?"Provisional placement displayed":"No valid placement";poseText.textContent=detail.placed?`${detail.pose.source==="constellation"?"Best constellation":"Best isolated correspondence; no valid constellation"} · rotation ${detail.pose.theta_deg}° · scale ${detail.pose.scale} · ${support} ${supportLabel}. Lines connect placed model keypoints to scene keypoints.`:"No admissible correspondence supports a pose proposal in this scene.";pose.append(poseTitle,poseText);
  const matchSection=document.createElement("section");matchSection.className="query-match-detail";
  const matchTitle=document.createElement("strong");matchTitle.textContent=`Displayed correspondences (${detail.matches.length})`;
  const matches=document.createElement("div");
  for(const match of detail.matches){
    const chip=document.createElement("button");chip.type="button";chip.textContent=`M${match.model_keypoint} ↔ S${match.scan_keypoint}`;
    chip.dataset.tooltip=match.descriptor_distance==null?"Geometric pose support":`Distance ${match.descriptor_distance}`;
    const rotation=match.theta_deg??detail.pose?.theta_deg,scale=match.scale??detail.pose?.scale;
    chip.dataset.tooltipDetail=`${rotation==null||scale==null?"":`Rotation ${rotation}° · scale ${scale}. `}The colored line shows where this model keypoint lands in the scene.`;
    matches.append(chip);
  }
  if(!detail.matches.length){const none=document.createElement("p");none.textContent=detail.diagnostics?"No local pair displayed at this stage. Inspect diagnostics for decision details.":"No correspondence below the current thresholds.";matches.append(none)}
  matchSection.append(matchTitle,matches);
  if(detail.diagnostics){
    poseTitle.textContent=detail.placed?"Saved pose":"No pose";
    poseText.textContent=`${detail.explanation||detail.status} ${detail.pose?`Rotation ${detail.pose.theta_deg}° · scale ${Number(detail.pose.scale).toFixed(3)}`:"No admissible transformation."}`;
    if(detail.geometry_note)poseText.textContent+=` ${detail.geometry_note}`;
    if(isFinalSelection(state.current?.scenes?.[sceneId],activeStepDefinition()))hide.textContent="Close diagnostics";
    const diagnostics=document.createElement("details"),caption=document.createElement("summary"),body=document.createElement("pre");
    caption.textContent="Detailed diagnostics";body.textContent=JSON.stringify(detail.diagnostics,null,2);body.style.cssText="white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;max-height:320px;overflow:auto";diagnostics.append(caption,body);
    summary.replaceChildren();const measurement=detail.diagnostics.pose_verification||detail.diagnostics.registration||detail.diagnostics;
    for(const [key,label] of [["coverage","Coverage"],["cov_reverse","Coverage inverse"],["balanced_score","Score harmonique"],["mean_surface_dist_m","Surface distance (m)"],["correspondences","Correspondences"],["constellations","Constellations"],["conflict_model","Conflit"]]){const value=measurement[key]??detail.diagnostics[key];if(value==null)continue;const cell=document.createElement("div"),name=document.createElement("span"),number=document.createElement("b");name.textContent=label;number.textContent=typeof value==="number"?String(Math.round(value*1000)/1000):value;cell.append(name,number);summary.append(cell)}
    panel.append(header,pose,summary,diagnostics,matchSection);
  }else panel.append(header,summary,pose,matchSection);bindTooltips();
}
async function loadQueryCandidate(sceneId,candidate){
  const selected=querySelection(sceneId),candidateIndex=Number(candidate.index),viewer=state.viewers.get(sceneId),runId=state.current?.manifest?.id,key=queryDetailKey(runId,sceneId,candidateIndex);
  let detail=state.queryDetails.get(key)||state.current?.scenes?.[sceneId]?.candidate_details?.[candidateIndex];
  if(detail)state.queryDetails.set(key,detail);
  if(!detail&&!state.queryLoading.has(key)){
    state.queryLoading.add(key);
    renderQueryCandidates();renderQueryCandidateDetail();
    try{const groupId=selectedQueryGroup(),groupArgument=groupId==="all"?"":`&group_id=${encodeURIComponent(groupId)}`;detail=await api(`/api/debug/query-candidate?run_id=${encodeURIComponent(runId)}&scene_id=${encodeURIComponent(sceneId)}&candidate_index=${candidateIndex}${groupArgument}`);state.queryDetails.set(key,detail)}catch(error){selected.delete(candidateIndex);if(state.queryActiveCandidate===candidateIndex)state.queryActiveCandidate=null;notify(error.message,true)}finally{state.queryLoading.delete(key)}
  }
  if(detail&&selected.has(candidateIndex))viewer?.setQueryCandidate(candidateIndex,detail,queryCandidateColor(candidateIndex));
  renderQueryCandidates();renderQueryCandidateDetail();
}
function queueQueryCandidateLoads(sceneId,candidates){
  const unique=[...new Map(candidates.map(candidate=>[Number(candidate.index),candidate])).values()];
  const previous=state.queryQueues.get(sceneId)||Promise.resolve(),run=async()=>{for(const candidate of unique){if(querySelection(sceneId).has(Number(candidate.index)))await loadQueryCandidate(sceneId,candidate)}};
  const queued=previous.then(run,error=>{console.warn("Previous candidate queue interrupted",error);return run()});
  state.queryQueues.set(sceneId,queued);return queued;
}
function applyQueryCandidateSelection(sceneId,candidates,position,event={}){
  const selected=querySelection(sceneId),candidate=candidates[position];if(!candidate)return;
  const candidateIndex=Number(candidate.index),anchor=state.querySelectionAnchors.get(sceneId),additive=Boolean(event.additive??(event.ctrlKey||event.metaKey)),range=Boolean(event.range??event.shiftKey);
  let added=[];
  if(range&&Number.isInteger(anchor)){
    const start=Math.min(anchor,position),end=Math.max(anchor,position),range=candidates.slice(start,end+1);
    if(!additive)clearSceneQueryCandidates(sceneId,{render:false});
    for(const item of range){const index=Number(item.index);if(!selected.has(index)){selected.add(index);added.push(item)}}
    state.querySelectionAnchors.set(sceneId,anchor);
  }else if(additive){
    if(selected.has(candidateIndex)){selected.delete(candidateIndex);state.viewers.get(sceneId)?.removeQueryCandidate(candidateIndex)}else{selected.add(candidateIndex);added=[candidate]}
    state.querySelectionAnchors.set(sceneId,position);
  }else{
    clearSceneQueryCandidates(sceneId,{render:false});selected.add(candidateIndex);added=[candidate];state.querySelectionAnchors.set(sceneId,position);
  }
  state.queryActiveCandidate=selected.has(candidateIndex)?candidateIndex:([...selected].at(-1)??null);
  renderQueryCandidates();renderQueryCandidateDetail();if(added.length)queueQueryCandidateLoads(sceneId,added);
}
function selectAllQueryCandidates(){
  const sceneId=querySceneId(),candidates=visibleQueryCandidates(),selected=querySelection(sceneId);for(const candidate of candidates)selected.add(Number(candidate.index));
  if(candidates.length){state.queryActiveCandidate=Number(candidates[0].index);state.querySelectionAnchors.set(sceneId,0)}
  renderQueryCandidates();renderQueryCandidateDetail();queueQueryCandidateLoads(sceneId,candidates);
}
function updateHistogram(){const sceneId=state.focusedScene&&state.current?.scenes?.[state.focusedScene]?state.focusedScene:Object.keys(state.current?.scenes||{})[0],result=state.current?.scenes?.[sceneId],kind=$("#histogram-kind").value,svg=$("#histogram");svg.replaceChildren();if(!result?.histograms?.[kind])return;const hist=result.histograms[kind],counts=hist.counts,max=Math.max(1,...counts),width=320,height=150,pad={l:12,r:8,t:8,b:18},plotW=width-pad.l-pad.r,plotH=height-pad.t-pad.b;const ns="http://www.w3.org/2000/svg";const axis=document.createElementNS(ns,"line");axis.setAttribute("x1",pad.l);axis.setAttribute("x2",width-pad.r);axis.setAttribute("y1",height-pad.b);axis.setAttribute("y2",height-pad.b);axis.setAttribute("class","hist-axis");svg.append(axis);counts.forEach((count,index)=>{const rect=document.createElementNS(ns,"rect"),barW=plotW/counts.length,x=pad.l+index*barW,y=pad.t+plotH*(1-count/max);rect.setAttribute("x",x+.5);rect.setAttribute("y",y);rect.setAttribute("width",Math.max(1,barW-1));rect.setAttribute("height",height-pad.b-y);rect.setAttribute("class","hist-bar");svg.append(rect)});const thresholdInputs={curvature:"curvature_threshold",harris:"harris_threshold",wall_score:"wall_reject_threshold",object_score:"wall_reject_max_object_score",descriptor_distance:"utility_distance_threshold",descriptor_margin:"utility_margin_threshold",descriptor_entropy:"utility_entropy_threshold",valid_models:"utility_max_valid_models"},threshold=Number($(`#param-${thresholdInputs[kind]}`)?.value),min=hist.edges[0],maxEdge=hist.edges.at(-1);if(Number.isFinite(threshold)&&maxEdge>min){const x=pad.l+plotW*Math.max(0,Math.min(1,(threshold-min)/(maxEdge-min))),line=document.createElementNS(ns,"line");line.setAttribute("x1",x);line.setAttribute("x2",x);line.setAttribute("y1",pad.t);line.setAttribute("y2",height-pad.b);line.setAttribute("class","hist-threshold");svg.append(line)}}
function renderQueryCandidates(){
  const list=$("#query-candidate-list"),count=$("#query-candidate-count");
  if(!list||!count)return;
  list.replaceChildren();
  if(!retrievalStages.includes(state.stage)){count.textContent="0 candidates";return}
  const sceneId=querySceneId(),result=state.current?.scenes?.[sceneId],step=activeStepDefinition()?.id;syncQueryGroupFilter();const allCandidates=allQueryCandidates();queryFilterController?.syncCategories(allCandidates);const candidates=visibleQueryCandidates();
  const selected=querySelection(sceneId),runId=state.current?.manifest?.id,visibleSelected=candidates.filter(candidate=>selected.has(Number(candidate.index))).length,loaded=[...selected].filter(index=>state.queryDetails.has(queryDetailKey(runId,sceneId,index))).length,loading=[...selected].filter(index=>state.queryLoading.has(queryDetailKey(runId,sceneId,index))).length;count.textContent=`${formatInt(candidates.length)}/${formatInt(allCandidates.length)} visible${selected.size?` · ${selected.size} selected${selected.size>1?"s":""}`:""}`;
  const progress=$("#query-selection-progress"),showAll=$("#query-show-all"),hideAll=$("#query-hide-all");if(progress)progress.textContent=loading?`${loaded}/${selected.size} computed · computing…`:(selected.size?`${loaded}/${selected.size} poses computed · Ctrl/Shift to change selection`:(step==="global_pool"?"Global pool: check whether a desk family survives before reranking.":step==="top_k"?"Final Top-k: only these models are passed to full matching.":"Click · Ctrl · Shift · long press then drag"));if(showAll)showAll.disabled=!candidates.length||visibleSelected===candidates.length;if(hideAll)hideAll.disabled=!selected.size;
  if(!candidates.length){const empty=document.createElement("p");empty.className="query-candidate-empty";empty.textContent=step==="database"?`${formatInt(result?.metrics?.database_models||0)} indexed models. Advance to build the shortlist.`:"No candidates available.";list.append(empty);return}
  candidates.forEach((candidate,position)=>{const row=document.createElement("button"),swatch=document.createElement("i"),rank=document.createElement("b"),copy=document.createElement("div"),name=document.createElement("strong"),meta=document.createElement("span"),selectedCandidate=selected.has(Number(candidate.index)),candidateLoading=state.queryLoading.has(queryDetailKey(runId,sceneId,candidate.index)),filteredGroup=selectedQueryGroup(),groupId=filteredGroup!=="all"?filteredGroup:candidate.best_group_id,groupRank=(candidate.group_hits||[]).find(hit=>String(hit.group_id)===String(groupId))?.rank,evidence=candidate.group_evidence||{},evidenceText=evidence.matches==null?"":` ${evidence.matches} distinct matches · coverage ${Math.round(Number(evidence.coverage||0)*100)}% · extent ${Math.round(Number(evidence.spatial_spread||0)*100)}%.`;row.type="button";row.className=`query-candidate-row ${selectedCandidate?"selected":""} ${candidateLoading?"loading":""}`;row.setAttribute("aria-pressed",String(selectedCandidate));row.dataset.sceneId=sceneId;row.dataset.queryPosition=String(position);row.dataset.tooltip=selectedCandidate?"Selected candidate":"Select and place";row.dataset.tooltipDetail=groupId==null?"This model comes from the global ranking.":`This model is displayed for object query ${groupId}${groupRank?` at global rank ${groupRank}`:""}. Group weight ${Number(candidate.group_weight??1).toFixed(2)}.${evidenceText} Placement compares only descriptors from this group.`;swatch.style.background=queryCandidateColorCss(candidate.index);rank.textContent=`#${candidate.rank}`;name.textContent=candidate.name;meta.textContent=`${candidate.status||candidate.synset} · ${candidate.pool_rank==null?"saved pose":`rang pool ${candidate.pool_rank}`}${groupId==null?"":` · groupe ${groupId}${groupRank?` #${groupRank}`:""}`}${candidateLoading?" · computing…":""}`;copy.append(name,meta);row.append(swatch,rank,copy);list.append(row)});
  if(result?.candidate_details){
    list.querySelectorAll(".query-candidate-row").forEach((row,position)=>{
      const detail=result.candidate_details[candidates[position].index];
      row.dataset.tooltipDetail=`${detail?.explanation||detail?.status||"Saved hypothesis"} Inspecting the saved pose without new matching.`;
    });
  }
  if(isFinalSelection(result,activeStepDefinition())){
    if(progress)progress.textContent="Final view: only retained models are displayed. Click a hypothesis to inspect its diagnostics.";
    if(showAll)showAll.disabled=true;if(hideAll)hideAll.disabled=true;
    list.querySelectorAll(".query-candidate-row").forEach(row=>row.dataset.tooltip="Read saved diagnostics");
  }
  bindTooltips();
}

function updateRunState(snapshot){state.status=snapshot.status;if(snapshot.run_id&&snapshot.run_id!==state.logRunId&&!state.logsCleared)setDebugLogs(snapshot.logs||[],snapshot.run_id);const dot=$("#status-dot");dot.className=`status-dot ${snapshot.status}`;$("#status-label").textContent={idle:"Ready",running:"Computing",completed:"Completed",failed:"Partiel",stopped:"Stopped"}[snapshot.status]||snapshot.status;$("#status-detail").textContent=snapshot.message||"";$("#run-debug").disabled=snapshot.status==="running";$("#stop-debug").disabled=snapshot.status!=="running"&&!state.runUntilActive;if(!snapshot.stage||snapshot.stage===state.stage)for(const[sceneId,scene]of Object.entries(snapshot.scenes||{}))setSceneStatus(sceneId,scene.status,scene.message);if(snapshot.status!=="running"&&state.pendingAuto){state.pendingAuto=false;setTimeout(runDebug,200)}}
async function runDebug(){try{const snapshot=await api("/api/debug/run",{method:"POST",body:JSON.stringify(collectPayload())});updateRunState(snapshot);$("#run-id").textContent=`${stageLabel()} · ${snapshot.run_id}`;return snapshot}catch(error){state.runUntilActive=false;renderChainControls();notify(error.message,true);throw error}}
async function applyExecutionParameters(){
  const button=$("#apply-execution-parameters"),status=$("#parameter-sync-status");
  button.disabled=true;
  try{
    const payload=await api("/api/debug/apply-execution-parameters",{method:"POST",body:JSON.stringify({stage:state.stage,parameters:collectPayload().parameters})});
    const applied=Object.keys(payload.applied||{}).length,ignored=(payload.ignored||[]).length;
    status.textContent=`${applied} settings saved${ignored?` · ${ignored} option${ignored>1?"s":""} Debug option ignored${ignored>1?"s":""}`:""}.`;
    status.classList.add("success");
    localStorage.setItem("objectsensing-execution-preset-updated",String(Date.now()));
    notify(`Run Console updated · ${applied} parameters applied`);
  }catch(error){
    status.textContent=error.message;status.classList.remove("success");notify(error.message,true);
  }finally{button.disabled=false}
}
async function stopDebug(){state.runUntilActive=false;state.pendingAuto=false;clearTimeout(state.autoTimer);renderChainControls();try{await api("/api/debug/stop",{method:"POST",body:"{}"})}catch(error){notify(error.message,true)}}
async function poll(){try{const payload=await api(`/api/debug/events?after=${state.eventId}`);for(const event of payload.events){state.eventId=Math.max(state.eventId,event.id||0);handleEvent(event)}updateRunState(payload.state)}catch(error){$("#status-detail").textContent="Serveur indisponible"}}
function handleEvent(event){
  const data=event.data||{},eventStage=data.stage||data.result?.stage;
  if(event.type==="run_started"){
    if(eventStage!==state.stage)return;
    setDebugLogs([],data.run_id);
    state.current={manifest:{id:data.run_id,stage:eventStage,sources:collectPayload().sources},scenes:{}};
    state.stageResults.set(eventStage,state.current);state.viewers.forEach(viewer=>viewer.clearResult());renderTimeline();$("#run-id").textContent=`${stageLabel()} · ${data.run_id}`;
  }else if(event.type==="debug_log"){
    if(eventStage===state.stage&&(!state.logRunId||data.run_id===state.logRunId))appendDebugLog(data.message);
  }else if(event.type==="scene_started"){
    if(eventStage===state.stage)setSceneStatus(data.scene.id,"running","Computation");
  }else if(event.type==="scene_completed"){
    if(eventStage!==state.stage)return;
    state.current=state.current||{manifest:{id:data.run_id,stage:state.stage,sources:defaultSources()},scenes:{}};
    state.current.scenes[data.scene_id]=data.result;state.stageResults.set(state.stage,state.current);renderTimeline({preserveStep:true});
    state.viewers.get(data.scene_id)?.setResult(data.result,state.stage==="keypoints"?state.baseline?.scenes?.[data.scene_id]:null);
    setSceneStatus(data.scene_id,"completed","Computed");updateTimelineMetrics();updateHistogram();updateComparison();
  }else if(event.type==="scene_failed"){
    if(eventStage===state.stage){setSceneStatus(data.scene_id,"failed","Failed");notify(`${data.scene_id}: ${data.message}`,true)}
  }else if(event.type==="run_completed"){
    if(eventStage===state.stage){(async()=>{try{await loadRun(data.run_id);await refreshDefaults();if(data.status==="completed")await continueRunUntil(data.run_id,eventStage);else{state.runUntilActive=false;renderChainControls()}}catch(error){state.runUntilActive=false;renderChainControls();reportLoadError("Cannot load or chain the completed run",error)}})()}else refreshDefaults().catch(error=>reportLoadError("Cannot refresh history",error));
  }else if(event.type==="code_changed")notify("Code changed: recomputing");
  else if(event.type==="baseline_changed")loadBaseline(data.run_id).catch(error=>reportLoadError("Cannot load baseline",error));
  else if(event.type==="run_metadata_changed"){refreshDefaults().catch(error=>reportLoadError("Cannot refresh history",error));if(state.current?.manifest?.id===data.run_id)loadRun(data.run_id).catch(error=>reportLoadError("Cannot load modified run",error))}
}

async function loadRun(runId){const payload=await api(`/api/debug/result?id=${encodeURIComponent(runId)}`);applyCurrentResults(payload);return payload}
async function loadBaseline(runId){if(!runId){applyBaseline(null);return}const payload=await api(`/api/debug/result?id=${encodeURIComponent(runId)}`);applyBaseline(payload)}
async function refreshDefaults(){const fresh=await api("/api/debug/defaults");state.defaults={...state.defaults,...fresh};renderStageNavigation();renderHistory();renderChainControls()}
function openDebugRename(runId){const run=(state.defaults.recent_runs||[]).find(item=>item.id===runId);if(!run)return;state.renameRunId=runId;$("#debug-rename-input").value=run.label||run.id;$("#debug-rename-dialog").hidden=false;$("#debug-rename-input").focus();$("#debug-rename-input").select()}
function closeDebugRename(){state.renameRunId=null;$("#debug-rename-dialog").hidden=true}
async function submitDebugRename(event){event.preventDefault();if(!state.renameRunId)return;try{const payload=await api("/api/debug/run/rename",{method:"POST",body:JSON.stringify({run_id:state.renameRunId,label:$("#debug-rename-input").value})});if(state.current?.manifest?.id===state.renameRunId)applyCurrentResults(payload);closeDebugRename();await refreshDefaults();notify("Run renamed")}catch(error){notify(error.message,true)}}
async function toggleDebugFavorite(run){const favorite=!run.favorite;try{const payload=await api("/api/debug/run/favorite",{method:"POST",body:JSON.stringify({run_id:run.id,favorite})});if(state.current?.manifest?.id===run.id)applyCurrentResults(payload);await refreshDefaults();notify(favorite?"Run added to favorites":"Run removed from favorites")}catch(error){notify(error.message,true)}}
function renderHistory(){const list=$("#history-list");list.replaceChildren();for(const run of state.defaults.recent_runs||[]){const row=document.createElement("div");row.className=`history-item ${state.current?.manifest?.id===run.id?"active":""} ${run.favorite?"favorite":""}`;const load=document.createElement("button");load.type="button";load.className="history-load";const runStage=stageLabel(run.stage||"keypoints"),runDate=new Date((run.created_at||0)*1000).toLocaleString("en-US"),sceneCount=Number(run.scene_count||0),chainInfo=run.chainable&&run.next_stage?` · Can chain to ${stageLabel(run.next_stage)} (${run.chainable_scene_count}/${sceneCount})`:"";load.dataset.tooltip=run.label||run.id;load.dataset.tooltipDetail=`Stage: ${runStage} · Status: ${run.status||"unknown"} · ${sceneCount} scene${sceneCount>1?"s":""} · Created: ${runDate} · ID: ${run.id}${chainInfo}`;const copy=document.createElement("div"),title=document.createElement("strong"),meta=document.createElement("span"),status=document.createElement("b");title.textContent=run.label||run.id;const identifier=run.label&&run.label!==run.id?`${run.id} · `:"";meta.textContent=`${runStage} · ${identifier}${sceneCount} scene${sceneCount>1?"s":""} · ${runDate}`;status.textContent=run.status;copy.append(title,meta);load.append(copy,status);load.addEventListener("click",async()=>{try{await loadRun(run.id)}catch(error){notify(error.message,true)}});const actions=document.createElement("div");actions.className="history-actions";if(run.chainable&&run.next_stage){const chain=document.createElement("button");chain.type="button";chain.className="history-action";chain.setAttribute("aria-label",`Use for ${stageLabel(run.next_stage)}`);chain.dataset.tooltip=`Chain to ${stageLabel(run.next_stage)}`;chain.dataset.tooltipDetail=`Use the ${run.chainable_scene_count} complete artifacts from this run as explicit inputs.`;chain.innerHTML='<i data-lucide="git-merge"></i>';chain.addEventListener("click",async()=>{try{await chainStageRun(run.id,run.next_stage)}catch(error){notify(error.message,true)}});actions.append(chain)}const favorite=document.createElement("button");favorite.type="button";favorite.className=`history-action favorite-action ${run.favorite?"active":""}`;favorite.setAttribute("aria-label",run.favorite?"Remove from favorites":"Add to favorites");favorite.setAttribute("aria-pressed",String(Boolean(run.favorite)));favorite.dataset.tooltip=run.favorite?"Remove from favorites":"Add to favorites";favorite.dataset.tooltipDetail="Pin this run at the top of the list and highlight it.";favorite.innerHTML='<i data-lucide="star"></i>';favorite.addEventListener("click",()=>toggleDebugFavorite(run));const rename=document.createElement("button");rename.type="button";rename.className="history-action";rename.setAttribute("aria-label","Rename run");rename.dataset.tooltip="Rename";rename.dataset.tooltipDetail="Change the display name while preserving the technical directory.";rename.innerHTML='<i data-lucide="pencil"></i>';rename.addEventListener("click",()=>openDebugRename(run.id));actions.append(favorite,rename);row.append(load,actions);list.append(row)}refreshIcons()}

function activateInspectorTab(name){$$('.inspector-tabs button').forEach(button=>button.classList.toggle("active",button.dataset.tab===name));$$('.inspector-content').forEach(panel=>panel.classList.toggle("active",panel.dataset.panel===name))}
function bindUi(){
  $("#chain-start").addEventListener("click",async()=>{try{await startToSelectedStage()}catch(error){notify(error.message,true)}});
  $("#run-debug").addEventListener("click",runDebug);$("#stop-debug").addEventListener("click",stopDebug);$("#reset-stage-parameters").addEventListener("click",resetStageParameters);$("#apply-execution-parameters").addEventListener("click",applyExecutionParameters);$("#previous-step").addEventListener("click",()=>setStep(state.step-1));$("#next-step").addEventListener("click",()=>setStep(state.step+1));
  $("#toggle-debug-logs").addEventListener("click",()=>toggleDebugLogs());$("#clear-debug-logs").addEventListener("click",()=>{state.logs=[];state.logsCleared=true;renderDebugLogs()});
  document.addEventListener("pointerdown",event=>{if(!event.target.closest(".scene-display-popover,.scene-display"))state.viewers.forEach(viewer=>viewer.setDisplayPopoverOpen(false))});
  document.addEventListener("keydown",event=>{if(event.key==="Escape"&&!$("#step-help-dialog").hidden){closeStepHelp();return}if(event.key==="Escape"&&!$("#debug-rename-dialog").hidden){closeDebugRename();return}if(event.key==="Escape"&&!$("#parameter-help-dialog").hidden){closeParameterHelp();return}const displayOpen=$$(".scene-display-popover").find(popover=>!popover.hidden);if(event.key==="Escape"&&displayOpen){state.viewers.forEach(viewer=>viewer.setDisplayPopoverOpen(false));return}if(event.key==="Escape"&&$(".visual-workspace").classList.contains("logs-open")){toggleDebugLogs(false);return}if(event.key==="Escape"&&state.featuredScene&&!document.fullscreenElement){setFeaturedScene(state.featuredScene);return}if(["INPUT","SELECT","TEXTAREA"].includes(document.activeElement?.tagName))return;const jump=event.shiftKey?10:1;if(event.key==="ArrowLeft"){event.preventDefault();setStep(state.step-jump)}if(event.key==="ArrowRight"){event.preventDefault();setStep(state.step+jump)}if(event.key==="Home"){event.preventDefault();setStep(0)}if(event.key==="End"){event.preventDefault();setStep(state.stageSteps.length-1)}});
  document.addEventListener("fullscreenchange",()=>{state.viewers.forEach(viewer=>{viewer.updateViewActions();viewer.resize()});bindTooltips()});
  bindLayerVisibility();
  queryFilterController=createCandidateFilterController({search:"#query-candidate-search",category:"#query-candidate-category",status:"#query-candidate-status",reset:"#reset-query-candidate-filters",getCategory:queryCandidateCategory,getStatus:queryCandidateStatus,getSearchText:candidate=>`${candidate.name} ${candidate.synset} ${queryCandidateCategory(candidate)} rang ${candidate.rank} pool ${candidate.pool_rank}`,onChange:()=>renderQueryCandidates()});
  $("#query-candidate-group").addEventListener("change",event=>{state.queryGroupByScene.set(querySceneId(),event.target.value);queryFilterController.reset();renderQueryCandidates();renderQueryCandidateDetail()});
  bindMultiSelectListInteractions({container:"#query-candidate-list",itemSelector:".query-candidate-row",getPosition:item=>Number(item.dataset.queryPosition),onActivate:(position,interaction)=>applyQueryCandidateSelection(querySceneId(),visibleQueryCandidates(),position,interaction),onPaint:(position)=>{const sceneId=querySceneId(),candidate=visibleQueryCandidates()[position];if(candidate)querySelection(sceneId).add(Number(candidate.index))},onPaintEnd:positions=>{const sceneId=querySceneId(),candidates=visibleQueryCandidates(),added=positions.map(position=>candidates[position]).filter(Boolean);if(added.length){state.queryActiveCandidate=Number(added.at(-1).index);state.querySelectionAnchors.set(sceneId,positions.at(-1));renderQueryCandidates();renderQueryCandidateDetail();queueQueryCandidateLoads(sceneId,added)}}});
  $("#reset-cameras").addEventListener("click",()=>{state.syncing=true;state.viewers.forEach(viewer=>{viewer.hasFit=false;viewer.fit();viewer.hasFit=true});state.syncing=false});
  $("#code-watch").addEventListener("change",async event=>{try{await api("/api/debug/watch",{method:"POST",body:JSON.stringify({enabled:event.target.checked})})}catch(error){event.target.checked=!event.target.checked;notify(error.message,true)}});
  $("#pin-baseline").addEventListener("click",async()=>{if(!state.current?.manifest?.id)return;try{await api("/api/debug/baseline",{method:"POST",body:JSON.stringify({run_id:state.current.manifest.id})});await loadBaseline(state.current.manifest.id);notify("Baseline pinned")}catch(error){notify(error.message,true)}});
  $("#histogram-kind").addEventListener("change",updateHistogram);$("#refresh-history").addEventListener("click",refreshDefaults);
  $("#query-show-all").addEventListener("click",selectAllQueryCandidates);$("#query-hide-all").addEventListener("click",()=>clearSceneQueryCandidates(querySceneId()));
  $("#upstream-fusion-run").addEventListener("change",event=>renderChainControls(event.target.value));$("#chain-fusion-run").addEventListener("click",async()=>{try{await chainStageRun($("#upstream-fusion-run").value,state.stage)}catch(error){notify(error.message,true)}});$("#chain-target-stage").addEventListener("change",event=>{state.runUntilStage=event.target.value;renderChainControls()});$("#chain-current-fusion").addEventListener("click",async()=>{try{await runToSelectedStage()}catch(error){notify(error.message,true)}});
  for(const id of ["#close-parameter-help","#confirm-parameter-help"])$(id).addEventListener("click",closeParameterHelp);$("#parameter-help-dialog").addEventListener("pointerdown",event=>{if(event.target===$("#parameter-help-dialog"))closeParameterHelp()});
  $("#open-step-help").addEventListener("click",openStepHelp);for(const id of ["#close-step-help","#confirm-step-help"])$(id).addEventListener("click",closeStepHelp);$("#step-help-dialog").addEventListener("pointerdown",event=>{if(event.target===$("#step-help-dialog"))closeStepHelp()});
  $("#debug-rename-form").addEventListener("submit",submitDebugRename);$("#close-debug-rename").addEventListener("click",closeDebugRename);$("#cancel-debug-rename").addEventListener("click",closeDebugRename);$("#debug-rename-dialog").addEventListener("pointerdown",event=>{if(event.target===$("#debug-rename-dialog"))closeDebugRename()});
  $$('.inspector-tabs button').forEach(button=>button.addEventListener("click",()=>activateInspectorTab(button.dataset.tab)));
  $("#toggle-left").addEventListener("click",()=>{const app=$("#debug-app");app.classList.toggle("left-collapsed");app.classList.remove("left-peeking");updatePanelToggles()});$("#toggle-right").addEventListener("click",()=>{const app=$("#debug-app");app.classList.toggle("right-collapsed");app.classList.remove("right-peeking");updatePanelToggles()});
  bindLayoutResizers();bindHoverPanels();bindTooltips();updatePanelToggles();
}

async function init(){
  if(innerWidth<=1000)$("#debug-app").classList.add("left-collapsed","right-collapsed");state.sceneWeights=loadSceneWeights();state.sceneDisplaySettings=loadSceneDisplaySettings();bindUi();state.defaults=await api("/api/debug/defaults");
  const currentMeta=(state.defaults.recent_runs||[]).find(run=>run.id===state.defaults.current_run_id),rememberedStage=localStorage.getItem(stageStorage),rememberedDefinition=stageDefinition(rememberedStage);state.stage=rememberedDefinition?.available?rememberedStage:(currentMeta?.stage||"keypoints");
  for(const source of defaultSources())for(const[stage,path]of Object.entries(source.stage_paths||{})){const key=sourcePathKey(stage,source.id);state.sourcePaths.set(key,path);state.sourceEnabled.set(key,source.enabled!==false)}
  renderStageNavigation();renderSources();renderParameters();renderTimeline();updateStageUi();$("#code-watch").checked=state.defaults.watch_enabled;renderHistory();refreshIcons();
  const latestForStage=(state.defaults.recent_runs||[]).find(run=>(run.stage||"keypoints")===state.stage),initialRunId=currentMeta?.stage===state.stage?state.defaults.current_run_id:(latestForStage?.id||state.defaults.current_run_id);
  const snapshot=await api("/api/debug/state");state.eventId=snapshot.last_event_id||0;updateRunState(snapshot);
  if(initialRunId){$("#run-id").textContent=`${stageLabel()} · loading ${initialRunId}…`;await loadRun(initialRunId).catch(error=>reportLoadError("Cannot load initial run",error))}
  if(state.defaults.baseline_run_id)loadBaseline(state.defaults.baseline_run_id).catch(error=>reportLoadError("Cannot load baseline",error));
  setInterval(poll,700);
}
init().catch(error=>notify(error.message,true));

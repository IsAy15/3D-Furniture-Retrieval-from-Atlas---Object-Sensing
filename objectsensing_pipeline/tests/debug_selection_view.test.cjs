const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/debug/debug_visualizer.js'), 'utf8');
function viewerHarness() {
  const context = vm.createContext({
    state: {stage: 'selection'}, retrievalStages: ['selection'],
    layerVisible: () => true,
    resultStepForDefinition: (result, definition) => result.steps.find(step => step.id === definition.id),
    resultStepIndex: () => 0,
    activeStepDefinition: () => ({id: 'final'}),
    $: () => ({}),
  });
  vm.runInContext(source.slice(source.indexOf('function isFinalSelection('), source.indexOf('function savedLayerVisibility(')), context);
  vm.runInContext(source.slice(source.indexOf('class SceneViewer{'), source.indexOf('function primaryLayerId(')) + '\nthis.Viewer=SceneViewer;', context);
  const layers = [
    {id: 'surface', role: 'surface'},
    {id: 'pose-0', display_role: 'model'},
    {id: 'pose-1', display_role: 'model', role: 'rejected'},
    {id: 'pose-0-keypoints', display_role: 'model-keypoints'},
    {id: 'query_keypoints', role: 'candidate'},
  ];
  const viewer = Object.create(context.Viewer.prototype);
  Object.assign(viewer, {
    result: {stage: 'selection', steps: [
      {id: 'pose-1', visible_layers: ['surface', 'pose-1']},
      {id: 'final', visible_layers: ['surface', 'pose-0', 'pose-0-keypoints']},
    ]},
    finalMode: 'both', finalKeypoints: false, hasFit: false,
    objects: new Map(layers.map(layer => [layer.id, {visible: true, userData: {layer}}])),
    baselineObjects: new Map(), queryObjects: new Map([[1, {visible: true}]]),
    viewport: {updateClipping() {}},
    syncFinalControls() {}, applyDisplaySettings() {}, fit() {}, updateHeader() {},
  });
  const visible = () => [...viewer.objects].filter(([, object]) => object.visible).map(([id]) => id);
  return {viewer, visible, context};
}

test('final modes show only retained geometry and keep model keypoints optional', () => {
  const {viewer, visible} = viewerHarness();
  viewer.renderStep({id: 'final'});
  assert.deepEqual(visible(), ['surface', 'pose-0']);
  assert.equal(viewer.queryObjects.get(1).visible, false, 'previously inspected rejected candidate is hidden');
  viewer.finalMode = 'models'; viewer.finalKeypoints = true;
  viewer.renderStep({id: 'final'});
  assert.deepEqual(visible(), ['pose-0', 'pose-0-keypoints']);
  viewer.finalMode = 'scene'; viewer.renderStep({id: 'final'});
  assert.deepEqual(visible(), ['surface']);
  viewer.renderStep({id: 'pose-1'});
  assert.deepEqual(visible(), ['surface', 'pose-1']);
  assert.equal(viewer.queryObjects.get(1).visible, true, 'hypotheses remain available in diagnostic steps');
});

test('empty final selection shows the scene and never revives an inspected hypothesis', () => {
  const {viewer, visible} = viewerHarness();
  viewer.result.steps[1].visible_layers = ['surface'];
  viewer.renderStep({id: 'final'});
  assert.deepEqual(visible(), ['surface']);
  assert.equal(viewer.queryObjects.get(1).visible, false);
});

test('entering the final view frames it once and preserves subsequent user camera movement', () => {
  const {viewer} = viewerHarness();
  let fits = 0;
  viewer.fit = () => {fits++;};
  viewer.hasFit = true;
  viewer.viewport.isAtHome = () => false;
  viewer.renderStep({id: 'final'});
  assert.equal(fits, 1);
  viewer.finalKeypoints = true;
  viewer.renderStep({id: 'final'});
  assert.equal(fits, 1, 'display toggle must preserve an adjusted viewpoint');
  viewer.viewport.isAtHome = () => true;
  viewer.renderStep({id: 'pose-1'});
  assert.equal(fits, 2, 'an unchanged home camera follows the current diagnostic step');
});

test('loading another final result resets its framing after a manual camera movement', () => {
  const {viewer, context} = viewerHarness();
  context.state.syncing = false;
  let fits = 0;
  Object.assign(viewer, {
    hasFit: true, finalSelectionWasActive: true,
    fit: () => {assert.equal(context.state.syncing, true); fits++;}, clearObjects() {}, clearQueryCandidates() {},
    buildLayers() {}, syncDisplayControls() {},
  });
  viewer.viewport.isAtHome = () => false;
  const nextResult = {...viewer.result};
  viewer.setResult(nextResult);
  assert.equal(fits, 1, 'new result is reframed even after a manual zoom');
  assert.equal(context.state.syncing, false, 'linked camera motion is restored after loading');
  viewer.setResult(nextResult);
  assert.equal(fits, 1, 'refreshing the same result preserves the camera');
});

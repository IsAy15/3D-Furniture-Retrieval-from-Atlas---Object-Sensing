const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../web/debug/debug_visualizer.js'), 'utf8');
function harness(sources) {
  const calls = [];
  const context = vm.createContext({
    state: {stage: 'fusion', status: 'idle', current: null, runUntilActive: false},
    collectPayload: () => ({sources}),
    $: () => ({value: 'selection'}),
    renderChainControls() {}, clearTimeout() {},
    runDebug: async () => calls.push('run'),
    notify() {}, stageLabel: value => value,
    executableStages: ['fusion', 'keypoints', 'descriptors', 'query', 'matching', 'verification', 'selection'],
    nextStageByStage: {fusion: 'keypoints', keypoints: 'descriptors'},
    chainStageRun: async (...args) => calls.push(args),
  });
  vm.runInContext(source.slice(source.indexOf('async function startToSelectedStage()'), source.indexOf('function chainFusionRun(')), context);
  vm.runInContext(source.slice(source.indexOf('async function continueRunUntil('), source.indexOf('async function runToSelectedStage(')), context);
  return {context, calls};
}

test('fresh fusion chain needs no prior run and keeps only checked scenes', async () => {
  const {context, calls} = harness([
    {id: 'office', path: 'office.zip', enabled: true},
    {id: 'single-chair', path: 'single-chair.zip', enabled: true},
    {id: 'ikea-table', path: 'ikea.zip', enabled: false},
  ]);
  await context.startToSelectedStage();
  assert.equal(context.state.runUntilActive, true);
  assert.deepEqual(Array.from(context.state.runUntilSceneIds), ['office', 'single-chair']);
  assert.deepEqual(calls, ['run']);
  await context.continueRunUntil('fusion-test', 'fusion');
  assert.deepEqual(Array.from(calls[1][2]), ['office', 'single-chair']);
  assert.equal(calls[1][1], 'keypoints');
  await context.continueRunUntil('selection-test', 'selection');
  assert.equal(context.state.runUntilActive, false);
  assert.equal(calls.length, 3);
});

test('invalid inputs and busy state cannot start a chain', async () => {
  const {context, calls} = harness([]);
  await assert.rejects(context.startToSelectedStage(), /Check at least one scene/);
  context.state.status = 'running';
  await context.startToSelectedStage();
  assert.deepEqual(calls, []);
});

test('start failure releases chain controls', async () => {
  const {context} = harness([{id: 'office', path: 'office.zip', enabled: true}]);
  context.runDebug = async () => {throw new Error('worker failed');};
  await assert.rejects(context.startToSelectedStage(), /worker failed/);
  assert.equal(context.state.runUntilActive, false);
});

test('stop during an awaited stage transition cannot launch the next worker', async () => {
  const {context, calls} = harness([{id: 'single-chair', path: 'chair.zip', enabled: true}]);
  await context.startToSelectedStage();
  let finishTransition;
  context.chainStageRun = () => new Promise(resolve => {finishTransition = resolve;});
  context.api = async () => {};
  vm.runInContext(source.slice(source.indexOf('async function stopDebug()'), source.indexOf('async function poll()')), context);
  const transition = context.continueRunUntil('fusion-test', 'fusion');
  await context.stopDebug();
  finishTransition();
  await transition;
  assert.equal(context.state.runUntilActive, false);
  assert.deepEqual(calls, ['run']);
});

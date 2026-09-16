export const SCENE3D_THEME = Object.freeze({
  background: 0x101315,
  gridMajor: 0x465159,
  gridMinor: 0x2a3136,
  surface: 0x91989d,
  fusion: 0x77b8d6,
  candidate: 0xdea04c,
  selected: 0x65c7d6,
  accepted: 0x55b887,
  rejected: 0xdf6570,
  keypointDetected: 0xff6b8a,
  keypointRetained: 0xf1cb62,
  cameraDirection: [0.9, 0.72, 1],
  pointSize: Object.freeze({
    surface: 0.014,
    fusion: 0.021,
    candidate: 0.028,
    keypointDetected: 0.03,
    keypointRetained: 0.066,
    keypointOverlayPixels: 7,
  }),
});

function elementFor(value) {
  return typeof value === "string" ? document.querySelector(value) : value;
}

function flatValues(values) {
  if (!values) return [];
  if (ArrayBuffer.isView(values)) return values;
  return Array.isArray(values[0]) ? values.flat() : values;
}

export function createPointCloud({
  THREE,
  positions,
  color = SCENE3D_THEME.surface,
  colors = null,
  size = SCENE3D_THEME.pointSize.surface,
  opacity = 1,
  depthWrite = false,
  renderOrder = 0,
} = {}) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(flatValues(positions), 3),
  );
  if (colors) {
    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(flatValues(colors), 3),
    );
  }
  const material = new THREE.PointsMaterial({
    color,
    vertexColors: Boolean(colors),
    size,
    sizeAttenuation: true,
    transparent: opacity < 1,
    opacity,
    depthWrite,
  });
  const object = new THREE.Points(geometry, material);
  object.renderOrder = renderOrder;
  return object;
}

export function createPolyline({
  THREE,
  positions,
  color = SCENE3D_THEME.fusion,
  opacity = 0.8,
  segments = false,
} = {}) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(flatValues(positions), 3),
  );
  const material = new THREE.LineBasicMaterial({
    color,
    transparent: opacity < 1,
    opacity,
  });
  return segments
    ? new THREE.LineSegments(geometry, material)
    : new THREE.Line(geometry, material);
}

export function disposeObject3D(object, {remove = true} = {}) {
  if (!object) return;
  if (remove) object.parent?.remove(object);
  object.traverse?.(child => {
    child.geometry?.dispose();
    if (Array.isArray(child.material)) {
      child.material.forEach(material => material?.dispose());
    } else {
      child.material?.dispose();
    }
  });
}

export function clearObjectGroup(group) {
  if (!group) return;
  while (group.children.length) {
    disposeObject3D(group.children[0]);
  }
}

export function validSceneBounds(box) {
  return Boolean(
    box
    && !box.isEmpty()
    && [
      box.min.x, box.min.y, box.min.z,
      box.max.x, box.max.y, box.max.z,
    ].every(Number.isFinite),
  );
}

export function createSceneViewport({
  THREE,
  OrbitControls,
  host,
  canvas = null,
  gridSize = 12,
  gridDivisions = 24,
  pixelRatio = 2,
  far = 1000,
  fov = 45,
  cameraPosition = [4, 2.8, 4],
  cameraTarget = [0, 1, 0],
  autoStart = true,
  onControlsChange = null,
} = {}) {
  const hostElement = elementFor(host);
  if (!hostElement) throw new Error("3D scene container not found");

  const renderer = new THREE.WebGLRenderer({
    canvas: canvas || undefined,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, pixelRatio));
  renderer.setClearColor(SCENE3D_THEME.background, 1);
  if (!canvas) hostElement.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(SCENE3D_THEME.background);
  const camera = new THREE.PerspectiveCamera(fov, 1, 0.01, far);
  camera.position.set(...cameraPosition);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(...cameraTarget);

  const root = new THREE.Group();
  scene.add(root);
  let grid = null;
  let homeCamera = null;
  let homeBox = null;
  let homeFrame = null;
  let homePadding = 1.12;
  let fitRadius = 1;
  let running = false;
  let frameId = null;
  const frameListeners = new Set();

  function isAtHome(tolerance = 1e-8) {
    return Boolean(
      homeCamera
      && camera.position.distanceToSquared(homeCamera.position) < tolerance
      && controls.target.distanceToSquared(homeCamera.target) < tolerance
    );
  }

  function replaceGrid({
    size = gridSize,
    divisions = gridDivisions,
    center = [0, 0, 0],
    y = 0,
  } = {}) {
    disposeObject3D(grid);
    grid = new THREE.GridHelper(
      Math.max(Number(size) || gridSize, 0.1),
      Math.max(2, Math.round(Number(divisions) || gridDivisions)),
      SCENE3D_THEME.gridMajor,
      SCENE3D_THEME.gridMinor,
    );
    grid.position.set(Number(center[0]) || 0, Number(y) || 0, Number(center[2]) || 0);
    scene.add(grid);
    return grid;
  }

  function resize() {
    const width = Math.max(1, hostElement.clientWidth);
    const height = Math.max(1, hostElement.clientHeight);
    const cameraWasHome = isAtHome();
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    if (cameraWasHome && homeFrame) {
      fitFrame(homeFrame, homePadding, {remember: true});
    } else if (cameraWasHome && homeBox) {
      fitBox(homeBox, homePadding, {remember: true});
    } else {
      updateClipping(root);
    }
  }

  function viewBasis() {
    const direction = new THREE.Vector3(...SCENE3D_THEME.cameraDirection).normalize();
    const right = new THREE.Vector3().crossVectors(camera.up, direction);
    if (right.lengthSq() < 1e-8) right.set(1, 0, 0);
    else right.normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize();
    return {direction, right, up};
  }

  function frameFromExtents(extents, box, basis) {
    const rightCenter = (extents.rightMin + extents.rightMax) / 2;
    const upCenter = (extents.upMin + extents.upMax) / 2;
    const depthCenter = (extents.depthMin + extents.depthMax) / 2;
    const center = basis.right.clone().multiplyScalar(rightCenter)
      .addScaledVector(basis.up, upCenter)
      .addScaledVector(basis.direction, depthCenter);
    const halfRight = Math.max((extents.rightMax - extents.rightMin) / 2, 0.01);
    const halfUp = Math.max((extents.upMax - extents.upMin) / 2, 0.01);
    const halfDepth = Math.max((extents.depthMax - extents.depthMin) / 2, 0.01);
    return {
      center,
      halfRight,
      halfUp,
      halfDepth,
      box: box.clone(),
      radius: Math.max(
        Math.hypot(halfRight, halfUp, halfDepth),
        0.5,
      ),
    };
  }

  function fitFrame(frame, padding = 1.12, {remember = false} = {}) {
    const safePadding = Math.max(padding, 1);
    const verticalFov = THREE.MathUtils.degToRad(
      camera.getEffectiveFOV?.() || camera.fov,
    );
    const horizontalFov = 2 * Math.atan(
      Math.tan(verticalFov / 2) * Math.max(camera.aspect, 0.01),
    );
    const tanVertical = Math.max(Math.tan(verticalFov / 2), 1e-4);
    const tanHorizontal = Math.max(Math.tan(horizontalFov / 2), 1e-4);
    const distance = Math.max(
      0.5,
      frame.halfDepth + Math.max(
        frame.halfRight * safePadding / tanHorizontal,
        frame.halfUp * safePadding / tanVertical,
      ),
    );
    const direction = new THREE.Vector3(...SCENE3D_THEME.cameraDirection).normalize();
    camera.up.set(0, 1, 0);
    controls.target.copy(frame.center);
    camera.position.copy(frame.center).addScaledVector(direction, distance);
    controls.minDistance = Math.max(0.02, frame.radius * 0.015);
    controls.maxDistance = Math.max(100, frame.radius * 50);
    updateClippingForBox(frame.box);
    controls.update();
    fitRadius = frame.radius;
    if (remember) {
      homeBox = frame.box.clone();
      homeFrame = {
        center: frame.center.clone(),
        halfRight: frame.halfRight,
        halfUp: frame.halfUp,
        halfDepth: frame.halfDepth,
        box: frame.box.clone(),
        radius: frame.radius,
      };
      homePadding = padding;
      homeCamera = {
        position: camera.position.clone(),
        target: controls.target.clone(),
        near: camera.near,
        far: camera.far,
      };
    }
    return true;
  }

  function fitBox(box, padding = 1.12, options = {}) {
    if (!validSceneBounds(box)) return false;
    const basis = viewBasis();
    const extents = {
      rightMin: Infinity, rightMax: -Infinity,
      upMin: Infinity, upMax: -Infinity,
      depthMin: Infinity, depthMax: -Infinity,
    };
    const point = new THREE.Vector3();
    for (const x of [box.min.x, box.max.x]) {
      for (const y of [box.min.y, box.max.y]) {
        for (const z of [box.min.z, box.max.z]) {
          point.set(x, y, z);
          const rightValue = point.dot(basis.right);
          const upValue = point.dot(basis.up);
          const depthValue = point.dot(basis.direction);
          extents.rightMin = Math.min(extents.rightMin, rightValue);
          extents.rightMax = Math.max(extents.rightMax, rightValue);
          extents.upMin = Math.min(extents.upMin, upValue);
          extents.upMax = Math.max(extents.upMax, upValue);
          extents.depthMin = Math.min(extents.depthMin, depthValue);
          extents.depthMax = Math.max(extents.depthMax, depthValue);
        }
      }
    }
    return fitFrame(frameFromExtents(extents, box, basis), padding, options);
  }

  function fitObjects(objects, padding = 1.12, options = {}) {
    const roots = (Array.isArray(objects) ? objects : [objects]).filter(Boolean);
    if (!roots.length) return false;
    const basis = viewBasis();
    const extents = {
      rightMin: Infinity, rightMax: -Infinity,
      upMin: Infinity, upMax: -Infinity,
      depthMin: Infinity, depthMax: -Infinity,
    };
    const box = new THREE.Box3();
    const point = new THREE.Vector3();
    let found = false;
    for (const object of roots) {
      object.updateWorldMatrix(true, true);
      object.traverse(child => {
        if (!child.visible) return;
        const attribute = child.geometry?.getAttribute?.("position");
        if (!attribute?.count) return;
        const start = Math.max(0, child.geometry.drawRange?.start || 0);
        const requested = child.geometry.drawRange?.count;
        const end = Math.min(
          attribute.count,
          start + (Number.isFinite(requested) ? requested : attribute.count),
        );
        for (let index = start; index < end; index += 1) {
          point.fromBufferAttribute(attribute, index).applyMatrix4(child.matrixWorld);
          if (![point.x, point.y, point.z].every(Number.isFinite)) continue;
          const rightValue = point.dot(basis.right);
          const upValue = point.dot(basis.up);
          const depthValue = point.dot(basis.direction);
          extents.rightMin = Math.min(extents.rightMin, rightValue);
          extents.rightMax = Math.max(extents.rightMax, rightValue);
          extents.upMin = Math.min(extents.upMin, upValue);
          extents.upMax = Math.max(extents.upMax, upValue);
          extents.depthMin = Math.min(extents.depthMin, depthValue);
          extents.depthMax = Math.max(extents.depthMax, depthValue);
          box.expandByPoint(point);
          found = true;
        }
      });
    }
    if (!found || !validSceneBounds(box)) return false;
    return fitFrame(frameFromExtents(extents, box, basis), padding, options);
  }

  function fitObject(object, padding = 1.12, options = {}) {
    if (!object) return false;
    return fitObjects([object], padding, options);
  }

  function rememberCamera() {
    homeBox = null;
    homeFrame = null;
    homeCamera = {
      position: camera.position.clone(),
      target: controls.target.clone(),
      near: camera.near,
      far: camera.far,
    };
  }

  function resetCamera() {
    if (!homeCamera) return false;
    camera.position.copy(homeCamera.position);
    controls.target.copy(homeCamera.target);
    camera.near = homeCamera.near;
    camera.far = homeCamera.far;
    camera.updateProjectionMatrix();
    controls.update();
    return true;
  }

  function updateClipping(object = root) {
    const box = new THREE.Box3().setFromObject(object);
    if (!validSceneBounds(box)) return false;
    return updateClippingForBox(box);
  }

  function updateClippingForBox(box) {
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 0.5);
    const distance = camera.position.distanceTo(center);
    const nearest = distance - radius * 1.15;
    camera.near = nearest > 0
      ? Math.max(0.005, Math.min(0.1, nearest * 0.08))
      : 0.005;
    camera.far = Math.max(100, distance + radius * 8);
    camera.updateProjectionMatrix();
    return true;
  }

  function renderFrame() {
    controls.update();
    frameListeners.forEach(listener => listener());
    renderer.render(scene, camera);
  }

  function animate() {
    if (!running) return;
    renderFrame();
    frameId = requestAnimationFrame(animate);
  }

  function start() {
    if (running) return;
    running = true;
    frameId = requestAnimationFrame(animate);
  }

  function stop() {
    running = false;
    if (frameId != null) cancelAnimationFrame(frameId);
    frameId = null;
  }

  function addFrameListener(listener) {
    frameListeners.add(listener);
    return () => frameListeners.delete(listener);
  }

  function dispose() {
    stop();
    observer.disconnect();
    clearObjectGroup(root);
    disposeObject3D(grid);
    controls.dispose();
    renderer.dispose();
  }

  controls.addEventListener("change", () => updateClipping(root));
  if (onControlsChange) controls.addEventListener("change", onControlsChange);
  replaceGrid();
  const observer = new ResizeObserver(resize);
  observer.observe(hostElement);
  resize();
  if (autoStart) start();

  return {
    renderer,
    scene,
    camera,
    controls,
    root,
    get grid() { return grid; },
    get fitRadius() { return fitRadius; },
    resize,
    replaceGrid,
    fitBox,
    fitObjects,
    fitObject,
    isAtHome,
    rememberCamera,
    resetCamera,
    updateClipping,
    renderFrame,
    start,
    stop,
    addFrameListener,
    dispose,
  };
}

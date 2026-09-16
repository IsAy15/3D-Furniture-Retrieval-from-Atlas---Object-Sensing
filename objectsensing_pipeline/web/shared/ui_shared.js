export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export function installSharedPrimitives(root = document) {
  const apply = target => {
    if (!(target instanceof Element) && target !== document) return;
    const controls = target.matches?.("input,select,textarea")
      ? [target]
      : [...target.querySelectorAll?.("input,select,textarea") || []];
    controls.forEach(control => {
      const type = String(control.type || "").toLowerCase();
      if (!["button", "checkbox", "color", "file", "hidden", "image", "radio", "range", "reset", "submit"].includes(type)) {
        control.classList.add("ui-control");
      }
    });
    const badges = target.matches?.(".status-badge")
      ? [target]
      : [...target.querySelectorAll?.(".status-badge") || []];
    badges.forEach(badge => badge.classList.add("ui-status-badge"));
  };
  apply(root);
  const observer = new MutationObserver(records => records.forEach(record => (
    record.addedNodes.forEach(node => apply(node))
  )));
  observer.observe(root === document ? document.documentElement : root, {
    childList: true,
    subtree: true,
  });
  return {refresh: () => apply(root), disconnect: () => observer.disconnect()};
}

function storedValue(key) {
  if (!key) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function persistValue(key, value) {
  if (!key) return;
  try {
    localStorage.setItem(key, String(value));
  } catch {
    // Persistence is optional in private or restricted browser contexts.
  }
}

export function bindResizableAccordions({container, storagePrefix}) {
  container.querySelectorAll(".ui-accordion").forEach(panel => {
    const body = panel.querySelector(".ui-accordion-body");
    const key = `${storagePrefix}-${panel.dataset.panel}`;
    try {
      const saved = localStorage.getItem(`${key}-open`);
      if (saved !== null) panel.open = saved === "true";
    } catch {}
    panel.addEventListener("toggle", () => {
      try { localStorage.setItem(`${key}-open`, String(panel.open)); } catch {}
    });
    const handle = document.createElement("div");
    handle.className = "ui-accordion-resizer ui-resizer";
    handle.tabIndex = 0;
    handle.setAttribute("role", "separator");
    handle.setAttribute("aria-orientation", "horizontal");
    const label = panel.querySelector("summary").textContent.trim();
    handle.setAttribute("aria-label", `Height: ${label}`);
    handle.title = "Drag or use arrow keys to resize. Double-click to reset.";
    panel.append(handle);
    bindResizableDimension({handle, container:body, root:panel,
      property:"--accordion-height", storageKey:`${key}-height`,
      defaultValue:Number(panel.dataset.height)||220, min:64, max:1000,
      axis:"y", edge:"start"});
  });
}

export function bindResizableDimension({
  handle,
  container,
  property,
  storageKey = null,
  defaultValue,
  min = 0,
  max = Infinity,
  getMax = null,
  axis = "x",
  edge = "start",
  disabledBelow = 0,
  root = document.documentElement,
  stateTarget = document.body,
  stateClass = axis === "x" ? "resizing-panels" : "resizing-timeline",
  step = 12,
  largeStep = 40,
  onChange = null,
} = {}) {
  const handleElement = typeof handle === "string" ? $(handle) : handle;
  const containerElement = () => (
    typeof container === "string" ? $(container) : container
  );
  if (!handleElement) return null;

  const maximum = () => {
    const dynamic = typeof getMax === "function" ? Number(getMax()) : Number(max);
    return Math.max(Number(min) || 0, Number.isFinite(dynamic) ? dynamic : Infinity);
  };

  function set(value, persist = true) {
    const lower = Number(min) || 0;
    const upper = maximum();
    const fallback = Number(defaultValue) || lower;
    const next = Math.round(Math.max(lower, Math.min(upper, Number(value) || fallback)));
    root.style.setProperty(property, `${next}px`);
    handleElement.setAttribute("aria-valuemin", String(lower));
    handleElement.setAttribute("aria-valuemax", String(upper));
    handleElement.setAttribute("aria-valuenow", String(next));
    if (persist) persistValue(storageKey, next);
    onChange?.(next);
    return next;
  }

  function valueFromPointer(event) {
    const bounds = containerElement()?.getBoundingClientRect();
    if (!bounds) return Number(defaultValue) || Number(min) || 0;
    if (axis === "y") {
      return edge === "bottom"
        ? bounds.bottom - event.clientY
        : event.clientY - bounds.top;
    }
    return edge === "end"
      ? bounds.right - event.clientX
      : event.clientX - bounds.left;
  }

  let active = false;
  const move = event => {
    if (active) set(valueFromPointer(event));
  };
  const finish = () => {
    if (!active) return;
    active = false;
    stateTarget?.classList.remove(stateClass);
    document.body.style.userSelect = "";
  };

  handleElement.addEventListener("pointerdown", event => {
    if (disabledBelow && innerWidth <= disabledBelow) return;
    event.preventDefault();
    active = true;
    handleElement.setPointerCapture(event.pointerId);
    stateTarget?.classList.add(stateClass);
    document.body.style.userSelect = "none";
    move(event);
  });
  handleElement.addEventListener("pointermove", move);
  handleElement.addEventListener("pointerup", finish);
  handleElement.addEventListener("pointercancel", finish);
  handleElement.addEventListener("dblclick", () => set(defaultValue));
  handleElement.addEventListener("keydown", event => {
    const keys = axis === "x"
      ? ["ArrowLeft", "ArrowRight", "Home"]
      : ["ArrowUp", "ArrowDown", "Home"];
    if (!keys.includes(event.key)) return;
    event.preventDefault();
    if (event.key === "Home") {
      set(defaultValue);
      return;
    }
    const current = Number(handleElement.getAttribute("aria-valuenow")) || Number(defaultValue);
    let direction;
    if (axis === "x") {
      direction = event.key === "ArrowRight" ? 1 : -1;
      if (edge === "end") direction *= -1;
    } else {
      direction = event.key === "ArrowUp" ? 1 : -1;
      if (edge !== "bottom") direction *= -1;
    }
    set(current + direction * (event.shiftKey ? largeStep : step));
  });

  set(storedValue(storageKey) || defaultValue, false);
  return {set, reset: () => set(defaultValue), refresh: () => set(
    handleElement.getAttribute("aria-valuenow") || defaultValue,
    false,
  )};
}

export function refreshIcons() {
  window.lucide?.createIcons({attrs: {width: 16, height: 16}});
}

export function createNotifier({selector = "#toast", duration = 3400} = {}) {
  let timer = null;
  return function notify(message, error = false) {
    const toast = $(selector);
    if (!toast) return;
    toast.textContent = message;
    toast.classList.toggle("error", error);
    toast.hidden = false;
    clearTimeout(timer);
    timer = setTimeout(() => {
      toast.hidden = true;
    }, duration);
  };
}

export function createCandidateFilterController({
  search,
  category,
  status,
  reset,
  getCategory = item => item?.category || "",
  getStatus = item => item?.status || "pending",
  getSearchText = item => JSON.stringify(item),
  seedCategories = [],
  allCategoryLabel = "All categories",
  locale = "fr",
  onChange = null,
} = {}) {
  const resolve = value => typeof value === "string" ? $(value) : value;
  const elements = {
    search: resolve(search),
    category: resolve(category),
    status: resolve(status),
    reset: resolve(reset),
  };
  if (!elements.search || !elements.category || !elements.status || !elements.reset) {
    return null;
  }

  const state = {query: "", category: "all", status: "all"};
  const knownCategories = new Set(seedCategories.filter(Boolean));

  function syncCategories(items = []) {
    items.forEach(item => {
      const value = String(getCategory(item) || "").trim();
      if (value && value !== "—") knownCategories.add(value);
    });
    const current = state.category;
    const categories = [...knownCategories].sort((a, b) => a.localeCompare(b, locale));
    elements.category.replaceChildren();
    const all = document.createElement("option");
    all.value = "all";
    all.textContent = allCategoryLabel;
    elements.category.append(all);
    categories.forEach(value => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      elements.category.append(option);
    });
    state.category = categories.includes(current) ? current : "all";
    elements.category.value = state.category;
  }

  function filter(items = []) {
    const query = state.query.trim().toLocaleLowerCase(locale);
    return items.filter(item => {
      const categoryValue = String(getCategory(item) || "");
      const statusValue = String(getStatus(item) || "pending");
      const haystack = String(getSearchText(item) || "").toLocaleLowerCase(locale);
      return (!query || haystack.includes(query))
        && (state.category === "all" || categoryValue === state.category)
        && (state.status === "all" || statusValue === state.status);
    });
  }

  function clear({notify = true} = {}) {
    state.query = "";
    state.category = "all";
    state.status = "all";
    elements.search.value = "";
    elements.category.value = "all";
    elements.status.value = "all";
    if (notify) onChange?.(state);
  }

  elements.search.addEventListener("input", event => {
    state.query = event.target.value;
    onChange?.(state);
  });
  elements.category.addEventListener("change", event => {
    state.category = event.target.value;
    onChange?.(state);
  });
  elements.status.addEventListener("change", event => {
    state.status = event.target.value;
    onChange?.(state);
  });
  elements.reset.addEventListener("click", () => clear());
  syncCategories();
  return {state, filter, syncCategories, reset: clear, elements};
}

export function createTooltipController({
  selector = "#tooltip",
  delay = 160,
  targetSelector = "[data-tooltip],button[aria-label],a[aria-label]",
  resolveContent = null,
} = {}) {
  let timer = null;
  let currentTarget = null;
  let bound = false;

  const tooltip = () => $(selector);
  const targetFor = target => target?.closest?.(targetSelector) || null;

  function position(target, clientX = null, clientY = null) {
    const element = tooltip();
    if (!element || !target) return;
    const rect = target.getBoundingClientRect();
    const margin = 10;
    element.style.left = "0px";
    element.style.top = "0px";
    const box = element.getBoundingClientRect();
    const anchorX = clientX ?? rect.left + rect.width / 2;
    const preferRight = rect.right + box.width + margin < innerWidth;
    let left = preferRight ? rect.right + margin : anchorX - box.width / 2;
    let top = clientY == null ? rect.bottom + margin : clientY + 14;
    if (top + box.height > innerHeight - margin) {
      top = Math.max(margin, rect.top - box.height - margin);
    }
    left = Math.max(margin, Math.min(innerWidth - box.width - margin, left));
    element.style.left = `${left}px`;
    element.style.top = `${top}px`;
  }

  function show(target, clientX = null, clientY = null) {
    if (!target || target.matches(":disabled")) return;
    clearTimeout(timer);
    currentTarget = target;
    timer = setTimeout(() => {
      if (currentTarget !== target) return;
      const element = tooltip();
      const custom = typeof resolveContent === "function"
        ? resolveContent(target)
        : null;
      const title = custom?.title
        || target.dataset.tooltip
        || target.getAttribute("aria-label");
      if (!element || !title) return;
      const detail = custom?.detail ?? target.dataset.tooltipDetail ?? "";
      const heading = $("strong", element);
      const body = $("span", element);
      if (heading) heading.textContent = title;
      if (body) {
        body.textContent = detail;
        body.hidden = !detail;
      }
      element.hidden = false;
      position(target, clientX, clientY);
    }, delay);
  }

  function hide(target = null) {
    if (target && currentTarget !== target) return;
    clearTimeout(timer);
    currentTarget = null;
    const element = tooltip();
    if (element) element.hidden = true;
  }

  function bind() {
    if (bound) return;
    bound = true;
    document.addEventListener("pointerover", event => {
      const target = targetFor(event.target);
      if (!target || target === currentTarget || target.contains(event.relatedTarget)) return;
      show(target, event.clientX, event.clientY);
    });
    document.addEventListener("pointerout", event => {
      const target = targetFor(event.target);
      if (!target || target.contains(event.relatedTarget)) return;
      hide(target);
    });
    document.addEventListener("focusin", event => {
      const target = targetFor(event.target);
      if (target) show(target);
    });
    document.addEventListener("focusout", event => {
      const target = targetFor(event.target);
      if (target) hide(target);
    });
    document.addEventListener("pointerdown", () => hide());
  }

  return {bind, hide, show, position};
}

export function bindMultiSelectListInteractions({
  container,
  itemSelector,
  getPosition = item => Number(item?.dataset?.position),
  onActivate,
  onPaint = null,
  onPaintEnd = null,
  longPressDelay = 420,
} = {}) {
  const host = typeof container === "string" ? $(container) : container;
  if (!host || !itemSelector || typeof onActivate !== "function") return null;
  let touch = null;
  let suppressClick = false;

  const itemAt = (x, y) => document.elementFromPoint(x, y)?.closest?.(itemSelector);
  const paint = item => {
    if (!touch?.active || !item || !host.contains(item)) return;
    const position = getPosition(item);
    if (!Number.isInteger(position) || touch.visited.has(position)) return;
    touch.visited.add(position);
    item.classList.add("touch-painted");
    (onPaint || onActivate)(position, {
      additive: true,
      range: false,
      touch: true,
      originalEvent: touch.event,
    });
  };

  host.addEventListener("click", event => {
    const item = event.target.closest?.(itemSelector);
    if (!item || !host.contains(item)) return;
    if (suppressClick) {
      event.preventDefault();
      return;
    }
    const position = getPosition(item);
    if (!Number.isInteger(position)) return;
    onActivate(position, {
      additive: Boolean(event.ctrlKey || event.metaKey),
      range: Boolean(event.shiftKey),
      touch: false,
      originalEvent: event,
    });
  });
  host.addEventListener("pointerdown", event => {
    if (event.pointerType !== "touch") return;
    const item = event.target.closest?.(itemSelector);
    if (!item || !host.contains(item)) return;
    touch = {
      pointerId: event.pointerId,
      event,
      active: false,
      visited: new Set(),
      timer: setTimeout(() => {
        if (!touch || touch.pointerId !== event.pointerId) return;
        touch.active = true;
        suppressClick = true;
        host.setPointerCapture?.(event.pointerId);
        paint(item);
      }, longPressDelay),
    };
  });
  host.addEventListener("pointermove", event => {
    if (!touch || touch.pointerId !== event.pointerId || !touch.active) return;
    event.preventDefault();
    paint(itemAt(event.clientX, event.clientY));
  });
  const finish = event => {
    if (!touch || touch.pointerId !== event.pointerId) return;
    clearTimeout(touch.timer);
    const wasActive = touch.active;
    const painted = [...touch.visited];
    touch = null;
    if (wasActive) {
      onPaintEnd?.(painted, {additive: true, touch: true, originalEvent: event});
      setTimeout(() => { suppressClick = false; }, 0);
    }
  };
  host.addEventListener("pointerup", finish);
  host.addEventListener("pointercancel", finish);
  return {cancel: () => {
    if (touch) clearTimeout(touch.timer);
    touch = null;
    suppressClick = false;
  }};
}

export function bindHoverPreviewPanels({app, panels, delay = 180}) {
  const appElement = typeof app === "string" ? $(app) : app;
  if (!appElement) return;
  const canHover = () => matchMedia("(hover:hover) and (pointer:fine)").matches;
  for (const descriptor of panels) {
    const panel = typeof descriptor.element === "string"
      ? $(descriptor.element)
      : descriptor.element;
    if (!panel) continue;
    let closeTimer = null;
    panel.addEventListener("pointerenter", () => {
      if (!canHover() || !appElement.classList.contains(descriptor.collapsedClass)) return;
      clearTimeout(closeTimer);
      appElement.classList.add(descriptor.peekClass);
    });
    panel.addEventListener("pointerleave", () => {
      clearTimeout(closeTimer);
      closeTimer = setTimeout(
        () => appElement.classList.remove(descriptor.peekClass),
        delay,
      );
    });
    panel.addEventListener("focusin", () => {
      if (appElement.classList.contains(descriptor.collapsedClass)) {
        appElement.classList.add(descriptor.peekClass);
      }
    });
    panel.addEventListener("focusout", event => {
      if (!panel.contains(event.relatedTarget)) {
        appElement.classList.remove(descriptor.peekClass);
      }
    });
  }
}

export const CANDIDATE_TIMELINE_PHASES = [
  {id: "query", label: "Query", icon: "list-filter"},
  {id: "matching", label: "Matching", icon: "waypoints"},
  {id: "verification", label: "Verification", icon: "badge-check"},
  {id: "selection", label: "Selection", icon: "check-check"},
];

function modelName(value) {
  if (typeof value === "string") return value;
  return value?.model || value?.model_id || value?.name || null;
}

function listValues(value) {
  if (value == null) return [];
  return Array.isArray(value) ? value : [value];
}

function candidateCheckpoint(candidate) {
  const value = candidate?.checkpoint_frames ?? candidate?.frame_count
    ?? candidate?.checkpoint;
  return value == null ? null : Number(value);
}

function eventCheckpoint(event) {
  const value = event?.checkpoint ?? event?.step?.frame_count
    ?? event?.data?.frame_count ?? event?.candidate?.checkpoint_frames;
  return value == null ? null : Number(value);
}

function eventModels(event) {
  const values = [
    event?.candidate,
    event?.data?.candidate,
    event?.data?.model,
    ...listValues(event?.relatedModels),
    ...listValues(event?.data?.selected),
    ...listValues(event?.data?.candidates),
    ...listValues(event?.data?.diagnostics),
  ];
  return new Set(values.map(modelName).filter(Boolean));
}

export function candidateTimelineGroups(
  events,
  candidate,
  {
    phaseOf = event => event?.chapter || event?.phase,
    matches = null,
  } = {},
) {
  const model = modelName(candidate);
  const checkpoint = candidateCheckpoint(candidate);
  const byPhase = new Map(CANDIDATE_TIMELINE_PHASES.map(item => [item.id, []]));
  if (!model) return CANDIDATE_TIMELINE_PHASES.map(item => ({...item, indices: []}));

  events.forEach((event, index) => {
    const phase = phaseOf(event);
    if (!byPhase.has(phase)) return;
    const currentCheckpoint = eventCheckpoint(event);
    if (
      checkpoint != null
      && currentCheckpoint != null
      && checkpoint !== currentCheckpoint
    ) return;
    const related = matches
      ? matches(event, candidate, model)
      : eventModels(event).has(model);
    if (related) byPhase.get(phase).push(index);
  });

  return CANDIDATE_TIMELINE_PHASES.map(item => ({
    ...item,
    indices: byPhase.get(item.id),
  }));
}

export function relatedTimelineIndices(groups) {
  return new Set(groups.flatMap(group => group.indices));
}

export function nextRelatedIndex(indices, currentIndex = -1) {
  if (!indices?.length) return null;
  return indices.find(index => index > currentIndex) ?? indices[0];
}

export function renderCandidateTimelineNavigation(
  container,
  groups,
  {
    currentIndex = -1,
    onNavigate = null,
    emptyText = "No stage recorded for this candidate.",
  } = {},
) {
  if (!container) return;
  container.replaceChildren();
  container.classList.add("ui-candidate-timeline");

  const available = groups.some(group => group.indices.length);
  if (!available) {
    const empty = document.createElement("p");
    empty.className = "ui-candidate-timeline-empty";
    empty.textContent = emptyText;
    container.append(empty);
    return;
  }

  for (const group of groups) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ui-candidate-step";
    button.disabled = !group.indices.length;
    button.dataset.candidatePhase = group.id;
    button.dataset.tooltip = group.indices.length
      ? `${group.label} linked to this candidate`
      : `${group.label} absent from trace`;
    button.dataset.tooltipDetail = group.indices.length
      ? `${group.indices.length} event(s). Click again to advance.`
      : "This trace cannot link the chapter to the candidate.";
    button.innerHTML = `
      <i data-lucide="${group.icon}" aria-hidden="true"></i>
      <span>${group.label}</span>
      <b>${group.indices.length || "—"}</b>
    `;
    if (group.indices.includes(currentIndex)) button.classList.add("active");
    button.addEventListener("click", () => {
      const target = nextRelatedIndex(group.indices, currentIndex);
      if (target != null) onNavigate?.(target, group);
    });
    container.append(button);
  }
}

export function applyRelatedTimelineState(
  root,
  indices,
  selector = "[data-event-index]",
) {
  if (!root) return;
  const related = indices instanceof Set ? indices : new Set(indices || []);
  root.classList.toggle("has-candidate-context", related.size > 0);
  root.querySelectorAll(selector).forEach(element => {
    element.classList.toggle(
      "candidate-related",
      related.has(Number(element.dataset.eventIndex)),
    );
  });
}

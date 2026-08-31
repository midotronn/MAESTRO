"use strict";
const $ = (id) => document.getElementById(id);
const EDITOR_UTILS = globalThis.MaestroEditorUtils;
if (!EDITOR_UTILS) throw new Error("editor utilities failed to load");
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
let ST = {
  sid: null,
  fps: 30,
  dur: 0,
  nframes: 0,
  beats: 0,
  head: null,
  sel: null,
  timeline: [],
  branchBase: null,
};
let MOTION_PREVIEW = null;
let PLAYHEAD_RAF = 0;
let SONGS = [];
let FULL_REVIEW_STATE = null;
let FULL_REVIEW_POLL_TIMER = 0;
let FULL_REVIEW_REFRESH_TIMER = 0;
let FULL_REVIEW_COMPLETED_SIGNATURE = null;
let COMPARE_POLL_TIMER = 0;
let COMPARE_CONTEXT = null;
let COMPARE_LOADING = false;

const METRIC_LABELS = { energy: "energy", bas: "beat align (BAS)", jerk: "smoothness", foot: "foot contact" };
const HIGHER_BETTER = { energy: null, bas: true, jerk: false, foot: true }; // null = neutral (raw metrics)
// Display transform: jerk (lower = better) is presented as SMOOTHNESS (higher = better) so the
// before/after values, the change sign, the arrow and the colour all agree (a smoother edit reads as
// an increase). 1/(1+jerk) is a bounded 0..1 smoothness score.
const metricDisp = (k, v) => (k === "jerk" ? 1 / (1 + v) : v);
const DISP_HIGHER_BETTER = { energy: null, bas: true, jerk: true, foot: true };
const METRIC_INFO = {
  energy: "How big and lively the movement is. Measured as the average frame-to-frame change in the body pose (root travel + all joint rotations). Higher = more energetic; there is no 'good' value, it just tells you the intensity.",
  bas: "Beat Alignment Score (0-1): how well the dance lands on the music's beats. We find each 'motion beat' (a moment the body decelerates into a pose) and score how close it is to the nearest music beat. Higher = tighter to the beat.",
  jerk: "Smoothness (higher = smoother). Shown as 1/(1+jerk), where jerk is the average 3rd derivative of the pose (how abruptly acceleration changes); less jitter -> higher smoothness.",
  foot: "Foot-plant consistency (0-1): of the frames a foot is marked on the floor, the fraction where the body is NOT sliding. Higher = less foot-skating (feet stay planted).",
};

function toast(msg) {
  let t = $("toast"); if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200);
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

const ACTIVITIES = new Map();
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function elapsedLabel(started) {
  const seconds = Math.max(0, Math.round((Date.now() - started) / 1000));
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${String(seconds % 60).padStart(2, "0")}s`;
}

function renderActivities() {
  const center = $("activityCenter"), list = $("activityList");
  if (!center || !list) return;
  const items = [...ACTIVITIES.entries()].sort((a, b) => a[1].started - b[1].started);
  center.hidden = items.length === 0;
  if (!items.length) { list.innerHTML = ""; return; }
  $("activityCount").textContent = items.length === 1 ? "1 operation" : `${items.length} operations`;
  list.innerHTML = items.map(([id, item]) => {
    const determinate = Number.isFinite(item.progress);
    const progress = determinate ? Math.max(0, Math.min(100, Math.round(item.progress))) : null;
    return `<div class="activity-row ${item.state}" data-activity="${escapeHtml(id)}" role="status">`
      + `<div class="activity-head"><strong>${escapeHtml(item.label)}</strong>`
      + `<span class="activity-percent">${progress == null ? "live" : progress + "%"}</span></div>`
      + `<div class="activity-track ${determinate ? "" : "indeterminate"}" role="progressbar"`
      + ` aria-valuemin="0" aria-valuemax="100"${determinate ? ` aria-valuenow="${progress}"` : ""}>`
      + `<i style="${determinate ? `width:${progress}%` : ""}"></i></div>`
      + `<div class="activity-meta"><span class="activity-detail">${escapeHtml(item.detail || "")}</span>`
      + `<span class="activity-elapsed">${elapsedLabel(item.started)}</span></div></div>`;
  }).join("");
}

function tickActivityElapsed() {
  const list = $("activityList");
  if (!list) return;
  list.querySelectorAll("[data-activity]").forEach((row) => {
    const item = ACTIVITIES.get(row.dataset.activity);
    const elapsed = row.querySelector(".activity-elapsed");
    if (item && elapsed) elapsed.textContent = elapsedLabel(item.started);
  });
}
setInterval(tickActivityElapsed, 1000);

function activityStart(id, label, detail, progress = null) {
  const old = ACTIVITIES.get(id);
  if (old && old.timer) clearTimeout(old.timer);
  ACTIVITIES.set(id, {
    label,
    detail: detail || "",
    progress: Number.isFinite(progress) ? progress : null,
    state: "active",
    started: old && old.state === "active" ? old.started : Date.now(),
    timer: null,
  });
  renderActivities();
}

function activityUpdate(id, patch) {
  const item = ACTIVITIES.get(id);
  if (!item) return;
  if (patch.label !== undefined) item.label = patch.label;
  if (patch.detail !== undefined) item.detail = patch.detail;
  if (patch.progress !== undefined) {
    item.progress = Number.isFinite(patch.progress) ? patch.progress : null;
  }
  item.state = patch.state || "active";
  renderActivities();
}

function activityFinish(id, state, detail, removeAfter) {
  const item = ACTIVITIES.get(id);
  if (!item) return;
  if (item.timer) clearTimeout(item.timer);
  item.state = state;
  item.detail = detail || item.detail;
  item.progress = 100;
  renderActivities();
  item.timer = setTimeout(() => {
    ACTIVITIES.delete(id);
    renderActivities();
  }, removeAfter);
}

function activityDone(id, detail) {
  activityFinish(id, "done", detail || "complete", 1800);
}

function activityFail(id, detail) {
  activityFinish(id, "error", detail || "failed", 7000);
}

function setProgressBar(barId, wrapId, progress) {
  const bar = $(barId), wrap = $(wrapId);
  if (!bar || !wrap) return;
  wrap.hidden = false;
  const determinate = Number.isFinite(progress);
  wrap.classList.toggle("indeterminate", !determinate);
  if (determinate) {
    const value = Math.max(0, Math.min(100, Math.round(progress)));
    bar.style.width = value + "%";
    wrap.setAttribute("aria-valuenow", String(value));
  } else {
    bar.style.width = "";
    wrap.removeAttribute("aria-valuenow");
  }
}

function mediaProgress(video) {
  if (video.readyState >= 3) return 1;
  if (video.duration && video.buffered && video.buffered.length) {
    try {
      return Math.max(0.2, Math.min(0.95, video.buffered.end(video.buffered.length - 1) / video.duration));
    } catch (e) {}
  }
  if (video.readyState >= 2) return 0.72;
  if (video.readyState >= 1) return 0.35;
  return 0.08;
}

function waitForMediaReady(videos, activityId, start, end, detail, timeout = 20000) {
  const media = videos.filter(Boolean);
  if (!media.length) return Promise.resolve(true);
  return new Promise((resolve) => {
    let settled = false;
    const cleanups = [];
    const finish = (ok) => {
      if (settled) return;
      settled = true;
      cleanups.forEach((fn) => fn());
      resolve(ok);
    };
    const update = () => {
      const ratio = media.reduce((sum, item) => sum + mediaProgress(item), 0) / media.length;
      activityUpdate(activityId, {
        progress: start + (end - start) * ratio,
        detail,
      });
      if (media.every((item) => item.readyState >= 3)) finish(true);
    };
    media.forEach((item) => {
      ["loadstart", "loadedmetadata", "progress", "loadeddata", "canplay"].forEach((name) => {
        item.addEventListener(name, update);
        cleanups.push(() => item.removeEventListener(name, update));
      });
      const fail = () => finish(false);
      item.addEventListener("error", fail, { once: true });
      cleanups.push(() => item.removeEventListener("error", fail));
    });
    const timer = setTimeout(() => finish(false), timeout);
    cleanups.push(() => clearTimeout(timer));
    update();
  });
}

function wireMediaBuffering(video, activityId, label) {
  if (!video || video.dataset.progressWired) return;
  video.dataset.progressWired = "1";
  const buffering = () => {
    if (!video.currentSrc || video.ended) return;
    activityStart(activityId, label, "Waiting for more video data", null);
  };
  const resumed = () => {
    const item = ACTIVITIES.get(activityId);
    if (item && item.state === "active") activityDone(activityId, "Playback resumed");
  };
  video.addEventListener("waiting", buffering);
  video.addEventListener("stalled", buffering);
  video.addEventListener("playing", resumed);
  video.addEventListener("canplay", resumed);
}

// -------------------------------------------------------------- review all edited Dynamite sections as one song
function clockLabel(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function shortSectionName(name) {
  const parts = String(name || "").split("\u2014");
  return (parts.length > 1 ? parts.slice(1).join("\u2014") : parts[0]).trim();
}

function setFullReviewPlaceholder(icon, title, detail) {
  const placeholder = $("fullReviewPlaceholder");
  placeholder.querySelector("span").textContent = icon;
  placeholder.querySelector("strong").textContent = title;
  placeholder.querySelector("p").textContent = detail;
}

function updateFullReviewButton(state) {
  const button = $("fullRenderBtn");
  if (!button) return;
  const meta = $("fullReviewButtonMeta");
  const ready = !!state?.render?.ready;
  button.classList.toggle("ready", ready);
  if (!state) {
    meta.textContent = "Combines all 5 sections";
    return;
  }
  meta.textContent = ready
    ? "Current 5-section preview ready"
    : `${state.edited_sections} of ${state.total_sections} sections edited`;
}

function renderFullReview(state) {
  if (!state) return;
  const render = state.render || {};
  const running = render.status === "queued" || render.status === "rendering";
  const currentSid = ST.sid;
  $("fullReviewTitle").textContent = `${state.source.name} \u2014 full song`;
  $("fullReviewCount").textContent =
    `${state.edited_sections} of ${state.total_sections} sections edited`;
  $("fullReviewSummary").textContent = render.ready
    ? "This preview matches every section\u2019s current history state."
    : state.edited_sections
      ? "Edited sections use their current version; the rest stay original."
      : "No section has been edited yet, so every section is still original.";
  $("fullReviewMediaTag").textContent =
    `full song \u00b7 ${clockLabel(state.source.duration_sec)}`;

  const sectionHost = $("fullReviewSections");
  sectionHost.innerHTML = state.sections.map((section) => {
    const current = section.sid === currentSid;
    const status = current ? "Current" : section.edited ? "Edited" : "Original";
    return `<button class="full-review-section ${section.edited ? "edited" : ""} ${current ? "current" : ""}" `
      + `type="button" data-sid="${escapeHtml(section.sid)}" `
      + `aria-label="Edit section ${section.order}: ${escapeHtml(shortSectionName(section.name))}, ${status}">`
      + `<span class="full-review-section-number">${String(section.order).padStart(2, "0")}</span>`
      + `<span class="full-review-section-copy"><strong>${escapeHtml(shortSectionName(section.name))}</strong>`
      + `<small>${clockLabel(section.start_sec)}\u2013${clockLabel(section.end_sec)} \u00b7 `
      + `${section.edited ? "current edit" : "original choreography"}</small></span>`
      + `<span class="full-review-section-state">${status}</span></button>`;
  }).join("");
  sectionHost.querySelectorAll("[data-sid]").forEach((button) => {
    button.onclick = () => openReviewSection(button.dataset.sid);
  });

  const video = $("fullReviewVideo");
  const placeholder = $("fullReviewPlaceholder");
  const status = $("fullReviewStatus");
  const progressWrap = $("fullReviewProgWrap");
  const renderButton = $("fullReviewRender");
  if (render.ready && render.video_url) {
    placeholder.hidden = true;
    video.hidden = false;
    if (video.dataset.signature !== state.signature) {
      video.dataset.signature = state.signature;
      video.src = `${render.video_url}&t=${Date.now()}`;
      video.load();
    }
    status.className = "full-review-status ok";
    status.textContent = `\u2714 Full-song preview ready${render.elapsed ? ` in ${render.elapsed}s` : ""}.`;
    progressWrap.hidden = true;
    renderButton.disabled = true;
    renderButton.textContent = "Preview up to date";
  } else {
    try { video.pause(); } catch (e) {}
    video.hidden = true;
    placeholder.hidden = false;
    renderButton.disabled = running;
    if (running) {
      const staleTitle = render.stale
        ? "Finishing an older full-song preview"
        : "Building your continuous full-song preview";
      const staleDetail = render.stale
        ? "Your section choices changed during rendering. When this finishes, update once to include them."
        : "MAESTRO is rendering the current version of all five sections with the full song audio.";
      setFullReviewPlaceholder("\u25b6", staleTitle, staleDetail);
      status.className = "full-review-status";
      status.textContent = render.stale
        ? "Current edits will need one update after this render finishes."
        : render.message || "Rendering the full song\u2026";
      setProgressBar("fullReviewProg", "fullReviewProgWrap", Number(render.progress || 0));
      renderButton.textContent = "Building preview\u2026";
    } else if (render.stale) {
      setFullReviewPlaceholder(
        "\u21bb",
        "Your section choices changed",
        "Update the preview to include the current history state from every section.",
      );
      status.className = "full-review-status";
      status.textContent = "The previous full-song preview is out of date.";
      progressWrap.hidden = true;
      renderButton.textContent = "Update full-song preview";
    } else if (render.status === "error") {
      setFullReviewPlaceholder(
        "!",
        "The preview could not be completed",
        "Your section edits are safe. Try the full-song render again.",
      );
      status.className = "full-review-status bad";
      status.textContent = render.message || "Full-song render failed.";
      progressWrap.hidden = true;
      renderButton.textContent = "Try again";
    } else {
      setFullReviewPlaceholder(
        "\u25b6",
        "See all five sections as one continuous dance",
        "Create a preview when you are ready. It uses each section\u2019s current history state and the full Dynamite audio.",
      );
      status.className = "full-review-status";
      status.textContent = "Ready to combine the five current section versions.";
      progressWrap.hidden = true;
      renderButton.textContent = "Create full-song preview";
    }
  }
  updateFullReviewButton(state);
}

async function loadFullReviewState(showError = false) {
  try {
    const state = await api("/api/full-song-review");
    FULL_REVIEW_STATE = state;
    updateFullReviewButton(state);
    if (!$("fullReview").hidden) renderFullReview(state);
    return state;
  } catch (e) {
    $("fullReviewButtonMeta").textContent = "Full-song review unavailable";
    $("fullRenderBtn").classList.remove("ready");
    if (!$("fullReview").hidden) {
      $("fullReviewStatus").className = "full-review-status bad";
      $("fullReviewStatus").textContent = `\u26a0 ${e.message}`;
    }
    if (showError) toast("Full-song review unavailable: " + e.message);
    return null;
  }
}

function scheduleFullReviewRefresh() {
  clearTimeout(FULL_REVIEW_REFRESH_TIMER);
  FULL_REVIEW_REFRESH_TIMER = setTimeout(() => loadFullReviewState(false), 250);
}

async function openFullReview() {
  const dialog = $("fullReview");
  dialog.hidden = false;
  document.body.classList.add("modal-open");
  $("fullReviewCount").textContent = "Loading section status\u2026";
  $("fullReviewSummary").textContent = "Checking each section\u2019s current history state.";
  $("fullReviewSections").innerHTML = "";
  $("fullReviewStatus").className = "full-review-status";
  $("fullReviewStatus").textContent = "Loading full-song review\u2026";
  const state = await loadFullReviewState(true);
  if (state) renderFullReview(state);
  $("fullReviewClose").focus();
}

function closeFullReview() {
  const dialog = $("fullReview");
  if (dialog.hidden) return;
  try { $("fullReviewVideo").pause(); } catch (e) {}
  dialog.hidden = true;
  document.body.classList.remove("modal-open");
  $("fullRenderBtn").focus();
}

async function openReviewSection(sid) {
  const select = $("song");
  if (!sid || ![...select.options].some((option) => option.value === sid)) {
    $("fullReviewStatus").className = "full-review-status bad";
    $("fullReviewStatus").textContent = "That section is not available in the editor.";
    return;
  }
  closeFullReview();
  select.value = sid;
  await openSession(sid);
  if (window.innerWidth <= 900) window.scrollTo({ top: 0, behavior: "smooth" });
  else document.querySelector(".stage-col").scrollTop = 0;
}

async function pollFullSongReview() {
  clearTimeout(FULL_REVIEW_POLL_TIMER);
  const state = await loadFullReviewState(false);
  if (!state) {
    activityUpdate("full-song-render", { detail: "Reconnecting to full-song render status\u2026" });
    FULL_REVIEW_POLL_TIMER = setTimeout(pollFullSongReview, 1200);
    return;
  }
  const render = state.render || {};
  if (render.status === "queued" || render.status === "rendering") {
    activityUpdate("full-song-render", {
      progress: Number(render.progress || 0),
      detail: render.message || "Rendering the full song\u2026",
    });
    FULL_REVIEW_POLL_TIMER = setTimeout(pollFullSongReview, 1000);
    return;
  }
  if (render.ready) {
    if (FULL_REVIEW_COMPLETED_SIGNATURE !== state.signature) {
      FULL_REVIEW_COMPLETED_SIGNATURE = state.signature;
      let mediaReady = true;
      if (!$("fullReview").hidden) {
        const video = $("fullReviewVideo");
        activityUpdate("full-song-render", {
          progress: 96,
          detail: "Loading the full-song preview\u2026",
        });
        mediaReady = await waitForMediaReady(
          [video],
          "full-song-render",
          96,
          100,
          "Loading the full-song preview\u2026",
          30000,
        );
        video.play().catch(() => {});
      }
      activityDone(
        "full-song-render",
        mediaReady ? "Full-song preview ready" : "Preview ready; video is still buffering",
      );
      toast("Full Dynamite preview ready");
    }
    return;
  }
  if (render.stale && render.status === "done") {
    activityDone("full-song-render", "Older preview finished; update needed");
    if (!$("fullReview").hidden) toast("Section edits changed \u2014 update the full-song preview");
    return;
  }
  if (render.status === "error") {
    activityFail("full-song-render", render.message || "Full-song render failed");
    toast("Full-song preview failed");
  }
}

async function startFullSongReview() {
  $("fullReviewRender").disabled = true;
  $("fullReviewStatus").className = "full-review-status";
  $("fullReviewStatus").textContent = "Preparing all five current section versions\u2026";
  setProgressBar("fullReviewProg", "fullReviewProgWrap", 3);
  activityStart(
    "full-song-render",
    "Rendering full Dynamite",
    "Combining the current version from all five sections",
    3,
  );
  try {
    const state = await api("/api/full-song-review", { method: "POST" });
    FULL_REVIEW_STATE = state;
    FULL_REVIEW_COMPLETED_SIGNATURE = null;
    renderFullReview(state);
    pollFullSongReview();
  } catch (e) {
    $("fullReviewRender").disabled = false;
    $("fullReviewStatus").className = "full-review-status bad";
    $("fullReviewStatus").textContent = `\u26a0 ${e.message}`;
    $("fullReviewProgWrap").hidden = true;
    activityFail("full-song-render", e.message);
  }
}

// -------------------------------------------------------------- before/after window comparison
function currentCompareHead() {
  return (ST.timeline || []).find((checkpoint) => checkpoint.is_head) || null;
}

function compareAncestors(head) {
  if (!head) return [];
  const byId = new Map((ST.timeline || []).map((checkpoint) => [checkpoint.id, checkpoint]));
  if (Array.isArray(head.lineage) && head.lineage.length) {
    return head.lineage.slice(0, -1).map((id) => byId.get(id)).filter(Boolean);
  }
  return (ST.timeline || [])
    .filter((checkpoint) => checkpoint.id !== head.id && checkpoint.is_ancestor_of_head)
    .sort((a, b) => Number(a.depth || 0) - Number(b.depth || 0));
}

function updateCompareAvailability() {
  const available = !!compareAncestors(currentCompareHead()).length;
  $("compareBtn").disabled = COMPARE_LOADING || !available;
  $("compareButtonMeta").textContent = available
    ? "Synchronized side by side"
    : "Available after your first edit";
  return available;
}

function populateCmpVersions(preserveSelection = false) {
  const sel = $("cmpVersion");
  const head = currentCompareHead();
  const options = compareAncestors(head);
  const previous = preserveSelection ? sel.value : "";
  sel.innerHTML = "";
  options.forEach((checkpoint) => {
    const o = document.createElement("option");
    o.value = checkpoint.id;
    if (!checkpoint.parent_id) o.textContent = "Original dance";
    else if (checkpoint.id === head.parent_id) {
      o.textContent = `Before latest edit \u2014 ${checkpoint.label || "earlier version"}`;
    } else {
      o.textContent = `Earlier version \u2014 ${checkpoint.label || "checkpoint"}`;
    }
    sel.appendChild(o);
  });
  if (previous && options.some((checkpoint) => checkpoint.id === previous)) {
    sel.value = previous;
  } else if (options.length) {
    sel.value = options[0].id;
  }
  sel.disabled = COMPARE_LOADING || options.length <= 1;
  return head;
}

function resetCompareMedia() {
  const videos = [$("cmpBefore"), $("cmpAfter")];
  videos.forEach((video) => {
    try { video.pause(); } catch (e) {}
    video.removeAttribute("src");
    video.load();
  });
  const audio = $("cmpAudio");
  try { audio.pause(); } catch (e) {}
  audio.removeAttribute("src");
  audio.load();
  $("cmpContent").hidden = true;
  $("cmpPlaceholder").hidden = false;
  $("cmpScrub").value = 0;
  $("cmpTime").textContent = "0:00 / 0:00";
  $("cmpPlay").textContent = "\u25b6 Play both";
}

function setCompareLoading(loading) {
  COMPARE_LOADING = loading;
  updateCompareAvailability();
  const sel = $("cmpVersion");
  sel.disabled = loading || sel.options.length <= 1;
}

function closeCompare(focusButton = true) {
  const panel = $("compare");
  if (panel.hidden) return;
  clearTimeout(COMPARE_POLL_TIMER);
  COMPARE_POLL_TIMER = 0;
  COMPARE_CONTEXT = null;
  setCompareLoading(false);
  resetCompareMedia();
  panel.hidden = true;
  if ($("fullReview").hidden) document.body.classList.remove("modal-open");
  if (focusButton) $("compareBtn").focus();
}

function compareError(message) {
  clearTimeout(COMPARE_POLL_TIMER);
  COMPARE_POLL_TIMER = 0;
  const status = $("cmpStatus");
  status.className = "compare-status bad";
  status.textContent = `\u26a0 ${message}`;
  $("cmpProgWrap").hidden = true;
  $("cmpContent").hidden = true;
  $("cmpPlaceholder").hidden = false;
  $("cmpPlaceholder").querySelector("strong").textContent = "Comparison unavailable";
  $("cmpPlaceholder").querySelector("span").textContent = message;
  setCompareLoading(false);
  activityFail("compare", message);
}

async function startCompare(preserveSelection = false) {
  const head = populateCmpVersions(preserveSelection);
  if (!head || !compareAncestors(head).length) {
    updateCompareAvailability();
    toast("Make an edit first, then compare Original and Edited");
    return;
  }
  const panel = $("compare");
  panel.hidden = false;
  document.body.classList.add("modal-open");
  resetCompareMedia();
  const status = $("cmpStatus");
  status.className = "compare-status";
  status.textContent = "Preparing synchronized Original and Edited clips\u2026";
  $("cmpPlaceholder").querySelector("strong").textContent = "Preparing Original and Edited clips\u2026";
  $("cmpPlaceholder").querySelector("span").textContent =
    "The comparison will appear here when both synchronized renders are ready.";
  setProgressBar("cmpProg", "cmpProgWrap", 3);
  activityStart("compare", "Rendering Original vs Edited", "Preparing synchronized clips", 3);
  setCompareLoading(true);
  const context = {
    sid: ST.sid,
    headId: head.id,
    comparisonId: null,
  };
  COMPARE_CONTEXT = context;
  try {
    const fromId = $("cmpVersion").value || null;
    const job = await api(`/api/session/${context.sid}/compare`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_id: fromId }) });
    if (COMPARE_CONTEXT !== context) return;
    context.comparisonId = job.comparison_id || null;
    pollCompare(context);
    $("cmpClose").focus();
  } catch (e) {
    if (COMPARE_CONTEXT === context) compareError(e.message);
  }
}

async function pollCompare(context) {
  clearTimeout(COMPARE_POLL_TIMER);
  if (
    COMPARE_CONTEXT !== context
    || $("compare").hidden
    || ST.sid !== context.sid
    || ST.head !== context.headId
  ) return;
  let j;
  try { j = await api(`/api/session/${context.sid}/compare`); }
  catch (e) {
    activityUpdate("compare", { detail: "Reconnecting to comparison status\u2026" });
    COMPARE_POLL_TIMER = setTimeout(() => pollCompare(context), 900);
    return;
  }
  if (
    context.comparisonId
    && j.comparison_id
    && context.comparisonId !== j.comparison_id
  ) {
    compareError("A newer comparison replaced this one. Start Original vs Edited again.");
    return;
  }
  const status = $("cmpStatus");
  status.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  setProgressBar("cmpProg", "cmpProgWrap", Number(j.progress || 0));
  activityUpdate("compare", {
    progress: Number(j.progress || 0),
    detail: j.message || j.status,
  });
  if (j.status === "done") {
    activityUpdate("compare", { progress: 96, detail: "Loading comparison videos\u2026" });
    let mediaReady;
    try {
      mediaReady = await setupCompareVideos(
        context.sid,
        j.before_video,
        j.after_video,
        j.metrics || {},
        j.audio,
        j.highlight || null,
        context.comparisonId,
      );
    } catch (error) {
      compareError(error instanceof Error ? error.message : String(error));
      return;
    }
    if (COMPARE_CONTEXT !== context) return;
    status.className = "compare-status ok";
    status.textContent = `\u2714 Original and Edited are ready${j.elapsed ? " in " + j.elapsed + "s" : ""}.`;
    $("cmpProgWrap").hidden = true;
    setCompareLoading(false);
    activityDone(
      "compare",
      mediaReady ? "Comparison ready" : "Comparison ready; videos are still buffering",
    );
    toast("Original vs Edited comparison ready");
    return;
  }
  if (j.status === "stale") {
    compareError(j.message || "The dance changed. Start a new comparison.");
    return;
  }
  if (j.status === "error") {
    compareError(j.message || "Comparison failed");
    toast("Compare failed");
    return;
  }
  COMPARE_POLL_TIMER = setTimeout(() => pollCompare(context), 650);
}

function showCompareMetrics(m) {
  const t = $("cmpMetrics"); t.innerHTML = "";
  const b = m.before || {}, a = m.after || {};
  if (m.window_sec) {
    $("cmpWin").textContent =
      `Edit window ${clockLabel(m.window_sec[0])}\u2013${clockLabel(m.window_sec[1])}`;
  }
  $("cmpBeforeDetail").textContent = m.before_is_original
    ? "Original dance"
    : m.before_label || "Earlier version";
  $("cmpAfterDetail").textContent = m.after_label || "Current LLM-guided edit";
  if (b.bas === undefined) return;
  t.appendChild(metricHeader());
  ["energy", "bas", "jerk", "foot"].forEach((k) => {
    if (b[k] === undefined || a[k] === undefined) return;
    t.appendChild(metricRow(k, b[k], a[k]));
  });
}

function compareTimeLabel(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

async function setupCompareVideos(
  sid,
  beforeName,
  afterName,
  metrics,
  audioName,
  highlight,
  comparisonId,
) {
  const A = $("cmpAfter"), B = $("cmpBefore"), AU = $("cmpAudio");
  if (!beforeName || !afterName) {
    throw new Error("The comparison render did not provide both videos.");
  }
  wireMediaBuffering(A, "compare-buffer", "Buffering comparison");
  wireMediaBuffering(B, "compare-buffer", "Buffering comparison");
  const bust = `?comparison=${encodeURIComponent(comparisonId || "")}&t=${Date.now()}`;
  A.src = `/api/session/${sid}/media/${afterName}${bust}`;
  B.src = `/api/session/${sid}/media/${beforeName}${bust}`;
  A.load(); B.load();
  const haveAudio = !!(audioName && AU);
  if (AU) {
    AU.src = haveAudio ? `/api/session/${sid}/media/${audioName}${bust}` : "";
    AU.muted = !haveAudio;
    if (haveAudio) AU.load();
  }
  showCompareMetrics(metrics);
  const changedParts = Array.isArray(highlight?.parts) ? highlight.parts : [];
  $("cmpChangeSummary").textContent = changedParts.length
    ? `Detected change: ${changedParts.join(" + ")}.`
    : "Both clips show the exact same time window; compare the movement directly.";
  const updateTime = () => {
    $("cmpTime").textContent =
      `${compareTimeLabel(A.currentTime)} / ${compareTimeLabel(A.duration)}`;
  };
  const syncSecondary = (force = false) => {
    if (!Number.isFinite(A.currentTime)) return;
    if (force || Math.abs((B.currentTime || 0) - A.currentTime) > 0.05) {
      try { B.currentTime = A.currentTime; } catch (e) {}
    }
    if (haveAudio && (force || Math.abs((AU.currentTime || 0) - A.currentTime) > 0.12)) {
      try { AU.currentTime = A.currentTime; } catch (e) {}
    }
  };
  const setPlayLabel = () => {
    $("cmpPlay").textContent = A.paused ? "\u25b6 Play both" : "\u23f8 Pause both";
  };
  const playBoth = async () => {
    syncSecondary(true);
    await Promise.allSettled([
      A.play(),
      B.play(),
      haveAudio ? AU.play() : Promise.resolve(),
    ]);
    setPlayLabel();
  };
  const pauseBoth = () => {
    A.pause();
    B.pause();
    if (haveAudio) AU.pause();
    setPlayLabel();
  };
  $("cmpPlay").onclick = () => { if (A.paused) playBoth(); else pauseBoth(); };
  A.onplay = () => {
    if (B.paused) B.play().catch(() => {});
    if (haveAudio && AU.paused) AU.play().catch(() => {});
    setPlayLabel();
  };
  A.onpause = () => {
    if (!B.paused) B.pause();
    if (haveAudio) AU.pause();
    setPlayLabel();
  };
  A.ontimeupdate = () => {
    const d = A.duration || 1;
    $("cmpScrub").value = Math.round((A.currentTime / d) * 1000);
    syncSecondary(false);
    updateTime();
  };
  $("cmpScrub").oninput = () => {
    const d = A.duration || 1, t = ($("cmpScrub").value / 1000) * d;
    try { A.currentTime = t; B.currentTime = t; if (haveAudio) AU.currentTime = t; } catch (e) {}
    updateTime();
  };
  A.onloadedmetadata = updateTime;
  const mediaReady = await waitForMediaReady(
    [A, B],
    "compare",
    96,
    100,
    "Loading comparison videos\u2026",
  );
  if (!mediaReady) {
    throw new Error("The comparison videos did not become ready. Try the comparison again.");
  }
  syncSecondary(true);
  updateTime();
  $("cmpPlaceholder").hidden = true;
  $("cmpContent").hidden = false;
  return mediaReady;
}

// -------------------------------------------------------------- load songs + session
async function runHistoryAction(label, endpoint) {
  activityStart("history", label, "Updating the current checkpoint", null);
  try {
    const st = await api(`/api/session/${ST.sid}/${endpoint}`, { method: "POST" });
    ST.branchBase = null;
    applyState(st);
    activityDone("history", `${label} complete`);
    return st;
  } catch (e) {
    activityFail("history", e.message);
    toast(`${label} failed: ${e.message}`);
    return null;
  }
}

function wireControls() {
  wireTimeline();
  wireUpload();
  $("apply").onclick = runEdit;
  $("fullRenderBtn").onclick = openFullReview;
  $("prevSection").onclick = () => moveSection(-1);
  $("nextSection").onclick = () => moveSection(1);
  $("fullReviewBackdrop").onclick = closeFullReview;
  $("fullReviewClose").onclick = closeFullReview;
  $("fullReviewContinue").onclick = closeFullReview;
  $("fullReviewRender").onclick = startFullSongReview;
  $("fullReviewVideo").addEventListener("error", () => {
    $("fullReviewVideo").dataset.signature = "";
    $("fullReviewStatus").className = "full-review-status bad";
    $("fullReviewStatus").textContent =
      "The full-song video could not be loaded. Your section edits are safe; reload the preview.";
    $("fullReviewRender").disabled = false;
    $("fullReviewRender").textContent = "Reload full-song preview";
  });
  $("compareBtn").onclick = () => startCompare(false);
  $("compareBackdrop").onclick = () => closeCompare();
  $("cmpClose").onclick = () => closeCompare();
  $("cmpVersion").onchange = () => {
    if (!$("compare").hidden) startCompare(true);
  };
  [$("cmpBefore"), $("cmpAfter")].forEach((video) => {
    video.addEventListener("error", () => {
      if (!$("compare").hidden) {
        compareError("One of the comparison videos could not be loaded. Try the comparison again.");
      }
    });
  });
  $("undo").onclick = () => runHistoryAction("Undoing edit", "undo");
  $("redo").onclick = () => runHistoryAction("Redoing edit", "redo");
  $("branchCancel").onclick = () => clearBranchBase(true);
  $("motionPreviewClose").onclick = closeMotionPreview;
  $("motionPreviewBackdrop").onclick = closeMotionPreview;
  $("motionPreviewUse").onclick = useMotionPreview;
  $("motionPreviewVideo").addEventListener("error", () => {
    const video = $("motionPreviewVideo");
    video.hidden = true;
    $("motionPreviewStatus").textContent =
      "This example video could not be loaded. The motion description and prompt action remain available.";
  });
  $("reset").onclick = async () => {
    if (!confirm("Clear the edit history and start over from the original dance? This cannot be undone.")) return;
    activityStart("history", "Resetting edit history", "Restoring the original dance", null);
    try {
      const st = await api(`/api/session/${ST.sid}/reset`, { method: "POST" });
      closeCompare(false);
      const video = $("video");
      video.src = st.preview_url + "?t=" + Date.now();       // back to the original dance
      video.load();
      activityUpdate("history", { progress: 70, detail: "Loading the original preview\u2026" });
      await waitForMediaReady([video], "history", 70, 100, "Loading the original preview\u2026");
      applyState(st);
      showCurrentMetrics(st.metrics || {});
      activityDone("history", "Original dance restored");
      toast("Edit history cleared \u2014 back to the original");
    } catch (e) {
      activityFail("history", e.message);
      toast("Reset failed: " + e.message);
    }
  };
  $("instruction").addEventListener("keydown", (e) => { if (e.key === "Enter") runEdit(); });
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("compare").hidden) {
      closeCompare();
      return;
    }
    if (e.key === "Escape" && !$("fullReview").hidden) {
      closeFullReview();
      return;
    }
    if (e.key === "Escape" && !$("motionPreview").hidden) closeMotionPreview();
  });
  wireTour();
}

// -------------------------------------------------------------- first-run walkthrough (skippable)
const TOUR_KEY = "maestro_onboarded_v1";
const TOUR_STEPS = [
  { el: "timeline", title: "1 · Pick the part to edit",
    text: "Drag across this bar to choose the window of the dance you want to change. The preview pauses at the shaded window's start so the displayed frame matches the start marker." },
  { el: "instruction", title: "2 · Say what you want",
    text: "Describe the change in plain English, for example \u201cmake it more energetic\u201d, \u201ctighten to the beat\u201d, \u201cclap to the right\u201d, or \u201cwave here\u201d. Named actions replace motion in the selected window; insertion is unavailable until it has its own visual audit. If you omit a direction, MAESTRO follows the dance flow." },
  { el: "motionPickerSummary", title: "3 · Browse 19 supported motions",
    text: "Open this collapsed catalog when you need it. Click a motion to watch the exact editor example, read its meaning, and add it to your prompt. Beat-hit motions land on the strongest beat in the selected window." },
  { el: "apply", title: "4 · Apply the edit",
    text: "The agent plans the right tools, applies them, and verifies the result actually hit your goal. If needed, it refines the edit." },
  { el: "compareBtn", title: "5 · Compare Original and Edited",
    text: "After an edit, open two clearly labeled videos of the same window. Play or scrub them together to compare the original dance directly with the current LLM-guided edit." },
  { el: "history", title: "6 · Iterate freely",
    text: "Every edit is a checkpoint. Restore one by clicking its row, or use its Branch button to make the next edit from that older state without deleting later work." },
  { el: "fullRenderBtn", title: "7 · Review the full song",
    text: "When you are satisfied, combine the current version from all five Dynamite sections into one continuous full-song preview. From the review, jump straight back to any section that needs more work." },
  { el: "song", title: "That\u2019s it. Have fun!",
    text: "Use the previous and next arrows or this section menu to move through Dynamite in order. Tap the \u201c?\u201d in the top bar to see this again anytime." },
];
let tourIdx = 0;

function showTourStep(i) {
  const step = TOUR_STEPS[i];
  if (!step) return endTour();
  const target = $(step.el), ring = $("tourRing"), card = $("tourCard");
  if (target) {
    if (target.tagName === "DETAILS") target.open = true;
    const r = target.getBoundingClientRect(), pad = 6;
    ring.style.display = "block";
    ring.style.left = (r.left - pad) + "px"; ring.style.top = (r.top - pad) + "px";
    ring.style.width = (r.width + 2 * pad) + "px"; ring.style.height = (r.height + 2 * pad) + "px";
    const cardW = 300, cardH = 190;
    let left = Math.min(Math.max(12, r.left), window.innerWidth - cardW - 12);
    let top = r.bottom + 14;
    if (top + cardH > window.innerHeight - 12) top = Math.max(12, r.top - cardH - 14);
    card.style.left = left + "px"; card.style.top = top + "px";
  } else {
    ring.style.display = "none";
    card.style.left = (window.innerWidth / 2 - 150) + "px"; card.style.top = "40%";
  }
  $("tourStep").textContent = `Step ${i + 1} of ${TOUR_STEPS.length}`;
  $("tourTitle").textContent = step.title;
  $("tourText").textContent = step.text;
  $("tourNext").textContent = i === TOUR_STEPS.length - 1 ? "Done" : "Next";
  $("tourDots").innerHTML = TOUR_STEPS.map((_s, k) => `<i class="${k === i ? "on" : ""}"></i>`).join("");
}

function startTour() { tourIdx = 0; $("tour").hidden = false; showTourStep(0); }
function endTour() { $("tour").hidden = true; try { localStorage.setItem(TOUR_KEY, "1"); } catch (e) {} }

function wireTour() {
  $("helpBtn").onclick = startTour;
  $("tourSkip").onclick = endTour;
  $("tourNext").onclick = () => { tourIdx += 1; tourIdx >= TOUR_STEPS.length ? endTour() : showTourStep(tourIdx); };
  window.addEventListener("resize", () => { if (!$("tour").hidden) showTourStep(tourIdx); });
  window.addEventListener("keydown", (e) => { if (!$("tour").hidden && e.key === "Escape") endTour(); });
}

function maybeAutoTour() {
  let seen = false;
  try { seen = !!localStorage.getItem(TOUR_KEY); } catch (e) {}
  if (!seen) setTimeout(startTour, 800);                    // once the first song has loaded in
}

function motionInstruction(motion) {
  const phrase = (motion.aliases && motion.aliases[0]) || motion.name || motion.id;
  return `add ${String(phrase).toLowerCase()} here`;
}

function closeMotionPreview() {
  const panel = $("motionPreview"), video = $("motionPreviewVideo");
  try { video.pause(); } catch (e) {}
  video.removeAttribute("src");
  video.load();
  panel.hidden = true;
  MOTION_PREVIEW = null;
}

function useMotionPreview() {
  if (!MOTION_PREVIEW) return;
  $("instruction").value = motionInstruction(MOTION_PREVIEW);
  closeMotionPreview();
  $("instruction").focus();
}

function openMotionPreview(motion) {
  MOTION_PREVIEW = motion;
  $("motionPreviewTitle").textContent = motion.name;
  $("motionPreviewDescription").textContent = motion.description || "";
  const timing = motion.default_anchor === "beat"
    ? "The main action lands on the strongest feasible beat in the selected window."
    : "The action is centered in the selected window.";
  const direction = motion.directions && motion.directions.length
    ? ` Directions: ${motion.directions.join(", ")}; auto follows the dance flow.`
    : "";
  $("motionPreviewMeta").textContent =
    `${timing}${direction} The example uses this same manifest clip and editor composition path.`;

  const panel = $("motionPreview"), video = $("motionPreviewVideo");
  const status = $("motionPreviewStatus");
  panel.hidden = false;
  video.hidden = true;
  video.removeAttribute("src");
  status.textContent = "";
  if (motion.preview_available) {
    status.textContent = "Loading the editor example\u2026";
    video.src = motion.preview_url;
    video.hidden = false;
    video.load();
    video.addEventListener("canplay", () => {
      if (MOTION_PREVIEW && MOTION_PREVIEW.id === motion.id) status.textContent = "";
    }, { once: true });
    video.play().catch(() => {});
  } else {
    status.textContent =
      "Example video is not generated on this host yet. This is not a placeholder: the GPU-pod "
      + "generation command produces the deterministic MP4 for this exact motion.";
  }
  $("motionPreviewClose").focus();
}

async function loadMotionBank() {
  const host = $("motionSuggestions");
  if (!host) return;
  activityStart("motion-catalog", "Loading motion catalog", "Fetching supported motions", null);
  try {
    const data = await api("/api/motions");
    host.innerHTML = "";
    const motions = data.motions || [];
    $("motionCount").textContent = String(motions.length);
    motions.forEach((motion) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "chip";
      button.textContent = motion.name;
      const timing = motion.default_anchor === "beat"
        ? "lands on the strongest beat in the selected window by default"
        : "centered in the selected window by default";
      const direction = motion.directions && motion.directions.length
        ? ` · directions: ${motion.directions.join(", ")}; auto follows the dance flow`
        : "";
      button.title = `${motion.description} · ${motion.category} · ${timing}${direction}`;
      button.setAttribute("aria-label", `Watch ${motion.name} example`);
      button.onclick = () => openMotionPreview(motion);
      host.appendChild(button);
    });
    activityDone("motion-catalog", `${motions.length} motions ready`);
  } catch (e) {
    host.textContent = "Named motions are unavailable.";
    activityFail("motion-catalog", "Motion catalog unavailable");
  }
}

async function loadSongs(maxAttempts = 20) {
  // Retry: the page can load while the pod editor is still starting, in which case the very first
  // /api/songs may fail or return empty. Poll until it answers so the page self-heals instead of
  // getting stuck on a dead, songless screen.
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const { songs } = await api("/api/songs");
      if (songs && songs.length) return songs;
    } catch (e) { /* editor still coming up -- keep polling */ }
    activityUpdate("startup", {
      progress: Math.min(45, 8 + i * 2),
      detail: `Waiting for the editor service${i ? `, retry ${i + 1}` : ""}`,
    });
    if (i === 0) toast("Loading songs\u2026 the editor may still be starting");
    await sleep(1500);
  }
  return [];
}

function updateSectionNavigation(sid = ST.sid) {
  const index = SONGS.findIndex((song) => song.sid === sid);
  const total = SONGS.length;
  $("sectionPosition").textContent =
    index >= 0 && total ? `Section ${index + 1} of ${total}` : "Section";
  $("prevSection").disabled = index <= 0;
  $("nextSection").disabled = index < 0 || index >= total - 1;
}

async function moveSection(delta) {
  const index = SONGS.findIndex((song) => song.sid === ST.sid);
  const target = SONGS[index + delta];
  if (!target) return;
  const select = $("song");
  select.value = target.sid;
  await openSession(target.sid);
}

async function init() {
  activityStart("startup", "Starting MAESTRO", "Loading songs and editor services", 5);
  const sel = $("song");
  sel.innerHTML = "<option>Loading\u2026</option>"; sel.disabled = true;
  wireControls();                                            // wire up-front so controls work at once
  loadMotionBank();
  const songs = await loadSongs();
  sel.innerHTML = "";
  if (!songs.length) {
    sel.innerHTML = "<option>no songs \u2014 refresh</option>";
    activityFail("startup", "Songs could not be loaded");
    toast("Couldn't load songs \u2014 the editor may still be starting. Please refresh in a moment.");
    return;
  }
  SONGS = songs;
  activityUpdate("startup", { progress: 55, detail: "Opening the first song" });
  sel.disabled = false;
  songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid; o.textContent = s.name || s.sid; sel.appendChild(o); });
  sel.onchange = () => openSession(sel.value);
  const opened = await openSession(songs[0].sid);
  if (!opened) {
    activityFail("startup", "The first song could not be opened");
    return;
  }
  activityDone("startup", "Editor ready");
  maybeAutoTour();                                          // first-run walkthrough (skippable)
}

async function openSession(sid) {
  const sel = $("song");
  const label = sel && sel.selectedOptions.length ? sel.selectedOptions[0].textContent : sid;
  activityStart("session", "Loading song", `Opening ${label}`, 8);
  if (sel) sel.disabled = true;
  $("compareBtn").disabled = true;
  $("compareButtonMeta").textContent = "Loading section\u2026";
  updateSectionNavigation(sid);
  $("prevSection").disabled = true;
  $("nextSection").disabled = true;
  let st = null;
  for (let i = 0; i < 4 && !st; i++) {
    try { st = await api(`/api/session/${sid}`, { method: "POST" }); }
    catch (e) {
      activityUpdate("session", {
        progress: 12 + i * 8,
        detail: `Waiting for the song session, attempt ${i + 2}`,
      });
      await sleep(1200);
    }                                                            // pod may still be warming
  }
  if (!st) {
    if (sel) {
      sel.value = ST.sid;
      sel.disabled = false;
    }
    updateSectionNavigation();
    updateCompareAvailability();
    activityFail("session", `Could not open ${label}`);
    toast(`Couldn't open ${sid} \u2014 please refresh in a moment.`);
    return false;
  }
  closeCompare(false);
  ST.sid = sid;
  ST.branchBase = null;
  updateSectionNavigation();
  applyState(st);
  activityUpdate("session", { progress: 55, detail: "Loading the dance preview\u2026" });
  const v = $("video");
  v.src = st.preview_url + "?t=" + Date.now();
  v.load();
  const mediaReady = await waitForMediaReady(
    [v],
    "session",
    55,
    100,
    "Loading the dance preview\u2026",
    12000,
  );
  if (sel) sel.disabled = false;
  updateSectionNavigation();
  activityDone("session", mediaReady ? `${label} ready` : `${label} loaded; preview is buffering`);
  toast(`Loaded ${sid}: ${st.duration}s, ${st.n_beats} beats, ${st.generator} generator`);
  return true;
}

function applyState(st, opts) {
  opts = opts || {};
  if (ST.head && ST.head !== st.head) closeCompare(false);
  ST.fps = st.fps; ST.dur = st.duration; ST.nframes = st.n_frames; ST.beats = st.n_beats; ST.head = st.head;
  ST.generator = st.generator;
  ST.timeline = st.timeline || [];
  if (ST.branchBase && !ST.timeline.some((c) => c.id === ST.branchBase)) ST.branchBase = null;
  setGenBadge(st.generator);
  $("undo").disabled = !st.can_undo; $("redo").disabled = !st.can_redo;
  drawBeatsTicks();
  renderHistory(st.timeline);
  renderBranchState();
  syncTimelineToVideo();
  syncPlayhead();
  updateCompareAvailability();
  if (!opts.keepMetrics && st.metrics && st.metrics.energy !== undefined) showCurrentMetrics(st.metrics);
  scheduleFullReviewRefresh();
}

const GEN_INFO = {
  live: ["live pod", "Live pod mode: every new seed runs a fresh LODGE/EDGE diffusion sample on the GPU pod (minutes per new take), so edits search an unbounded space. Falls back to the local bank if the pod can't answer."],
  bank: ["bank", "Editing selects from a pre-generated bank of real LODGE/EDGE takes for this song. Instant, and works with the pod switched off, but only as diverse as the number of cached seeds."],
  mock: ["offline demo", "No real backbone takes for this song yet, using the offline stand-in generator. Connect a GPU pod (bank or live mode) for real search."],
};
function setGenBadge(kind) {
  const el = $("genbadge"); if (!el) return;
  const [label, tip] = GEN_INFO[kind] || GEN_INFO.mock;
  el.textContent = label; el.title = tip;
  el.className = "genbadge " + (kind || "mock");
}

function checkpointDisplayLabel(checkpoint) {
  if (!checkpoint) return "unknown checkpoint";
  return checkpoint.label
    ? String(checkpoint.label).replace(/\s*\[[^\]]*\]/, "").replace(/_/g, " ")
    : "original";
}

function clearBranchBase(announce) {
  const hadBase = ST.branchBase;
  ST.branchBase = null;
  renderBranchState();
  renderHistory(ST.timeline);
  if (announce && hadBase) toast("Branch selection cleared");
}

function selectBranchBase(checkpointId) {
  const checkpoint = ST.timeline.find((item) => item.id === checkpointId);
  if (!checkpoint || checkpoint.id === ST.head) return;
  ST.branchBase = checkpoint.id;
  renderBranchState();
  renderHistory(ST.timeline);
  toast(`Next edit will branch from ${checkpointDisplayLabel(checkpoint)}`);
}

function renderBranchState() {
  const state = $("branchState");
  if (!state) return;
  const base = ST.timeline.find((item) => item.id === ST.branchBase);
  if (!base || base.id === ST.head) {
    state.hidden = true;
    if (!base) ST.branchBase = null;
    return;
  }
  const lineageIds = EDITOR_UTILS.checkpointLineage(ST.timeline, base.id);
  const labels = lineageIds
    .map((id) => checkpointDisplayLabel(ST.timeline.find((item) => item.id === id)))
    .filter(Boolean);
  $("branchBaseLabel").textContent = checkpointDisplayLabel(base);
  $("branchLineage").textContent =
    `${labels.join(" \u2192 ")}. The preview and metrics still show the current head until this edit is applied.`;
  state.hidden = false;
}

// -------------------------------------------------------------- timeline drawing + selection
function pct(sec) {
  const value = Number(sec);
  return ST.dur && Number.isFinite(value)
    ? Math.max(0, Math.min(100, (value / ST.dur) * 100))
    : 0;
}
function drawBeatsTicks() {
  const ticks = $("ticks"); ticks.innerHTML = "";
  const step = ST.dur > 200 ? 30 : ST.dur > 60 ? 15 : 5;
  for (let t = 0; t <= ST.dur + 0.1; t += step) {
    const b = document.createElement("b"); b.style.left = pct(t) + "%"; b.textContent = t + "s"; ticks.appendChild(b);
  }
}
function drawSelection() {
  const s = $("selection");
  if (!ST.sel) { s.style.display = "none"; return; }
  const [a, b] = ST.sel;
  s.style.display = "block"; s.style.left = pct(a) + "%"; s.style.width = (pct(b) - pct(a)) + "%";
  $("aSec").value = a.toFixed(1); $("bSec").value = b.toFixed(1);
}
function seekPreviewToSelectionStart() {
  const video = $("video");
  const start = ST.sel && Number(ST.sel[0]);
  if (!video || video.readyState < 1 || !Number.isFinite(start)) return;
  video.pause();
  const mediaDuration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : ST.dur;
  video.currentTime = Math.max(0, Math.min(start, mediaDuration));
  syncPlayhead();
}
function setSel(a, b) {
  a = Math.max(0, Math.min(a, ST.dur)); b = Math.max(0, Math.min(b, ST.dur));
  ST.sel = [Math.min(a, b), Math.max(a, b)];
  drawSelection();
  seekPreviewToSelectionStart();
}

function syncTimelineToVideo() {
  const wrap = $("tlWrap"), viewer = $("viewer"), video = $("video");
  if (!wrap || !viewer || !video) return;
  const viewerRect = viewer.getBoundingClientRect();
  const videoRect = video.getBoundingClientRect();
  const media = EDITOR_UTILS.renderedMediaBox(videoRect, video.videoWidth, video.videoHeight);
  const width = media.width > 0 ? media.width : viewerRect.width;
  wrap.style.width = `${Math.max(1, Math.min(viewerRect.width, width))}px`;
}

function syncPlayhead() {
  const video = $("video"), playhead = $("playhead");
  if (!video || !playhead) return;
  const mediaDuration = Number.isFinite(video.duration) && video.duration > 0
    ? video.duration
    : ST.dur;
  const fraction = mediaDuration > 0
    ? Math.max(0, Math.min(1, video.currentTime / mediaDuration))
    : 0;
  playhead.style.left = `${fraction * 100}%`;
}

function startPlayheadLoop() {
  if (PLAYHEAD_RAF) return;
  const tick = () => {
    PLAYHEAD_RAF = 0;
    syncPlayhead();
    const video = $("video");
    if (video && !video.paused && !video.ended) PLAYHEAD_RAF = requestAnimationFrame(tick);
  };
  PLAYHEAD_RAF = requestAnimationFrame(tick);
}

function wireTimeline() {
  const tl = $("timeline");
  let pointerId = null, startSec = 0;
  const secAt = (e) => EDITOR_UTILS.timelineFraction(
    e.clientX,
    tl.getBoundingClientRect(),
    {
      clientLeft: tl.clientLeft,
      clientWidth: tl.clientWidth,
      offsetWidth: tl.offsetWidth,
    },
  ) * ST.dur;
  const finishDrag = (e) => {
    if (pointerId === null || (e && e.pointerId !== pointerId)) return;
    if (e) setSel(startSec, secAt(e));
    try { tl.releasePointerCapture(pointerId); } catch (err) {}
    pointerId = null;
  };
  tl.addEventListener("pointerdown", (e) => {
    if (e.pointerType === "mouse" && e.button !== 0) return;
    e.preventDefault();
    pointerId = e.pointerId;
    startSec = secAt(e);
    tl.setPointerCapture(pointerId);
    setSel(startSec, startSec);
  });
  tl.addEventListener("pointermove", (e) => {
    if (e.pointerId === pointerId) setSel(startSec, secAt(e));
  });
  tl.addEventListener("pointerup", finishDrag);
  tl.addEventListener("pointercancel", finishDrag);
  tl.addEventListener("lostpointercapture", () => { pointerId = null; });
  const v = $("video");
  wireMediaBuffering(v, "viewer-buffer", "Buffering dance preview");
  ["loadedmetadata", "durationchange", "resize"].forEach((name) =>
    v.addEventListener(name, () => {
      syncTimelineToVideo();
      syncPlayhead();
    }));
  ["timeupdate", "seeking", "seeked", "pause", "ended"].forEach((name) =>
    v.addEventListener(name, syncPlayhead));
  v.addEventListener("play", startPlayheadLoop);
  if (typeof ResizeObserver === "function") {
    const observer = new ResizeObserver(() => {
      syncTimelineToVideo();
      drawSelection();
      syncPlayhead();
    });
    observer.observe($("viewer"));
    observer.observe(v);
  }
  window.addEventListener("resize", () => {
    syncTimelineToVideo();
    drawSelection();
    syncPlayhead();
  });
  [$("aSec"), $("bSec")].forEach((inp) => inp.addEventListener("change", () => setSel(parseFloat($("aSec").value) || 0, parseFloat($("bSec").value) || 0)));
}

// -------------------------------------------------------------- edit (WebSocket w/ live progress)
// -------------------------------------------------------------- song upload (processed on the pod)
function newRequestId() {
  if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  return `upload-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function uploadAudio(file) {
  return new Promise((resolve, reject) => {
    const requestId = newRequestId();
    const browserStartedAtMs = Date.now();
    const browserStartedPerf = performance.now();
    let uploadCompletedPerf = null;
    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append("file", file);
    fd.append("request_id", requestId);
    fd.append("client_started_at_ms", String(browserStartedAtMs));
    xhr.open("POST", "/api/upload");
    xhr.setRequestHeader("X-MAESTRO-Request-ID", requestId);
    xhr.responseType = "json";
    xhr.timeout = 5 * 60 * 1000;
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) {
        activityUpdate("upload", { progress: null, detail: `Uploading ${file.name}` });
        setProgressBar("progbar", "agentProgWrap", null);
        return;
      }
      const progress = Math.round(100 * event.loaded / Math.max(1, event.total));
      activityUpdate("upload", {
        progress,
        detail: `Uploading ${file.name} (${progress}%)`,
      });
      setProgressBar("progbar", "agentProgWrap", progress);
      $("progtext").textContent = `uploading audio: ${progress}%`;
    };
    xhr.upload.onload = () => {
      uploadCompletedPerf = performance.now();
      activityUpdate("upload", {
        progress: null,
        detail: "Upload complete; preparing audio for generation",
      });
      setProgressBar("progbar", "agentProgWrap", null);
      $("progtext").textContent = "preparing audio for generation\u2026";
    };
    xhr.onload = () => {
      const body = xhr.response || {};
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve({
          ...body,
          browser_timing: {
            request_id: body.request_id || requestId,
            browser_started_at_ms: browserStartedAtMs,
            browser_started_perf: browserStartedPerf,
            browser_upload_seconds: (
              (uploadCompletedPerf || performance.now()) - browserStartedPerf
            ) / 1000,
            source_bytes: file.size,
          },
        });
      }
      else reject(new Error(body.detail || body.error || xhr.statusText || "upload failed"));
    };
    xhr.onerror = () => reject(new Error("network error while uploading"));
    xhr.ontimeout = () => reject(new Error("audio preparation timed out"));
    xhr.send(fd);
  });
}

function wireUpload() {
  const btn = $("uploadBtn"), input = $("uploadInput");
  if (!btn || !input) return;
  btn.onclick = () => input.click();
  input.onchange = async () => {
    const f = input.files && input.files[0];
    input.value = "";
    if (!f) return;
    btn.disabled = true;
    activityStart("upload", "Uploading audio", `Uploading ${f.name}`, 0);
    $("goal").innerHTML = `<b>Uploading “${escapeHtml(f.name)}”</b>`;
    $("feedback").textContent = "";
    $("feedback").className = "feedback";
    setProgressBar("progbar", "agentProgWrap", 0);
    toast(`Uploading ${f.name}…`);
    let job;
    try {
      job = await uploadAudio(f);
      activityDone("upload", "Audio received");
    } catch (e) {
      btn.disabled = false;
      activityFail("upload", e.message);
      setProgressBar("progbar", "agentProgWrap", 100);
      $("progtext").textContent = "upload failed";
      $("feedback").textContent = "\u26a0 " + e.message;
      $("feedback").className = "feedback bad";
      toast("Upload failed: " + e.message);
      return;
    }
    btn.disabled = false;
    if (job.error) {
      activityFail("upload", job.error);
      setProgressBar("progbar", "agentProgWrap", 100);
      $("progtext").textContent = "upload failed";
      toast(job.error);
      return;
    }
    pollJob(job.sid, f.name, job.browser_timing);
  };
}

async function watchBank(sid, name, initialJob) {
  const activityId = `bank:${sid}`;
  activityStart(
    activityId,
    "Expanding editing library",
    initialJob.bank_message || `Generating more alternatives for ${name}`,
    Number.isFinite(initialJob.bank_progress) ? initialJob.bank_progress : null,
  );
  while (true) {
    let job;
    try {
      job = await api(`/api/jobs/${sid}`);
    } catch (e) {
      activityUpdate(activityId, { detail: "Reconnecting to editing-library status\u2026" });
      await sleep(3000);
      continue;
    }
    if (job.bank_status === "ready") {
      activityDone(activityId, "Additional editing alternatives ready");
      return;
    }
    if (job.bank_status === "error") {
      activityFail(activityId, job.bank_error || job.bank_message || "Editing-library build failed");
      return;
    }
    activityUpdate(activityId, {
      progress: Number.isFinite(job.bank_progress) ? job.bank_progress : null,
      detail: job.bank_message || "Generating additional editing alternatives",
    });
    await sleep(5000);
  }
}

async function pollJob(sid, name, browserTiming = null) {
  const goal = $("goal"), pt = $("progtext"), fb = $("feedback");
  fb.textContent = ""; fb.className = "feedback";
  goal.innerHTML = `<b>Processing “${escapeHtml(name)}”</b> on the GPU pod`;
  setProgressBar("progbar", "agentProgWrap", 5);
  pt.textContent = "queued\u2026";
  const activityId = `process:${sid}`;
  activityStart(activityId, `Generating ${name}`, "Queued on the GPU pod", 5);
  while (true) {
    let j;
    try {
      j = await api(`/api/jobs/${sid}`);
    } catch (e) {
      activityUpdate(activityId, { detail: "Reconnecting to generation status\u2026" });
      await sleep(1500);
      continue;
    }
    const progress = Number(j.progress || 0);
    setProgressBar("progbar", "agentProgWrap", progress);
    const elapsed = j.elapsed ? ` \u00b7 ${elapsedLabel(Date.now() - Number(j.elapsed) * 1000)}` : "";
    pt.textContent = (j.message || j.status) + elapsed;
    activityUpdate(activityId, {
      progress,
      detail: j.message || j.status,
    });
    if (j.status === "done") {
      if (browserTiming) {
        const completedAtMs = Date.now();
        const totalSeconds = (
          performance.now() - browserTiming.browser_started_perf
        ) / 1000;
        try {
          await api(`/api/jobs/${sid}/browser-timing`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              request_id: browserTiming.request_id,
              browser_started_at_ms: browserTiming.browser_started_at_ms,
              browser_completed_at_ms: completedAtMs,
              browser_upload_seconds: browserTiming.browser_upload_seconds,
              browser_total_seconds: totalSeconds,
              source_bytes: browserTiming.source_bytes,
            }),
          });
        } catch (error) {
          console.warn("Could not persist browser timing", error);
        }
      }
      setProgressBar("progbar", "agentProgWrap", 100);
      fb.textContent = "\u2714 Ready"; fb.className = "feedback ok";
      activityDone(activityId, `${name} is ready`);
      if (j.bank_status === "building") watchBank(sid, name, j);
      const { songs } = await api("/api/songs");
      SONGS = songs;
      const sel = $("song"); sel.innerHTML = "";
      songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid;
        o.textContent = s.name || s.sid; sel.appendChild(o); });
      sel.value = sid; await openSession(sid);
      toast(`“${name}” is ready`);
      return;
    } else if (j.status === "error") {
      setProgressBar("progbar", "agentProgWrap", 100);
      fb.textContent = "\u26a0 " + (j.message || "processing failed"); fb.className = "feedback bad";
      activityFail(activityId, j.message || "Song generation failed");
      return;
    }
    await sleep(1000);
  }
}

function runEdit() {
  if (!ST.sel) { toast("Select a window on the timeline first"); return; }
  const instruction = $("instruction").value.trim();
  if (!instruction) { toast("Type an instruction"); return; }
  const [a, b] = ST.sel;
  const fromId = ST.branchBase || null;
  $("apply").disabled = true; $("feedback").textContent = ""; $("feedback").className = "feedback";
  $("goal").innerHTML = "";
  setProgressBar("progbar", "agentProgWrap", null);
  $("progtext").textContent = "connecting\u2026";
  activityStart("edit", "Applying edit", "Connecting to the editing agent", null);

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/session/${ST.sid}/edit_ws`);
  let sent = false, finished = false, fallbackStarted = false;
  ws.onopen = () => {
    activityUpdate("edit", { progress: 3, detail: "Sending the edit request" });
    ws.send(JSON.stringify({ a_sec: a, b_sec: b, instruction, from_id: fromId }));
    sent = true;
  };
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "progress") onProgress(ev);
    else if (ev.type === "final") {
      finished = true;
      onFinal(ev);
      ws.close();
    }
    else if (ev.type === "error") {
      finished = true;
      activityFail("edit", ev.message);
      setProgressBar("progbar", "agentProgWrap", 100);
      $("progtext").textContent = "edit failed";
      toast("Edit failed: " + ev.message);
      $("apply").disabled = false;
      ws.close();
    }
  };
  ws.onerror = () => {
    if (finished || fallbackStarted) return;
    if (!sent) {
      fallbackStarted = true;
      $("progtext").textContent = "socket unavailable; retrying over HTTP\u2026";
      activityUpdate("edit", { progress: null, detail: "Retrying the edit over HTTP" });
      runEditHTTP(a, b, instruction, fromId);
      return;
    }
    finished = true;
    setProgressBar("progbar", "agentProgWrap", 100);
    $("progtext").textContent = "connection lost during edit";
    activityFail("edit", "Connection lost after the edit started; check history before retrying");
    $("apply").disabled = false;
    toast("Edit connection lost; check history before retrying");
  };
}

async function runEditHTTP(a, b, instruction, fromId) {
  try {
    const r = await api(`/api/session/${ST.sid}/edit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a_sec: a, b_sec: b, instruction, from_id: fromId }) });
    onFinal(r);
  } catch (e) {
    activityFail("edit", e.message);
    setProgressBar("progbar", "agentProgWrap", 100);
    $("progtext").textContent = "edit failed";
    toast("Edit failed: " + e.message);
  } finally { $("apply").disabled = false; }
}

function onProgress(ev) {
  if (ev.phase === "plan") {
    $("goal").innerHTML = `<b>Agent plan:</b> ${escapeHtml(ev.summary || "")}`;
    $("checks").innerHTML = ""; $("reasoning").hidden = true;
    setProgressBar("progbar", "agentProgWrap", 8);
    const detail = `planning ${(ev.steps || []).length} step(s)\u2026`;
    $("progtext").textContent = detail;
    activityUpdate("edit", { progress: 8, detail });
  } else if (ev.phase === "refine") {
    setProgressBar("progbar", "agentProgWrap", 30);
    const detail = `refining (attempt ${ev.cycle}): ${ev.summary || ""}`;
    $("progtext").textContent = detail;
    activityUpdate("edit", { progress: 30, detail });
  } else if (ev.phase === "step") {
    const p = ev.n_steps ? Math.round(20 + (ev.step / ev.n_steps) * 60) : 50;
    setProgressBar("progbar", "agentProgWrap", p);
    const tag = ev.status === "rejected" ? " \u21a9 rejected" : ev.status === "applied" ? " \u2713" : "";
    const detail = `attempt ${ev.cycle || 1} \u00b7 step ${ev.step}/${ev.n_steps}: ${ev.tool}${tag}`;
    $("progtext").textContent = detail;
    activityUpdate("edit", { progress: p, detail });
  } else if (ev.phase === "verify") {
    const actionVerified = (ev.checks || []).some((check) =>
      check.metric === "semantic" && check.met);
    const detail = `attempt ${ev.cycle}: ${ev.ok
      ? (actionVerified ? "action verified \u2714" : "goals met \u2714")
      : "short of goal, refining\u2026"}`;
    setProgressBar("progbar", "agentProgWrap", ev.ok ? 94 : 86);
    $("progtext").textContent = detail;
    activityUpdate("edit", { progress: ev.ok ? 94 : 86, detail });
  } else if (ev.phase === "candidate") {   // legacy best-of-K generators
    const p = ev.total ? Math.round((ev.done / ev.total) * 100) : 0;
    setProgressBar("progbar", "agentProgWrap", p);
    const detail = `trying ${ev.backbone} take #${ev.seed} (${ev.done}/${ev.total})`;
    $("progtext").textContent = detail;
    activityUpdate("edit", { progress: p, detail });
  }
}

function metricDeltaStr(before, after) {
  if (!before || !after) return "";
  const keys = ["energy", "bas", "jerk", "foot"];
  const parts = [];
  for (const k of keys) {
    if (before[k] === undefined || after[k] === undefined) continue;
    const bv = metricDisp(k, before[k]), av = metricDisp(k, after[k]);
    if (Math.abs(av - bv) < 1e-3) continue;
    const lbl = { energy: "energy", bas: "beat", jerk: "smooth.", foot: "foot" }[k];
    parts.push(`${lbl} ${bv.toFixed(2)}\u2192${av.toFixed(2)}`);
  }
  return parts.join("  ");
}

// compact one-line-per-goal chips: did each thing the user asked for actually happen?
const ARROW = { improved: "\u25b2", regressed: "\u25bc", held: "\u2192" };
function renderChecks(trace) {
  const el = $("checks");
  const checks = (trace && trace.final && trace.final.checks) || [];
  if (!checks.length) { el.innerHTML = ""; return; }
  el.innerHTML = checks.map((c) => {
    const bv = metricDisp(c.metric, c.before), av = metricDisp(c.metric, c.after);
    return `<span class="chk ${c.met ? "met" : "miss"}" title="${c.met ? "goal met" : "not fully met"}">`
      + `${escapeHtml(c.label)} <b>${bv.toFixed(2)}${ARROW[c.status] || ""}${av.toFixed(2)}</b></span>`;
  }).join("");
}

function paramsStr(p) {
  if (!p || !Object.keys(p).length) return "";
  return ` <code class="rz-params">${escapeHtml(JSON.stringify(p))}</code>`;
}

// the expandable walk-through: for each attempt, the PLANNER's plan, the EXECUTOR's applied/rejected
// steps (with metric deltas), and the VERIFY checks. Collapsed by default to keep the panel calm.
const PLANNER_LABEL = { llm: "AI agent (LLM)", keyword: "offline keyword planner", keyword_fallback: "offline fallback (LLM failed)" };
function renderTrace(trace) {
  const det = $("reasoning"), body = $("reasoningBody");
  if (!trace || !(trace.attempts || []).length) { det.hidden = true; body.innerHTML = ""; return; }
  det.hidden = false;
  const goalLine = (trace.goals || []).map((g) => `${g.label} ${g.dir === "up" ? "\u2191" : "\u2193"}`).join(", ");
  let html = goalLine ? `<div class="rz-goals">Goals: ${escapeHtml(goalLine)}</div>` : "";
  if (trace.planner_note) html += `<div class="rz-goals">Planned by: ${escapeHtml(trace.planner_note)}</div>`;
  for (const at of trace.attempts) {
    const v = at.verify || {};
    const pl = at.plan.planner || "llm";
    const goalOk = v.goal_ok === undefined ? !!v.ok : !!v.goal_ok;
    const qualityOk = v.quality_ok === undefined
      ? !(v.checks || []).some((check) => check.guard && !check.met)
      : !!v.quality_ok;
    const actionVerified = (v.checks || []).some((check) =>
      check.metric === "semantic" && check.met);
    const verdictClass = !goalOk ? "bad" : qualityOk ? "ok" : "warn";
    const verdictText = !goalOk
      ? "short of goal"
      : !qualityOk
        ? (actionVerified ? "action applied, quality warning" : "goal met, quality warning")
        : (actionVerified ? "action verified" : "goals met");
    html += `<div class="rz-attempt"><div class="rz-head">Attempt ${at.n}`
      + `<span class="rz-verdict ${verdictClass}">${verdictText}</span></div>`;
    html += `<div class="rz-sub">\ud83e\udde0 Planner <span class="rz-src ${pl}">${escapeHtml(PLANNER_LABEL[pl] || pl)}</span></div>`
      + `<div class="rz-plan">\u201c${escapeHtml(at.plan.summary || "")}\u201d</div>`;
    html += `<ol class="rz-steps">` + (at.plan.steps || []).map((s) =>
      `<li><span class="rz-tool">${escapeHtml(s.tool)}</span>${s.why ? ": " + escapeHtml(s.why) : ""}${paramsStr(s.params)}</li>`
    ).join("") + `</ol>`;
    html += `<div class="rz-sub">\u2699\ufe0f Executor</div><ol class="rz-steps">` + (at.steps || []).map((s) => {
      const delta = metricDeltaStr(s.metrics_before, s.metrics_after);
      const note = escapeHtml(s.reject_reason || s.note || "");
      return `<li class="ex-${s.status}"><span class="rz-badge ${s.status}">${s.status}</span>`
        + `<span class="rz-tool">${escapeHtml(s.tool)}</span>`
        + `<span class="rz-note">${note}${delta ? ` <span class="ldelta">${delta}</span>` : ""}</span></li>`;
    }).join("") + `</ol>`;
    if ((v.checks || []).length) {
      html += `<div class="rz-sub">\u2713 Verify</div><ul class="rz-checks">` + v.checks.map((c) =>
        `<li class="${c.met ? "met" : "miss"}">${escapeHtml(c.label)}: ${metricDisp(c.metric, c.before).toFixed(3)}\u2192${metricDisp(c.metric, c.after).toFixed(3)} <em>(${c.status})</em></li>`
      ).join("") + `</ul>`;
    }
    html += `</div>`;
  }
  body.innerHTML = html;
}

function onFinal(payload) {
  const res = payload.result, st = payload.state;
  const branchBase = ST.branchBase
    ? ST.timeline.find((checkpoint) => checkpoint.id === ST.branchBase)
    : null;
  ST.branchBase = null;
  setProgressBar("progbar", "agentProgWrap", 100);
  $("progtext").textContent = "";
  if (res.agent_summary) {
    let g = `<b>Agent plan:</b> ${escapeHtml(res.agent_summary)}`;
    const pl = res.trace && res.trace.planner;
    if (pl && pl !== "llm") {
      const note = (res.trace && res.trace.planner_note) || "offline keyword planner";
      g += ` <span class="planner-tag" title="${escapeHtml(note)}">\u2699 ${pl === "keyword_fallback" ? "offline fallback" : "offline planner"}</span>`;
      if (pl === "keyword_fallback") toast("LLM planning failed, used the offline keyword planner");
    }
    $("goal").innerHTML = g;
  }
  renderChecks(res.trace);
  renderTrace(res.trace);
  const fb = $("feedback");
  fb.textContent = (res.ok ? "\u2714 " : "\u26a0 ") + res.feedback;
  fb.className = "feedback " + (res.ok ? "ok" : "bad");
  applyState(st, { keepMetrics: true });          // update history/toolbar but keep the before->after table
  showMetrics(res.metrics_before, res.metrics_after);
  $("apply").disabled = false;
  activityDone("edit", res.ok ? "Edit applied and verified" : "Best-effort edit checkpointed");
  if (branchBase) {
    toast(`Branch created from ${checkpointDisplayLabel(branchBase)}`);
  } else {
    toast(res.ok ? "Edit applied + checkpointed" : "Best-effort edit checkpointed");
  }
  // NOTE: to see the edit as video, hit 🎬 Render. Metric deltas + history reflect the edit now.
}

// -------------------------------------------------------------- metrics + history
function labelCell(k) {
  return `<td><span class="mlabel">${METRIC_LABELS[k] || k}`
    + `<span class="info" tabindex="0" data-tip="${(METRIC_INFO[k] || "").replace(/"/g, "&quot;")}">i</span>`
    + `</span></td>`;
}
function metricHeader() {
  const tr = document.createElement("tr"); tr.className = "mhead";
  tr.innerHTML = `<td></td><td class="num">before</td><td class="num">after</td><td class="num">change</td>`;
  return tr;
}
function metricRow(k, before, after) {
  const tr = document.createElement("tr");
  const bef = metricDisp(k, before), aft = metricDisp(k, after);
  const d = Math.round((aft - bef) * 1000) / 1000;         // on the DISPLAYED value (smoothness for jerk)
  const better = DISP_HIGHER_BETTER[k];
  let cls = "", arrow = "";
  if (d !== 0) {
    if (better === null) {
      arrow = d > 0 ? " \u25b2" : " \u25bc";               // neutral metric (energy): raw direction, no colour
    } else {
      const improved = (d > 0) === better;                 // higher-is-better after the display transform
      cls = improved ? "up" : "down";
      arrow = improved ? " \u25b2" : " \u25bc";             // arrow, sign and colour now all agree
    }
  }
  tr.innerHTML = labelCell(k)
    + `<td class="num">${bef.toFixed(3)}</td>`
    + `<td class="num">${aft.toFixed(3)}</td>`
    + `<td class="num delta ${cls}">${d >= 0 ? "+" : ""}${d.toFixed(3)}${arrow}</td>`;
  return tr;
}
function showMetrics(before, after) {
  const t = $("metrics"); t.innerHTML = ""; t.appendChild(metricHeader());
  ["energy", "bas", "jerk", "foot"].forEach((k) => {
    if (before[k] === undefined) return; t.appendChild(metricRow(k, before[k], after[k]));
  });
}
function showCurrentMetrics(m) {
  const t = $("metrics"); t.innerHTML = "";
  const hr = document.createElement("tr"); hr.className = "mhead";
  hr.innerHTML = `<td></td><td class="num" colspan="3">whole-dance (current)</td>`;
  t.appendChild(hr);
  ["energy", "bas", "jerk", "foot"].forEach((k) => { if (m[k] === undefined) return;
    const tr = document.createElement("tr");
    tr.innerHTML = labelCell(k) + `<td></td><td class="num">${metricDisp(k, m[k]).toFixed(3)}</td><td></td>`;
    t.appendChild(tr); });
}

function renderHistory(timeline) {
  const ol = $("history"); ol.innerHTML = "";
  const list = Array.isArray(timeline) ? timeline : [];
  const tree = EDITOR_UTILS.checkpointTree(list);

  const appendCheckpoint = (node, container, parentNode, depth) => {
    const c = node.checkpoint;
    const parent = parentNode ? parentNode.checkpoint : null;
    const siblings = parentNode ? parentNode.children : tree;
    const siblingIndex = Math.max(0, siblings.indexOf(node));
    const children = node.children;
    const li = document.createElement("li");
    const classes = [];
    classes.push("history-node");
    if (!parent) classes.push("root");
    if (children.length) classes.push("has-children");
    if (children.length > 1) classes.push("has-fork");
    if (c.is_head) classes.push("head");
    if (c.id === ST.branchBase) classes.push("branch-base");
    if (c.is_ancestor_of_head) classes.push("ancestor");
    if (c.is_branch) classes.push("branch");
    li.className = classes.join(" ");
    li.dataset.checkpointId = c.id;
    li.dataset.depth = String(depth);
    const ed = c.edit || {};
    const objective = ed.objective == null ? "" : String(ed.objective);
    const badge = objective
      ? `<span class="badge2 ${ed.ok === false ? "bad" : ""}">${escapeHtml(objective.replace(/_/g, " "))}</span>`
      : "";
    const interval = EDITOR_UTILS.formatCheckpointInterval(c, ST.fps, ST.dur);
    const label = checkpointDisplayLabel(c);
    const displayLabel = parent ? label : "Original dance";
    const relationship = parent
      ? siblings.length > 1
        ? `Branch ${siblingIndex + 1} of ${siblings.length} from ${checkpointDisplayLabel(parent)}`
        : `After ${checkpointDisplayLabel(parent)}`
      : "Starting point";
    const forkBadge = children.length > 1
      ? `<span class="lineage-badge">${children.length} branches</span>`
      : "";
    const currentBadge = c.is_head
      ? `<span class="history-current">Current</span>`
      : "";
    const branchButton = c.is_head
      ? ""
      : `<button class="branch-btn" type="button" aria-pressed="${c.id === ST.branchBase}" `
        + `aria-label="Branch next edit from ${escapeHtml(displayLabel)}">`
        + `${c.id === ST.branchBase ? "Selected" : "Branch here"}</button>`;
    li.innerHTML = `<div class="history-row" aria-current="${c.is_head ? "true" : "false"}">`
      + `<span class="history-marker" aria-hidden="true"></span><span class="history-copy">`
      + `<span class="history-title"><span class="lbl">${escapeHtml(displayLabel)}</span>${currentBadge}</span>`
      + `<span class="history-lineage"><span>${escapeHtml(relationship)}</span>`
      + `${interval ? `<small>${escapeHtml(interval)}</small>` : ""}${badge}${forkBadge}</span></span>`
      + `<span class="history-actions">${branchButton}</span></div>`;
    const row = li.querySelector(".history-row");
    if (!c.is_head) {
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.setAttribute("aria-label", `Restore ${displayLabel}${interval ? `, ${interval}` : ""}`);
    }

    const restore = async () => {
      if (c.id === ST.head) return;
      activityStart("history", "Restoring version", `Loading ${c.label || "original"}`, null);
      try {
        const state = await api(`/api/session/${ST.sid}/restore`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ckpt_id: c.id }),
        });
        ST.branchBase = null;
        applyState(state);
        activityDone("history", `${c.label || "Original"} restored`);
        toast("Rolled back to: " + (c.label || "original"));
      } catch (e) {
        activityFail("history", e.message);
        toast("Restore failed: " + e.message);
      }
    };
    row.onclick = restore;
    row.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        restore();
      }
    };
    const button = row.querySelector(".branch-btn");
    if (button) {
      button.onclick = (e) => {
        e.stopPropagation();
        selectBranchBase(c.id);
      };
      button.onkeydown = (e) => e.stopPropagation();
    }
    container.appendChild(li);

    if (children.length) {
      const group = document.createElement("ol");
      group.className = `history-children ${children.length > 1 ? "fork-group" : "linear-group"}`;
      group.setAttribute("aria-label", `Versions after ${displayLabel}`);
      children.forEach((child) => appendCheckpoint(child, group, node, depth + 1));
      li.appendChild(group);
    }
  };

  tree.forEach((node) => appendCheckpoint(node, ol, null, 0));
  const current = ol.querySelector(".history-node.head > .history-row");
  if (current) {
    requestAnimationFrame(() => {
      const listBox = ol.getBoundingClientRect();
      const rowBox = current.getBoundingClientRect();
      if (rowBox.bottom > listBox.bottom) ol.scrollTop += rowBox.bottom - listBox.bottom + 8;
      if (rowBox.top < listBox.top) ol.scrollTop -= listBox.top - rowBox.top + 8;
    });
  }
}

init().catch((e) => toast("init failed: " + e.message));

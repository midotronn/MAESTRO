"use strict";
const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
let ST = { sid: null, fps: 30, dur: 0, nframes: 0, beats: 0, head: null, sel: null };
const CMP_HIGHLIGHT = {
  mode: "highlight",
  raf: 0,
  holdOriginal: false,
  failed: false,
  beforeCanvas: document.createElement("canvas"),
  afterCanvas: document.createElement("canvas"),
  overlayCanvas: document.createElement("canvas"),
  overlayImage: null,
  metadata: null,
};

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

// -------------------------------------------------------------- render the complete edited dance (Blender/pod)
async function startFullRender() {
  ST.lastRenderScope = "full";
  $("fullRenderBtn").disabled = true;
  const st = $("renderStatus"); st.style.display = "block"; st.className = "render-status";
  st.textContent = "starting the full-dance render with music\u2026";
  setProgressBar("renderProg", "renderProgWrap", 3);
  activityStart("render", "Rendering full dance", "Starting full-quality render", 3);
  try {
    await api(`/api/session/${ST.sid}/render`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope: "full" }) });
    pollRender();
  } catch (e) {
    st.textContent = "\u26a0 " + e.message;
    st.className = "render-status bad";
    $("renderProgWrap").hidden = true;
    $("fullRenderBtn").disabled = false;
    activityFail("render", e.message);
  }
}

async function pollRender() {
  let j;
  try { j = await api(`/api/session/${ST.sid}/render`); }
  catch (e) {
    activityUpdate("render", { detail: "Reconnecting to render status\u2026" });
    setTimeout(pollRender, 1000);
    return;
  }
  const st = $("renderStatus");
  st.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  setProgressBar("renderProg", "renderProgWrap", Number(j.progress || 0));
  activityUpdate("render", {
    progress: Number(j.progress || 0),
    detail: j.message || j.status,
  });
  if (j.status === "done") {
    const v = $("video");
    v.src = `/api/session/${ST.sid}/media/${j.video}?t=` + Date.now();
    v.load();
    activityUpdate("render", { progress: 96, detail: "Loading the rendered video\u2026" });
    const mediaReady = await waitForMediaReady(
      [v],
      "render",
      96,
      100,
      "Loading the rendered video\u2026",
    );
    v.play().catch(() => {});
    $("viewerTag").textContent = ST.lastRenderScope === "full" ? "full dance + music" : "edited render";
    st.className = "render-status ok";
    st.textContent = `\u2714 full dance ready${j.elapsed ? " in " + j.elapsed + "s" : ""}`;
    $("renderProgWrap").hidden = true;
    $("fullRenderBtn").disabled = false;
    activityDone("render", mediaReady ? "Full dance ready" : "Render ready; video is still buffering");
    toast("Full dance ready");
    setTimeout(() => { st.style.display = "none"; }, 5000);
    return;
  }
  if (j.status === "error") {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + (j.message || "render failed");
    $("renderProgWrap").hidden = true;
    $("fullRenderBtn").disabled = false;
    activityFail("render", j.message || "Render failed");
    toast("Render failed");
    return;
  }
  setTimeout(pollRender, 1000);
}

// -------------------------------------------------------------- before/after window comparison
function populateCmpVersions() {
  const sel = $("cmpVersion");
  if (!sel) return;
  const tl = ST.timeline || [];
  const head = tl.find((c) => c.is_head) || tl[tl.length - 1];
  if (!head) { sel.innerHTML = ""; return; }
  // candidate "before" versions = every checkpoint except the current one, newest first; default is
  // the state right before the last edit (the head's parent).
  const opts = tl.filter((c) => c.id !== head.id).slice().reverse();
  const prev = sel.value;
  sel.innerHTML = "";
  opts.forEach((c) => {
    const o = document.createElement("option");
    o.value = c.id;
    const isParent = c.id === head.parent_id;
    o.textContent = (c.label || "original") + (isParent ? " (pre-edit)" : "");
    sel.appendChild(o);
  });
  // keep the previous choice if still valid, else default to the head's parent
  if (prev && opts.some((c) => c.id === prev)) sel.value = prev;
  else if (head.parent_id) sel.value = head.parent_id;
  sel.disabled = opts.length <= 1;
}

async function startCompare() {
  const panel = $("compare"); panel.hidden = false;
  populateCmpVersions();
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
  const st = $("cmpStatus"); st.style.display = "block"; st.className = "render-status";
  st.textContent = "starting comparison render\u2026";
  setProgressBar("cmpProg", "cmpProgWrap", 3);
  activityStart("compare", "Rendering edit comparison", "Starting before and after render", 3);
  $("compareBtn").disabled = true;
  try {
    const fromId = ($("cmpVersion") && $("cmpVersion").value) || null;
    await api(`/api/session/${ST.sid}/compare`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_id: fromId }) });
    pollCompare();
  } catch (e) {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + e.message;
    $("cmpProgWrap").hidden = true;
    $("compareBtn").disabled = false;
    activityFail("compare", e.message);
  }
}

async function pollCompare() {
  let j;
  try { j = await api(`/api/session/${ST.sid}/compare`); }
  catch (e) {
    activityUpdate("compare", { detail: "Reconnecting to comparison status\u2026" });
    setTimeout(pollCompare, 750);
    return;
  }
  const st = $("cmpStatus");
  st.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  setProgressBar("cmpProg", "cmpProgWrap", Number(j.progress || 0));
  activityUpdate("compare", {
    progress: Number(j.progress || 0),
    detail: j.message || j.status,
  });
  if (j.status === "done") {
    activityUpdate("compare", { progress: 96, detail: "Loading comparison videos\u2026" });
    const mediaReady = await setupCompareVideos(
      j.before_video,
      j.after_video,
      j.metrics || {},
      j.audio,
      j.highlight || null,
    );
    st.className = "render-status ok";
    st.textContent = `\u2714 comparison ready${j.elapsed ? " in " + j.elapsed + "s" : ""}`;
    $("cmpProgWrap").hidden = true;
    $("compareBtn").disabled = false;
    activityDone(
      "compare",
      mediaReady ? "Comparison ready" : "Comparison ready; videos are still buffering",
    );
    setTimeout(() => { st.style.display = "none"; }, 5000);
    toast("Comparison ready");
    return;
  }
  if (j.status === "error") {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + (j.message || "compare failed");
    $("cmpProgWrap").hidden = true;
    $("compareBtn").disabled = false;
    activityFail("compare", j.message || "Comparison failed");
    toast("Compare failed");
    return;
  }
  setTimeout(pollCompare, 500);
}

function showCompareMetrics(m) {
  const t = $("cmpMetrics"); t.innerHTML = "";
  const b = m.before || {}, a = m.after || {};
  if (m.window_sec) $("cmpWin").textContent = `(${m.window_sec[0]}\u2013${m.window_sec[1]}s)`;
  if (m.before_label && $("cmpCapBefore")) $("cmpCapBefore").textContent = "Before \u2014 " + m.before_label;
  if (b.bas === undefined) return;
  t.appendChild(metricHeader());
  ["energy", "bas", "jerk", "foot"].forEach((k) => {
    if (b[k] === undefined || a[k] === undefined) return;
    t.appendChild(metricRow(k, b[k], a[k]));
  });
}

function stopCompareHighlightLoop() {
  if (CMP_HIGHLIGHT.raf) cancelAnimationFrame(CMP_HIGHLIGHT.raf);
  CMP_HIGHLIGHT.raf = 0;
}

function failCompareHighlight(message) {
  if (CMP_HIGHLIGHT.failed) return;
  CMP_HIGHLIGHT.failed = true;
  setCompareMode("side");
  const st = $("cmpStatus");
  st.style.display = "block";
  st.className = "render-status bad";
  st.textContent = "\u26a0 change highlighting unavailable: " + message;
  toast("Change highlighting unavailable; showing side by side");
}

function ensureCompareHighlightCanvases(width, height) {
  const canvases = [
    $("cmpHighlightCanvas"),
    CMP_HIGHLIGHT.beforeCanvas,
    CMP_HIGHLIGHT.afterCanvas,
    CMP_HIGHLIGHT.overlayCanvas,
  ];
  canvases.forEach((canvas) => {
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
  });
  const overlayCtx = CMP_HIGHLIGHT.overlayCanvas.getContext("2d");
  if (!CMP_HIGHLIGHT.overlayImage
      || CMP_HIGHLIGHT.overlayImage.width !== width
      || CMP_HIGHLIGHT.overlayImage.height !== height) {
    CMP_HIGHLIGHT.overlayImage = overlayCtx.createImageData(width, height);
  }
}

function drawProjectedBodyHighlights(ctx, width, height, currentTime) {
  const metadata = CMP_HIGHLIGHT.metadata;
  if (!metadata || !Array.isArray(metadata.frames) || !metadata.frames.length) return null;
  const fps = Number(metadata.fps) || 30;
  const frame = Math.max(0, Math.min(metadata.frames.length - 1, Math.round(currentTime * fps)));
  const markers = metadata.frames[frame] || [];
  const labels = [];

  markers.forEach((marker) => {
    const x = Number(marker.x) * width;
    const y = Number(marker.y) * height;
    const rx = Math.max(20, Number(marker.rx) * width);
    const ry = Math.max(24, Number(marker.ry) * height);
    const strength = Math.max(0.2, Math.min(1, Number(marker.strength) || 0));
    if (![x, y, rx, ry].every(Number.isFinite)) return;

    ctx.save();
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(24, 211, 238, ${0.07 + 0.09 * strength})`;
    ctx.shadowColor = "rgba(24, 211, 238, .95)";
    ctx.shadowBlur = 12 + 18 * strength;
    ctx.lineWidth = 2.5 + 2.5 * strength;
    ctx.strokeStyle = `rgba(24, 211, 238, ${0.62 + 0.30 * strength})`;
    ctx.fill();
    ctx.stroke();
    ctx.restore();

    const label = String(marker.label || "Changed area");
    labels.push(label);
    ctx.save();
    ctx.font = `600 ${Math.max(11, Math.round(width / 40))}px "Noto Sans", sans-serif`;
    const textWidth = ctx.measureText(label).width;
    const lx = Math.max(6, Math.min(width - textWidth - 18, x - textWidth / 2 - 9));
    const ly = Math.max(25, y - ry - 9);
    ctx.fillStyle = "rgba(15, 23, 42, .82)";
    ctx.fillRect(lx, ly - 20, textWidth + 18, 25);
    ctx.fillStyle = "#8ff3ff";
    ctx.fillText(label, lx + 9, ly - 3);
    ctx.restore();
  });
  return [...new Set(labels)];
}

function renderCompareHighlight() {
  if (CMP_HIGHLIGHT.mode !== "highlight" || CMP_HIGHLIGHT.failed || $("compare").hidden) return;
  const A = $("cmpAfter"), B = $("cmpBefore"), canvas = $("cmpHighlightCanvas");
  if (A.readyState < 2 || B.readyState < 2 || !A.videoWidth || !A.videoHeight) return;

  try {
    const width = A.videoWidth, height = A.videoHeight;
    ensureCompareHighlightCanvases(width, height);
    const ctx = canvas.getContext("2d");
    const badge = $("cmpHighlightBadge");

    if (CMP_HIGHLIGHT.holdOriginal) {
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(B, 0, 0, width, height);
      badge.textContent = "Original · release to see highlighted edit";
      return;
    }

    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(A, 0, 0, width, height);
    const partLabels = drawProjectedBodyHighlights(ctx, width, height, A.currentTime || 0);
    if (partLabels) {
      badge.textContent = partLabels.length
        ? "After · highlighting " + partLabels.join(" + ")
        : "After · edited body areas will glow cyan";
      return;
    }

    const beforeCtx = CMP_HIGHLIGHT.beforeCanvas.getContext("2d", { willReadFrequently: true });
    const afterCtx = CMP_HIGHLIGHT.afterCanvas.getContext("2d", { willReadFrequently: true });
    const overlayCtx = CMP_HIGHLIGHT.overlayCanvas.getContext("2d");
    beforeCtx.drawImage(B, 0, 0, width, height);
    afterCtx.drawImage(A, 0, 0, width, height);
    const before = beforeCtx.getImageData(0, 0, width, height).data;
    const after = afterCtx.getImageData(0, 0, width, height).data;
    const kernel = window.MAESTRO_COMPARE_HIGHLIGHT;
    if (!kernel || typeof kernel.colorizeChangedPixels !== "function") {
      failCompareHighlight("pixel comparison module did not load");
      return;
    }
    const changed = kernel.colorizeChangedPixels(
      before,
      after,
      CMP_HIGHLIGHT.overlayImage.data,
      kernel.DEFAULT_THRESHOLD,
    );
    overlayCtx.putImageData(CMP_HIGHLIGHT.overlayImage, 0, 0);

    ctx.save();
    ctx.globalAlpha = 0.58;
    ctx.filter = "blur(5px)";
    ctx.drawImage(CMP_HIGHLIGHT.overlayCanvas, 0, 0);
    ctx.filter = "none";
    ctx.globalAlpha = 0.92;
    ctx.drawImage(CMP_HIGHLIGHT.overlayCanvas, 0, 0);
    ctx.restore();
    badge.textContent = changed
      ? "After · cyan = changed body area"
      : "After · no visible pixel change at this frame";
  } catch (error) {
    failCompareHighlight(error instanceof Error ? error.message : String(error));
  }
}

function startCompareHighlightLoop() {
  stopCompareHighlightLoop();
  if (CMP_HIGHLIGHT.mode !== "highlight" || CMP_HIGHLIGHT.failed) return;
  const tick = () => {
    CMP_HIGHLIGHT.raf = 0;
    renderCompareHighlight();
    if (!$("cmpAfter").paused && CMP_HIGHLIGHT.mode === "highlight" && !$("compare").hidden) {
      CMP_HIGHLIGHT.raf = requestAnimationFrame(tick);
    }
  };
  CMP_HIGHLIGHT.raf = requestAnimationFrame(tick);
}

function setCompareMode(mode) {
  const useHighlight = mode === "highlight" && !CMP_HIGHLIGHT.failed;
  CMP_HIGHLIGHT.mode = useHighlight ? "highlight" : "side";
  $("cmpHighlight").hidden = !useHighlight;
  $("cmpSideBySide").hidden = useHighlight;
  $("cmpHoldBefore").hidden = !useHighlight;
  $("cmpModeHighlight").classList.toggle("active", useHighlight);
  $("cmpModeHighlight").setAttribute("aria-pressed", String(useHighlight));
  $("cmpModeSide").classList.toggle("active", !useHighlight);
  $("cmpModeSide").setAttribute("aria-pressed", String(!useHighlight));
  if (useHighlight) startCompareHighlightLoop();
  else stopCompareHighlightLoop();
}

function setCompareOriginalHeld(held) {
  CMP_HIGHLIGHT.holdOriginal = held;
  if (CMP_HIGHLIGHT.mode === "highlight") renderCompareHighlight();
}

function setupCompareVideos(beforeName, afterName, metrics, audioName, highlight) {
  const A = $("cmpAfter"), B = $("cmpBefore"), AU = $("cmpAudio");
  wireMediaBuffering(A, "compare-buffer", "Buffering comparison");
  const bust = "?t=" + Date.now();
  A.src = `/api/session/${ST.sid}/media/${afterName}${bust}`;
  B.src = `/api/session/${ST.sid}/media/${beforeName}${bust}`;
  A.load(); B.load();
  const mediaReady = waitForMediaReady(
    [A, B],
    "compare",
    96,
    100,
    "Loading comparison videos\u2026",
  );
  // window music: a small clip the same length as the window, looped in sync with the (looping)
  // videos. The clips and the audio are all 0-based over the window, so audio.currentTime == A.time.
  const haveAudio = !!(audioName && AU);
  if (AU) {
    AU.src = haveAudio ? `/api/session/${ST.sid}/media/${audioName}${bust}` : "";
    AU.muted = !haveAudio;
    if (haveAudio) AU.load();
  }
  showCompareMetrics(metrics);
  CMP_HIGHLIGHT.failed = false;
  CMP_HIGHLIGHT.holdOriginal = false;
  CMP_HIGHLIGHT.metadata = highlight || null;
  setCompareMode("highlight");
  const setPlayLabel = () => { $("cmpPlay").textContent = A.paused ? "\u25b6 Play both" : "\u23f8 Pause"; };
  const playAudio = () => { if (haveAudio) { try { AU.currentTime = A.currentTime || 0; } catch (e) {} AU.play().catch(() => {}); } };
  const playBoth = () => { A.play().catch(() => {}); B.play().catch(() => {}); playAudio(); setPlayLabel(); };
  const pauseBoth = () => { A.pause(); B.pause(); if (haveAudio) AU.pause(); setPlayLabel(); };
  $("cmpPlay").onclick = () => { A.paused ? playBoth() : pauseBoth(); };
  A.onplay = () => {
    if (B.paused) B.play().catch(() => {});
    playAudio();
    setPlayLabel();
    startCompareHighlightLoop();
  };
  A.onpause = () => {
    if (!B.paused) B.pause();
    if (haveAudio) AU.pause();
    setPlayLabel();
    stopCompareHighlightLoop();
    renderCompareHighlight();
  };
  A.ontimeupdate = () => {                                  // keep "before" + music locked to "after"
    const d = A.duration || 1;
    $("cmpScrub").value = Math.round((A.currentTime / d) * 1000);
    if (isFinite(A.currentTime) && Math.abs((B.currentTime || 0) - A.currentTime) > 0.08) {
      try { B.currentTime = A.currentTime; } catch (e) {}
    }
    if (haveAudio && isFinite(A.currentTime) && Math.abs((AU.currentTime || 0) - A.currentTime) > 0.18) {
      try { AU.currentTime = A.currentTime; } catch (e) {}
    }
    if (!CMP_HIGHLIGHT.raf) renderCompareHighlight();
  };
  $("cmpScrub").oninput = () => {
    const d = A.duration || 1, t = ($("cmpScrub").value / 1000) * d;
    try { A.currentTime = t; B.currentTime = t; if (haveAudio) AU.currentTime = t; } catch (e) {}
    requestAnimationFrame(renderCompareHighlight);
  };
  A.onseeked = renderCompareHighlight;
  B.onseeked = renderCompareHighlight;
  B.onloadeddata = renderCompareHighlight;
  A.onloadeddata = () => { renderCompareHighlight(); playBoth(); };
  return mediaReady;
}

// -------------------------------------------------------------- load songs + session
async function runHistoryAction(label, endpoint) {
  activityStart("history", label, "Updating the current checkpoint", null);
  try {
    const st = await api(`/api/session/${ST.sid}/${endpoint}`, { method: "POST" });
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
  $("fullRenderBtn").onclick = startFullRender;
  $("compareBtn").onclick = startCompare;
  $("cmpModeHighlight").onclick = () => setCompareMode("highlight");
  $("cmpModeSide").onclick = () => setCompareMode("side");
  const holdBefore = $("cmpHoldBefore");
  const releaseOriginal = () => setCompareOriginalHeld(false);
  holdBefore.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    setCompareOriginalHeld(true);
  });
  ["pointerup", "pointercancel", "pointerleave"].forEach((name) =>
    holdBefore.addEventListener(name, releaseOriginal));
  holdBefore.addEventListener("keydown", (e) => {
    if (e.key === " " || e.key === "Enter") {
      e.preventDefault();
      setCompareOriginalHeld(true);
    }
  });
  holdBefore.addEventListener("keyup", (e) => {
    if (e.key === " " || e.key === "Enter") releaseOriginal();
  });
  window.addEventListener("blur", releaseOriginal);
  $("cmpClose").onclick = () => {
    $("compare").hidden = true;
    stopCompareHighlightLoop();
    try { $("cmpAfter").pause(); $("cmpBefore").pause(); $("cmpAudio").pause(); } catch (e) {}
  };
  const cmpVer = $("cmpVersion");
  if (cmpVer) cmpVer.onchange = () => { if (!$("compare").hidden) startCompare(); };
  $("undo").onclick = () => runHistoryAction("Undoing edit", "undo");
  $("redo").onclick = () => runHistoryAction("Redoing edit", "redo");
  $("reset").onclick = async () => {
    if (!confirm("Clear the edit history and start over from the original dance? This cannot be undone.")) return;
    activityStart("history", "Resetting edit history", "Restoring the original dance", null);
    try {
      const st = await api(`/api/session/${ST.sid}/reset`, { method: "POST" });
      $("compare").hidden = true;
      stopCompareHighlightLoop();
      try { $("cmpAfter").pause(); $("cmpBefore").pause(); $("cmpAudio").pause(); } catch (e) {}
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
  wireTour();
}

// -------------------------------------------------------------- first-run walkthrough (skippable)
const TOUR_KEY = "maestro_onboarded_v1";
const TOUR_STEPS = [
  { el: "timeline", title: "1 · Pick the part to edit",
    text: "Drag across this bar to choose the window of the dance you want to change. The shaded band is your window." },
  { el: "instruction", title: "2 · Say what you want",
    text: "Describe the change in plain English, for example \u201cmake it more energetic\u201d, \u201ctighten to the beat\u201d, \u201cclap to the right\u201d, or \u201cwave here\u201d. Named actions replace motion in the selected window; insertion is unavailable until it has its own visual audit. If you omit a direction, MAESTRO follows the dance flow." },
  { el: "motionPicker", title: "3 · Browse 20 supported motions",
    text: "Open this catalog to see every common motion MAESTRO supports. Click one to add it to your prompt. Named motions are slightly exaggerated so they read clearly, and beat-hit motions land on the strongest beat in the selected window." },
  { el: "apply", title: "4 · Apply the edit",
    text: "The agent plans the right tools, applies them, and verifies the result actually hit your goal. If needed, it refines the edit." },
  { el: "compareBtn", title: "5 · Review the result",
    text: "Review the edited dancer with changed body areas glowing cyan. Hold for the original or switch to side by side, all synchronized to the music." },
  { el: "history", title: "6 · Iterate freely",
    text: "Every edit is a checkpoint. Undo, redo, compare versions, or reset to start over. Edit, listen, refine." },
  { el: "song", title: "That\u2019s it. Have fun!",
    text: "Switch songs here or upload your own. Tap the \u201c?\u201d in the top bar to see this again anytime." },
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

async function loadMotionBank() {
  const host = $("motionSuggestions");
  if (!host) return;
  activityStart("motion-catalog", "Loading motion catalog", "Fetching supported motions", null);
  try {
    const data = await api("/api/motions");
    host.innerHTML = "";
    const motions = data.motions || [];
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
      button.title = `${motion.category} · ${timing}${direction}`;
      button.onclick = () => {
        $("instruction").value = `add a ${motion.name.toLowerCase()} here`;
        $("instruction").focus();
      };
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
    if (sel) sel.disabled = false;
    activityFail("session", `Could not open ${label}`);
    toast(`Couldn't open ${sid} \u2014 please refresh in a moment.`);
    return false;
  }
  ST.sid = sid;
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
  activityDone("session", mediaReady ? `${label} ready` : `${label} loaded; preview is buffering`);
  toast(`Loaded ${sid}: ${st.duration}s, ${st.n_beats} beats, ${st.generator} generator`);
  return true;
}

function applyState(st, opts) {
  opts = opts || {};
  ST.fps = st.fps; ST.dur = st.duration; ST.nframes = st.n_frames; ST.beats = st.n_beats; ST.head = st.head;
  ST.generator = st.generator;
  ST.timeline = st.timeline || [];
  setGenBadge(st.generator);
  $("undo").disabled = !st.can_undo; $("redo").disabled = !st.can_redo;
  drawBeatsTicks();
  renderHistory(st.timeline);
  if (!$("compare").hidden) populateCmpVersions();          // keep the version picker current
  if (!opts.keepMetrics && st.metrics && st.metrics.energy !== undefined) showCurrentMetrics(st.metrics);
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

// -------------------------------------------------------------- timeline drawing + selection
function pct(sec) { return ST.dur ? (sec / ST.dur) * 100 : 0; }
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
function setSel(a, b) {
  a = Math.max(0, Math.min(a, ST.dur)); b = Math.max(0, Math.min(b, ST.dur));
  ST.sel = [Math.min(a, b), Math.max(a, b)]; drawSelection();
}
function wireTimeline() {
  const tl = $("timeline");
  let dragging = false, startSec = 0;
  const secAt = (e) => { const r = tl.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * ST.dur; };
  tl.addEventListener("mousedown", (e) => { dragging = true; startSec = secAt(e); setSel(startSec, startSec); });
  window.addEventListener("mousemove", (e) => { if (dragging) setSel(startSec, secAt(e)); });
  window.addEventListener("mouseup", () => { dragging = false; });
  const v = $("video");
  wireMediaBuffering(v, "viewer-buffer", "Buffering dance preview");
  v.addEventListener("timeupdate", () => { $("playhead").style.left = pct(v.currentTime) + "%"; });
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
    ws.send(JSON.stringify({ a_sec: a, b_sec: b, instruction }));
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
      runEditHTTP(a, b, instruction);
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

async function runEditHTTP(a, b, instruction) {
  try {
    const r = await api(`/api/session/${ST.sid}/edit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a_sec: a, b_sec: b, instruction }) });
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
  toast(res.ok ? "Edit applied + checkpointed" : "Best-effort edit checkpointed");
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
  (timeline || []).slice().reverse().forEach((c) => {
    const li = document.createElement("li"); if (c.is_head) li.className = "head";
    const ed = c.edit || {};
    const badge = ed.objective ? `<span class="badge2 ${ed.ok === false ? "bad" : ""}">${ed.objective.replace(/_/g, " ")}</span>` : "";
    const win = ed.window ? `<small>${(ed.window[0] / ST.fps).toFixed(0)}-${(ed.window[1] / ST.fps).toFixed(0)}s</small>` : "";
    const label = c.label ? c.label.replace(/\s*\[[^\]]*\]/, "").replace(/_/g, " ") : "original";
    li.innerHTML = `<span class="dot"></span><span class="lbl">${label}${win}</span>${badge}`;
    li.onclick = async () => {
      activityStart("history", "Restoring version", `Loading ${c.label || "original"}`, null);
      try {
        const state = await api(`/api/session/${ST.sid}/restore`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ckpt_id: c.id }),
        });
        applyState(state);
        activityDone("history", `${c.label || "Original"} restored`);
        toast("Rolled back to: " + (c.label || "original"));
      } catch (e) {
        activityFail("history", e.message);
        toast("Restore failed: " + e.message);
      }
    };
    ol.appendChild(li);
  });
}

init().catch((e) => toast("init failed: " + e.message));

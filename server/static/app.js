"use strict";
const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
let ST = { sid: null, fps: 30, dur: 0, nframes: 0, beats: 0, head: null, sel: null };

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

// -------------------------------------------------------------- render the complete edited dance (Blender/pod)
async function startFullRender() {
  ST.lastRenderScope = "full";
  $("fullRenderBtn").disabled = true;
  const st = $("renderStatus"); st.style.display = "block"; st.className = "render-status";
  st.textContent = "starting the full-dance render with music\u2026";
  $("renderProgWrap").hidden = false; $("renderProg").style.width = "3%";
  try {
    await api(`/api/session/${ST.sid}/render`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope: "full" }) });
    pollRender();
  } catch (e) { st.textContent = "\u26a0 " + e.message; $("fullRenderBtn").disabled = false; }
}

async function pollRender() {
  let j;
  try { j = await api(`/api/session/${ST.sid}/render`); }
  catch (e) { setTimeout(pollRender, 3000); return; }
  const st = $("renderStatus");
  st.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  $("renderProgWrap").hidden = false; $("renderProg").style.width = (j.progress || 0) + "%";
  if (j.status === "done") {
    const v = $("video");
    v.src = `/api/session/${ST.sid}/media/${j.video}?t=` + Date.now();
    v.load(); v.play().catch(() => {});
    $("viewerTag").textContent = ST.lastRenderScope === "full" ? "full dance + music" : "edited render";
    st.className = "render-status ok";
    st.textContent = `\u2714 full dance ready${j.elapsed ? " in " + j.elapsed + "s" : ""}`;
    $("renderProgWrap").hidden = true;
    $("fullRenderBtn").disabled = false;
    toast("Full dance ready");
    setTimeout(() => { st.style.display = "none"; }, 5000);
    return;
  }
  if (j.status === "error") {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + (j.message || "render failed");
    $("renderProgWrap").hidden = true;
    $("fullRenderBtn").disabled = false;
    toast("Render failed");
    return;
  }
  setTimeout(pollRender, 3000);
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
  $("cmpProgWrap").hidden = false; $("cmpProg").style.width = "3%";
  $("compareBtn").disabled = true;
  try {
    const fromId = ($("cmpVersion") && $("cmpVersion").value) || null;
    await api(`/api/session/${ST.sid}/compare`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from_id: fromId }) });
    pollCompare();
  } catch (e) {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + e.message;
    $("compareBtn").disabled = false;
  }
}

async function pollCompare() {
  let j;
  try { j = await api(`/api/session/${ST.sid}/compare`); }
  catch (e) { setTimeout(pollCompare, 3000); return; }
  const st = $("cmpStatus");
  st.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  $("cmpProgWrap").hidden = false; $("cmpProg").style.width = (j.progress || 0) + "%";
  if (j.status === "done") {
    setupCompareVideos(j.before_video, j.after_video, j.metrics || {}, j.audio);
    st.className = "render-status ok";
    st.textContent = `\u2714 comparison ready${j.elapsed ? " in " + j.elapsed + "s" : ""}`;
    $("cmpProgWrap").hidden = true;
    $("compareBtn").disabled = false;
    setTimeout(() => { st.style.display = "none"; }, 5000);
    toast("Comparison ready");
    return;
  }
  if (j.status === "error") {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + (j.message || "compare failed");
    $("cmpProgWrap").hidden = true;
    $("compareBtn").disabled = false; toast("Compare failed");
    return;
  }
  setTimeout(pollCompare, 3000);
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

function setupCompareVideos(beforeName, afterName, metrics, audioName) {
  const A = $("cmpAfter"), B = $("cmpBefore"), AU = $("cmpAudio");
  const bust = "?t=" + Date.now();
  A.src = `/api/session/${ST.sid}/media/${afterName}${bust}`;
  B.src = `/api/session/${ST.sid}/media/${beforeName}${bust}`;
  A.load(); B.load();
  // window music: a small clip the same length as the window, looped in sync with the (looping)
  // videos. The clips and the audio are all 0-based over the window, so audio.currentTime == A.time.
  const haveAudio = !!(audioName && AU);
  if (AU) {
    AU.src = haveAudio ? `/api/session/${ST.sid}/media/${audioName}${bust}` : "";
    AU.muted = !haveAudio;
    if (haveAudio) AU.load();
  }
  showCompareMetrics(metrics);
  const setPlayLabel = () => { $("cmpPlay").textContent = A.paused ? "\u25b6 Play both" : "\u23f8 Pause"; };
  const playAudio = () => { if (haveAudio) { try { AU.currentTime = A.currentTime || 0; } catch (e) {} AU.play().catch(() => {}); } };
  const playBoth = () => { A.play().catch(() => {}); B.play().catch(() => {}); playAudio(); setPlayLabel(); };
  const pauseBoth = () => { A.pause(); B.pause(); if (haveAudio) AU.pause(); setPlayLabel(); };
  $("cmpPlay").onclick = () => { A.paused ? playBoth() : pauseBoth(); };
  A.onplay = () => { if (B.paused) B.play().catch(() => {}); playAudio(); setPlayLabel(); };
  A.onpause = () => { if (!B.paused) B.pause(); if (haveAudio) AU.pause(); setPlayLabel(); };
  A.ontimeupdate = () => {                                  // keep "before" + music locked to "after"
    const d = A.duration || 1;
    $("cmpScrub").value = Math.round((A.currentTime / d) * 1000);
    if (isFinite(A.currentTime) && Math.abs((B.currentTime || 0) - A.currentTime) > 0.08) {
      try { B.currentTime = A.currentTime; } catch (e) {}
    }
    if (haveAudio && isFinite(A.currentTime) && Math.abs((AU.currentTime || 0) - A.currentTime) > 0.18) {
      try { AU.currentTime = A.currentTime; } catch (e) {}
    }
  };
  $("cmpScrub").oninput = () => {
    const d = A.duration || 1, t = ($("cmpScrub").value / 1000) * d;
    try { A.currentTime = t; B.currentTime = t; if (haveAudio) AU.currentTime = t; } catch (e) {}
  };
  A.onloadeddata = () => { playBoth(); };
}

// -------------------------------------------------------------- load songs + session
function wireControls() {
  wireTimeline();
  wireUpload();
  $("apply").onclick = runEdit;
  $("fullRenderBtn").onclick = startFullRender;
  $("compareBtn").onclick = startCompare;
  $("cmpClose").onclick = () => {
    $("compare").hidden = true;
    try { $("cmpAfter").pause(); $("cmpBefore").pause(); $("cmpAudio").pause(); } catch (e) {}
  };
  const cmpVer = $("cmpVersion");
  if (cmpVer) cmpVer.onchange = () => { if (!$("compare").hidden) startCompare(); };
  $("undo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/undo`, { method: "POST" }));
  $("redo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/redo`, { method: "POST" }));
  $("reset").onclick = async () => {
    if (!confirm("Clear the edit history and start over from the original dance? This cannot be undone.")) return;
    const st = await api(`/api/session/${ST.sid}/reset`, { method: "POST" });
    $("compare").hidden = true;
    try { $("cmpAfter").pause(); $("cmpBefore").pause(); $("cmpAudio").pause(); } catch (e) {}
    $("video").src = st.preview_url + "?t=" + Date.now();     // back to the original dance
    applyState(st);
    showCurrentMetrics(st.metrics || {});
    toast("Edit history cleared \u2014 back to the original");
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
    text: "Describe the change in plain English, for example \u201cmake it more energetic\u201d, \u201ctighten to the beat\u201d, \u201cadd a clap here\u201d, or \u201cinsert a wave before the next move\u201d. The agent chooses the right edit." },
  { el: "motionPicker", title: "3 · Browse 20 supported motions",
    text: "Open this catalog to see every common motion MAESTRO supports. Click one to add it to your prompt. Named motions are slightly exaggerated so they read clearly, and beat-hit motions land on the strongest beat in the selected window." },
  { el: "apply", title: "4 · Apply the edit",
    text: "The agent plans the right tools, applies them, and verifies the result actually hit your goal. If needed, it refines the edit." },
  { el: "compareBtn", title: "5 · Review the result",
    text: "Review the edited window beside an earlier version, synchronized to the music. Use Render full dance only when you want a slower final review of the complete performance." },
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
      button.title = `${motion.category} · ${timing}`;
      button.onclick = () => {
        $("instruction").value = `add a ${motion.name.toLowerCase()} here`;
        $("instruction").focus();
      };
      host.appendChild(button);
    });
  } catch (e) {
    host.textContent = "Named motions are unavailable.";
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
    if (i === 0) toast("Loading songs\u2026 the editor may still be starting");
    await new Promise((r) => setTimeout(r, 1500));
  }
  return [];
}

async function init() {
  const sel = $("song");
  sel.innerHTML = "<option>Loading\u2026</option>"; sel.disabled = true;
  wireControls();                                            // wire up-front so controls work at once
  loadMotionBank();
  const songs = await loadSongs();
  sel.innerHTML = "";
  if (!songs.length) {
    sel.innerHTML = "<option>no songs \u2014 refresh</option>";
    toast("Couldn't load songs \u2014 the editor may still be starting. Please refresh in a moment.");
    return;
  }
  sel.disabled = false;
  songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid; o.textContent = s.name || s.sid; sel.appendChild(o); });
  sel.onchange = () => openSession(sel.value);
  await openSession(songs[0].sid);
  maybeAutoTour();                                          // first-run walkthrough (skippable)
}

async function openSession(sid) {
  let st = null;
  for (let i = 0; i < 4 && !st; i++) {
    try { st = await api(`/api/session/${sid}`, { method: "POST" }); }
    catch (e) { await new Promise((r) => setTimeout(r, 1200)); }   // pod may still be warming
  }
  if (!st) { toast(`Couldn't open ${sid} \u2014 please refresh in a moment.`); return; }
  ST.sid = sid;
  const v = $("video");
  v.src = st.preview_url + "?t=" + Date.now();
  applyState(st);
  toast(`Loaded ${sid}: ${st.duration}s, ${st.n_beats} beats, ${st.generator} generator`);
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
  v.addEventListener("timeupdate", () => { $("playhead").style.left = pct(v.currentTime) + "%"; });
  [$("aSec"), $("bSec")].forEach((inp) => inp.addEventListener("change", () => setSel(parseFloat($("aSec").value) || 0, parseFloat($("bSec").value) || 0)));
}

// -------------------------------------------------------------- edit (WebSocket w/ live progress)
// -------------------------------------------------------------- song upload (processed on the pod)
function wireUpload() {
  const btn = $("uploadBtn"), input = $("uploadInput");
  if (!btn || !input) return;
  btn.onclick = () => input.click();
  input.onchange = async () => {
    const f = input.files && input.files[0];
    input.value = "";
    if (!f) return;
    const fd = new FormData(); fd.append("file", f);
    toast(`Uploading ${f.name}…`);
    let job;
    try { job = await (await fetch("/api/upload", { method: "POST", body: fd })).json(); }
    catch (e) { toast("Upload failed: " + e.message); return; }
    if (job.error) { toast(job.error); return; }
    pollJob(job.sid, f.name);
  };
}

function pollJob(sid, name) {
  const goal = $("goal"), pt = $("progtext"), pb = $("progbar"), fb = $("feedback");
  fb.textContent = ""; fb.className = "feedback";
  goal.innerHTML = `<b>Processing “${name}”</b> on the GPU pod`;
  pb.style.width = "8%"; pt.textContent = "queued…";
  const iv = setInterval(async () => {
    let j; try { j = await api(`/api/jobs/${sid}`); } catch { return; }
    pb.style.width = (j.progress || 10) + "%";
    pt.textContent = j.message || j.status;
    if (j.status === "done") {
      clearInterval(iv); pb.style.width = "100%";
      fb.textContent = "\u2714 Ready"; fb.className = "feedback ok";
      const { songs } = await api("/api/songs");
      const sel = $("song"); sel.innerHTML = "";
      songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid;
        o.textContent = s.name || s.sid; sel.appendChild(o); });
      sel.value = sid; await openSession(sid);
      toast(`“${name}” is ready`);
    } else if (j.status === "error") {
      clearInterval(iv); pb.style.width = "0%";
      fb.textContent = "\u26a0 " + (j.message || "processing failed"); fb.className = "feedback bad";
    }
  }, 2500);
}

function runEdit() {
  if (!ST.sel) { toast("Select a window on the timeline first"); return; }
  const instruction = $("instruction").value.trim();
  if (!instruction) { toast("Type an instruction"); return; }
  const [a, b] = ST.sel;
  $("apply").disabled = true; $("feedback").textContent = ""; $("feedback").className = "feedback";
  $("goal").innerHTML = ""; $("progbar").style.width = "0%"; $("progtext").textContent = "connecting...";

  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/session/${ST.sid}/edit_ws`);
  ws.onopen = () => ws.send(JSON.stringify({ a_sec: a, b_sec: b, instruction }));
  ws.onmessage = (msg) => {
    const ev = JSON.parse(msg.data);
    if (ev.type === "progress") onProgress(ev);
    else if (ev.type === "final") { onFinal(ev); ws.close(); }
    else if (ev.type === "error") { toast("Edit failed: " + ev.message); $("apply").disabled = false; ws.close(); }
  };
  ws.onerror = () => { $("progtext").textContent = "socket error; retrying over HTTP..."; runEditHTTP(a, b, instruction); };
}

async function runEditHTTP(a, b, instruction) {
  try {
    const r = await api(`/api/session/${ST.sid}/edit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a_sec: a, b_sec: b, instruction }) });
    onFinal(r);
  } catch (e) { toast("Edit failed: " + e.message); } finally { $("apply").disabled = false; }
}

function onProgress(ev) {
  if (ev.phase === "plan") {
    $("goal").innerHTML = `<b>Agent plan:</b> ${escapeHtml(ev.summary || "")}`;
    $("checks").innerHTML = ""; $("reasoning").hidden = true;
    $("progbar").style.width = "8%";
    $("progtext").textContent = `planning ${(ev.steps || []).length} step(s)\u2026`;
  } else if (ev.phase === "refine") {
    $("progbar").style.width = "30%";
    $("progtext").textContent = `refining (attempt ${ev.cycle}): ${ev.summary || ""}`;
  } else if (ev.phase === "step") {
    const p = ev.n_steps ? Math.round(20 + (ev.step / ev.n_steps) * 60) : 50;
    $("progbar").style.width = p + "%";
    const tag = ev.status === "rejected" ? " \u21a9 rejected" : ev.status === "applied" ? " \u2713" : "";
    $("progtext").textContent = `attempt ${ev.cycle || 1} \u00b7 step ${ev.step}/${ev.n_steps}: ${ev.tool}${tag}`;
  } else if (ev.phase === "verify") {
    $("progtext").textContent = `attempt ${ev.cycle}: ${ev.ok ? "goals met \u2714" : "short of goal, refining\u2026"}`;
  } else if (ev.phase === "candidate") {   // legacy best-of-K generators
    const p = ev.total ? Math.round((ev.done / ev.total) * 100) : 0;
    $("progbar").style.width = p + "%";
    $("progtext").textContent = `trying ${ev.backbone} take #${ev.seed} (${ev.done}/${ev.total})`;
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
    html += `<div class="rz-attempt"><div class="rz-head">Attempt ${at.n}`
      + `<span class="rz-verdict ${v.ok ? "ok" : "bad"}">${v.ok ? "goals met" : "short of goal"}</span></div>`;
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
  $("progbar").style.width = "100%";
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
    li.onclick = async () => { applyState(await api(`/api/session/${ST.sid}/restore`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ckpt_id: c.id }) }));
      toast("Rolled back to: " + (c.label || "original")); };
    ol.appendChild(li);
  });
}

init().catch((e) => toast("init failed: " + e.message));

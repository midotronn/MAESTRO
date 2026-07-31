"use strict";
const $ = (id) => document.getElementById(id);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
};
let ST = { sid: null, fps: 30, dur: 0, nframes: 0, beats: 0, head: null, sel: null };

const METRIC_LABELS = { energy: "energy", bas: "beat align (BAS)", jerk: "smoothness (jerk)", foot: "foot contact" };
const HIGHER_BETTER = { energy: null, bas: true, jerk: false, foot: true }; // null = neutral
const METRIC_INFO = {
  energy: "How big and lively the movement is. Measured as the average frame-to-frame change in the body pose (root travel + all joint rotations). Higher = more energetic; there is no 'good' value, it just tells you the intensity.",
  bas: "Beat Alignment Score (0-1): how well the dance lands on the music's beats. We find each 'motion beat' (a moment the body decelerates into a pose) and score how close it is to the nearest music beat. Higher = tighter to the beat.",
  jerk: "Smoothness, shown as jerk = the average 3rd derivative of the pose (how abruptly acceleration changes). Lower = smoother, less jittery motion; higher = sharper, snappier.",
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

// -------------------------------------------------------------- render the edited dance (Blender/pod)
async function startRender() {
  const scope = $("renderScope").value;
  const body = { scope };
  // window render targets the LAST EDIT by default; a valid drag selection overrides it
  if (scope === "window" && ST.sel && ST.sel[1] > ST.sel[0] + 0.1) {
    body.a_sec = ST.sel[0]; body.b_sec = ST.sel[1];
  }
  $("renderBtn").disabled = true;
  const st = $("renderStatus"); st.style.display = "block"; st.className = "render-status";
  st.textContent = "starting render\u2026";
  try {
    await api(`/api/session/${ST.sid}/render`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    pollRender();
  } catch (e) { st.textContent = "\u26a0 " + e.message; $("renderBtn").disabled = false; }
}

async function pollRender() {
  let j;
  try { j = await api(`/api/session/${ST.sid}/render`); }
  catch (e) { setTimeout(pollRender, 3000); return; }
  const st = $("renderStatus");
  st.textContent = (j.status === "rendering" ? "\u{1F3AC} " : "") + (j.message || j.status);
  if (j.status === "done") {
    const v = $("video");
    v.src = `/api/session/${ST.sid}/media/${j.video}?t=` + Date.now();
    v.load(); v.play().catch(() => {});
    $("viewerTag").textContent = "HQ render";
    st.className = "render-status ok";
    st.textContent = `\u2714 rendered${j.elapsed ? " in " + j.elapsed + "s" : ""}`;
    $("renderBtn").disabled = false;
    $("viewVid").disabled = false;
    showView("video");
    toast("HQ render ready");
    setTimeout(() => { st.style.display = "none"; }, 5000);
    return;
  }
  if (j.status === "error") {
    st.className = "render-status bad"; st.textContent = "\u26a0 " + (j.message || "render failed");
    $("renderBtn").disabled = false;
    toast("Render failed");
    return;
  }
  setTimeout(pollRender, 3000);
}

// -------------------------------------------------------------- load songs + session
async function init() {
  initSkeletonViewer();                    // set up the 3D viewer BEFORE the first openSession loads it
  const { songs } = await api("/api/songs");
  const sel = $("song");
  sel.innerHTML = "";
  songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid; o.textContent = s.name || s.sid; sel.appendChild(o); });
  sel.onchange = () => openSession(sel.value);
  if (songs.length) await openSession(songs[0].sid);
  wireTimeline();
  wireUpload();
  $("apply").onclick = runEdit;
  $("renderBtn").onclick = startRender;
  $("undo").onclick = async () => { applyState(await api(`/api/session/${ST.sid}/undo`, { method: "POST" })); loadSkeleton(); };
  $("redo").onclick = async () => { applyState(await api(`/api/session/${ST.sid}/redo`, { method: "POST" })); loadSkeleton(); };
  $("instruction").addEventListener("keydown", (e) => { if (e.key === "Enter") runEdit(); });
}

// -------------------------------------------------------------- 3D stick-figure preview (instant, local FK)
function initSkeletonViewer() {
  if (window.Skel3D) window.Skel3D.init("skel3d");
  $("skPlayBtn").onclick = () => { const p = window.Skel3D.toggle(); $("skPlayBtn").textContent = p ? "\u275a\u275a" : "\u25b6"; };
  $("skScrub").addEventListener("input", (e) => {
    if (!window.Skel3D) return;
    window.Skel3D.pause(); $("skPlayBtn").textContent = "\u25b6";
    const f = Math.round((parseFloat(e.target.value) / 100) * (window.Skel3D.total() - 1));
    window.Skel3D.setFrame(f);
  });
  if (window.Skel3D) window.Skel3D.onFrame((cur, total, tsec) => {
    if (!$("skScrub").matches(":active")) $("skScrub").value = total > 1 ? (cur / (total - 1)) * 100 : 0;
    $("skTime").textContent = tsec.toFixed(1) + "s";
  });
  $("view3d").onclick = () => showView("3d");
  $("viewVid").onclick = () => showView("video");
}

async function loadSkeleton() {
  if (!window.Skel3D || !ST.sid) return;
  try {
    const resp = await fetch(`/api/session/${ST.sid}/skeleton.bin?fps=20`);
    if (!resp.ok) throw new Error("skeleton fetch failed");
    const joints = new Float32Array(await resp.arrayBuffer());
    window.Skel3D.load({
      joints,
      n_frames: +resp.headers.get("X-Frames"),
      n_joints: +resp.headers.get("X-Joints"),
      fps: +resp.headers.get("X-Fps"),
      bones: JSON.parse(resp.headers.get("X-Bones") || "[]"),
    });
    window.Skel3D.resize();
    $("skPlayBtn").textContent = "\u275a\u275a";
    showView("3d");
  } catch (e) { /* keep last */ }
}

function showView(mode) {
  const is3d = mode === "3d";
  $("skel3d").style.display = is3d ? "block" : "none";
  $("skelPlay").style.display = is3d ? "flex" : "none";
  $("video").style.display = is3d ? "none" : "block";
  $("view3d").classList.toggle("active", is3d);
  $("viewVid").classList.toggle("active", !is3d);
  $("viewerTag").textContent = is3d ? "3D preview" : "HQ render";
  if (is3d && window.Skel3D) window.Skel3D.resize();
}

async function openSession(sid) {
  const st = await api(`/api/session/${sid}`, { method: "POST" });
  ST.sid = sid;
  const v = $("video");
  v.src = st.preview_url + "?t=" + Date.now();
  applyState(st);
  await loadSkeleton();
  toast(`Loaded ${sid}: ${st.duration}s, ${st.n_beats} beats, ${st.generator} generator`);
}

function applyState(st, opts) {
  opts = opts || {};
  ST.fps = st.fps; ST.dur = st.duration; ST.nframes = st.n_frames; ST.beats = st.n_beats; ST.head = st.head;
  ST.generator = st.generator;
  setGenBadge(st.generator);
  $("undo").disabled = !st.can_undo; $("redo").disabled = !st.can_redo;
  drawBeatsTicks();
  renderHistory(st.timeline);
  if (!opts.keepMetrics && st.metrics && st.metrics.energy !== undefined) showCurrentMetrics(st.metrics);
}

const GEN_INFO = {
  live: ["live pod", "Live pod mode: every new seed runs a fresh LODGE/EDGE diffusion sample on the GPU pod (minutes per new take), so edits search an unbounded space. Falls back to the local bank if the pod can't answer."],
  bank: ["bank", "Editing selects from a pre-generated bank of real LODGE/EDGE takes for this song. Instant, and works with the pod switched off, but only as diverse as the number of cached seeds."],
  mock: ["offline demo", "No real backbone takes for this song yet — using the offline stand-in generator. Connect a GPU pod (bank or live mode) for real search."],
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
    $("agentlog").innerHTML = "";
    $("progbar").style.width = "8%";
    $("progtext").textContent = `planning ${(ev.steps || []).length} step(s)...`;
  } else if (ev.phase === "refine") {
    $("progbar").style.width = "30%";
    $("progtext").textContent = `refining (attempt ${ev.cycle}): ${ev.summary || ""}`;
  } else if (ev.phase === "step") {
    const p = ev.n_steps ? Math.round(20 + (ev.step / ev.n_steps) * 60) : 50;
    $("progbar").style.width = p + "%";
    $("progtext").textContent = `attempt ${ev.cycle || 1} \u00b7 step ${ev.step}/${ev.n_steps}: ${ev.tool}`;
  } else if (ev.phase === "verify") {
    $("progtext").textContent = `attempt ${ev.cycle}: ${ev.ok ? "goal met \u2714" : "short of goal, refining\u2026"} (${ev.feedback || ""})`;
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
    const d = after[k] - before[k];
    if (Math.abs(d) < 1e-3) continue;
    const lbl = { energy: "energy", bas: "beat", jerk: "smooth.", foot: "foot" }[k];
    parts.push(`${lbl} ${before[k].toFixed(2)}\u2192${after[k].toFixed(2)}`);
  }
  return parts.join("  ");
}

function renderAgentLog(log) {
  const el = $("agentlog");
  if (!log || !log.length) { el.innerHTML = ""; return; }
  let html = "";
  let lastCycle = null;
  const multi = log.some((e) => (e.cycle || 1) > 1);
  for (const e of log) {
    const cyc = e.cycle || 1;
    if (multi && cyc !== lastCycle) {
      html += `<li class="cyc-div">${cyc === 1 ? "attempt 1" : "refine \u2192 attempt " + cyc}</li>`;
      lastCycle = cyc;
    }
    const delta = metricDeltaStr(e.metrics_before, e.metrics_after);
    html += `<li class="done"><span class="lstep">${e.step}</span>`
      + `<span class="ltool">${escapeHtml(e.tool)}</span>`
      + `<span class="lnote">${escapeHtml(e.note || e.why || "")}`
      + (delta ? ` <span class="ldelta">${delta}</span>` : "")
      + `</span></li>`;
  }
  el.innerHTML = html;
}

function onFinal(payload) {
  const res = payload.result, st = payload.state;
  $("progbar").style.width = "100%";
  $("progtext").textContent = res.agent_summary ? `plan: ${res.agent_summary}` : "edit applied";
  if (res.agent_summary) $("goal").innerHTML = `<b>Agent plan:</b> ${escapeHtml(res.agent_summary)}`;
  renderAgentLog(res.log);
  const fb = $("feedback");
  fb.textContent = (res.ok ? "\u2714 " : "\u26a0 ") + res.feedback;
  fb.className = "feedback " + (res.ok ? "ok" : "bad");
  applyState(st, { keepMetrics: true });          // update history/toolbar but keep the before->after table
  showMetrics(res.metrics_before, res.metrics_after);
  $("apply").disabled = false;
  toast(res.ok ? "Edit applied + checkpointed" : "Best-effort edit checkpointed");
  loadSkeleton();     // auto-refresh the instant 3D preview with the edited motion
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
  const d = (after - before);
  const better = HIGHER_BETTER[k];
  let cls = ""; if (better !== null && Math.abs(d) > 1e-6) cls = ((d > 0) === better) ? "up" : "down";
  const arrow = Math.abs(d) < 1e-6 ? "" : (d > 0 ? " \u25b2" : " \u25bc");
  tr.innerHTML = labelCell(k)
    + `<td class="num">${before.toFixed(3)}</td>`
    + `<td class="num">${after.toFixed(3)}</td>`
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
    tr.innerHTML = labelCell(k) + `<td></td><td class="num">${m[k].toFixed(3)}</td><td></td>`;
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
      loadSkeleton(); toast("Rolled back to: " + (c.label || "original")); };
    ol.appendChild(li);
  });
}

init().catch((e) => toast("init failed: " + e.message));

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

// -------------------------------------------------------------- load songs + session
async function init() {
  const { songs } = await api("/api/songs");
  const sel = $("song");
  sel.innerHTML = "";
  songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid; o.textContent = s.name || s.sid; sel.appendChild(o); });
  sel.onchange = () => openSession(sel.value);
  if (songs.length) await openSession(songs[0].sid);
  wireTimeline();
  wireUpload();
  $("apply").onclick = runEdit;
  $("undo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/undo`, { method: "POST" }));
  $("redo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/redo`, { method: "POST" }));
  $("instruction").addEventListener("keydown", (e) => { if (e.key === "Enter") runEdit(); });
}

async function openSession(sid) {
  const st = await api(`/api/session/${sid}`, { method: "POST" });
  ST.sid = sid;
  const v = $("video");
  v.src = st.preview_url + "?t=" + Date.now();
  applyState(st);
  toast(`Loaded ${sid}: ${st.duration}s, ${st.n_beats} beats, ${st.generator} generator`);
}

function applyState(st, opts) {
  opts = opts || {};
  ST.fps = st.fps; ST.dur = st.duration; ST.nframes = st.n_frames; ST.beats = st.n_beats; ST.head = st.head;
  $("undo").disabled = !st.can_undo; $("redo").disabled = !st.can_redo;
  drawBeatsTicks();
  renderHistory(st.timeline);
  if (!opts.keepMetrics && st.metrics && st.metrics.energy !== undefined) showCurrentMetrics(st.metrics);
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
  if (ev.phase === "parsed") {
    const g = ev.goal; $("goal").innerHTML =
      `<span class="tag ${g.backbone}">${g.backbone}</span> objective: <b>${g.objective.replace(/_/g, " ")}</b>`;
    $("progtext").textContent = "planning the edit...";
  } else if (ev.phase === "candidate") {
    const p = ev.total ? Math.round((ev.done / ev.total) * 100) : 0;
    $("progbar").style.width = p + "%";
    $("progtext").textContent = `cycle ${ev.cycle} \u00b7 trying ${ev.backbone} take #${ev.seed}  (candidate ${ev.done}/${ev.total}, best so far ${ev.best_reward})`;
  } else if (ev.phase === "verify") {
    $("progtext").textContent = `checked cycle ${ev.cycle}: ${ev.ok ? "goal met \u2714" : "not there yet, refining..."}`;
  }
}

function onFinal(payload) {
  const res = payload.result, st = payload.state, ncand = payload.cycles ? payload.cycles.length : 0;
  $("progbar").style.width = "100%";
  const src = st.generator === "bank" ? "real backbone takes" : (st.generator === "remote" ? "live pod generations" : "candidates");
  $("progtext").textContent = `evaluated ${ncand} ${src} across ${res.n_cycles || 0} cycle(s)`;
  const fb = $("feedback");
  let msg = (res.ok ? "\u2714 " : "\u26a0 ") + res.feedback;
  if (!res.ok && st.generator === "bank") msg += "  \u2014 the bank has limited takes for this window; a richer bank (more seeds) or live pod mode would search harder.";
  fb.textContent = msg;
  fb.className = "feedback " + (res.ok ? "ok" : "bad");
  applyState(st, { keepMetrics: true });          // update history/toolbar but keep the before->after table
  showMetrics(res.metrics_before, res.metrics_after);
  $("apply").disabled = false;
  toast(res.ok ? "Edit applied + checkpointed" : "Best-effort edit checkpointed");
  // NOTE: re-rendering the edited motion to video needs the GPU worker; the preview stays the
  // base take. Metric deltas + history reflect the real edited motion immediately.
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
      toast("Rolled back to: " + (c.label || "original")); };
    ol.appendChild(li);
  });
}

init().catch((e) => toast("init failed: " + e.message));

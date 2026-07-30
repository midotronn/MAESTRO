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

function toast(msg) {
  let t = $("toast"); if (!t) { t = document.createElement("div"); t.id = "toast"; t.className = "toast"; document.body.appendChild(t); }
  t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 2200);
}

// -------------------------------------------------------------- load songs + session
async function init() {
  const { songs } = await api("/api/songs");
  const sel = $("song");
  sel.innerHTML = "";
  songs.forEach((s) => { const o = document.createElement("option"); o.value = s.sid; o.textContent = s.sid + (s.has_bank ? "  (bank)" : ""); sel.appendChild(o); });
  sel.onchange = () => openSession(sel.value);
  if (songs.length) await openSession(songs[0].sid);
  wireTimeline();
  $("apply").onclick = runEdit;
  $("undo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/undo`, { method: "POST" }));
  $("redo").onclick = async () => applyState(await api(`/api/session/${ST.sid}/redo`, { method: "POST" }));
  $("instruction").addEventListener("keydown", (e) => { if (e.key === "Enter") runEdit(); });
  renderChips();
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
  const g = st.generator === "bank" ? "real backbone bank" : (st.generator === "remote" ? "live pod generation" : "offline mock");
  $("genkind").textContent = g;
  $("genkind").title = st.generator === "bank"
    ? "Candidates come from a pre-generated bank of real LODGE/EDGE takes. The heavy generation ran once when the bank was built, so edits are instant selection + splice."
    : (st.generator === "remote" ? "Each edit runs real LODGE/EDGE generation on the pod (slower)."
       : "No bank found: using the offline synthetic generator.");
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
function renderChips() {
  const ex = ["make this more energetic", "make this calmer", "tighten this to the beat",
    "make this smoother", "make this sharper", "reverse this section", "mirror this"];
  const c = $("chips"); c.innerHTML = "";
  ex.forEach((t) => { const s = document.createElement("span"); s.className = "chip"; s.textContent = t;
    s.onclick = () => { $("instruction").value = t; }; c.appendChild(s); });
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
  tr.innerHTML = `<td>${METRIC_LABELS[k] || k}</td>`
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
    tr.innerHTML = `<td>${METRIC_LABELS[k]}</td><td></td><td class="num">${m[k].toFixed(3)}</td><td></td>`;
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

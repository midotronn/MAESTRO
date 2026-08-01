/* MAESTRO project page: render structure timelines, metrics, and storyboards from JSON. */
(function () {
  "use strict";
  const DATA = "static/data/";
  const VID = "static/videos/";

  const pct = (x) => (x * 100).toFixed(1) + "%";
  const round = (x, d = 3) => (x == null ? "" : Number(x).toFixed(d));

  async function getJSON(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error("fetch " + path + " -> " + r.status);
    return r.json();
  }

  function srcClass(src) {
    if (src === "lodge") return "lodge";
    if (src === "edge") return "edge";
    return "reuse"; // reuse:N motif recall
  }
  function srcLabel(src) {
    if (src === "lodge") return "LODGE";
    if (src === "edge") return "EDGE";
    if (src && src.indexOf("reuse") === 0) return "recap";
    return src || "";
  }

  function structureTimeline(schedule) {
    if (!schedule || !schedule.length) return "";
    const total = schedule[schedule.length - 1][1] || 1;
    const segs = schedule.map(([a, b, src, role]) => {
      const w = ((b - a) / total) * 100;
      const cls = srcClass(src);
      return `<div class="seg ${cls}" style="flex:0 0 ${w}%" title="${role} · ${srcLabel(src)} · ${(a / 30).toFixed(1)}-${(b / 30).toFixed(1)}s">
        <span class="role">${role || ""}</span><span class="src">${srcLabel(src)}</span></div>`;
    }).join("");
    return `<div class="struct-wrap">
      <div class="struct-label">Detected structure &amp; per-section source</div>
      <div class="timeline">${segs}</div>
      <div class="legend">
        <span><span class="sw" style="background:var(--src-lodge)"></span>LODGE section</span>
        <span><span class="sw" style="background:var(--src-edge)"></span>EDGE section</span>
        <span><span class="sw" style="background:var(--src-reuse)"></span>Motif recap (reused, may be mirrored/retrograded)</span>
      </div></div>`;
  }

  const ROWS = [
    ["beat_alignment", "Beat alignment (BAS)"],
    ["beat_coverage", "Beat coverage"],
    ["foot_contact_consistency", "Foot-contact consistency"],
  ];
  function metricsTable(cols) {
    if (!cols) return "";
    const L = cols.LODGE_1seed, E = cols.EDGE_1seed, M = cols.AgentLODGE_multiseed;
    if (!L || !E || !M) return "";
    const body = ROWS.map(([k, label]) => {
      const vals = [L[k], E[k], M[k]];
      const best = Math.max(vals[0], vals[1], vals[2]);
      const cell = (v, extra) => `<td class="${v === best ? "win" : ""} ${extra || ""}">${round(v)}</td>`;
      return `<tr><td>${label}</td>${cell(L[k])}${cell(E[k])}${cell(M[k], "maestro")}</tr>`;
    }).join("");
    return `<table class="mtable">
      <thead><tr><th>Metric (whole dance)</th><th>LODGE</th><th>EDGE</th><th class="maestro">MAESTRO</th></tr></thead>
      <tbody>${body}</tbody></table>
      <p class="vid-cap">LODGE and EDGE are single best-seed baselines; MAESTRO is the structure-aware assembly. Higher is better.</p>`;
  }

  function storyboard(story) {
    if (!story) return "";
    const plans = (story.plans || []).map((p) => {
      const v = p.variation || {};
      const genCls = p.generator_bias === "lodge" ? "src-lodge" : p.generator_bias === "edge" ? "src-edge" : "";
      const chips = [`<span class="chip ${genCls}">${(p.generator_bias || "auto")}</span>`];
      if (p.reuse_of != null) {
        const flags = [];
        if (v.mirror) flags.push("mirror");
        if (v.retrograde) flags.push("retrograde");
        if (v.retime && v.retime !== 1) flags.push("retime " + v.retime + "x");
        chips.push(`<span class="chip motif">recap of #${p.reuse_of}${flags.length ? " · " + flags.join(" · ") : ""}</span>`);
      }
      return `<div class="plan">
        <div class="role">${p.role || ""}</div>
        <div class="meta">${p.vocabulary || ""}</div>
        <div class="bar"><i style="width:${pct(p.target_intensity || 0)}"></i></div>
        <div class="chips">${chips.join("")}</div>
      </div>`;
    }).join("");
    return `<div class="struct-wrap">
      <div class="struct-label">LLM storyboard</div>
      ${story.arc ? `<div class="arc">${story.arc}</div>` : ""}
      ${story.reasoning ? `<p class="prose" style="font-size:0.95rem">${story.reasoning}</p>` : ""}
      <div class="plans">${plans}</div></div>`;
  }

  async function renderSong(song) {
    const parts = [];
    parts.push(`<div class="song-head"><div class="song-title">${song.title}</div></div>`);
    parts.push(`<div class="song-tag">${song.tagline || ""}</div>`);
    parts.push(`<video controls playsinline preload="metadata" src="${VID}${song.video}"></video>`);
    parts.push(`<div class="vid-cap">Side by side: LODGE &nbsp;|&nbsp; EDGE &nbsp;|&nbsp; MAESTRO (structure-aware)</div>`);

    let metrics = null, story = null;
    try { if (song.metrics) metrics = await getJSON(DATA + song.metrics); } catch (e) { console.warn(e); }
    try { if (song.story) story = await getJSON(DATA + song.story); } catch (e) { console.warn(e); }

    if (metrics) parts.push(structureTimeline(metrics.schedule));
    if (story) parts.push(storyboard(story));
    if (metrics) parts.push(metricsTable(metrics.columns));

    const el = document.createElement("div");
    el.className = "song";
    el.innerHTML = parts.join("");
    return el;
  }

  // On a static host (e.g. GitHub Pages) there is no Python backend, so the "Launch live editor"
  // links can't point at "/". Send them to the repo instead. When the page is served BY the editor
  // app (localhost or a deployed server), "/" is the editor, so leave the links alone.
  function fixEditorLinks() {
    const h = location.hostname;
    const localish = h === "localhost" || h === "127.0.0.1" || h === "0.0.0.0" || h === "";
    if (localish || !h.endsWith("github.io")) return;
    const repo = "https://github.com/midotronn/AgentLODGE";
    document.querySelectorAll('a[href="/"]').forEach((a) => {
      a.setAttribute("href", repo);
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener");
      a.setAttribute("title", "The live editor runs from the Python server; see the repo to run it locally.");
    });
  }

  async function main() {
    fixEditorLinks();
    const host = document.getElementById("songs");
    if (!host) return;
    let songs;
    try { songs = await getJSON(DATA + "songs.json"); }
    catch (e) { host.innerHTML = `<p class="loading">Could not load demo data.</p>`; return; }
    host.innerHTML = "";
    for (const s of songs) {
      try { host.appendChild(await renderSong(s)); }
      catch (e) { console.warn("song " + s.id, e); }
    }
  }
  document.addEventListener("DOMContentLoaded", main);
})();

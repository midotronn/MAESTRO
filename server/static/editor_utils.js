(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.MaestroEditorUtils = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const number = (value) => {
    if (value === null || value === "" || typeof value === "boolean") return null;
    const parsed = typeof value === "number" ? value : Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  };

  const unitName = (value, fallback) => {
    const unit = String(value || fallback || "frames").toLowerCase();
    return unit === "s" || unit.startsWith("sec") ? "seconds" : "frames";
  };

  function rawPair(value, fallbackUnit) {
    if (Array.isArray(value) && value.length >= 2) {
      return { start: number(value[0]), end: number(value[1]), unit: unitName(null, fallbackUnit) };
    }
    if (!value || typeof value !== "object") return null;
    if (Array.isArray(value.seconds)) return rawPair(value.seconds, "seconds");
    if (Array.isArray(value.frames)) return rawPair(value.frames, "frames");

    const forms = [
      ["start_sec", "end_sec", "seconds"],
      ["a_sec", "b_sec", "seconds"],
      ["start_seconds", "end_seconds", "seconds"],
      ["start_frame", "end_frame", "frames"],
      ["a_frame", "b_frame", "frames"],
      ["startFrame", "endFrame", "frames"],
      ["start", "end", unitName(value.unit, fallbackUnit)],
      ["a", "b", unitName(value.unit, fallbackUnit)],
      ["from", "to", unitName(value.unit, fallbackUnit)],
    ];
    for (const [startKey, endKey, unit] of forms) {
      if (value[startKey] !== undefined || value[endKey] !== undefined) {
        return { start: number(value[startKey]), end: number(value[endKey]), unit };
      }
    }
    return null;
  }

  function checkpointInterval(checkpoint, fps, duration) {
    const c = checkpoint && typeof checkpoint === "object" ? checkpoint : {};
    const edit = c.edit && typeof c.edit === "object" ? c.edit : {};
    const rate = number(fps) > 0 ? number(fps) : 30;
    const candidates = [
      [c.interval, "frames", "checkpoint.interval"],
      [edit.interval_seconds, "seconds", "edit.interval_seconds"],
      [edit.window_seconds, "seconds", "edit.window_seconds"],
      [edit.window_sec, "seconds", "edit.window_sec"],
      [c.interval_seconds, "seconds", "checkpoint.interval_seconds"],
      [c.window_seconds, "seconds", "checkpoint.window_seconds"],
      [c.window_sec, "seconds", "checkpoint.window_sec"],
      [edit, "frames", "edit fields"],
      [c, "frames", "checkpoint fields"],
      [edit.window, "frames", "edit.window"],
      [edit.interval, "frames", "edit.interval"],
      [edit.frame_interval, "frames", "edit.frame_interval"],
      [c.window, "frames", "checkpoint.window"],
      [c.frame_interval, "frames", "checkpoint.frame_interval"],
    ];

    let found = null;
    for (const [value, fallbackUnit, source] of candidates) {
      const pair = rawPair(value, fallbackUnit);
      if (pair && pair.start !== null && pair.end !== null) {
        found = { ...pair, source };
        break;
      }
    }

    if (!found) {
      const label = String(c.label || "");
      const match = label.match(/\[\s*(\d+(?:\.\d+)?)\s*[-\u2013]\s*(\d+(?:\.\d+)?)\s*s\s*\]/i);
      if (match) {
        found = {
          start: Number(match[1]),
          end: Number(match[2]),
          unit: "seconds",
          source: "label",
        };
      }
    }
    if (!found) return null;

    let startSec = found.unit === "seconds" ? found.start : found.start / rate;
    let endSec = found.unit === "seconds" ? found.end : found.end / rate;
    if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) return null;
    if (endSec < startSec) [startSec, endSec] = [endSec, startSec];
    const limit = number(duration);
    if (limit !== null && limit >= 0) {
      startSec = Math.max(0, Math.min(startSec, limit));
      endSec = Math.max(startSec, Math.min(endSec, limit));
    } else {
      startSec = Math.max(0, startSec);
      endSec = Math.max(startSec, endSec);
    }
    return {
      startSec,
      endSec,
      startFrame: Math.round(startSec * rate),
      endFrame: Math.round(endSec * rate),
      source: found.source,
    };
  }

  function formatSeconds(value) {
    if (!Number.isFinite(value)) return "";
    const rounded = Math.round(value);
    if (Math.abs(value - rounded) < 0.05) return String(rounded);
    return value.toFixed(1).replace(/\.0$/, "");
  }

  function formatCheckpointInterval(checkpoint, fps, duration) {
    const interval = checkpointInterval(checkpoint, fps, duration);
    if (!interval) return "";
    return `${formatSeconds(interval.startSec)}\u2013${formatSeconds(interval.endSec)}s`;
  }

  function timelineFraction(clientX, rect, box) {
    if (!rect || !Number.isFinite(rect.left) || !Number.isFinite(rect.width) || rect.width <= 0) {
      return 0;
    }
    const metrics = box || {};
    const offsetWidth = number(metrics.offsetWidth);
    const scale = offsetWidth && offsetWidth > 0 ? rect.width / offsetWidth : 1;
    const border = Math.max(0, number(metrics.clientLeft) || 0) * scale;
    const contentWidth = Math.max(
      1,
      (number(metrics.clientWidth) || Math.max(1, rect.width / scale - 2 * border)) * scale,
    );
    const fraction = (Number(clientX) - rect.left - border) / contentWidth;
    return Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
  }

  function renderedMediaBox(rect, videoWidth, videoHeight) {
    if (!rect) return { left: 0, top: 0, width: 0, height: 0 };
    const width = Math.max(0, number(rect.width) || 0);
    const height = Math.max(0, number(rect.height) || 0);
    const sourceWidth = number(videoWidth);
    const sourceHeight = number(videoHeight);
    if (!sourceWidth || !sourceHeight || !width || !height) {
      return { left: rect.left || 0, top: rect.top || 0, width, height };
    }
    const scale = Math.min(width / sourceWidth, height / sourceHeight);
    const mediaWidth = sourceWidth * scale;
    const mediaHeight = sourceHeight * scale;
    return {
      left: (rect.left || 0) + (width - mediaWidth) / 2,
      top: (rect.top || 0) + (height - mediaHeight) / 2,
      width: mediaWidth,
      height: mediaHeight,
    };
  }

  function checkpointLineage(timeline, checkpointId) {
    const list = Array.isArray(timeline) ? timeline : [];
    const byId = new Map(list.map((item) => [item.id, item]));
    const chosen = byId.get(checkpointId);
    if (chosen && Array.isArray(chosen.lineage) && chosen.lineage.length) {
      return chosen.lineage.filter((id) => byId.has(id));
    }
    const lineage = [];
    const seen = new Set();
    let current = chosen;
    while (current && !seen.has(current.id)) {
      lineage.push(current.id);
      seen.add(current.id);
      current = current.parent_id ? byId.get(current.parent_id) : null;
    }
    return lineage.reverse();
  }

  function checkpointTree(timeline) {
    const source = Array.isArray(timeline) ? timeline : [];
    const checkpoints = [];
    const byId = new Map();
    source.forEach((checkpoint, sourceIndex) => {
      if (!checkpoint || checkpoint.id == null || byId.has(checkpoint.id)) return;
      const entry = { checkpoint, sourceIndex };
      checkpoints.push(entry);
      byId.set(checkpoint.id, entry);
    });

    const childrenByParent = new Map();
    checkpoints.forEach((entry) => {
      const parentId = entry.checkpoint.parent_id;
      if (!parentId || parentId === entry.checkpoint.id || !byId.has(parentId)) return;
      if (!childrenByParent.has(parentId)) childrenByParent.set(parentId, []);
      childrenByParent.get(parentId).push(entry);
    });

    const compare = (parentId) => {
      const declared = byId.get(parentId)?.checkpoint?.children;
      const declaredOrder = new Map(
        (Array.isArray(declared) ? declared : []).map((id, index) => [id, index]),
      );
      return (left, right) => {
        const leftDeclared = declaredOrder.has(left.checkpoint.id);
        const rightDeclared = declaredOrder.has(right.checkpoint.id);
        if (leftDeclared !== rightDeclared) return leftDeclared ? -1 : 1;
        if (leftDeclared) {
          return declaredOrder.get(left.checkpoint.id) - declaredOrder.get(right.checkpoint.id);
        }
        const leftTs = number(left.checkpoint.ts);
        const rightTs = number(right.checkpoint.ts);
        if (leftTs !== null && rightTs !== null && leftTs !== rightTs) return leftTs - rightTs;
        return left.sourceIndex - right.sourceIndex;
      };
    };
    childrenByParent.forEach((children, parentId) => children.sort(compare(parentId)));

    const roots = checkpoints
      .filter((entry) => {
        const parentId = entry.checkpoint.parent_id;
        return !parentId || parentId === entry.checkpoint.id || !byId.has(parentId);
      })
      .sort((left, right) => left.sourceIndex - right.sourceIndex);
    const visited = new Set();
    const build = (entry) => {
      if (!entry || visited.has(entry.checkpoint.id)) return null;
      visited.add(entry.checkpoint.id);
      return {
        checkpoint: entry.checkpoint,
        children: (childrenByParent.get(entry.checkpoint.id) || [])
          .map(build)
          .filter(Boolean),
      };
    };
    const tree = roots.map(build).filter(Boolean);
    checkpoints.forEach((entry) => {
      if (!visited.has(entry.checkpoint.id)) {
        const node = build(entry);
        if (node) tree.push(node);
      }
    });
    return tree;
  }

  return {
    checkpointInterval,
    checkpointLineage,
    checkpointTree,
    formatCheckpointInterval,
    renderedMediaBox,
    timelineFraction,
  };
});

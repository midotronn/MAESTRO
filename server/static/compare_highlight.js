(function exposeCompareHighlight(global) {
  "use strict";

  const DEFAULT_THRESHOLD = 42;

  function colorizeChangedPixels(before, after, overlay, threshold = DEFAULT_THRESHOLD) {
    if (before.length !== after.length || before.length !== overlay.length) {
      throw new TypeError("comparison pixel buffers must have matching lengths");
    }
    if (before.length % 4 !== 0) {
      throw new TypeError("comparison pixel buffers must contain RGBA pixels");
    }

    let changed = 0;
    for (let i = 0; i < before.length; i += 4) {
      const dr = Math.abs(after[i] - before[i]);
      const dg = Math.abs(after[i + 1] - before[i + 1]);
      const db = Math.abs(after[i + 2] - before[i + 2]);
      const strongest = Math.max(dr, dg, db);
      const average = (dr + dg + db) / 3;

      if (strongest < threshold && average < threshold * 0.72) {
        overlay[i] = 0;
        overlay[i + 1] = 0;
        overlay[i + 2] = 0;
        overlay[i + 3] = 0;
        continue;
      }

      const strength = Math.max(strongest, average);
      overlay[i] = 24;
      overlay[i + 1] = 211;
      overlay[i + 2] = 238;
      overlay[i + 3] = Math.min(220, 70 + Math.round((strength - threshold) * 1.8));
      changed += 1;
    }
    return changed;
  }

  global.MAESTRO_COMPARE_HIGHLIGHT = {
    DEFAULT_THRESHOLD,
    colorizeChangedPixels,
  };
})(typeof window === "undefined" ? globalThis : window);

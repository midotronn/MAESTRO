"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {
  checkpointInterval,
  checkpointLineage,
  formatCheckpointInterval,
  renderedMediaBox,
  timelineFraction,
} = require("../server/static/editor_utils.js");

test("checkpoint intervals normalize every persisted shape without error text", () => {
  const cases = [
    [{ interval: { start_sec: 2.25, end_sec: 4.75 } }, [2.25, 4.75]],
    [{ edit: { window: [60, 150] } }, [2, 5]],
    [{ edit: { interval: { start_frame: 90, end_frame: 180 } } }, [3, 6]],
    [{ edit: { window_sec: [1.5, 2.5] } }, [1.5, 2.5]],
    [{ edit: { a_sec: 7, b_sec: 9 } }, [7, 9]],
    [{ window: { start: 330, end: 300, unit: "frames" } }, [10, 11]],
    [{ label: "older edit [12.5-14s]" }, [12.5, 14]],
  ];
  for (const [checkpoint, expected] of cases) {
    const interval = checkpointInterval(checkpoint, 30, 20);
    assert.ok(interval);
    assert.deepEqual([interval.startSec, interval.endSec], expected);
    const text = formatCheckpointInterval(checkpoint, 30, 20);
    assert.match(text, /^\d+(?:\.\d+)?\u2013\d+(?:\.\d+)?s$/);
    assert.doesNotMatch(text, /NaN|undefined|error/i);
  }

  for (const checkpoint of [
    {},
    { edit: null },
    { edit: { window: [null, "nope"] } },
    { interval: { start_sec: Number.NaN, end_sec: 2 } },
  ]) {
    assert.equal(formatCheckpointInterval(checkpoint, 30, 20), "");
  }
});

test("timeline coordinates use the scaled content box and clamp drags", () => {
  const rect = { left: 100, width: 408 };
  const box = { clientLeft: 2, clientWidth: 200, offsetWidth: 204 };
  assert.equal(timelineFraction(104, rect, box), 0);
  assert.equal(timelineFraction(304, rect, box), 0.5);
  assert.equal(timelineFraction(504, rect, box), 1);
  assert.equal(timelineFraction(-500, rect, box), 0);
  assert.equal(timelineFraction(900, rect, box), 1);
});

test("rendered media box follows letterboxing at any layout size", () => {
  assert.deepEqual(
    renderedMediaBox({ left: 10, top: 20, width: 800, height: 400 }, 400, 400),
    { left: 210, top: 20, width: 400, height: 400 },
  );
  assert.deepEqual(
    renderedMediaBox({ left: 0, top: 0, width: 300, height: 600 }, 1920, 1080),
    { left: 0, top: 215.625, width: 300, height: 168.75 },
  );
});

test("checkpoint lineage is root first for backend and legacy timeline shapes", () => {
  const timeline = [
    { id: "root", parent_id: null },
    { id: "a", parent_id: "root" },
    { id: "b", parent_id: "a", lineage: ["root", "a", "b"] },
    { id: "fork", parent_id: "a" },
  ];
  assert.deepEqual(checkpointLineage(timeline, "b"), ["root", "a", "b"]);
  assert.deepEqual(checkpointLineage(timeline, "fork"), ["root", "a", "fork"]);
  assert.deepEqual(checkpointLineage(timeline, "missing"), []);
});

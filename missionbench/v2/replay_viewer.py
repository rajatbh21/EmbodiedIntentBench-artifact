"""The static HTML replay viewer: one self-contained page over a replay bundle.

``build_viewer_html`` embeds the bundle's manifest and events JSON directly into the
page (so it works from ``file://`` with no server and no build step) and references the
frame/background PNGs by relative path. Plain HTML/CSS/JS — no framework, no external
assets, no online map service.

Presentation decisions (all viewer-side; benchmark semantics untouched):

- **Real map background.** The mission map draws over ``manifest.map.background`` — a
  pre-rendered crop of the mission's own world raster with an exact meter extent — so
  the trajectory, UAV, detections, and skips register on the actual mission
  environment. Absent the background (an early 1.0 bundle), the map falls back to the
  schematic drawing.
- **Continuous playback over mission time.** Play advances a mission-time clock via
  ``requestAnimationFrame`` (1x = real mission seconds); the UAV position and path
  progress are interpolated from the manifest's waypoints and ``drone_speed_mps`` —
  the same constant-speed-polyline rule the simulator's trajectory uses — while the
  discrete observation panels switch exactly at their event boundaries. Stepping
  (prev/next/slider) still pins the timeline to an observation's completion state.

Information hierarchy (deliberate, and the reason this exists): mission outcome first,
hard-constraint status second, policy/system behaviour third, perception diagnostics
last and visually subordinate. Ground truth appears only behind controls labelled
"Debug GT" — it is evaluator-side data, never something the policy could see.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["build_viewer_html"]


def build_viewer_html(payload: dict[str, Any]) -> str:
    """Return the viewer page with ``payload`` (manifest + replay docs) embedded."""
    data = json.dumps(payload, sort_keys=True, allow_nan=False)
    # A literal "</script>" inside the JSON would end the data block early.
    data = data.replace("</", "<\\/")
    title = f"EmbodiedIntentBench legacy visual replay — {payload['manifest']['mission_id']}"
    return _TEMPLATE.replace("__TITLE__", title).replace("__REPLAY_DATA_JSON__", data)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --page: #f9f9f7;
    --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
    --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
    --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
    --s5: #e87ba4; --s6: #008300; --s7: #4a3aa7; --s8: #e34948;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--page); color: var(--ink);
    font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header { padding: 10px 16px 0; }
  header h1 { font-size: 16px; margin: 0 0 6px; }
  header .sub { color: var(--ink-2); font-size: 12px; }
  #mission-status-banner {
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 14px; margin: 10px 16px;
  }
  #mission-status-banner .big { font-size: 20px; font-weight: 700; }
  .layout-top {
    display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 1fr); gap: 12px;
    padding: 0 16px; align-items: start;
  }
  @media (max-width: 1000px) { .layout-top { grid-template-columns: 1fr; } }
  .panel {
    background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
    padding: 10px 12px;
  }
  .panel h2 {
    font-size: 13px; margin: 0 0 8px; color: var(--ink-2);
    text-transform: uppercase; letter-spacing: .04em;
  }
  canvas { display: block; width: 100%; background: var(--surface); }
  #mission-map { border: 1px solid var(--grid); border-radius: 4px; }
  .map-status {
    display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px;
    font-size: 13px; font-variant-numeric: tabular-nums;
  }
  .map-status .lbl { color: var(--muted); margin-right: 5px; }
  .legend {
    display: flex; flex-wrap: wrap; gap: 6px 14px; margin-top: 8px;
    font-size: 12px; color: var(--ink-2);
  }
  .legend .sw {
    display: inline-block; width: 10px; height: 10px; border-radius: 50%;
    margin-right: 4px; vertical-align: -1px;
  }
  .kv {
    display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; font-size: 12.5px;
  }
  .kv dt { color: var(--muted); }
  .kv dd { margin: 0; font-variant-numeric: tabular-nums; }
  #frame-stack {
    position: relative; border: 1px solid var(--grid); border-radius: 4px;
    overflow: hidden;
  }
  #frame-stack img { display: block; width: 100%; image-rendering: pixelated; }
  #frame-stack img.overlay { position: absolute; inset: 0; }
  .toggles { display: flex; gap: 14px; margin: 8px 0; font-size: 12.5px; flex-wrap: wrap; }
  .pill {
    display: inline-flex; align-items: center; gap: 6px; padding: 1px 8px;
    border-radius: 10px; font-size: 11.5px; font-weight: 600;
    border: 1px solid var(--border);
  }
  .pill::before {
    content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor;
  }
  .st-SAFE { color: var(--good); background: rgba(12,163,12,.09); }
  .st-AT_RISK { color: #8a6100; background: rgba(250,178,25,.16); }
  .st-VIOLATED { color: var(--critical); background: rgba(208,59,59,.10); }
  .st-NOT_APPLICABLE, .st-UNKNOWN { color: var(--muted); background: rgba(137,135,129,.12); }
  #mission-dashboard { margin: 12px 16px 0; }
  .dash-grid {
    display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr) minmax(0, 1fr);
    gap: 18px; align-items: start;
  }
  @media (max-width: 1000px) { .dash-grid { grid-template-columns: 1fr; } }
  table.constraints { width: 100%; border-collapse: collapse; font-size: 12.5px; }
  table.constraints th {
    text-align: left; color: var(--muted); font-weight: 500;
    border-bottom: 1px solid var(--grid); padding: 3px 6px;
  }
  table.constraints td {
    padding: 4px 6px; border-bottom: 1px solid var(--grid);
    font-variant-numeric: tabular-nums;
  }
  .dash-grid h3 { font-size: 12px; color: var(--muted); margin: 0 0 6px; }
  section.diag {
    color: var(--ink-2); font-size: 12px; border-left: 1px dashed var(--grid);
    padding-left: 16px;
  }
  @media (max-width: 1000px) {
    section.diag { border-left: 0; padding-left: 0; border-top: 1px dashed var(--grid); padding-top: 8px; }
  }
  .controls {
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
    padding: 10px 16px 4px; margin: 8px 16px 0;
  }
  .controls button {
    font: inherit; padding: 4px 12px; border-radius: 6px;
    border: 1px solid var(--border); background: var(--surface); cursor: pointer;
  }
  .controls button:hover { background: var(--grid); }
  #timeline-slider { flex: 1; min-width: 180px; }
  #config-strip {
    height: 26px; border: 1px solid var(--grid); border-radius: 4px;
    margin: 0 16px 14px; width: calc(100% - 32px);
  }
  .strip-legend {
    display: flex; gap: 14px; font-size: 12px; color: var(--ink-2);
    padding: 0 16px 16px; flex-wrap: wrap;
  }
  .gt-note { color: var(--serious); font-size: 11.5px; }
  footer { color: var(--muted); font-size: 11.5px; padding: 0 16px 18px; }
</style>
</head>
<body>
<header>
  <h1>EmbodiedIntentBench legacy visual replay viewer</h1>
  <div class="sub" id="mission-sub"></div>
</header>

<div id="mission-status-banner">
  <span class="big" id="mission-status"></span>
  <span class="pill" id="overall-pill"></span>
  <span id="evidence-line"></span>
  <span id="termination-line" class="sub"></span>
</div>

<div class="layout-top">
  <div class="panel" id="panel-map">
    <h2>Mission map</h2>
    <canvas id="mission-map" width="960" height="480"></canvas>
    <div class="map-status">
      <span><span class="lbl">time</span><span id="map-time"></span></span>
      <span><span class="lbl">path progress</span><span id="map-progress"></span></span>
      <span><span class="lbl">executed config</span><span id="map-config"></span></span>
    </div>
    <div class="legend" id="map-legend"></div>
  </div>

  <div class="panel" id="perception-view">
    <h2>Current perception view</h2>
    <div id="frame-stack">
      <img id="frame-rgb" alt="observation RGB">
      <img id="frame-pred" class="overlay" alt="prediction overlay">
      <img id="frame-gt" class="overlay" alt="Debug GT overlay" hidden>
    </div>
    <div class="toggles">
      <label><input type="checkbox" id="toggle-pred" checked> prediction overlay</label>
      <label class="gt-note"><input type="checkbox" id="toggle-gt"> Debug GT (evaluator-only; not policy-visible)</label>
    </div>
    <dl class="kv" id="frame-meta"></dl>
  </div>
</div>

<div class="panel" id="mission-dashboard">
  <h2>Mission dashboard</h2>
  <div class="dash-grid">
    <div>
      <h3>Hard constraints</h3>
      <table class="constraints" id="constraint-table">
        <thead><tr><th>constraint</th><th>status</th><th>value</th><th>limit</th><th>margin</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="behaviour">
      <h3>Policy &amp; system behaviour</h3>
      <dl class="kv" id="behaviour-meta"></dl>
    </div>
    <section class="diag">
      <h3>Perception diagnostics (subordinate to mission outcome)</h3>
      <dl class="kv" id="perception-meta"></dl>
    </section>
  </div>
</div>

<div class="controls">
  <button id="btn-prev" title="previous observation">&#9664;</button>
  <button id="btn-play">Play</button>
  <button id="btn-next" title="next observation">&#9654;</button>
  <input type="range" id="timeline-slider" min="0" value="0">
  <span id="obs-counter"></span>
  <label>speed <select id="speed-select">
    <option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option>
    <option value="2">2&times;</option><option value="4">4&times;</option>
    <option value="8">8&times;</option>
  </select></label>
</div>
<canvas id="config-strip"></canvas>
<div class="strip-legend" id="strip-legend"></div>

<footer id="honesty-footer"></footer>

<script id="replay-data" type="application/json">__REPLAY_DATA_JSON__</script>
<script>
"use strict";
const DATA = JSON.parse(document.getElementById("replay-data").textContent);
const M = DATA.manifest, R = DATA.replay, EVENTS = R.events, FINAL = R.final;
const N = EVENTS.length;
const CONFIG_HEX = ["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#4a3aa7","#e34948"];
const configIndex = {};
M.executor_configs.forEach((c, i) => { configIndex[c.config_id] = i; });
const configHex = id => CONFIG_HEX[configIndex[id] % CONFIG_HEX.length];
const END_T = FINAL.final_time_s || EVENTS[N-1].completion_time_s;

// ---- replay clock (presentation layer over mission time; semantics live in events) -----
let activeIdx = 0;
let replayTime = EVENTS[0].completion_time_s;
let playing = false, lastFrameTs = null;

// ---- header / identity ----------------------------------------------------------------
document.getElementById("mission-sub").textContent =
  `scenario ${M.scenario_id} · policy ${M.policy} · contract ${M.contract.contract_id}` +
  (M.evaluation_purpose ? ` · purpose ${M.evaluation_purpose}` : "");
document.getElementById("honesty-footer").textContent =
  Object.values(M.honesty.notes).join("  ·  ") + "  ·  " + M.honesty.nondeterminism;

// ---- map registration -----------------------------------------------------------------
const map = document.getElementById("mission-map"), mctx = map.getContext("2d");
const wps = M.map.waypoints_m;
const BG = M.map.background || null;
let bgImage = null;
if (BG) {
  bgImage = new Image();
  bgImage.onload = () => renderContinuous();
  bgImage.src = BG.file;
}
let ext;
if (BG) {
  ext = BG.extent_m;
} else {  // early 1.0 bundle without a background: schematic fallback bounds
  const pts = wps.concat(EVENTS.map(e => e.capture_position_m),
                         R.skipped_observations.map(s => s.position_m));
  const px = M.map.footprint_width_m, py = M.map.footprint_height_m;
  ext = [Math.min(...pts.map(p => p[0])) - px, Math.min(...pts.map(p => p[1])) - py,
         Math.max(...pts.map(p => p[0])) + px, Math.max(...pts.map(p => p[1])) + py];
}
const extW = ext[2] - ext[0], extH = ext[3] - ext[1];
function fitCanvas() {
  const w = Math.max(320, map.clientWidth || 960);
  map.width = w;
  map.height = Math.round(w * Math.min(Math.max(extH / extW, 0.22), 1.3));
}
fitCanvas();
const sx = x => (x - ext[0]) / extW * map.width;
const sy = y => (y - ext[1]) / extH * map.height;

// ---- trajectory interpolation (same rule as the simulator: constant speed along the
// waypoint polyline, clamped at the ends) -----------------------------------------------
const SPEED = M.map.drone_speed_mps || null;
const cumLen = [0];
for (let i = 1; i < wps.length; i++)
  cumLen.push(cumLen[i-1] + Math.hypot(wps[i][0]-wps[i-1][0], wps[i][1]-wps[i-1][1]));
const totalLen = cumLen[cumLen.length - 1];
function pointAtDistance(d) {
  d = Math.min(Math.max(d, 0), totalLen);
  for (let i = 1; i < wps.length; i++) {
    if (d <= cumLen[i] || i === wps.length - 1) {
      const seg = cumLen[i] - cumLen[i-1];
      const f = seg > 0 ? (d - cumLen[i-1]) / seg : 0;
      return [wps[i-1][0] + (wps[i][0]-wps[i-1][0]) * f,
              wps[i-1][1] + (wps[i][1]-wps[i-1][1]) * f];
    }
  }
  return wps[wps.length - 1];
}
function posAtTime(t) {
  if (!SPEED) return EVENTS[activeIdx].completion_position_m;
  return pointAtDistance(SPEED * t);
}
function progressAtTime(t) {
  if (!SPEED || totalLen === 0)
    return EVENTS[activeIdx].runtime.after_completion.path_progress;
  return Math.min(SPEED * t, totalLen) / totalLen;
}
function indexForTime(t) {
  let idx = 0;
  for (let i = 0; i < N; i++) { if (EVENTS[i].capture_time_s <= t) idx = i; else break; }
  return idx;
}
function partialPathPts(progress) {
  const target = progress * totalLen, pts = [wps[0]];
  for (let i = 1; i < wps.length; i++) {
    if (cumLen[i] <= target) { pts.push(wps[i]); continue; }
    pts.push(pointAtDistance(target));
    break;
  }
  return pts;
}

// ---- map drawing ----------------------------------------------------------------------
function poly(pts, colour, widthPx) {
  mctx.strokeStyle = colour; mctx.lineWidth = widthPx;
  mctx.lineJoin = "round"; mctx.lineCap = "round"; mctx.beginPath();
  pts.forEach((p, i) => i ? mctx.lineTo(sx(p[0]), sy(p[1])) : mctx.moveTo(sx(p[0]), sy(p[1])));
  mctx.stroke();
}
function dot(p, r, fill, stroke, strokeW) {
  mctx.beginPath(); mctx.arc(sx(p[0]), sy(p[1]), r, 0, 7);
  if (fill) { mctx.fillStyle = fill; mctx.fill(); }
  if (stroke) { mctx.strokeStyle = stroke; mctx.lineWidth = strokeW || 2; mctx.stroke(); }
}
function cross(p, r, colour) {
  const x = sx(p[0]), y = sy(p[1]);
  mctx.strokeStyle = colour; mctx.lineWidth = 2.5; mctx.lineCap = "round"; mctx.beginPath();
  mctx.moveTo(x-r, y-r); mctx.lineTo(x+r, y+r); mctx.moveTo(x-r, y+r); mctx.lineTo(x+r, y-r);
  mctx.stroke();
}
function drawMap(t) {
  const e = EVENTS[activeIdx], showGT = gtToggle.checked;
  mctx.clearRect(0, 0, map.width, map.height);
  if (bgImage && bgImage.complete && bgImage.naturalWidth)
    mctx.drawImage(bgImage, 0, 0, map.width, map.height);

  // Remaining path (light over imagery), then completed path with a dark halo.
  poly(wps, "rgba(252,252,251,0.75)", 3);
  const done = partialPathPts(progressAtTime(t));
  poly(done, "rgba(11,11,11,0.65)", 6);
  poly(done, "#2a78d6", 3.5);

  dot(wps[0], 7, null, "#0ca30c", 3);                          // start
  dot(wps[wps.length-1], 7, null, "#e87ba4", 3);               // final waypoint
  EVENTS.forEach(ev => {                                        // observation positions
    if (ev.capture_time_s <= t) dot(ev.capture_position_m, 3, "#2a78d6", "#fcfcfb", 1.5);
    else dot(ev.capture_position_m, 3, null, "rgba(252,252,251,0.85)", 1.5);
  });
  R.skipped_observations.forEach(s => {                         // skipped captures
    if (s.capture_time_s <= t) cross(s.position_m, 5, "#eb6834");
  });
  EVENTS.forEach(ev => {                                        // detection events
    if (ev.completion_time_s <= t && ev.newly_found_target_ids.length)
      dot(ev.capture_position_m, 7, "#008300", "#fcfcfb", 2.5);
  });
  if (showGT) {                                                 // Debug GT layer
    M.debug_gt.targets.forEach(g =>
      dot(g.position_m, 8, null, g.found ? "#0ca30c" : "#d03b3b", 3));
    M.debug_gt.distractors.forEach(d => dot(d.position_m, 6, null, "#eda100", 3));
  }
  const uav = posAtTime(t);                                     // the UAV
  dot(uav, 12, null, "rgba(252,252,251,0.9)", 3);
  dot(uav, 8, configHex(e.executed_config_id), "#0b0b0b", 2);

  document.getElementById("map-time").textContent =
    `${t.toFixed(1)} s / deadline ${M.contract.deadline_s.toFixed(0)} s`;
  document.getElementById("map-progress").textContent =
    `${(progressAtTime(t) * 100).toFixed(1)} %`;
  document.getElementById("map-config").textContent = e.executed_config_id;
}
document.getElementById("map-legend").innerHTML = [
  ["#2a78d6", "completed path / processed obs"], ["rgba(160,160,155,1)", "remaining path (white on map)"],
  ["#eb6834", "skipped capture (&times;)"], ["#008300", "detection event"],
  ["#0ca30c", "start"], ["#e87ba4", "destination"],
  ["#d03b3b", "Debug GT: missed target (toggle)"], ["#eda100", "Debug GT: distractor (toggle)"],
].map(([c, t]) => `<span><span class="sw" style="background:${c}"></span>${t}</span>`).join("");

// ---- perception panel -----------------------------------------------------------------
const predToggle = document.getElementById("toggle-pred");
const gtToggle = document.getElementById("toggle-gt");
function drawFrame() {
  const e = EVENTS[activeIdx];
  document.getElementById("frame-rgb").src = e.frames.rgb;
  const pred = document.getElementById("frame-pred");
  pred.src = e.frames.prediction; pred.hidden = !predToggle.checked;
  const gt = document.getElementById("frame-gt");
  gt.src = e.frames.gt_debug; gt.hidden = !gtToggle.checked;
  const missedHere = e.score.visible_target_ids.filter(t => !e.score.matched_target_ids.includes(t));
  const rows = [
    ["requested config", e.requested_config_id ?? "(policy failure)"],
    ["executed config", e.executed_config_id + (e.fallback_used ? "  (fallback)" : "")],
    ["inference latency", `${e.execution.mission_latency_s.toFixed(2)} s ` +
      `(${(e.execution.measurement_provenance.mission_latency_s || "simulated").split(" ")[0]})`],
    ["wall clock (diagnostic)", `${e.execution.measured_wall_clock_s.toFixed(4)} s`],
    ["frames skipped after", `${e.skipped_after.length}`],
    ["matched (TP)", `${e.score.matched_components}`],
    ["false positives", `${e.score.false_positive_components}`],
    ["matched target ids", e.score.matched_target_ids.join(", ") || "—"],
    ["visible-but-missed here", missedHere.join(", ") || "—"],
  ];
  // Remote transport diagnostics (V3 P1): system behaviour, never perception quality.
  const remote = (e.execution.diagnostics && e.execution.diagnostics.remote_status)
    ? e.execution.diagnostics : null;
  if (remote) {
    const net = remote.network;
    rows.push(
      ["remote status", remote.remote_status +
        (remote.failure_reason ? ` — ${remote.failure_reason}` : "")],
      ["network regime", `${net.regime_id} · ${net.uplink_mbps}/${net.downlink_mbps} Mbps · ` +
        `${net.rtt_ms} ms · loss ${net.packet_loss_frac}`],
      ["transfer", `up ${remote.uploaded_mb.toFixed(2)} MB · down ` +
        `${remote.downloaded_mb.toFixed(2)} MB · radio ` +
        `${(e.execution.communication_energy_j ?? 0).toFixed(1)} J`],
    );
    if (remote.fallback) rows.push(
      ["remote fallback", `${remote.fallback.fallback_config_id} ran on the same capture ` +
        `(+${remote.fallback.fallback_latency_s.toFixed(2)} s)`]);
  }
  kv("frame-meta", rows);
}

// ---- dashboard ------------------------------------------------------------------------
function pill(status) { return `<span class="pill st-${status}">${status}</span>`; }
function kv(id, rows) {
  document.getElementById(id).innerHTML =
    rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}
function fmt(v) { return v === null || v === undefined ? "n/a" : (+v).toFixed(3); }
function drawBanner() {
  const atEnd = activeIdx === N - 1;
  const status = atEnd ? (FINAL.mission_success ? "MISSION SUCCESS" : "MISSION FAILED")
                       : "MISSION IN PROGRESS";
  const el = document.getElementById("mission-status");
  el.textContent = status;
  el.style.color = atEnd ? (FINAL.mission_success ? "var(--good)" : "var(--critical)") : "var(--ink)";
  const overall = atEnd ? FINAL.constraint_status.overall
                        : EVENTS[activeIdx].constraint_status.overall.status;
  document.getElementById("overall-pill").outerHTML =
    `<span class="pill st-${overall}" id="overall-pill">constraints ${overall}</span>`;
  const cq = EVENTS[activeIdx].cumulative_quality;
  const iv = EVENTS[activeIdx].constraint_status.quality.interim_value;
  document.getElementById("evidence-line").textContent =
    `evidence: ${cq.unique_targets_found}/${cq.total_unique_targets} targets · ` +
    `${M.contract.quality_metric} ${fmt(iv)} (need ${M.contract.quality_operator} ${M.contract.quality_threshold})`;
  document.getElementById("termination-line").textContent =
    atEnd ? `terminated: ${FINAL.termination_reason}` : "";
}
function drawConstraints() {
  const cs = EVENTS[activeIdx].constraint_status, atEnd = activeIdx === N - 1;
  const st = name => atEnd ? FINAL.constraint_status[name] : cs[name].status;
  // Privacy is active since V3 P1: real status + declared level where a remote path
  // exists; NOT_APPLICABLE (older bundles / local-only scenarios) otherwise.
  const privacyValue = cs.privacy.privacy_level ?? "no remote path";
  const privacyMargin = cs.privacy.violations_so_far !== undefined
    ? `${cs.privacy.violations_so_far} violation(s)` : "—";
  const rows = [
    ["quality", st("quality"), fmt(cs.quality.interim_value),
      `${cs.quality.operator} ${cs.quality.threshold}`, "—"],
    ["deadline", st("deadline"), `${cs.deadline.elapsed_s.toFixed(1)} s`,
      `${cs.deadline.deadline_s.toFixed(0)} s`, `${cs.deadline.remaining_s.toFixed(1)} s left`],
    ["battery", st("battery"), `${(cs.battery.battery_frac * 100).toFixed(1)} %`,
      `&ge; ${(cs.battery.min_final_battery_frac * 100).toFixed(0)} % at end`,
      `${(cs.battery.margin_frac * 100).toFixed(1)} pt`],
    ["communication", st("communication"), `${cs.communication.used_mb.toFixed(2)} MB`,
      `${cs.communication.budget_mb.toFixed(1)} MB`, `${cs.communication.remaining_mb.toFixed(2)} MB left`],
    ["privacy", cs.privacy.status, privacyValue, "no violation attempts", privacyMargin],
  ];
  document.querySelector("#constraint-table tbody").innerHTML = rows.map(
    ([n, s, v, l, m]) => `<tr><td>${n}</td><td>${pill(s)}</td><td>${v}</td><td>${l}</td><td>${m}</td></tr>`
  ).join("");
}
function drawBehaviour() {
  const e = EVENTS[activeIdx];
  let switches = 0, fallbacks = 0, invalid = 0, skips = 0;
  for (let i = 0; i <= activeIdx; i++) {
    if (EVENTS[i].config_switched) switches++;
    if (EVENTS[i].fallback_used) fallbacks++;
    if (!EVENTS[i].action_valid) invalid++;
    skips += EVENTS[i].skipped_after.length;
  }
  const net = e.runtime.at_capture.network;
  kv("behaviour-meta", [
    ["requested &rarr; executed", `${e.requested_config_id ?? "(failure)"} &rarr; ${e.executed_config_id}`],
    ["config switches so far", `${switches}`],
    ["fallback / invalid actions", `${fallbacks} / ${invalid}`],
    ["frames skipped so far", `${skips}`],
    ["cumulative energy", `${e.runtime.after_completion.cumulative_energy_j.toFixed(1)} J (simulated)`],
    ["battery", `${(e.runtime.after_completion.battery_frac * 100).toFixed(1)} %`],
    ["communication", `${e.runtime.after_completion.cumulative_communication_mb.toFixed(2)} / ${M.contract.communication_budget_mb.toFixed(1)} MB`],
    ["network (policy-visible)", `${net.bandwidth_mbps} Mbps · ${net.rtt_ms} ms · loss ${net.packet_loss_frac}`],
    ["policy-believed targets", `${e.runtime.at_capture.evidence_summary.predicted_unique_targets}`],
  ]);
}
function drawPerception() {
  let matched = 0, preds = 0, fps = 0;
  for (let i = 0; i <= activeIdx; i++) {
    matched += EVENTS[i].score.matched_components;
    preds += EVENTS[i].score.predicted_components;
    fps += EVENTS[i].score.false_positive_components;
  }
  const cq = EVENTS[activeIdx].cumulative_quality, atEnd = activeIdx === N - 1;
  kv("perception-meta", [
    ["target recall (cumulative)", fmt(cq.target_recall)],
    ["detection precision (cumulative)", preds ? (matched / preds).toFixed(3) : "1.000"],
    ["matched detections (TP)", `${matched}`],
    ["false positives", `${fps}`],
    ["missed targets (final)", atEnd ? `${FINAL.quality.total_unique_targets - FINAL.quality.unique_targets_found}` : "known at mission end"],
    ["matching IoU threshold", `${M.matching_iou_threshold}`],
  ]);
}

// ---- timeline strip -------------------------------------------------------------------
const slider = document.getElementById("timeline-slider");
slider.max = N - 1;
const strip = document.getElementById("config-strip"), sctx = strip.getContext("2d");
function drawStrip(t) {
  strip.width = strip.clientWidth; strip.height = 26;
  const tx = time => time / END_T * strip.width;
  sctx.clearRect(0, 0, strip.width, strip.height);
  EVENTS.forEach(e => {
    sctx.fillStyle = configHex(e.executed_config_id);
    sctx.fillRect(tx(e.capture_time_s), 4,
                  Math.max(2, tx(e.completion_time_s) - tx(e.capture_time_s) - 1), 18);
  });
  R.skipped_observations.forEach(s => {
    sctx.fillStyle = "#898781";
    sctx.fillRect(tx(s.capture_time_s), 0, 1.5, 4);
  });
  sctx.fillStyle = "#0b0b0b"; sctx.fillRect(tx(t) - 1, 0, 2, strip.height);
}
document.getElementById("strip-legend").innerHTML =
  M.executor_configs.map((c, i) =>
    `<span><span class="sw" style="background:${CONFIG_HEX[i % 8]}"></span>${c.config_id} (${c.kind}, ${c.mission_latency_s}s)</span>`
  ).join("") + `<span><span class="sw" style="background:#898781"></span>skipped capture tick</span>`;

// ---- render + playback ----------------------------------------------------------------
function renderContinuous() { drawMap(replayTime); drawStrip(replayTime); }
function renderPanels() {
  slider.value = activeIdx;
  document.getElementById("obs-counter").textContent =
    `observation ${activeIdx + 1} / ${N} (id ${EVENTS[activeIdx].observation_id})`;
  drawBanner(); drawFrame(); drawConstraints(); drawBehaviour(); drawPerception();
}
function setIndex(i) {
  activeIdx = Math.min(N - 1, Math.max(0, i));
  replayTime = EVENTS[activeIdx].completion_time_s;
  renderPanels(); renderContinuous();
}
function speedFactor() { return parseFloat(document.getElementById("speed-select").value); }
function tick(ts) {
  if (playing) {
    if (lastFrameTs !== null)
      replayTime = Math.min(END_T, replayTime + (ts - lastFrameTs) / 1000 * speedFactor());
    lastFrameTs = ts;
    const idx = indexForTime(replayTime);
    if (idx !== activeIdx) { activeIdx = idx; renderPanels(); }
    renderContinuous();
    if (replayTime >= END_T) { activeIdx = N - 1; renderPanels(); setPlaying(false); }
  }
  requestAnimationFrame(tick);
}
function setPlaying(on) {
  if (on && replayTime >= END_T) { activeIdx = 0; replayTime = 0; renderPanels(); }
  playing = on; lastFrameTs = null;
  document.getElementById("btn-play").textContent = on ? "Pause" : "Play";
}
document.getElementById("btn-prev").onclick = () => { setPlaying(false); setIndex(activeIdx - 1); };
document.getElementById("btn-next").onclick = () => { setPlaying(false); setIndex(activeIdx + 1); };
document.getElementById("btn-play").onclick = () => setPlaying(!playing);
slider.oninput = () => { setPlaying(false); setIndex(parseInt(slider.value, 10)); };
predToggle.onchange = drawFrame;
gtToggle.onchange = () => { drawFrame(); renderContinuous(); };
window.addEventListener("resize", () => { fitCanvas(); renderPanels(); renderContinuous(); });
setIndex(0);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""

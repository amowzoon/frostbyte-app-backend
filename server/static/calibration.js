'use strict';

// ── STATE ─────────────────────────────────────────────────────────────────────
const state = {
  sessionId:    null,
  irS3Path:     null,
  rgbS3Path:    null,
  irImgNatural: { w: 0, h: 0 },   // natural (native) pixel dimensions
  rgbImgNatural:{ w: 0, h: 0 },
  pairs:        [],                 // [{ir:{x,y}, rgb:{x,y}}]
  pendingIr:    null,               // {x, y} in canvas-display coords (normalised below)
  awaitingRgb:  false,
  homographyMatrix: null,
};

// Pair colour palette
const PAIR_COLORS = [
  '#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#ff9f43',
  '#a29bfe','#fd79a8','#00cec9','#e17055','#55efc4',
];
function pairColor(i) { return PAIR_COLORS[i % PAIR_COLORS.length]; }

// ── DOM REFS ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const irCanvas    = $('ir-canvas');
const rgbCanvas   = $('rgb-canvas');
const irCtx       = irCanvas.getContext('2d');
const rgbCtx      = rgbCanvas.getContext('2d');
const irImg       = new Image();
const rgbImg      = new Image();

// ── CANVAS SETUP ──────────────────────────────────────────────────────────────
function resizeCanvas(canvas, container) {
  canvas.width  = container.clientWidth  || 600;
  canvas.height = container.clientHeight || 400;
}

const irContainer  = $('ir-container');
const rgbContainer = $('rgb-container');

const ro = new ResizeObserver(() => {
  resizeCanvas(irCanvas,  irContainer);
  resizeCanvas(rgbCanvas, rgbContainer);
  redrawAll();
});
ro.observe(irContainer);
ro.observe(rgbContainer);

// ── IMAGE DRAWING ─────────────────────────────────────────────────────────────
function drawImageContained(ctx, img) {
  if (!img.complete || !img.naturalWidth) return null;
  const cw = ctx.canvas.width, ch = ctx.canvas.height;
  const iw = img.naturalWidth,  ih = img.naturalHeight;
  const scale = Math.min(cw / iw, ch / ih);
  const dw = iw * scale, dh = ih * scale;
  const dx = (cw - dw) / 2, dy = (ch - dh) / 2;
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, dx, dy, dw, dh);
  return { dx, dy, dw, dh, scale };
}

// Convert display canvas coords → image natural coords
function canvasToNatural(cx, cy, layout) {
  return {
    x: (cx - layout.dx) / layout.scale,
    y: (cy - layout.dy) / layout.scale,
  };
}

// Convert image natural coords → display canvas coords
function naturalToCanvas(nx, ny, layout) {
  return {
    x: nx * layout.scale + layout.dx,
    y: ny * layout.scale + layout.dy,
  };
}

let irLayout  = null;
let rgbLayout = null;

// ── DRAW OVERLAYS ─────────────────────────────────────────────────────────────
function drawOverlay(ctx, img, layout, points, pendingPoint, activeColor) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  const l = drawImageContained(ctx, img);
  if (!l) return null;

  // Draw each confirmed pair point
  points.forEach((pt, i) => {
    const c = pairColor(i);
    const cv = naturalToCanvas(pt.x, pt.y, l);
    drawMarker(ctx, cv.x, cv.y, i + 1, c);
  });

  // Draw pending point (awaiting RGB click)
  if (pendingPoint) {
    const cv = naturalToCanvas(pendingPoint.x, pendingPoint.y, l);
    drawPendingMarker(ctx, cv.x, cv.y, activeColor);
  }

  return l;
}

function drawMarker(ctx, x, y, label, color) {
  // Crosshair lines
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1;
  ctx.globalAlpha = 0.85;
  ctx.beginPath(); ctx.moveTo(x - 10, y); ctx.lineTo(x + 10, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y - 10); ctx.lineTo(x, y + 10); ctx.stroke();

  // Circle
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 6, 0, Math.PI * 2);
  ctx.stroke();

  // Filled dot
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, 2.5, 0, Math.PI * 2);
  ctx.fill();

  // Label
  ctx.fillStyle   = color;
  ctx.font        = 'bold 10px JetBrains Mono, monospace';
  ctx.textBaseline= 'bottom';
  ctx.fillText(label, x + 8, y - 4);
  ctx.globalAlpha = 1;
}

function drawPendingMarker(ctx, x, y, color) {
  ctx.strokeStyle = color;
  ctx.lineWidth   = 1;
  ctx.setLineDash([3, 3]);
  ctx.globalAlpha = 0.9;
  ctx.beginPath(); ctx.moveTo(x - 14, y); ctx.lineTo(x + 14, y); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x, y - 14); ctx.lineTo(x, y + 14); ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;

  ctx.strokeStyle = color;
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = color;
  ctx.globalAlpha = 0.3;
  ctx.beginPath();
  ctx.arc(x, y, 8, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;

  ctx.fillStyle   = color;
  ctx.font        = 'bold 10px JetBrains Mono, monospace';
  ctx.textBaseline= 'bottom';
  ctx.fillText('?', x + 10, y - 5);
}

function redrawAll() {
  const nextPairColor = pairColor(state.pairs.length);

  // IR pairs stored in native 160x120 — convert to JPEG space for drawing
  irLayout = drawOverlay(
    irCtx, irImg, irLayout,
    state.pairs.map(p => irNativeToJpeg(p.ir.x, p.ir.y)),
    state.pendingIr ? irNativeToJpeg(state.pendingIr.x, state.pendingIr.y) : null,
    nextPairColor
  );

  // RGB pairs already in native 4608x2592 space
  rgbLayout = drawOverlay(
    rgbCtx, rgbImg, rgbLayout,
    state.pairs.map(p => p.rgb),
    null,
    nextPairColor
  );
}

// ── SESSION LOADING ───────────────────────────────────────────────────────────
async function fetchRecentSessions() {
  const dd = $('sessions-dropdown');
  dd.style.display = 'flex';
  $('sessions-list').innerHTML = '<div style="padding:12px;color:var(--text-dim);font-size:11px;">Loading…</div>';

  try {
    const res  = await fetch('/api/sessions?limit=20');
    const data = await res.json();
    renderSessionsList(data.sessions || []);
  } catch (e) {
    $('sessions-list').innerHTML = `<div style="padding:12px;color:var(--red);font-size:11px;">Error: ${e.message}</div>`;
  }
}

function renderSessionsList(sessions) {
  const list = $('sessions-list');
  if (!sessions.length) {
    list.innerHTML = '<div style="padding:12px;color:var(--text-dim);font-size:11px;">No sessions found.</div>';
    return;
  }
  list.innerHTML = sessions.map(s => {
    const sensors = (s.captured_sensors || []).map(st =>
      `<span class="sensor-chip ${st}">${st}</span>`
    ).join('');
    const time = s.captured_at
      ? new Date(s.captured_at).toLocaleString('en-US', {
          timeZone: 'America/New_York', month:'2-digit', day:'2-digit',
          hour:'2-digit', minute:'2-digit', hour12:true
        })
      : '—';
    return `
      <div class="session-row" onclick="selectSession('${s.session_id}')">
        <span class="session-row-id">${s.session_id.substring(0,8)}… <span style="color:var(--text-dim)">${s.capture_id || ''}</span></span>
        <div class="session-row-sensors">${sensors}</div>
        <span class="session-row-time">${time}</span>
      </div>`;
  }).join('');
}

function selectSession(sessionId) {
  $('session-id-input').value = sessionId;
  closeSessionsDropdown();
  loadSession();
}

function closeSessionsDropdown() {
  $('sessions-dropdown').style.display = 'none';
}

async function loadSession() {
  const sessionId = $('session-id-input').value.trim();
  if (!sessionId) return;

  setInstruction('Loading session…');

  try {
    const res  = await fetch(`/api/sessions/${sessionId}`);
    if (!res.ok) throw new Error(`Session not found (${res.status})`);
    const data = await res.json();

    state.sessionId = sessionId;

    // Find IR and RGB sensor paths
    const sensors = data.sensors || [];
    const ir  = sensors.find(s => s.sensor_type === 'ir');
    const rgb = sensors.find(s => s.sensor_type === 'rgb');

    if (!ir)  throw new Error('No IR data in this session');
    if (!rgb) throw new Error('No RGB data in this session');

    state.irS3Path  = ir.s3_path;
    state.rgbS3Path = rgb.s3_path;

    // Hide placeholders
    $('ir-placeholder').style.display  = 'none';
    $('rgb-placeholder').style.display = 'none';

    // Load images
    await Promise.all([
      loadImage(irImg,  `/api/images/${state.irS3Path}`,  'ir-mode-badge'),
      loadImage(rgbImg, `/api/images/${state.rgbS3Path}`, 'rgb-mode-badge'),
    ]);

    state.irImgNatural  = { w: irImg.naturalWidth,  h: irImg.naturalHeight  };
    state.rgbImgNatural = { w: rgbImg.naturalWidth, h: rgbImg.naturalHeight };

    redrawAll();
    updatePairsList();
    updateInstruction();
    setCalibStatus('session loaded', 'warn');

    // Check if existing calibration exists for this session
    checkExistingCalibration();

  } catch (e) {
    setInstruction(`Error: ${e.message}`);
    setCalibStatus('load failed', 'err');
  }
}

function loadImage(imgEl, src, badgeId) {
  return new Promise((resolve, reject) => {
    imgEl.onload  = () => { setBadge(badgeId, 'loaded', 'done'); resolve(); };
    imgEl.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    imgEl.src     = src;
  });
}

async function checkExistingCalibration() {
  try {
    const res  = await fetch('/api/calibration/homography');
    if (res.ok) {
      const data = await res.json();
      if (data.exists) {
        setCalibStatus(`calibration on file (${data.pair_count} pts)`, 'ok');
      }
    }
  } catch (_) {}
}

// ── COORDINATE HELPERS ────────────────────────────────────────────────────────
// The IR image served from S3 is the JPEG visualization (e.g. 640×480),
// but the homography must be computed in native sensor space (160×120).
// All pairs are stored in native space; we rescale at draw time.
const IR_NATIVE_W = 160;
const IR_NATIVE_H = 120;

function irJpegToNative(x, y) {
  return {
    x: x * IR_NATIVE_W / state.irImgNatural.w,
    y: y * IR_NATIVE_H / state.irImgNatural.h,
  };
}

function irNativeToJpeg(x, y) {
  return {
    x: x * state.irImgNatural.w / IR_NATIVE_W,
    y: y * state.irImgNatural.h / IR_NATIVE_H,
  };
}

// ── CANVAS CLICK HANDLERS ─────────────────────────────────────────────────────
irCanvas.addEventListener('click', e => {
  if (!irImg.complete || !irImg.naturalWidth) return;
  if (state.awaitingRgb) {
    // User clicked IR again — replace pending point
    state.pendingIr = null;
    state.awaitingRgb = false;
  }

  const rect = irCanvas.getBoundingClientRect();
  const cx   = (e.clientX - rect.left) * (irCanvas.width  / rect.width);
  const cy   = (e.clientY - rect.top)  * (irCanvas.height / rect.height);

  if (!irLayout) return;
  const jpeg = canvasToNatural(cx, cy, irLayout);  // coords in JPEG image space

  // Clamp to JPEG image bounds
  if (jpeg.x < 0 || jpeg.y < 0 || jpeg.x > state.irImgNatural.w || jpeg.y > state.irImgNatural.h) return;

  // Store in native sensor space (160×120) — this is what the mapper uses
  state.pendingIr   = irJpegToNative(jpeg.x, jpeg.y);
  state.awaitingRgb = true;
  updateInstruction();
  redrawAll();
  updateActivePairIndicator();
});

rgbCanvas.addEventListener('click', e => {
  if (!state.awaitingRgb || !state.pendingIr) return;
  if (!rgbImg.complete || !rgbImg.naturalWidth) return;

  const rect = rgbCanvas.getBoundingClientRect();
  const cx   = (e.clientX - rect.left) * (rgbCanvas.width  / rect.width);
  const cy   = (e.clientY - rect.top)  * (rgbCanvas.height / rect.height);

  if (!rgbLayout) return;
  const nat = canvasToNatural(cx, cy, rgbLayout);  // RGB natural = native (4608×2592)

  if (nat.x < 0 || nat.y < 0 || nat.x > state.rgbImgNatural.w || nat.y > state.rgbImgNatural.h) return;

  // Commit: IR stored in native 160×120, RGB stored in native 4608×2592
  state.pairs.push({
    ir:  { x: state.pendingIr.x, y: state.pendingIr.y },
    rgb: { x: nat.x,             y: nat.y             },
  });

  state.pendingIr   = null;
  state.awaitingRgb = false;

  updatePairsList();
  updateInstruction();
  updateActivePairIndicator();
  redrawAll();
  updateComputeButton();
});

// ── PAIRS LIST UI ─────────────────────────────────────────────────────────────
function updatePairsList() {
  const list = $('pairs-list');
  $('pair-count').textContent = state.pairs.length;

  if (!state.pairs.length) {
    list.innerHTML = '<div class="pairs-empty">No points yet.<br>Click IR image first,<br>then RGB image.</div>';
    return;
  }

  list.innerHTML = state.pairs.map((p, i) => {
    const c = pairColor(i);
    return `
      <div class="pair-row">
        <div class="pair-dot" style="background:${c}"></div>
        <span class="pair-index">${i+1}</span>
        <div class="pair-coords">
          <span>IR  ${Math.round(p.ir.x)},${Math.round(p.ir.y)}</span>
          <span>RGB ${Math.round(p.rgb.x)},${Math.round(p.rgb.y)}</span>
        </div>
        <button class="pair-delete" onclick="deletePair(${i})" title="Remove">✕</button>
      </div>`;
  }).join('');
}

function deletePair(i) {
  state.pairs.splice(i, 1);
  state.homographyMatrix = null;
  $('preview-section').style.display = 'none';
  $('save-section').style.display    = 'none';
  updatePairsList();
  updateComputeButton();
  updateInstruction();
  redrawAll();
}

function undoLastPair() {
  if (state.awaitingRgb) {
    // Cancel pending IR point
    state.pendingIr   = null;
    state.awaitingRgb = false;
    updateInstruction();
    redrawAll();
    return;
  }
  if (state.pairs.length) {
    state.pairs.pop();
    state.homographyMatrix = null;
    $('preview-section').style.display = 'none';
    $('save-section').style.display    = 'none';
    updatePairsList();
    updateComputeButton();
    updateInstruction();
    redrawAll();
  }
}

function clearAllPairs() {
  if (!state.pairs.length && !state.pendingIr) return;
  state.pairs           = [];
  state.pendingIr       = null;
  state.awaitingRgb     = false;
  state.homographyMatrix= null;
  $('preview-section').style.display = 'none';
  $('save-section').style.display    = 'none';
  updatePairsList();
  updateComputeButton();
  updateInstruction();
  updateActivePairIndicator();
  redrawAll();
}

// ── COMPUTE HOMOGRAPHY ────────────────────────────────────────────────────────
async function computeHomography() {
  if (state.pairs.length < 4) return;

  const btn = $('btn-compute');
  btn.disabled = true;
  btn.textContent = 'Computing…';

  try {
    const payload = {
      session_id: state.sessionId,
      pairs: state.pairs.map(p => ({
        ir_x:  p.ir.x,  ir_y:  p.ir.y,
        rgb_x: p.rgb.x, rgb_y: p.rgb.y,
      })),
      ir_natural_width:   state.irImgNatural.w,
      ir_natural_height:  state.irImgNatural.h,
      rgb_natural_width:  state.rgbImgNatural.w,
      rgb_natural_height: state.rgbImgNatural.h,
    };

    const res  = await fetch('/api/calibration/compute', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Compute failed');
    }

    const data = await res.json();
    state.homographyMatrix = data.homography;

    // Show reprojection error
    const errEl = $('reproj-error');
    const reprojErr = data.reprojection_error ?? null;
    if (reprojErr !== null) {
      errEl.textContent = `Reprojection error: ${reprojErr.toFixed(2)} px`;
      errEl.className   = 'error-badge ' + (reprojErr < 5 ? '' : reprojErr < 15 ? 'warn' : 'err');
    }

    // Show preview
    if (data.preview_url) {
      $('preview-img').src = data.preview_url + '?t=' + Date.now();
    }
    $('preview-section').style.display = 'block';
    $('save-section').style.display    = 'block';
    $('save-status').textContent       = '';

    setCalibStatus('homography computed', 'warn');

  } catch (e) {
    alert(`Compute failed: ${e.message}`);
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Compute Homography';
    updateComputeButton();
  }
}

// ── SAVE CALIBRATION ──────────────────────────────────────────────────────────
async function saveCalibration() {
  if (!state.homographyMatrix) return;

  const btn = $('btn-save');
  btn.disabled    = true;
  btn.textContent = 'Saving…';

  try {
    const res = await fetch('/api/calibration/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        homography:  state.homographyMatrix,
        pair_count:  state.pairs.length,
        session_id:  state.sessionId,
        pairs: state.pairs.map(p => ({
          ir_x:  p.ir.x,  ir_y:  p.ir.y,
          rgb_x: p.rgb.x, rgb_y: p.rgb.y,
        })),
      }),
    });

    if (!res.ok) throw new Error('Save failed');

    $('save-status').textContent = '✓ Saved — mapping will use this calibration';
    $('save-status').style.color = 'var(--green)';
    setCalibStatus(`calibration saved (${state.pairs.length} pts)`, 'ok');

  } catch (e) {
    $('save-status').textContent = `✗ ${e.message}`;
    $('save-status').style.color = 'var(--red)';
  } finally {
    btn.disabled    = false;
    btn.textContent = 'Save Calibration';
  }
}

// ── UI HELPERS ────────────────────────────────────────────────────────────────
function updateComputeButton() {
  const btn  = $('btn-compute');
  const n    = state.pairs.length;
  btn.disabled = n < 4;
  btn.textContent = n >= 4
    ? `Compute Homography (${n} pts)`
    : `Compute Homography (need ${4 - n} more)`;
}

function updateInstruction() {
  const el = $('instruction-text');
  if (!state.sessionId) {
    el.textContent = 'Load a session to begin';
    el.style.color = 'var(--text-dim)';
    return;
  }
  if (state.awaitingRgb) {
    el.textContent = `Pair ${state.pairs.length + 1}: now click the matching point on RGB →`;
    el.style.color = 'var(--yellow)';
  } else {
    el.textContent = `Pair ${state.pairs.length + 1}: click a point on IR ←`;
    el.style.color = 'var(--green)';
  }
}

function updateActivePairIndicator() {
  const el = $('active-pair-indicator');
  if (state.awaitingRgb) {
    el.textContent = `#${state.pairs.length + 1}`;
    el.style.color = pairColor(state.pairs.length);
  } else {
    el.textContent = state.pairs.length ? `${state.pairs.length} done` : '—';
    el.style.color = 'var(--accent)';
  }
}

function setInstruction(msg) {
  $('instruction-text').textContent = msg;
}

function setCalibStatus(msg, cls) {
  const el = $('calib-status');
  el.textContent = msg;
  el.className   = `status-pill ${cls || ''}`;
}

function setBadge(id, text, cls) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.className   = `panel-mode-badge ${cls || ''}`;
}

// ── INIT ──────────────────────────────────────────────────────────────────────
resizeCanvas(irCanvas,  irContainer);
resizeCanvas(rgbCanvas, rgbContainer);
updateComputeButton();
updateInstruction();

// If URL has ?session_id=... pre-load it
const urlParams = new URLSearchParams(window.location.search);
const preloadSession = urlParams.get('session_id');
if (preloadSession) {
  $('session-id-input').value = preloadSession;
  loadSession();
}

// ═══════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════
const STAGES = ['command_sent','sensor_capture_complete','upload_received','s3_stored','db_stored'];
let currentCaptureId    = null;
let currentSessionId    = null;
let currentSession      = null;
let ws                  = null;
let awaitingCaptureId   = false;
let pendingCheckpoints  = [];
let userSelectedSession = false; // true only when user picks from the import modal
let captureInProgress   = false; // true while a capture is underway
let pendingSensor       = 'all'; // which sensor(s) are being captured

// Processing state
let procResults = {
  roadMask: null,
  irMapped: null,
  radarMapped: null,
  segments: null,
  featuresDF: null,
  confidenceMap: null,
  overlayRGB: null,
  heatmap: null,
  regions: null,
  numRegions: 0,
  regionFeatures: {}
};
let currentView        = 'rgb';
let showSegmentOverlay = false;
let showHeatmapOverlay = false;
let heatmapThreshold   = 0;
let currentWeights     = getDefaultWeights();

// ═══════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════
// ================================
// CAPTURE OPTIONS DICT
// ================================
let captureOptions = {
  focus:         null,
  exposure_time: null,
  gain:          null,
  awb_mode:      null,
  brightness:    null,
  contrast:      null,
  colormap:      null,
  save_raw:      true,
  ir_timeout:    2.0,
  radar_timeout: 2.0,
};

window.addEventListener('DOMContentLoaded', async () => {
  initSensorOpts();
  LBL.init();
  connectWS();
  await loadDevices();
  await loadLastCapture();
  const saved = localStorage.getItem('frostbyte_captureOptions');
  if (saved) {
    try {
      captureOptions = { ...captureOptions, ...JSON.parse(saved) };
      // sync sliders to loaded values here
    } catch(e) {}
  }
  updateCaptureOpts();
  fetchCameraStatus();
});

// ═══════════════════════════════════════════════════════════
// EXPOSURE HELPERS (log-scale slider → µs, photography display)
// ═══════════════════════════════════════════════════════════
// Slider 0..1000 → 100µs .. 20,000,000µs (20s)
// Exponential: us = 100 * 200000^(v/1000)
function expSliderToUs(v) {
  return Math.round(100 * Math.pow(200000, v / 1000));
}

function formatShutter(us) {
  const sec = us / 1e6;
  if (sec >= 1) return sec % 1 === 0 ? sec + '"' : sec.toFixed(1) + '"';
  // Photography fraction: 1/X
  const denom = Math.round(1 / sec);
  return '1/' + denom;
}

function updateCaptureOpts() {
  // RGB focus
  const focusEn = document.getElementById('opt-rgb-focus-en');
  captureOptions.focus = focusEn && focusEn.checked
    ? parseFloat(document.getElementById('opt-rgb-focus-sl').value) / 10
    : null;

  // RGB exposure (log-scale slider)
  const expEn = document.getElementById('opt-rgb-exp-en');
  if (expEn && expEn.checked) {
    captureOptions.exposure_time = expSliderToUs(parseInt(document.getElementById('opt-rgb-exp-sl').value));
    captureOptions.gain          = parseFloat(document.getElementById('opt-rgb-gain-sl').value) / 10;
  } else {
    captureOptions.exposure_time = null;
    captureOptions.gain          = null;
  }

  // RGB AWB
  const awb = document.getElementById('opt-rgb-awb');
  captureOptions.awb_mode = awb && awb.value ? awb.value : null;

  // RGB brightness / contrast
  const bright = parseInt(document.getElementById('opt-rgb-bright-sl').value);
  captureOptions.brightness = bright !== 0 ? bright / 100 : null;
  const contrast = parseInt(document.getElementById('opt-rgb-contrast-sl').value);
  captureOptions.contrast = contrast !== 100 ? contrast / 100 : null;

  // IR
  const cm = document.getElementById('opt-ir-colormap');
  captureOptions.colormap  = cm && cm.value ? cm.value : null;
  const rawEl = document.getElementById('opt-ir-raw');
  captureOptions.save_raw  = rawEl ? rawEl.checked : true;
  captureOptions.ir_timeout = parseFloat(document.getElementById('opt-ir-timeout-sl').value) / 10;

  // Radar
  captureOptions.radar_timeout = parseFloat(document.getElementById('opt-radar-timeout-sl').value) / 10;
  
  // persistent local storage
  localStorage.setItem('frostbyte_captureOptions', JSON.stringify(captureOptions));
}

// ═══════════════════════════════════════════════════════════
// CAMERA CONTROL HELPERS
// ═══════════════════════════════════════════════════════════
function onExpToggle(checked) {
  document.getElementById('opt-rgb-exp-wrap').style.display = checked ? 'block' : 'none';
  updateCaptureOpts();
  // When switching back to auto, send a lightweight set_controls call
  // to clear sticky manual exposure state — no full reinit needed.
  if (!checked) {
    const devId = document.getElementById('device-select').value;
    if (devId) {
      fetch(`/api/devices/${devId}/api/camera/auto_exposure`, { method: 'POST' })
        .then(r => r.json())
        .then(d => console.log('Auto-exposure restored:', d))
        .catch(e => console.warn('Auto-exposure restore failed:', e));
    }
  }
}

// ═══════════════════════════════════════════════════════════
// CAMERA STATUS + ORIENTATION
// ═══════════════════════════════════════════════════════════
async function fetchCameraStatus() {
  const devId = document.getElementById('device-select').value;
  if (!devId) return;
  try {
    const r = await fetch(`/api/devices/${devId}/api/camera/status`);
    if (!r.ok) return;
    const d = await r.json();
    const s = d.settings || {};

    // Populate rotation dropdown
    const rotEl = document.getElementById('opt-rgb-rotation');
    if (rotEl) rotEl.value = String(s.rotation || 0);

    // Populate mirror toggles
    const hEl = document.getElementById('opt-rgb-mirror-h');
    const vEl = document.getElementById('opt-rgb-mirror-v');
    if (hEl) hEl.checked = !!s.hflip;
    if (vEl) vEl.checked = !!s.vflip;

    updateOrientStatus(s);
  } catch(e) {
    console.warn('Failed to fetch camera status:', e);
  }
}

function updateOrientStatus(settings) {
  const el = document.getElementById('orient-status');
  if (!el) return;
  const rot = settings.rotation || 0;
  const h = !!settings.hflip;
  const v = !!settings.vflip;

  let pills = '';
  pills += `<span class="orient-pill${rot !== 0 ? ' active' : ''}">${rot}°</span>`;
  pills += `<span class="orient-pill${h ? ' active' : ''}">H</span>`;
  pills += `<span class="orient-pill${v ? ' active' : ''}">V</span>`;
  el.innerHTML = pills;
}

async function applyRotation() {
  const devId = document.getElementById('device-select').value;
  const angle = parseInt(document.getElementById('opt-rgb-rotation').value);
  if (!devId) { alert('Select a device first.'); return; }
  try {
    const r = await fetch(`/api/devices/${devId}/api/camera/rotation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ angle })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || d.error || r.statusText);
    console.log('Rotation applied:', d);
    await fetchCameraStatus();
  } catch(e) {
    showCaptureError('rgb', 'Rotation: ' + e.message);
  }
}

async function applyMirror() {
  const devId = document.getElementById('device-select').value;
  const horizontal = document.getElementById('opt-rgb-mirror-h').checked;
  const vertical   = document.getElementById('opt-rgb-mirror-v').checked;
  if (!devId) { alert('Select a device first.'); return; }
  try {
    const r = await fetch(`/api/devices/${devId}/api/camera/mirror`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ horizontal, vertical })
    });
    const d = await r.json();
    if (!r.ok) throw new Error(d.detail || d.error || r.statusText);
    console.log('Mirror applied:', d);
    await fetchCameraStatus();
  } catch(e) {
    showCaptureError('rgb', 'Mirror: ' + e.message);
  }
}

// ═══════════════════════════════════════════════════════════
// CAPTURE INFO BAR RESIZER
// ═══════════════════════════════════════════════════════════
(function initCaptureResizer() {
  document.addEventListener('DOMContentLoaded', () => {
    const resizer = document.getElementById('capture-resizer');
    const infoBar = document.getElementById('sensor-info-bar');
    const panel   = document.getElementById('panel-capture');
    if (!resizer || !infoBar || !panel) return;

    let dragging = false;

    resizer.addEventListener('mousedown', e => {
      dragging = true;
      resizer.classList.add('dragging');
      document.body.style.cursor     = 'ns-resize';
      document.body.style.userSelect = 'none';
      e.preventDefault();
    });

    document.addEventListener('mousemove', e => {
      if (!dragging) return;
      const panelRect = panel.getBoundingClientRect();
      // Distance from bottom of the capture tab panel to the mouse
      const fromBottom = panelRect.bottom - e.clientY;
      const clamped = Math.min(Math.max(fromBottom, 80), panelRect.height * 0.65);
      infoBar.style.height = clamped + 'px';
    });

    document.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor     = '';
      document.body.style.userSelect = '';
    });
  });
})();

// ═══════════════════════════════════════════════════════════
// SIB TAB SWITCHER
// ═══════════════════════════════════════════════════════════
function switchSibTab(sensor, panel, btn) {
  // Deactivate all tabs and panels in this cell
  const cell = document.getElementById('info-cell-' + sensor);
  cell.querySelectorAll('.sib-tab').forEach(t => t.classList.remove('active'));
  cell.querySelectorAll('.sib-panel').forEach(p => p.classList.remove('active'));

  // Activate selected
  btn.classList.add('active');
  document.getElementById('sib-' + sensor + '-' + panel).classList.add('active');
}

// ═══════════════════════════════════════════════════════════
// TABS
// ═══════════════════════════════════════════════════════════
function switchTab(tab) {
  ['capture','processing','labeling'].forEach(t => {
    document.getElementById(`panel-${t}`).classList.toggle('active', t === tab);
    document.getElementById(`tab-${t}`).classList.toggle('active', t === tab);
  });
  if (tab === 'processing') refreshProcCanvas();
  if (tab === 'labeling')   onLabelingTabActivated();
}

// ═══════════════════════════════════════════════════════════
// SENSOR OPTION CLICKS
// ═══════════════════════════════════════════════════════════
function initSensorOpts() {
  document.querySelectorAll('.sensor-opt').forEach(el => {
    el.addEventListener('click', () => {
      document.querySelectorAll('.sensor-opt').forEach(o => o.classList.remove('selected'));
      el.classList.add('selected');
    });
  });
}

function selectedSensor() {
  return document.querySelector('input[name=sensor]:checked')?.value || 'all';
}

// ═══════════════════════════════════════════════════════════
// WEBSOCKET
// ═══════════════════════════════════════════════════════════
function connectWS() {
  ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/dashboard`);
  ws.onopen    = () => {
    setWsDot(true);
    // Subscribe to the currently selected device
    const devId = document.getElementById('device-select').value;
    if (devId) wsSubscribe(devId);
  };
  ws.onclose   = () => { setWsDot(false); setTimeout(connectWS, 3000); };
  ws.onerror   = () => setWsDot(false);
  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type !== 'checkpoint') return;
    if (awaitingCaptureId) {
      pendingCheckpoints.push(msg);
    } else if (captureInProgress) {
      // During active capture, accept any checkpoint from our device
      // (device scoping in step 3 ensures we only get our Pi's events)
      handleCheckpoint(msg);
    } else if (msg.capture_id === currentCaptureId) {
      handleCheckpoint(msg);
    }
  };
}

let _subscribedDevice = null;

function wsSubscribe(deviceId) {
  if (_subscribedDevice === deviceId) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    if (_subscribedDevice) {
      ws.send(JSON.stringify({ action: 'unsubscribe', device_id: _subscribedDevice }));
    }
    ws.send(JSON.stringify({ action: 'subscribe', device_id: deviceId }));
    _subscribedDevice = deviceId;
  }
}

function setWsDot(on) {
  document.getElementById('ws-dot').className = 'dot ' + (on ? 'on' : 'off');
  document.getElementById('ws-lbl').textContent = on ? 'live' : 'disconnected';
}

// ═══════════════════════════════════════════════════════════
// PIPELINE STATUS
// ═══════════════════════════════════════════════════════════
function resetPipeline() {
  STAGES.forEach(s => setStage(s, 'pending', ''));
}

function setStage(name, status, time) {
  const dot = document.getElementById(`pd-${name}`);
  const lbl = document.getElementById(`pl-${name}`);
  const t   = document.getElementById(`pt-${name}`);
  if (!dot) return;
  dot.className = `p-dot ${status === 'pending' ? '' : status}`;
  lbl.className = `p-lbl ${status === 'pending' ? '' : status}`;
  if (t) t.textContent = time;
}

function handleCheckpoint(msg) {
  const ok     = !(msg.data && msg.data.error);
  const status = ok ? 'success' : 'failed';
  const time   = msg.timestamp
    ? new Date(msg.timestamp).toLocaleTimeString('en-US', {hour12: false})
    : '';
  setStage(msg.stage, status, time);

  // Show errors visibly on the capture page
  if (!ok && msg.data?.error) {
    showCaptureError(msg.data.sensor || msg.stage, msg.data.error);
  }

  if (msg.stage === 'db_stored' && msg.data?.session_id) {
    // Each sensor upload triggers a db_stored event — reload to show it
    if (captureInProgress || !userSelectedSession) {
      currentSessionId = msg.data.session_id;
      loadSession(msg.data.session_id);
    }
    if (msg.data.is_complete) {
      // All expected sensors arrived — capture is done
      captureInProgress   = false;
      pendingSensor       = 'all';
      userSelectedSession = false;
      clearTimeout(_captureTimeout);
      document.getElementById('capture-btn').disabled = false;
      document.getElementById('run-proc-btn').disabled = false;
      document.getElementById('proc-tab-dot').style.background = 'var(--green)';
    }
  }
  if (msg.stage === 'sensor_capture_complete' && !ok) {
    // Sensor failed on the Pi — re-enable capture button
    captureInProgress = false;
    pendingSensor     = 'all';
    clearTimeout(_captureTimeout);
    document.getElementById('capture-btn').disabled = false;
  }
}

// Safety timeout handle — re-enables capture button if WS events are lost
let _captureTimeout = null;

// ═══════════════════════════════════════════════════════════
// DEVICES
// ═══════════════════════════════════════════════════════════
async function loadDevices() {
  try {
    const r = await fetch('/api/devices');
    const d = await r.json();
    const sel = document.getElementById('device-select');
    sel.innerHTML = '';
    if (!d.devices.length) {
      sel.innerHTML = '<option value="">No devices registered</option>';
      return;
    }
    d.devices.forEach(dev => {
      const o = document.createElement('option');
      o.value = dev.id;
      o.textContent = `${dev.id}  ${dev.connected ? '●' : '○'}`;
      if (dev.connected) o.selected = true;
      sel.appendChild(o);
    });
    onDeviceChange();
  } catch(e) { console.error(e); }
}

function onDeviceChange() {
  const v = document.getElementById('device-select').value;
  document.getElementById('dev-lbl').textContent = v || '—';
  if (v) {
    wsSubscribe(v);
    fetchCameraStatus();
  }
}

// ═══════════════════════════════════════════════════════════
// LOAD / DISPLAY SESSION
// ═══════════════════════════════════════════════════════════
async function loadLastCapture() {
  try {
    const devId = document.getElementById('device-select').value;
    const url   = `/api/sessions?limit=1${devId ? '&device_id=' + devId : ''}`;
    const r     = await fetch(url);
    const d     = await r.json();
    if (d.sessions.length) {
      await loadSession(d.sessions[0].session_id, false);
    }
  } catch(e) { console.error(e); }
}

async function loadSession(sessionId, fromUserPick = false) {
  try {
    fetch('/api/processing/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId })
    }).catch(e => console.warn('Cache reset failed (non-fatal):', e));

    const r = await fetch(`/api/sessions/${sessionId}`);
    const s = await r.json();

    currentSessionId = sessionId;
    currentSession   = s;
    if (fromUserPick) userSelectedSession = true;

    procResults = {
      roadMask: null, irMapped: null, radarMapped: null,
      segments: null, featuresDF: null, confidenceMap: null,
      overlayRGB: null, heatmap: null, regions: null,
      numRegions: 0, regionFeatures: {}
    };
    updateProcStepIndicators();
    LBL.reset();
    displaySession(s);

    if (s.sensors.length >= 2) {
      document.getElementById('run-proc-btn').disabled = false;
      document.getElementById('proc-session-lbl').textContent = `session: ${sessionId.substring(0,8)}…`;
    }
  } catch(e) { console.error(e); }
}

// Track what's currently rendered to avoid redundant DOM updates (prevents flashing)
let _renderedSensors = {};  // sensor_type -> s3_path currently displayed

function displaySession(session) {
  document.getElementById('hdr-capture-id').textContent = session.capture_id;

  // Build a map of what the session currently has
  const sensorMap = {};
  session.sensors.forEach(s => { sensorMap[s.sensor_type] = s; });
  console.log('sensorMap keys:', Object.keys(sensorMap));
  console.log('full sensors:', session.sensors);
  ['rgb','ir','radar','temperature'].forEach(t => {
    const chip = document.getElementById(`chip-${t}`);
    console.log(`chip-${t}:`, chip, 'in sensorMap:', !!sensorMap[t]); 
    chip.className = `chip ${t} ${sensorMap[t] ? 'present' : ''}`;

    if (sensorMap[t]) {
      // Sensor exists — only re-render if the s3_path changed
      if (_renderedSensors[t] !== sensorMap[t].s3_path) {
        renderSensor(sensorMap[t]);
        _renderedSensors[t] = sensorMap[t].s3_path;
      }
    } else if (_renderedSensors[t]) {
      // Sensor was previously rendered but is no longer in this session.
      // Only clear if this session actually expects it (i.e. it's missing,
      // not just irrelevant).  For single-sensor captures the server sets
      // expected_sensors to e.g. ['rgb'], so IR/radar/temp stay untouched.
      const expected = session.expected_sensors || [];
      if (expected.length === 0 || expected.includes(t)) {
        if(t !== 'temperature'){
        document.getElementById(`img-${t}`).innerHTML =
          '<div class="sp-empty"><div class="sp-empty-icon">[ ]</div><div class="sp-empty-txt">no data</div></div>';
        document.getElementById(`meta-${t}`).innerHTML = '';
        } else {
           document.getElementById('temp-readout').style.display = 'none';
        }
        delete _renderedSensors[t];
      }
    }
    // If neither present nor previously rendered, leave the existing "no data" placeholder alone
  });
}

function resetRenderedSensors() {
  _renderedSensors = {};
  ['rgb','ir','radar'].forEach(t => {
    document.getElementById(`img-${t}`).innerHTML =
      '<div class="sp-empty"><div class="sp-empty-icon">[ ]</div><div class="sp-empty-txt">no data</div></div>';
    document.getElementById(`meta-${t}`).innerHTML = '';
  });
  document.getElementById('temp-readout').style.display = 'none';
}

function resetSensorPanel(sensor) {
  if (sensor === 'temperature') {
    document.getElementById('temp-readout').style.display = 'none';
  } else {
    document.getElementById(`img-${sensor}`).innerHTML =
      '<div class="sp-empty"><div class="sp-empty-icon">[ ]</div><div class="sp-empty-txt">capturing…</div></div>';
    document.getElementById(`meta-${sensor}`).innerHTML = '';
  }
  delete _renderedSensors[sensor];
}

function showCaptureError(sensor, error) {
  const container = document.getElementById('capture-errors');
  if (!container) return;
  const div = document.createElement('div');
  div.className = 'capture-error';
  div.innerHTML =
    '<span class="ce-sensor">' + sensor + '</span> ' +
    error +
    ' <button onclick="this.parentElement.remove()">✕</button>';
  container.appendChild(div);
  setTimeout(() => { if (div.parentElement) div.remove(); }, 15000);
}

function clearCaptureErrors() {
  const container = document.getElementById('capture-errors');
  if (container) container.innerHTML = '';
}

function renderTemperature(s) {
  const meta = s.metadata || {};
  const c1 = meta.temperature_c;
  const c2 = meta.temperature_c_s2;
  if (c1 === undefined) return;

  const f1 = parseFloat(meta.temperature_f || (c1 * 9/5 + 32)).toFixed(1);
  const f2 = c2 !== undefined ? parseFloat(meta.temperature_f_s2 || (c2 * 9/5 + 32)).toFixed(1) : null;
  const dc = meta.temperature_delta_c !== undefined ? Math.abs(meta.temperature_delta_c) : undefined;
  const df = dc !== undefined ? (dc * 9/5).toFixed(2) : null;

  const c2color = c2 === undefined ? 'var(--text-pri)'
    : c2 > 85 ? 'var(--red)'
    : c2 > 70 ? 'var(--yellow)'
    : 'var(--green)';

  if (c2 !== undefined && c2 > 85) {
    alert('WARNING: Interior temperature is ' + c2.toFixed(1) + 'C — exceeds safe operating limit!');
  }

  document.getElementById('temp-readout').style.display = 'block';
  document.getElementById('temp-display').innerHTML =
    '<div style="margin-bottom:6px">' +
      '<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px">Exterior</div>' +
      '<div style="font-size:16px;color:var(--text-pri)">' + c1.toFixed(1) + '°C / ' + f1 + '°F</div>' +
    '</div>' +
    '<div style="margin-bottom:6px">' +
      '<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px">Interior</div>' +
      '<div style="font-size:16px;color:' + c2color + '">' + (f2 ? c2.toFixed(1) + '°C / ' + f2 + '°F' : '—') + '</div>' +
    '</div>' +
    '<div>' +
      '<div style="font-size:10px;color:var(--text-dim);letter-spacing:0.1em;text-transform:uppercase;margin-bottom:3px">Difference</div>' +
      '<div style="font-size:16px;color:var(--text-pri)">' + (dc !== undefined ? dc.toFixed(2) + '°C / ' + df + '°F' : '—') + '</div>' +
    '</div>';
  document.getElementById('temp-display-f').innerHTML = '';
}

function renderSensor(s) {
  const t = s.sensor_type;

  if (t === 'radar') {
    document.getElementById('img-radar').innerHTML =
      '<canvas id="cap-radar-cv" data-s3="' + s.s3_path + '" width="300" height="300"' +
      ' style="max-width:100%;max-height:100%;cursor:zoom-in"' +
      ' onclick="openCanvasModal(\'cap-radar-cv\')"></canvas>';
    setTimeout(() => {
      const cv = document.getElementById('cap-radar-cv');
      if (cv) drawRadarHeatmap(cv, s.s3_path);
    }, 0);
  } else if (t === 'temperature') {
    renderTemperature(s);
  } else {
    const src = `/api/images/${s.s3_path}?t=${Date.now()}`;
    document.getElementById(`img-${t}`).innerHTML =
      `<img src="${src}" alt="${t}" onclick="openModal('${src}')">`;
  }

  if (t !== 'temperature' && s.metadata) {
    const skip = new Set(['capture_id','sensor_type']);
    const rows = Object.entries(s.metadata)
      .filter(([k]) => !skip.has(k))
      .slice(0, 9)
      .map(([k, v]) =>
        `<div class="meta-kv"><span class="meta-k">${k}</span><span class="meta-v">${formatVal(v)}</span></div>`)
      .join('');
    document.getElementById(`meta-${t}`).innerHTML = rows;
  }
}
// ═══════════════════════════════════════════════════════════
// CAPTURE
// ═══════════════════════════════════════════════════════════
async function triggerCapture() {
  const devId  = document.getElementById('device-select').value;
  const sensor = selectedSensor();
  if (!devId) { alert('Select a device first.'); return; }

  resetPipeline();
  clearCaptureErrors();
  document.getElementById('capture-btn').disabled  = true;
  document.getElementById('run-proc-btn').disabled = true;
  currentCaptureId    = null;
  awaitingCaptureId   = true;
  pendingCheckpoints  = [];
  captureInProgress   = true;
  userSelectedSession = false;
  pendingSensor       = sensor;

  procResults = {
    roadMask: null, irMapped: null, radarMapped: null,
    segments: null, featuresDF: null, confidenceMap: null,
    overlayRGB: null, heatmap: null, regions: null,
    numRegions: 0, regionFeatures: {}
  };
  LBL.reset();
  updateProcStepIndicators();

  // For "all" reset every panel; for single sensor only reset that one
  if (sensor === 'all') {
    resetRenderedSensors();
  } else {
    resetSensorPanel(sensor);
  }

  // Safety timeout — scale to capture type
  const maxSensorTimeout = Math.max(captureOptions.ir_timeout || 2, captureOptions.radar_timeout || 2);
  const safetyMs = sensor === 'all' ? (maxSensorTimeout * 1000 + 15000) : 12000;
  clearTimeout(_captureTimeout);
  _captureTimeout = setTimeout(() => {
    if (captureInProgress) {
      captureInProgress = false;
      pendingSensor     = 'all';
      document.getElementById('capture-btn').disabled = false;
      console.warn('Capture safety timeout — re-enabled capture button');
    }
  }, safetyMs);

  try {
    // Build per-sensor options to forward to the Pi
    const opts = {
      focus:         captureOptions.focus,
      exposure_time: captureOptions.exposure_time,
      gain:          captureOptions.gain,
      awb_mode:      captureOptions.awb_mode,
      brightness:    captureOptions.brightness,
      contrast:      captureOptions.contrast,
      colormap:      captureOptions.colormap,
      save_raw:      captureOptions.save_raw,
    };

    // Send per-sensor timeouts so the Pi gets the right value for each
    if (sensor === 'ir') {
      opts.timeout = captureOptions.ir_timeout;
    } else if (sensor === 'radar') {
      opts.timeout = captureOptions.radar_timeout;
    } else if (sensor === 'all') {
      // For "all" captures the Pi uses a single timeout for each sensor
      opts.timeout = maxSensorTimeout;
    }

    const r = await fetch(`/api/devices/${devId}/capture/${sensor}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(opts)
    });
    if (!r.ok) {
      const e = await r.json();
      throw new Error(e.detail || r.statusText);
    }
    const d = await r.json();
    currentCaptureId  = d.capture_id;
    awaitingCaptureId = false;
    setStage('command_sent', 'success', new Date().toLocaleTimeString('en-US', {hour12: false}));

    // Flush any checkpoints that arrived while we were waiting for capture_id
    const buffered = pendingCheckpoints.splice(0);
    buffered.forEach(msg => {
      if (msg.capture_id === currentCaptureId) handleCheckpoint(msg);
    });

    // From here, handleCheckpoint drives everything:
    //   - Each db_stored event triggers loadSession() to show new sensors
    //   - is_complete re-enables capture button and pipeline button

  } catch(e) {
    awaitingCaptureId = false;
    captureInProgress = false;
    pendingSensor     = 'all';
    clearTimeout(_captureTimeout);
    showCaptureError(sensor, e.message);
    document.getElementById('capture-btn').disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════
// PROCESSING PIPELINE
// ═══════════════════════════════════════════════════════════
function getConfig() {
  return {
    baseline: parseFloat(document.getElementById('param-baseline').value),
    height:   parseFloat(document.getElementById('param-height').value),
    angle:    parseFloat(document.getElementById('param-angle').value),
    airTemp:  parseFloat(document.getElementById('param-air-temp').value),
    segmentation: {
      min_distance:  parseInt(document.getElementById('param-min-dist').value),
      threshold_rel: parseFloat(document.getElementById('param-thresh-rel').value) / 100,
      lbp_r:         parseInt(document.getElementById('param-lbp-r').value),
      lbp_p:         parseInt(document.getElementById('param-lbp-p').value),
    },
    model: document.getElementById('model-select').value,
    weights: currentWeights,
    ensemble: {
      use_logistic:      document.getElementById('ens-logistic').checked,
      use_naive_bayes:   document.getElementById('ens-nb').checked,
      use_random_forest: document.getElementById('ens-rf').checked,
      gaussian_sigma:    parseFloat(document.getElementById('ens-sigma').value) / 10,
      median_size:       parseInt(document.getElementById('ens-median').value),
    }
  };
}

async function runFullPipeline() {
  if (!currentSessionId) return;
  switchTab('processing');
  await runStep('road');
  await runStep('mapping');
  await runStep('segment');
  await runStep('features');
  await runStep('inference');
}

async function runStep(step) {
  if (!currentSessionId) { alert('No session loaded.'); return; }

  const numEl   = document.getElementById(`step-num-${step}`);
  const btnEl   = document.getElementById(`btn-${step}`);
  const pipeEl  = document.getElementById(`pd-proc-${step}`);
  const pipeLbl = document.getElementById(`pl-proc-${step}`);

  if (numEl) numEl.classList.add('running');
  if (btnEl) { btnEl.disabled = true; btnEl.textContent = '⟳ Running…'; }
  if (pipeEl)  pipeEl.className  = 'p-dot running';
  if (pipeLbl) pipeLbl.className = 'p-lbl running';

  const t0 = Date.now();

  try {
    const r = await fetch(`/api/processing/${step}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: currentSessionId, config: getConfig() })
    });

    const elapsed = ((Date.now() - t0) / 1000).toFixed(2) + 's';

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || r.statusText);
    }

    const data = await r.json();
    handleStepResult(step, data);

    if (numEl)  { numEl.classList.remove('running'); numEl.classList.add('done'); }
    if (btnEl)  { btnEl.disabled = false; btnEl.textContent = '✓ Done'; btnEl.classList.add('success'); }
    if (pipeEl)  pipeEl.className  = 'p-dot success';
    if (pipeLbl) pipeLbl.className = 'p-lbl success';
    const ptEl = document.getElementById(`pt-proc-${step}`);
    if (ptEl) ptEl.textContent = elapsed;

  } catch(e) {
    if (numEl)  numEl.classList.remove('running');
    if (btnEl)  { btnEl.disabled = false; btnEl.textContent = '✗ Retry'; }
    if (pipeEl)  pipeEl.className  = 'p-dot failed';
    if (pipeLbl) pipeLbl.className = 'p-lbl failed';
    setStatusErr(step, e.message);
    console.error(`Step ${step} failed:`, e);
  }
}

function handleStepResult(step, data) {
  if (step === 'road') {
    procResults.roadMask = data.mask_url;
    setStatus('road', 'ok', `road: ${data.road_pixels?.toLocaleString() || '?'} px`);
    if (currentView === 'road-mask') refreshProcCanvas();
  }
  if (step === 'mapping') {
    procResults.irMapped    = data.ir_mapped_url;
    procResults.radarMapped = data.radar_mapped_url;
    setStatus('mapping', 'ok', 'mapping: done');
    if (['ir-mapped','radar-mapped'].includes(currentView)) refreshProcCanvas();
  }
  if (step === 'segment') {
    procResults.segments   = data.segment_overlay_url;
    procResults.numRegions = data.num_regions || 0;
    setStatus('segments', 'ok', `segments: ${data.num_regions} regions`);
    if (currentView === 'segments') refreshProcCanvas();
    updateClickHint();
  }
  if (step === 'features') {
    procResults.regionFeatures = data.region_features || {};
    setStatus('inference', 'ok', `features: ${Object.keys(procResults.regionFeatures).length} regions`);
    updateClickHint();
  }
  if (step === 'inference') {
    procResults.confidenceMap = data.confidence_map_url;
    procResults.overlayRGB    = data.overlay_rgb_url;
    procResults.heatmap       = data.heatmap_url;
    setStatus('inference', 'ok', `ice: ${(data.mean_confidence * 100).toFixed(1)}% mean conf`);
    document.getElementById('proc-tab-dot').style.background = 'var(--yellow)';
    setView('overlay');
  }
}

function setStatus(key, level, msg) {
  const dot = document.getElementById(`sd-${key}`);
  const lbl = document.getElementById(`ss-${key}`);
  if (dot) dot.className = `status-dot ${level}`;
  if (lbl) lbl.textContent = msg;
}

function setStatusErr(step, msg) {
  const keyMap = { road:'road', mapping:'mapping', segment:'segments', features:'inference', inference:'inference' };
  setStatus(keyMap[step] || step, 'err', `${step}: ${msg.substring(0,40)}`);
}

function updateProcStepIndicators() {
  const labels = {
    road:      'Run Road Isolation',
    mapping:   'Run Mapping',
    segment:   'Run Segmentation',
    features:  'Run Feature Extraction',
    inference: 'Run Inference'
  };
  ['road','mapping','segment','features','inference'].forEach(s => {
    const n = document.getElementById(`step-num-${s}`);
    if (n) n.classList.remove('done','running');
    const btn = document.getElementById(`btn-${s}`);
    if (btn) { btn.classList.remove('success'); btn.textContent = labels[s]; btn.disabled = false; }
  });
}

// ═══════════════════════════════════════════════════════════
// PROCESSING VISUALIZATION
// ═══════════════════════════════════════════════════════════
function setView(view) {
  currentView = view;
  document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(`vbtn-${view}`);
  if (btn) btn.classList.add('active');
  refreshProcCanvas();
}

async function refreshProcCanvas() {
  const canvas      = document.getElementById('proc-canvas');
  const placeholder = document.getElementById('proc-placeholder');

  let imgUrl = null;
  if      (currentView === 'rgb' && currentSession) {
    const rgb = currentSession.sensors.find(s => s.sensor_type === 'rgb');
    if (rgb) imgUrl = `/api/images/${rgb.s3_path}`;
  }
  else if (currentView === 'ir-mapped')    { imgUrl = procResults.irMapped; }
  else if (currentView === 'radar-mapped') { imgUrl = procResults.radarMapped; }
  else if (currentView === 'road-mask')    { imgUrl = procResults.roadMask; }
  else if (currentView === 'segments')     { imgUrl = procResults.segments; }
  else if (currentView === 'heatmap')      { imgUrl = procResults.heatmap; }
  else if (currentView === 'overlay')      { imgUrl = procResults.overlayRGB; }

  if (!imgUrl) {
    canvas.style.display = 'none';
    placeholder.style.display = 'flex';
    return;
  }

  placeholder.style.display = 'none';
  canvas.style.display = 'block';

  const img = new Image();
  img.onload = () => {
    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    if (showSegmentOverlay && procResults.segments) {
      drawOverlay(ctx, procResults.segments, 0.4, canvas.width, canvas.height);
    }
    if (showHeatmapOverlay && procResults.heatmap && currentView !== 'heatmap') {
      drawHeatmapOverlay(ctx, procResults.heatmap, 0.5, canvas.width, canvas.height, heatmapThreshold);
    }
  };
  img.src = imgUrl;
}

function drawOverlay(ctx, overlayUrl, alpha, w, h) {
  const img = new Image();
  img.onload = () => {
    ctx.globalAlpha = alpha;
    ctx.drawImage(img, 0, 0, w, h);
    ctx.globalAlpha = 1.0;
  };
  img.src = overlayUrl;
}

function drawHeatmapOverlay(ctx, heatmapUrl, alpha, w, h, threshold) {
  const img = new Image();
  img.onload = () => {
    const off  = document.createElement('canvas');
    off.width  = w;
    off.height = h;
    const octx = off.getContext('2d');
    octx.drawImage(img, 0, 0, w, h);

    if (threshold > 0) {
      const imageData = octx.getImageData(0, 0, w, h);
      const d = imageData.data;
      for (let i = 0; i < d.length; i += 4) {
        const v = (d[i] * 0.299 + d[i+1] * 0.587 + d[i+2] * 0.114) / 255;
        if (v < threshold / 100) d[i+3] = 0;
      }
      octx.putImageData(imageData, 0, 0);
    }

    ctx.globalAlpha = alpha;
    ctx.drawImage(off, 0, 0);
    ctx.globalAlpha = 1.0;
  };
  img.src = heatmapUrl;
}

function toggleSegmentOverlay() {
  const segEl  = document.getElementById('toggle-segments');
  const featEl = document.getElementById('toggle-feat-regions');
  showSegmentOverlay = segEl ? segEl.checked : false;
  if (featEl) featEl.checked = showSegmentOverlay;
  updateClickHint();
  refreshProcCanvas();
}

function toggleHeatmapOverlay() {
  showHeatmapOverlay = document.getElementById('toggle-heatmap').checked;
  refreshProcCanvas();
}

function updateThreshold(val) {
  heatmapThreshold = parseInt(val);
  document.getElementById('threshold-val').textContent = val + '%';
  if (showHeatmapOverlay || currentView === 'heatmap') refreshProcCanvas();
}

// ═══════════════════════════════════════════════════════════
// REGION CLICK — feature inspection
// ═══════════════════════════════════════════════════════════
function updateClickHint() {
  const hint = document.getElementById('click-hint');
  if (!hint) return;
  const canInspect = showSegmentOverlay && Object.keys(procResults.regionFeatures).length > 0;
  hint.classList.toggle('visible', canInspect);
}

async function onCanvasClick(event) {
  if (!showSegmentOverlay) return;
  if (!currentSessionId)   return;

  const canvas = document.getElementById('proc-canvas');
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.floor((event.clientX - rect.left) * scaleX);
  const y = Math.floor((event.clientY - rect.top)  * scaleY);

  try {
    const r = await fetch(`/api/processing/region_at?session_id=${currentSessionId}&x=${x}&y=${y}`);
    if (!r.ok) return;
    const data = await r.json();

    if (!data.region_id || data.region_id === 0) {
      hideRegionTooltip();
      return;
    }

    const backendFeatures = data.features || {};
    const cachedFeatures  = procResults.regionFeatures[data.region_id] || {};
    const merged = { ...cachedFeatures, ...backendFeatures };
    showRegionTooltip(event, data.region_id, merged);
  } catch(e) {
    console.warn('Region inspect failed:', e);
  }
}

function showRegionTooltip(event, regionId, features) {
  const tt = document.getElementById('region-tooltip');
  document.getElementById('rt-title').textContent = `Region ${regionId}`;

  // Priority keys matched to new IRFeatureExtractor / HeuristicWeights naming
  const priorityKeys = [
    'ir_mean_temp',
    'ir_near_freezing',
    'ir_in_danger_zone',
    'ir_colder_than_air_fraction',
    'shininess_ratio',
    'roughness_lap_s1',
    'color_bluish_score',
    'color_whitish_score',
    'wetness_darkening',
    'spatial_area_fraction',
  ];

  const rows = priorityKeys
    .filter(k => features[k] !== undefined)
    .map(k => {
      const v       = features[k];
      const display = typeof v === 'number' ? v.toFixed(3) : v;
      return `<div class="region-feat-row">
        <span class="region-feat-k">${k}</span>
        <span class="region-feat-v">${display}</span>
      </div>`;
    })
    .join('');

  const conf = features['_confidence'];
  let confHtml = '';
  if (conf !== undefined) {
    const pct   = Math.round(conf * 100);
    const color = conf > 0.65 ? 'var(--red)' : conf > 0.35 ? 'var(--yellow)' : 'var(--green)';
    confHtml = `
      <div class="region-conf-bar">
        <div class="region-conf-label">ice confidence: ${pct}%</div>
        <div class="region-conf-track">
          <div class="region-conf-fill" style="width:${pct}%;background:${color}"></div>
        </div>
      </div>`;
  }

  document.getElementById('rt-body').innerHTML = rows + confHtml;

  const area  = document.getElementById('proc-canvas-area');
  const aRect = area.getBoundingClientRect();
  let left = event.clientX - aRect.left + 12;
  let top  = event.clientY - aRect.top  + 12;
  if (left + 230 > aRect.width)  left = left - 242;
  if (top  + 220 > aRect.height) top  = top  - 230;

  tt.style.left = left + 'px';
  tt.style.top  = top  + 'px';
  tt.classList.add('visible');

  clearTimeout(tt._hideTimer);
  tt._hideTimer = setTimeout(() => tt.classList.remove('visible'), 6000);
}

function hideRegionTooltip() {
  document.getElementById('region-tooltip').classList.remove('visible');
}

document.addEventListener('click', e => {
  if (!e.target.closest('#proc-canvas') && !e.target.closest('#region-tooltip')) {
    hideRegionTooltip();
  }
});

// ═══════════════════════════════════════════════════════════
// MODEL CONFIG
// ═══════════════════════════════════════════════════════════
function onModelChange() {
  const v = document.getElementById('model-select').value;
  document.getElementById('heuristic-config').style.display = v === 'heuristic' ? 'block' : 'none';
  document.getElementById('ensemble-config').style.display  = v === 'ensemble'  ? 'block' : 'none';
}

// ═══════════════════════════════════════════════════════════
// HEURISTIC WEIGHTS
// ═══════════════════════════════════════════════════════════
function getDefaultWeights() {
  return {
    // IR Temperature
    ir_mean_temp_weight:                    -0.15,
    ir_near_freezing_weight:                 0.80,
    ir_at_or_below_freezing_weight:          0.60,
    ir_in_danger_zone_weight:                0.40,
    ir_diff_from_air_mean_weight:           -0.05,
    ir_colder_than_air_fraction_weight:      0.30,
    ir_temp_var_weight:                     -0.10,
    ir_gradient_mean_weight:                -0.20,
    ir_gradient_smooth_fraction_weight:      0.25,
    ir_dist_from_dew_point_weight:          -0.15,
    ir_below_dew_point_fraction_weight:      0.35,
    ir_emissivity_proxy_weight:              0.10,
    // Thermal Texture
    ir_glcm_homogeneity_weight:              0.40,
    ir_glcm_contrast_weight:                -0.20,
    ir_lbp_uniformity_weight:                0.30,
    // Temporal Temperature
    medium_temp_rate_weight:                -1.50,
    medium_temp_accel_weight:               -0.80,
    in_cooling_plateau_weight:               1.20,
    temp_trajectory_score_weight:            1.00,
    // Roughness / Texture
    roughness_lap_s1_weight:                -0.20,
    roughness_lap_s4_weight:                -0.15,
    roughness_lbp_uniformity_weight:         0.30,
    glcm_homogeneity_mean_weight:            0.40,
    glcm_energy_mean_weight:                 0.30,
    glcm_contrast_mean_weight:              -0.25,
    structure_isotropy_weight:               0.30,
    wavelet_hf_fraction_weight:             -0.25,
    // Specularity
    shininess_ratio_weight:                  0.60,
    highlight_density_weight:                0.30,
    nearby_highlights_weight:                0.40,
    specular_lobe_peak_weight:               0.25,
    // Wetness
    wetness_saturation_mean_weight:          0.20,
    wetness_spec_diffuse_ratio_weight:       0.35,
    wetness_reflection_coherence_weight:     0.25,
    wetness_darkening_weight:                0.40,
    wetness_v_variance_weight:               0.20,
    // Temporal Texture
    texture_stability_1min_weight:           0.50,
    highlight_persistence_1min_weight:       0.40,
    // Color
    color_bluish_score_weight:               0.30,
    color_whitish_score_weight:              0.40,
    color_brightness_mean_weight:            0.10,
    // Spatial
    spatial_area_fraction_weight:            0.05,
    spatial_compactness_weight:              0.10,
    spatial_boundary_smoothness_weight:      0.15,
    // Thresholds
    ice_threshold:                           0.50,
    uncertain_threshold_low:                 0.35,
    uncertain_threshold_high:                0.65,
  };
}

const PRESETS = {
  default:      getDefaultWeights(),
  conservative: {
    ...getDefaultWeights(),
    ice_threshold:               0.70,
    ir_near_freezing_weight:     1.20,
    in_cooling_plateau_weight:   1.50,
    medium_temp_rate_weight:    -2.00,
  },
  aggressive: {
    ...getDefaultWeights(),
    ice_threshold:               0.30,
    ir_mean_temp_weight:        -0.25,
    shininess_ratio_weight:      0.90,
    medium_temp_rate_weight:    -1.00,
  },
  texture: {
    ...getDefaultWeights(),
    shininess_ratio_weight:               1.00,
    nearby_highlights_weight:             0.80,
    spatial_boundary_smoothness_weight:   0.60,
    roughness_lap_s1_weight:             -0.50,
    ir_mean_temp_weight:                 -0.02,
  },
  temperature: {
    ...getDefaultWeights(),
    ir_mean_temp_weight:                 -0.30,
    ir_near_freezing_weight:              1.20,
    medium_temp_rate_weight:             -2.50,
    in_cooling_plateau_weight:            2.00,
    shininess_ratio_weight:               0.05,
    wetness_spec_diffuse_ratio_weight:    0.05,
  },
};

function applyPreset(name) {
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`preset-${name}`)?.classList.add('active');
  currentWeights = { ...PRESETS[name] };
  syncWeightSlidersFromWeights();
}

function syncWeightSlidersFromWeights() {
  document.querySelectorAll('.weight-slider').forEach(slider => {
    const key = slider.dataset.key;
    if (currentWeights[key] !== undefined) {
      slider.value = Math.round(currentWeights[key] * 100);
      const valEl = slider.nextElementSibling;
      if (valEl) valEl.textContent = currentWeights[key].toFixed(2);
    }
  });
}

function onWeightChange(slider) {
  const key = slider.dataset.key;
  const val = parseInt(slider.value) / 100;
  currentWeights[key] = val;
  const valEl = slider.nextElementSibling;
  if (valEl) valEl.textContent = val.toFixed(2);
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
}

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════
function formatVal(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}

function openModal(src) {
  document.getElementById('modal-img').src = src;
  document.getElementById('modal').classList.add('open');
}
function openCanvasModal(canvasId) {
  const c = document.getElementById(canvasId);
  if (c) openModal(c.toDataURL('image/png'));
}
function closeModal() {
  document.getElementById('modal').classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

// ═══════════════════════════════════════════════════════════
// RADAR RENDERING
// ═══════════════════════════════════════════════════════════
const RP = { TX: 2, RX: 4, BINS: 256, ABINS: 64, RES: 0.04360212053571429 };

function fftInPlace(re, im) {
  const N = re.length;
  for (let i = 1, j = 0; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i],re[j]]=[re[j],re[i]]; [im[i],im[j]]=[im[j],im[i]]; }
  }
  for (let len = 2; len <= N; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let cRe = 1, cIm = 0;
      const half = len >> 1;
      for (let j = 0; j < half; j++) {
        const uRe=re[i+j],uIm=im[i+j],vRe=re[i+j+half]*cRe-im[i+j+half]*cIm,vIm=re[i+j+half]*cIm+im[i+j+half]*cRe;
        re[i+j]=uRe+vRe;im[i+j]=uIm+vIm;re[i+j+half]=uRe-vRe;im[i+j+half]=uIm-vIm;
        const nr=cRe*wRe-cIm*wIm;cIm=cRe*wIm+cIm*wRe;cRe=nr;
      }
    }
  }
}

function jetColor(v) {
  v = Math.max(0, Math.min(1, v));
  return [
    Math.min(1, Math.max(0, 1.5 - Math.abs(4*v - 3))) * 255 | 0,
    Math.min(1, Math.max(0, 1.5 - Math.abs(4*v - 2))) * 255 | 0,
    Math.min(1, Math.max(0, 1.5 - Math.abs(4*v - 1))) * 255 | 0
  ];
}

async function drawRadarHeatmap(canvas, s3Path) {
  try {
    const resp = await fetch(`/api/images/${s3Path}`);
    const data = await resp.json();
    const raw  = data.azimuth_static;
    const { TX, RX, BINS, ABINS, RES } = RP;
    const VA   = TX * RX;
    const mag  = new Float32Array(BINS * ABINS);
    const re   = new Float64Array(ABINS);
    const im   = new Float64Array(ABINS);
    const half = ABINS >> 1;

    for (let r = 0; r < BINS; r++) {
      re.fill(0); im.fill(0);
      for (let v = 0; v < VA; v++) {
        const idx = (r * VA + v) * 2;
        re[v] = raw[idx]; im[v] = raw[idx+1];
      }
      fftInPlace(re, im);
      for (let k = 0; k < half; k++) {
        let t = re[k]; re[k] = re[k+half]; re[k+half] = t;
        t = im[k]; im[k] = im[k+half]; im[k+half] = t;
      }
      for (let k = 0; k < ABINS; k++) mag[r*ABINS+k] = Math.sqrt(re[k]*re[k] + im[k]*im[k]);
    }

    let maxMag = 0;
    for (let i = 0; i < mag.length; i++) if (mag[i] > maxMag) maxMag = mag[i];
    if (maxMag === 0) maxMag = 1;

    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(W, H);
    const px = imgData.data;
    const rangeDepth = BINS * RES, rangeWidth = rangeDepth / 2;
    const BG = [13, 13, 26];

    for (let py = 0; py < H; py++) {
      for (let px2 = 0; px2 < W; px2++) {
        const x     = (px2 / (W-1)) * 2 * rangeWidth - rangeWidth;
        const y     = (1 - py / (H-1)) * rangeDepth;
        const rng   = Math.sqrt(x*x + y*y);
        const theta = Math.atan2(x, y);
        const i4    = (py * W + px2) * 4;
        if (y < 0 || rng > rangeDepth || theta < -Math.PI/2 || theta > Math.PI/2) {
          px[i4]=BG[0]; px[i4+1]=BG[1]; px[i4+2]=BG[2]; px[i4+3]=255; continue;
        }
        const rBin = Math.min(BINS-1,  rng / RES | 0);
        const aBin = Math.min(ABINS-1, (theta + Math.PI/2) / Math.PI * ABINS | 0);
        const [cr,cg,cb] = jetColor(mag[rBin*ABINS+aBin] / maxMag);
        px[i4]=cr; px[i4+1]=cg; px[i4+2]=cb; px[i4+3]=255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  } catch(e) { console.error('Radar render error:', e); }
}

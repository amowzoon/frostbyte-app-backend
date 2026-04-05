/* ── HELPERS ─────────────────────────────────────── */
const $ = id => document.getElementById(id);

function getDeviceId() {
  return $('device-id').value.trim() || 'pi-001';
}

function formatESTTimeOnly(ts) {
  if (!ts) return 'N/A';
  return new Date(ts).toLocaleString('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true
  });
}

/* ── RESIZER ─────────────────────────────────────── */
(function initResizer() {
  const resizer    = $('resizer');
  const topSection = $('top-section');
  const mainLayout = $('main-layout');
  let dragging = false;

  resizer.addEventListener('mousedown', () => {
    dragging = true;
    resizer.classList.add('dragging');
    document.body.style.cursor     = 'ns-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const rect = mainLayout.getBoundingClientRect();
    const pct  = Math.min(Math.max((e.clientY - rect.top) / rect.height * 100, 15), 85);
    topSection.style.height = pct + '%';
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove('dragging');
    document.body.style.cursor     = '';
    document.body.style.userSelect = '';
  });
})();

/* ── ENDPOINT ACCORDION ──────────────────────────── */
function toggleEndpoint(hdr) {
  const body    = hdr.nextElementSibling;
  const chevron = hdr.querySelector('.api-chevron');
  chevron.textContent = body.classList.toggle('open') ? '▼' : '▶';
}

/* ── GENERIC API CALL ────────────────────────────── */
async function callApi(method, path, body, btn) {
  const respEl = btn.nextElementSibling;
  btn.disabled = true;
  respEl.className  = 'api-response visible';
  respEl.textContent = '…';

  // All sensor paths are proxied through the device route on the server
  const SENSOR_PREFIXES = ['/api/capture', '/api/camera', '/api/ir', '/api/radar', '/api/temperature'];
  const needsProxy = SENSOR_PREFIXES.some(p => path.startsWith(p));
  const url = needsProxy ? `/api/devices/${getDeviceId()}${path}` : path;

  try {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body !== null) opts.body = JSON.stringify(body);
    const res  = await fetch(url, opts);
    const data = await res.json();
    respEl.textContent = JSON.stringify(data, null, 2);
    respEl.className   = `api-response visible ${res.ok ? 'ok' : 'err'}`;
  } catch (e) {
    respEl.textContent = 'fetch error: ' + e.message;
    respEl.className   = 'api-response visible err';
  } finally {
    btn.disabled = false;
  }
}

/* ── CAPTURE HELPERS ─────────────────────────────── */
function sendRgbCapture(btn) {
  const body = { type: 'rgb' };
  const focus   = $('rgb-cap-focus').value;
  const exp     = $('rgb-cap-exp').value;
  const gain    = $('rgb-cap-gain').value;
  const awb     = $('rgb-cap-awb').value;
  const bright  = $('rgb-cap-bright').value;
  const contrast= $('rgb-cap-contrast').value;
  if (focus)    body.focus          = +focus;
  if (exp)      body.exposure_time  = +exp;
  if (gain)     body.gain           = +gain;
  if (awb)      body.awb_mode       = awb;
  if (bright)   body.brightness     = +bright;
  if (contrast) body.contrast       = +contrast;
  callApi('POST', '/api/capture', body, btn);
}

function sendIrCapture(btn) {
  const body = {
    type:     'ir',
    save_raw: $('ir-cap-raw').value === 'true',
    timeout:  +$('ir-cap-timeout').value,
  };
  const cmap = $('ir-cap-colormap').value;
  if (cmap) body.colormap = cmap;
  callApi('POST', '/api/capture', body, btn);
}

function sendTempCapture(btn) {
  callApi('POST', '/api/capture', {
    type:      'temperature',
    temp_mode: $('temp-cap-mode').value,
    temp_n:    +$('temp-cap-n').value,
    timeout:   +$('temp-cap-timeout').value,
  }, btn);
}

/* ── TOOLBAR QUICK-CAPTURE ───────────────────────── */
async function sendCapture(sensor) {
  try {
    const res  = await fetch(`/api/devices/${getDeviceId()}/capture/${sensor}`, { method: 'POST' });
    const data = await res.json();
    console.log('capture triggered:', data);
  } catch (e) {
    console.error('capture failed:', e);
    alert('Failed to send capture: ' + e.message);
  }
}

/* ── WEBSOCKETS ──────────────────────────────────── */
const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws     = new WebSocket(`${wsProto}://${location.host}/ws/dashboard`);
const logsWs = new WebSocket(`${wsProto}://${location.host}/ws/logs`);

let _dashSubscribed = null;

function dashSubscribe(deviceId) {
  if (_dashSubscribed === deviceId) return;
  if (ws.readyState === WebSocket.OPEN) {
    if (_dashSubscribed) ws.send(JSON.stringify({ action: 'unsubscribe', device_id: _dashSubscribed }));
    ws.send(JSON.stringify({ action: 'subscribe', device_id: deviceId }));
    _dashSubscribed = deviceId;
  }
}

ws.onopen = () => {
  setStatus('ws-status', true, 'pipeline');
  dashSubscribe(getDeviceId());
};
ws.onclose   = () => setStatus('ws-status',  false, 'pipeline');
ws.onmessage = e => {
  const data = JSON.parse(e.data);
  if (data.type === 'checkpoint') updateCapture(data);
};

// Re-subscribe when user changes the device ID input
$('device-id').addEventListener('change', () => dashSubscribe(getDeviceId()));

logsWs.onopen    = () => setStatus('log-status', true,  'logs');
logsWs.onclose   = () => setStatus('log-status', false, 'logs');
logsWs.onmessage = e => addLogEntry(JSON.parse(e.data));

function setStatus(elemId, ok, label) {
  const el = $(elemId);
  el.className   = 'status-pill ' + (ok ? 'ok' : 'err');
  el.textContent = label + ': ' + (ok ? 'live' : 'off');
}

/* ── LOG PANEL ───────────────────────────────────── */
const MAX_LOGS = 200;

function addLogEntry(entry) {
  const container = $('log-container');
  const ph = container.querySelector('.log-placeholder');
  if (ph) ph.remove();

  while (container.children.length >= MAX_LOGS) {
    container.removeChild(container.firstChild);
  }

  const div = document.createElement('div');
  div.className = `log-entry ${entry.level || ''}`;

  const time = formatESTTimeOnly(entry.timestamp);
  let html = `<span class="log-time">[${time}]</span>`;
  html    += `<span class="log-level">${entry.level}</span>`;
  if (entry.file) html += `<span class="log-loc">[${entry.file}:${entry.line}]</span>`;
  html    += `<span class="log-msg">${entry.message}</span>`;
  if (entry.traceback) html += `<div class="log-tb">${entry.traceback}</div>`;

  div.innerHTML = html;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

/* ── CAPTURE PIPELINE CARDS ──────────────────────── */
const captures = {};

const STAGES = [
  { key: 'command_sent',            label: 'command sent' },
  { key: 'sensor_capture_complete', label: 'sensor capture' },
  { key: 'upload_received',         label: 'upload received' },
  { key: 's3_stored',               label: 's3 stored' },
  { key: 'db_stored',               label: 'db stored' },
];

function updateCapture(checkpoint) {
  const cid = checkpoint.capture_id;
  if (!captures[cid]) {
    captures[cid] = { id: cid, stages: {}, startTime: Date.now(), collapsed: false };
    renderCaptureCard(cid);
  }
  captures[cid].stages[checkpoint.stage] = {
    status:      checkpoint.data.error ? 'failed' : 'success',
    timestamp:   checkpoint.timestamp,
    data:        checkpoint.data,
    duration_ms: checkpoint.duration_ms,
    file:        checkpoint.file,
    function:    checkpoint.function,
    line:        checkpoint.line,
  };
  updateCaptureCard(cid);
}

function renderCaptureCard(cid) {
  const container = $('captures-container');
  const ph = container.querySelector('.placeholder');
  if (ph) ph.remove();

  const card = document.createElement('div');
  card.id        = `capture-${cid}`;
  card.className = 'capture-card';
  container.insertBefore(card, container.firstChild);
  updateCaptureCard(cid);
}

function toggleCapture(cid) {
  captures[cid].collapsed = !captures[cid].collapsed;
  updateCaptureCard(cid);
}

function updateCaptureCard(cid) {
  const capture  = captures[cid];
  const card     = $(`capture-${cid}`);
  const expanded = !capture.collapsed;

  let overall = 'in-progress';
  for (const s of STAGES) {
    if (capture.stages[s.key]?.status === 'failed') { overall = 'failed'; break; }
  }
  if (overall !== 'failed' && capture.stages.db_stored?.status === 'success') overall = 'success';

  const shortId = cid.substring(0, 16) + '…';
  const timeStr = formatESTTimeOnly(capture.startTime);

  let html = `
    <div class="capture-card-hdr" onclick="toggleCapture('${cid}')">
      <span class="cc-arrow">${expanded ? '▼' : '▶'}</span>
      <span class="cc-id">${shortId}</span>
      <span class="cc-badge ${overall}">${overall}</span>
      <span class="cc-time">${timeStr}</span>
    </div>`;

  if (expanded) {
    html += `<div class="capture-card-body">`;
    for (const stage of STAGES) {
      const sd = capture.stages[stage.key];
      const st = sd ? sd.status : 'pending';
      html += `
        <div class="stage-row">
          <div class="stage-dot ${st}"></div>
          <div class="stage-info">
            <div class="stage-name ${st}">${stage.label}</div>`;
      if (sd) {
        let detail = formatESTTimeOnly(sd.timestamp);
        if (sd.duration_ms) detail += ` (${sd.duration_ms}ms)`;
        if (sd.file)        detail += `  ${sd.file}:${sd.line}`;
        html += `<div class="stage-detail">${detail}</div>`;
        if (sd.data?.error)      html += `<div class="stage-detail error">error: ${sd.data.error}</div>`;
        if (sd.data?.s3_path)    html += `<div class="stage-detail">${sd.data.s3_path}</div>`;
        if (sd.data?.size_bytes) html += `<div class="stage-detail">${(sd.data.size_bytes / 1024 / 1024).toFixed(2)} MB</div>`;
        html += `<details><summary>raw json</summary><pre>${JSON.stringify(sd.data, null, 2)}</pre></details>`;
      }
      html += `</div></div>`;
    }
    html += `</div>`;
  }

  card.innerHTML = html;
}

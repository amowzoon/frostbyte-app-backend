// ═══════════════════════════════════════════════════════════
// LABELING MODULE
// All state and logic for the labeling tab
// ═══════════════════════════════════════════════════════════

const LBL = (() => {

  // ── Constants ───────────────────────────────────────────
  const CLASSES = {
    ice:     { name: 'Ice',        color: '#4db8ff', rgba: [77,  184, 255, 180] },
    not_ice: { name: 'Not Ice',    color: '#e8a838', rgba: [232, 168,  56, 180] },
    void:    { name: 'Not Roadway',color: '#9060d0', rgba: [144,  96, 208, 180] },
  };

  const TOOLS = { BRUSH: 'brush', POLY: 'poly', BUCKET: 'bucket' };

  // ── State ────────────────────────────────────────────────
  let activeClass  = 'ice';
  let activeTool   = TOOLS.BRUSH;
  let brushRadius  = 18;
  let labelOpacity = 0.55;
  let labelView    = 'rgb'; // which base image to show

  // Canvas references
  let baseCanvas   = null;   // background image
  let drawCanvas   = null;   // label painting
  let cursorCanvas = null;   // brush cursor preview
  let drawCtx      = null;
  let cursorCtx    = null;
  let baseCtx      = null;

  // Current base image URL
  let baseImageUrl = null;

  // Cached offscreen canvas of the segment overlay (for bucket fill without needing overlay visible)
  let segmentCanvas = null;
  let segmentCanvasUrl = null;

  // Painting state
  let isPainting   = false;
  let lastX = 0, lastY = 0;
  let rafPending   = false;   // RAF throttle for brush
  let pendingMove  = null;    // latest mousemove coords waiting for next frame
  let pendingCursor = null;   // latest cursor position waiting for next frame
  let cachedRect   = null;    // cached getBoundingClientRect to avoid reflows

  // Polygon state
  let polyVertices = [];
  let polyHoverPt  = null;
  const POLY_CLOSE_DIST = 14; // px to snap-close

  // History (for undo/redo)
  const MAX_HISTORY = 30;
  let history = [];
  let historyIdx = -1;

  // Segment map (from processing) — parallel to procResults.segments image
  // We store pixel→regionId for bucket fill
  let segmentData = null; // ImageData of segments overlay, null if unavailable

  // ── Init ─────────────────────────────────────────────────
  function init() {
    baseCanvas   = document.getElementById('label-base-canvas');
    drawCanvas   = document.getElementById('label-draw-canvas');
    cursorCanvas = document.getElementById('label-cursor-canvas');
    drawCtx      = drawCanvas.getContext('2d');
    cursorCtx    = cursorCanvas.getContext('2d');
    baseCtx      = baseCanvas.getContext('2d');

    // Wire canvas events
    drawCanvas.addEventListener('mousedown',  onMouseDown);
    drawCanvas.addEventListener('mousemove',  onMouseMove);
    drawCanvas.addEventListener('mouseup',    onMouseUp);
    drawCanvas.addEventListener('mouseleave', onMouseLeave);
    drawCanvas.addEventListener('contextmenu', e => { e.preventDefault(); cancelPoly(); });

    // Keyboard shortcuts
    document.addEventListener('keydown', onKeyDown);
    // Invalidate cached bounding rect on resize so canvasXY stays accurate
    window.addEventListener('resize', () => { cachedRect = null; });
  }

  // ── Reset on new session ─────────────────────────────────
  function reset() {
    baseImageUrl     = null;
    segmentCanvas    = null;
    segmentCanvasUrl = null;
    pendingSnapshot  = false;
    history = [];
    historyIdx = -1;
    if (drawCtx && drawCanvas.width > 0) {
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    }
    updateUndoButtons();
    updateStats();
  }

  // ── Load base image ──────────────────────────────────────
  function loadBaseImage(url, force) {
    if (!url) return;
    if (!force && url === baseImageUrl && baseCanvas.width > 0) return;
    baseImageUrl = url;

    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth;
      const h = img.naturalHeight;

      // Save existing label data before resize wipes the canvas
      let savedLabels = null;
      if (drawCanvas.width > 0 && drawCanvas.height > 0) {
        savedLabels = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
      }

      // Size all three canvases identically
      [baseCanvas, drawCanvas, cursorCanvas].forEach(c => {
        c.width  = w;
        c.height = h;
      });

      // Restore label data if same dimensions (switching views on same session)
      if (savedLabels && savedLabels.width === w && savedLabels.height === h) {
        drawCtx.putImageData(savedLabels, 0, 0);
      }

      // Draw base image
      baseCtx.drawImage(img, 0, 0);

      // Invalidate rect cache since canvas may have reflowed
      cachedRect = null;

      updateStats();
      showCanvases();
    };
    img.src = url + (url.includes('?') ? '&' : '?') + '_cb=' + Date.now();
  }

  function showCanvases() {
    document.getElementById('label-placeholder').style.display = 'none';
    document.getElementById('label-canvas-wrap').style.display = 'inline-block';
  }

  // ── Tool / class selection ───────────────────────────────
  function setClass(cls) {
    activeClass = cls;
    document.querySelectorAll('.label-class-btn').forEach(b => {
      b.classList.remove('active-ice','active-not-ice','active-void');
    });
    const map = { ice:'active-ice', not_ice:'active-not-ice', void:'active-void' };
    document.getElementById(`lbl-class-${cls}`)?.classList.add(map[cls]);
    updateCursor();
    updateStatusBar();
  }

  function setTool(tool) {
    activeTool = tool;
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`lbl-tool-${tool}`)?.classList.add('active');

    // Update canvas area class for CSS cursor rules
    const area = document.getElementById('label-canvas-area');
    area.className = `label-canvas-area tool-${tool}`;

    // Show/hide tool-specific controls
    document.getElementById('brush-controls').style.display = tool === TOOLS.BRUSH  ? 'block' : 'none';
    document.getElementById('poly-controls').style.display  = tool === TOOLS.POLY   ? 'block' : 'none';

    // Cancel any in-progress polygon when switching tools
    if (tool !== TOOLS.POLY) cancelPoly();

    updateCursor();
    updateStatusBar();
  }

  // ── Brush radius ─────────────────────────────────────────
  function setBrushRadius(r) {
    brushRadius = parseInt(r);
    document.getElementById('lbl-brush-val').textContent = r + 'px';
    updateBrushPreview();
    updateCursor();
  }

  function updateBrushPreview() {
    const preview = document.getElementById('brush-preview-dot');
    const wrap    = document.getElementById('brush-preview-wrap');
    if (!preview || !wrap) return;
    // Scale dot inside 36px preview box — max display radius = 16px
    const displayR = Math.min(16, brushRadius * 0.6);
// TEMP DEBUG — remove after checking console
console.log('First session object:', JSON.stringify(d.sessions[0], null, 2));    const size = Math.max(3, displayR * 2);
    preview.style.width  = size + 'px';
    preview.style.height = size + 'px';
    preview.style.background = CLASSES[activeClass]?.color || 'var(--text-sec)';
  }

  // ── Opacity ──────────────────────────────────────────────
  function setOpacity(val) {
    labelOpacity = parseFloat(val);
    drawCanvas.style.opacity = labelOpacity;
    document.getElementById('lbl-opacity-val').textContent = Math.round(labelOpacity * 100) + '%';
  }

  // ── View selection ───────────────────────────────────────
  function setLabelView(view) {
    labelView = view;
    document.querySelectorAll('.lbl-view-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`lvbtn-${view}`)?.classList.add('active');
    refreshBaseImage(true);
  }

  function refreshBaseImage(force) {
    // Pull URLs from the shared procResults / currentSession globals
    let url = null;
    if (labelView === 'rgb' && typeof currentSession !== 'undefined' && currentSession) {
      const rgb = currentSession.sensors?.find(s => s.sensor_type === 'rgb');
      if (rgb) url = `/api/images/${rgb.s3_path}`;
    } else if (typeof procResults !== 'undefined') {
      const map = {
        'ir-mapped':    procResults.irMapped,
        'radar-mapped': procResults.radarMapped,
        'road-mask':    procResults.roadMask,
        'segments':     procResults.segments,
        'heatmap':      procResults.heatmap,
        'overlay':      procResults.overlayRGB,
      };
      url = map[labelView] || null;
    }
    if (url) loadBaseImage(url, force);
    else {
      document.getElementById('label-placeholder').style.display = 'flex';
      document.getElementById('label-canvas-wrap').style.display = 'none';
    }
  }

  // ── Mouse events ─────────────────────────────────────────
  function getRect() {
    // Cache the bounding rect — invalidated on window resize
    if (!cachedRect) cachedRect = drawCanvas.getBoundingClientRect();
    return cachedRect;
  }

  function canvasXY(e) {
    const rect   = getRect();
    const scaleX = drawCanvas.width  / rect.width;
    const scaleY = drawCanvas.height / rect.height;
    return [
      Math.floor((e.clientX - rect.left) * scaleX),
      Math.floor((e.clientY - rect.top)  * scaleY)
    ];
  }

  function onMouseDown(e) {
    if (e.button !== 0) return;
    const [x, y] = canvasXY(e);

    if (activeTool === TOOLS.BRUSH) {
      pushHistory();
      isPainting = true;
      lastX = x; lastY = y;
      paintCircle(x, y);
    } else if (activeTool === TOOLS.POLY) {
      handlePolyClick(x, y);
    } else if (activeTool === TOOLS.BUCKET) {
      pushHistory();
      bucketFill(x, y);
    }
  }

  function onMouseMove(e) {
    const [x, y] = canvasXY(e);
    updateCoord(x, y);

    if (activeTool === TOOLS.BRUSH) {
      pendingCursor = [x, y];
      if (isPainting) pendingMove = [x, y];
      if (!rafPending) {
        rafPending = true;
        requestAnimationFrame(flushBrushMove);
      }
    } else if (activeTool === TOOLS.POLY) {
      polyHoverPt = [x, y];
      drawPolyPreview();
    } else if (activeTool === TOOLS.BUCKET) {
      clearCursor();
    }
  }

  function flushBrushMove() {
    rafPending = false;
    // Paint stroke
    if (isPainting && pendingMove) {
      const [x, y] = pendingMove;
      pendingMove = null;
      paintLine(lastX, lastY, x, y);
      lastX = x; lastY = y;
    }
    // Update cursor preview — only once per frame regardless of mousemove rate
    if (pendingCursor) {
      const [cx, cy] = pendingCursor;
      pendingCursor = null;
      drawCursorPreview(cx, cy);
    }
  }

  function onMouseUp(e) {
    if (activeTool === TOOLS.BRUSH && isPainting) {
      isPainting = false;
      scheduleUpdateStats();
    }
  }

  function onMouseLeave() {
    isPainting = false;
    polyHoverPt = null;
    clearCursor();
    updateCoord(null, null);
    if (activeTool === TOOLS.POLY) drawPolyPreview();
  }

  // ── Keyboard shortcuts ───────────────────────────────────
  function onKeyDown(e) {
    // Only active when labeling tab is shown
    if (!document.getElementById('panel-labeling')?.classList.contains('active')) return;

    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

    switch (e.key) {
      case '1': setClass('ice');     break;
      case '2': setClass('not_ice'); break;
      case '3': setClass('void');    break;
      case 'b': case 'B': setTool(TOOLS.BRUSH);  break;
      case 'p': case 'P': setTool(TOOLS.POLY);   break;
      case 'f': case 'F': setTool(TOOLS.BUCKET); break;
      case 'Escape': cancelPoly(); break;
      case 'Enter':  if (activeTool === TOOLS.POLY) closePoly(); break;
      case 'z':
        if (e.ctrlKey || e.metaKey) { e.preventDefault(); e.shiftKey ? redo() : undo(); }
        break;
      case '[': setBrushRadius(Math.max(2,  brushRadius - 4)); syncBrushSlider(); break;
      case ']': setBrushRadius(Math.min(120, brushRadius + 4)); syncBrushSlider(); break;
    }
  }

  function syncBrushSlider() {
    const sl = document.getElementById('lbl-brush-radius');
    if (sl) sl.value = brushRadius;
    updateBrushPreview();
  }

  // ── Painting ─────────────────────────────────────────────
  function classColor() {
    const [r, g, b] = CLASSES[activeClass].rgba;
    return `rgba(${r},${g},${b},1)`; // full alpha on draw canvas (opacity set on element)
  }

  function paintCircle(x, y) {
    // Use the same stroke path as paintLine so a click looks the same as a drag
    drawCtx.globalCompositeOperation = 'source-over';
    drawCtx.strokeStyle = classColor();
    drawCtx.lineWidth   = brushRadius * 2;
    drawCtx.lineCap     = 'round';
    drawCtx.beginPath();
    drawCtx.moveTo(x, y);
    drawCtx.lineTo(x, y);
    drawCtx.stroke();
  }

  function paintLine(x0, y0, x1, y1) {
    // Single stroked path — one draw call instead of N arc+fill calls
    drawCtx.globalCompositeOperation = 'source-over';
    drawCtx.strokeStyle = classColor();
    drawCtx.lineWidth   = brushRadius * 2;
    drawCtx.lineCap     = 'round';
    drawCtx.lineJoin    = 'round';
    drawCtx.beginPath();
    drawCtx.moveTo(x0, y0);
    drawCtx.lineTo(x1, y1);
    drawCtx.stroke();
  }

  // ── Brush cursor preview ─────────────────────────────────
  function drawCursorPreview(x, y) {
    clearCursor();
    const c = CLASSES[activeClass].color;
    cursorCtx.strokeStyle = c;
    cursorCtx.lineWidth   = 1.5;
    cursorCtx.globalAlpha = 0.8;
    cursorCtx.beginPath();
    cursorCtx.arc(x, y, brushRadius, 0, Math.PI * 2);
    cursorCtx.stroke();

    // Center dot
    cursorCtx.fillStyle   = c;
    cursorCtx.globalAlpha = 0.9;
    cursorCtx.beginPath();
    cursorCtx.arc(x, y, 2, 0, Math.PI * 2);
    cursorCtx.fill();
    cursorCtx.globalAlpha = 1;
  }

  function clearCursor() {
    cursorCtx.clearRect(0, 0, cursorCanvas.width, cursorCanvas.height);
  }

  // ── Polygon tool ─────────────────────────────────────────
  function handlePolyClick(x, y) {
    if (polyVertices.length >= 3) {
      // Check if close to first vertex to close polygon
      const [fx, fy] = polyVertices[0];
      if (Math.hypot(x - fx, y - fy) < POLY_CLOSE_DIST) {
        closePoly();
        return;
      }
    }
    polyVertices.push([x, y]);
    updatePolyCount();
    drawPolyPreview();
  }

  function drawPolyPreview() {
    clearCursor();
    if (polyVertices.length === 0) return;

    const c = CLASSES[activeClass].color;
    cursorCtx.strokeStyle = c;
    cursorCtx.lineWidth   = 1.5;
    cursorCtx.setLineDash([5, 4]);
    cursorCtx.globalAlpha = 0.85;

    cursorCtx.beginPath();
    cursorCtx.moveTo(...polyVertices[0]);
    for (let i = 1; i < polyVertices.length; i++) {
      cursorCtx.lineTo(...polyVertices[i]);
    }

    // Line to hover point
    if (polyHoverPt) {
      cursorCtx.lineTo(...polyHoverPt);
    }
    cursorCtx.stroke();
    cursorCtx.setLineDash([]);

    // Draw vertices
    polyVertices.forEach(([vx, vy], i) => {
      cursorCtx.fillStyle = i === 0 && polyVertices.length >= 3 ? '#fff' : c;
      cursorCtx.globalAlpha = 0.9;
      cursorCtx.beginPath();
      cursorCtx.arc(vx, vy, i === 0 ? 5 : 3, 0, Math.PI * 2);
      cursorCtx.fill();
    });

    // Snap indicator on first vertex
    if (polyVertices.length >= 3 && polyHoverPt) {
      const [fx, fy] = polyVertices[0];
      if (Math.hypot(polyHoverPt[0] - fx, polyHoverPt[1] - fy) < POLY_CLOSE_DIST) {
        cursorCtx.strokeStyle = '#fff';
        cursorCtx.lineWidth   = 2;
        cursorCtx.globalAlpha = 0.6;
        cursorCtx.beginPath();
        cursorCtx.arc(fx, fy, 9, 0, Math.PI * 2);
        cursorCtx.stroke();
      }
    }

    cursorCtx.globalAlpha = 1;
  }

  function closePoly() {
    if (polyVertices.length < 3) return;
    pushHistory();

    drawCtx.globalCompositeOperation = 'source-over';
    drawCtx.fillStyle = classColor();
    drawCtx.beginPath();
    drawCtx.moveTo(...polyVertices[0]);
    for (let i = 1; i < polyVertices.length; i++) {
      drawCtx.lineTo(...polyVertices[i]);
    }
    drawCtx.closePath();
    drawCtx.fill();

    cancelPoly();
    updateStats();
  }

  function cancelPoly() {
    polyVertices = [];
    polyHoverPt  = null;
    updatePolyCount();
    clearCursor();
  }

  function updatePolyCount() {
    const el = document.getElementById('poly-vertex-count');
    if (el) el.textContent = polyVertices.length > 0
      ? `${polyVertices.length} vert${polyVertices.length !== 1 ? 'ices' : 'ex'} — click first to close`
      : 'click to place vertices';
  }

  // ── Bucket fill ──────────────────────────────────────────
  // Region-aware: if segment data is available, fills the clicked region.
  // Otherwise falls back to flood fill on the draw canvas.
  function bucketFill(x, y) {
    if (procResults?.segments) {
      // Try region fill using the segments image
      regionFill(x, y);
    } else {
      floodFill(x, y);
    }
    updateStats();
  }

  function regionFill(clickX, clickY) {
    const doFill = (off) => {
      const offCtx = off.getContext('2d');
      const segData = offCtx.getImageData(0, 0, off.width, off.height).data;

      // Sample the clicked pixel's color as the "region color"
      const idx = (clickY * off.width + clickX) * 4;
      const tr = segData[idx], tg = segData[idx+1], tb = segData[idx+2];

      // Tolerance for color matching (segments have distinct colors)
      const TOL = 20;

      // Paint all pixels matching that region color
      drawCtx.globalCompositeOperation = 'source-over';
      drawCtx.fillStyle = classColor();

      const drawData = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height);
      const [lr, lg, lb, la] = CLASSES[activeClass].rgba;

      for (let py = 0; py < off.height; py++) {
        let runStart = -1;
        for (let px2 = 0; px2 <= off.width; px2++) {
          const si = (py * off.width + px2) * 4;
          const match = px2 < off.width &&
            Math.abs(segData[si]   - tr) < TOL &&
            Math.abs(segData[si+1] - tg) < TOL &&
            Math.abs(segData[si+2] - tb) < TOL;

          if (match && runStart === -1) {
            runStart = px2;
          } else if (!match && runStart !== -1) {
            // Fill run [runStart, px2) on row py
            for (let rx = runStart; rx < px2; rx++) {
              const di = (py * drawCanvas.width + rx) * 4;
              drawData.data[di]   = lr;
              drawData.data[di+1] = lg;
              drawData.data[di+2] = lb;
              drawData.data[di+3] = la;
            }
            runStart = -1;
          }
        }
      }
      drawCtx.putImageData(drawData, 0, 0);
      updateStats();
    };

    // Use cached segment canvas if available, otherwise fetch and cache
    if (segmentCanvas && segmentCanvasUrl === procResults.segments) {
      doFill(segmentCanvas);
    } else {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const off = document.createElement('canvas');
        off.width  = drawCanvas.width;
        off.height = drawCanvas.height;
        off.getContext('2d').drawImage(img, 0, 0, off.width, off.height);
        segmentCanvas    = off;
        segmentCanvasUrl = procResults.segments;
        doFill(off);
      };
      img.src = procResults.segments;
    }
  }

  function floodFill(startX, startY) {
    const w = drawCanvas.width, h = drawCanvas.height;
    const imageData = drawCtx.getImageData(0, 0, w, h);
    const data = imageData.data;

    const idx = (startY * w + startX) * 4;
    const tr = data[idx], tg = data[idx+1], tb = data[idx+2], ta = data[idx+3];

    const [fr, fg, fb, fa] = CLASSES[activeClass].rgba;

    // Don't fill if already this color
    if (tr === fr && tg === fg && tb === fb && ta === fa) return;

    const TOL = 30;
    const matches = (i) =>
      Math.abs(data[i]   - tr) <= TOL &&
      Math.abs(data[i+1] - tg) <= TOL &&
      Math.abs(data[i+2] - tb) <= TOL &&
      Math.abs(data[i+3] - ta) <= TOL;

    const stack = [[startX, startY]];
    const visited = new Uint8Array(w * h);

    while (stack.length) {
      const [cx, cy] = stack.pop();
      if (cx < 0 || cx >= w || cy < 0 || cy >= h) continue;
      const ci = cy * w + cx;
      if (visited[ci]) continue;
      const di = ci * 4;
      if (!matches(di)) continue;

      visited[ci] = 1;
      data[di]   = fr; data[di+1] = fg; data[di+2] = fb; data[di+3] = fa;

      stack.push([cx+1,cy],[cx-1,cy],[cx,cy+1],[cx,cy-1]);
    }
    drawCtx.putImageData(imageData, 0, 0);
  }

  // ── History (undo/redo) ───────────────────────────────────
  // Lazy snapshot: we mark a checkpoint dirty on mousedown but only do the
  // expensive getImageData readback when undo is actually requested.
  // This removes the GPU stall from the hot painting path entirely.
  let pendingSnapshot = false;  // true = need to snapshot before next undo

  function pushHistory() {
    // Just mark that the next undo should capture current state first.
    // The actual getImageData is deferred until undo() is called.
    pendingSnapshot = true;
    // Trim redo history so redo is no longer valid after new stroke
    history = history.slice(0, historyIdx + 1);
    historyIdx = history.length - 1;
    updateUndoButtons();
  }

  function commitSnapshot() {
    if (!drawCanvas || !pendingSnapshot) return;
    pendingSnapshot = false;
    history = history.slice(0, historyIdx + 1);
    history.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
    if (history.length > MAX_HISTORY) history.shift();
    historyIdx = history.length - 1;
  }

  function undo() {
    // Commit any in-flight stroke as a snapshot first so we can undo it
    commitSnapshot();
    if (historyIdx <= 0) {
      drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
      historyIdx = -1;
    } else {
      historyIdx--;
      drawCtx.putImageData(history[historyIdx], 0, 0);
    }
    updateStats();
    updateUndoButtons();
  }

  function redo() {
    if (historyIdx >= history.length - 1) return;
    historyIdx++;
    drawCtx.putImageData(history[historyIdx], 0, 0);
    updateStats();
    updateUndoButtons();
  }

  function updateUndoButtons() {
    const undoBtn = document.getElementById('lbl-undo');
    const redoBtn = document.getElementById('lbl-redo');
    if (undoBtn) undoBtn.disabled = historyIdx < 0;
    if (redoBtn) redoBtn.disabled = historyIdx >= history.length - 1;
  }

  // ── Label stats ──────────────────────────────────────────
  let statsDebounceTimer = null;
  function scheduleUpdateStats() {
    clearTimeout(statsDebounceTimer);
    statsDebounceTimer = setTimeout(updateStats, 600);
  }

  function updateStats() {
    if (!drawCanvas || drawCanvas.width === 0) return;

    const data  = drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height).data;
    const total = drawCanvas.width * drawCanvas.height;
    const counts = { ice: 0, not_ice: 0, void: 0, unlabeled: 0 };

    for (let i = 0; i < data.length; i += 4) {
      const a = data[i+3];
      if (a < 10) { counts.unlabeled++; continue; }
      const r = data[i], g = data[i+1], b = data[i+2];
      // Match against class colors
      if      (Math.abs(r - 77) < 30  && b > 200)  counts.ice++;
      else if (r > 200 && Math.abs(g - 168) < 30)   counts.not_ice++;
      else if (Math.abs(r - 144) < 30 && Math.abs(b - 208) < 30) counts.void++;
      else counts.unlabeled++;
    }

    const pct = (n) => total > 0 ? ((n / total) * 100).toFixed(1) : '0.0';

    const statMap = [
      { key: 'ice',      id: 'stat-ice',      color: '#4db8ff', count: counts.ice },
      { key: 'not_ice',  id: 'stat-not-ice',  color: '#e8a838', count: counts.not_ice },
      { key: 'void',     id: 'stat-void',      color: '#9060d0', count: counts.void },
      { key: 'unlabeled',id: 'stat-unlabeled', color: '#40406a', count: counts.unlabeled },
    ];

    statMap.forEach(({ id, color, count }) => {
      const valEl  = document.getElementById(`${id}-val`);
      const barEl  = document.getElementById(`${id}-bar`);
      const p = pct(count);
      if (valEl) valEl.textContent = p + '%';
      if (barEl) { barEl.style.width = p + '%'; barEl.style.background = color; }
    });
  }

  // ── UI helpers ───────────────────────────────────────────
  function updateCursor() {
    updateBrushPreview();
  }

  function updateStatusBar() {
    const el = document.getElementById('lbl-status-txt');
    if (!el) return;
    const toolNames = { brush: 'Brush', poly: 'Polygon Fill', bucket: 'Region Fill' };
    el.textContent = `${toolNames[activeTool] || activeTool}  ·  ${CLASSES[activeClass]?.name || activeClass}`;
  }

  function updateCoord(x, y) {
    const el = document.getElementById('lbl-coord');
    if (!el) return;
    el.textContent = x !== null ? `${x}, ${y}` : '';
  }

  // ── Clear all labels ─────────────────────────────────────
  function clearAll() {
    if (!confirm('Clear all labels on this image?')) return;
    pushHistory();
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
    updateStats();
  }

  // ── Import modal ─────────────────────────────────────────
  async function openImportModal() {
  const modal   = document.getElementById('lbl-import-modal');
  const loading = document.getElementById('lbl-import-loading');
  const list    = document.getElementById('lbl-import-list');

  // Reset state
  loading.style.display = 'block';
  list.style.display    = 'none';
  list.innerHTML        = '';
  modal.classList.add('open');

  try {
    const r = await fetch('/api/sessions?limit=50');
    const d = await r.json();

    loading.style.display = 'none';
    list.style.display    = 'flex'; // flex so gap works between items

    if (!d.sessions?.length) {
      list.innerHTML = '<div class="lbl-import-loading">No captures found.</div>';
      return;
    }

    d.sessions.forEach(s => {
      console.log('First session object:', JSON.stringify(d.sessions[0], null, 2));
      const isCurrent = s.session_id === currentSessionId;
      const captured  = s.captured_sensors || [];
      const hasRgb    = captured.includes('rgb');
      const hasIr     = captured.includes('ir');
      const hasRadar  = captured.includes('radar');
      const ts        = s.created_at
        ? new Date(s.created_at).toLocaleString('en-US', {
            month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit', hour12: false
          })
        : '—';

      const row = document.createElement('div');
      row.className = 'lbl-import-item' + (isCurrent ? ' current' : '');
      row.innerHTML = `
        <div class="lbl-import-meta">
          <div class="lbl-import-id">${s.capture_id || s.session_id}</div>
          <div class="lbl-import-time">${ts} · session: ${s.session_id.substring(0, 8)}…</div>
        </div>
        <div class="lbl-import-chips">
          <span class="lbl-import-chip ${hasRgb   ? 'has-rgb'   : ''}">RGB</span>
          <span class="lbl-import-chip ${hasIr    ? 'has-ir'    : ''}">IR</span>
          <span class="lbl-import-chip ${hasRadar ? 'has-radar' : ''}">RADAR</span>
        </div>
        ${isCurrent
          ? '<span style="font-family:var(--mono);font-size:10px;color:var(--text-dim);flex-shrink:0">current</span>'
          : `<button class="lbl-import-load-btn" onclick="LBL.importSession('${s.session_id}')">Load →</button>`
        }`;
      list.appendChild(row);
    });

  } catch (e) {
    loading.textContent = 'Failed to load captures.';
    console.error(e);
  }
}

  function closeImportModal() {
    document.getElementById('lbl-import-modal')?.classList.remove('open');
  }

  async function importSession(sessionId) {
    closeImportModal();
    try {
      const r = await fetch(`/api/sessions/${sessionId}`);
      const s = await r.json();

      // Update the shared globals so procResults and currentSession are in sync
      // We do this by calling back into capture.js's loadSession if available,
      // otherwise update directly
      if (typeof loadSession === 'function') {
        await loadSession(sessionId, true);
      } else {
        // Fallback: update globals directly
        window.currentSession   = s;
        window.currentSessionId = sessionId;
        window.procResults = {
          roadMask: null, irMapped: null, radarMapped: null,
          segments: null, featuresDF: null, confidenceMap: null,
          overlayRGB: null, heatmap: null, regions: null,
          numRegions: 0, regionFeatures: {}
        };
        reset();
        // Try to populate procResults from session data if proc results stored
        if (s.proc_results) {
          Object.assign(window.procResults, s.proc_results);
        }
        refreshBaseImage(true);
      }
    } catch(e) {
      alert('Failed to load capture: ' + e.message);
    }
  }

  // ── Public API ───────────────────────────────────────────
  return {
    init,
    reset,
    setClass,
    setTool,
    setBrushRadius,
    setOpacity,
    setLabelView,
    refreshBaseImage: (force) => refreshBaseImage(force),
    openImportModal,
    closeImportModal,
    importSession,
    closePoly,
    cancelPoly,
    undo,
    redo,
    clearAll,
    updateStats,
    TOOLS,
  };
})();

// ── Called when labeling tab becomes active ──────────────
function onLabelingTabActivated() {
  LBL.refreshBaseImage(true);
  LBL.updateStats();
}


// ── Label save stub (backend engineer to implement) ──────
function onLabelSave() {
  if (!currentSessionId) { alert('No session loaded.'); return; }
  const btn = document.getElementById('lbl-save-btn');
  btn.textContent = '… Saving';
  btn.disabled = true;
  // TODO: backend engineer — serialize drawCanvas pixel data and POST to save endpoint
  // Example:
  // const canvas = document.getElementById('label-draw-canvas');
  // canvas.toBlob(blob => {
  //   const fd = new FormData();
  //   fd.append('session_id', currentSessionId);
  //   fd.append('label_mask', blob, 'labels.png');
  //   fetch('/api/labels/save', { method: 'POST', body: fd })
  //     .then(r => r.json()).then(() => { btn.textContent = '✓ Saved'; btn.disabled = false; });
  // }, 'image/png');
  setTimeout(() => { btn.textContent = '▲ Save Labels'; btn.disabled = false; }, 800);
}

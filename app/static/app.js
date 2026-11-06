'use strict';

// ---------------------------------------------------------------- 状态

const STATUS_TEXT = {
  pending: '待检测',
  detecting: '检测中…',
  ok: '已检测',
  failed: '需手动调整',
  accepted: '已接受',
  rejected: '已丢弃',
  error: '导入失败',
};

const state = { items: [], selectedId: null };
const view = { item: null, img: null, dragging: null, dragMode: null, moveStart: null };

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const el = {
  fileInput: document.getElementById('fileInput'),
  fileList: document.getElementById('fileList'),
  countBadge: document.getElementById('countBadge'),
  editorTitle: document.getElementById('editorTitle'),
  metaInfo: document.getElementById('metaInfo'),
  regionChips: document.getElementById('regionChips'),
  btnRedetect: document.getElementById('btnRedetect'),
  btnRotate: document.getElementById('btnRotate'),
  btnAccept: document.getElementById('btnAccept'),
  btnReject: document.getElementById('btnReject'),
  btnAcceptAll: document.getElementById('btnAcceptAll'),
  btnClear: document.getElementById('btnClear'),
  canvasWrap: document.getElementById('canvasWrap'),
  emptyHint: document.getElementById('emptyHint'),
  statusText: document.getElementById('statusText'),
  cropPreview: document.getElementById('cropPreview'),
  resultGrid: document.getElementById('resultGrid'),
  resultCount: document.getElementById('resultCount'),
  btnExport: document.getElementById('btnExport'),
  exportStatus: document.getElementById('exportStatus'),
  exportResult: document.getElementById('exportResult'),
  optFormat: document.getElementById('optFormat'),
  optQuality: document.getElementById('optQuality'),
  qualityValue: document.getElementById('qualityValue'),
  optMaxEdge: document.getElementById('optMaxEdge'),
  optWmText: document.getElementById('optWmText'),
  optWmPos: document.getElementById('optWmPos'),
  optWmOpacity: document.getElementById('optWmOpacity'),
  opacityValue: document.getElementById('opacityValue'),
  optLabel: document.getElementById('optLabel'),
  optTemplate: document.getElementById('optTemplate'),
  optOutDir: document.getElementById('optOutDir'),
  optSidecar: document.getElementById('optSidecar'),
};

// ---------------------------------------------------------------- 基础工具

function setStatus(text, kind) {
  el.statusText.textContent = text || '';
  el.statusText.className = 'status' + (kind ? ' ' + kind : '');
}

function setExportStatus(text, kind) {
  el.exportStatus.textContent = text || '';
  el.exportStatus.className = 'status' + (kind ? ' ' + kind : '');
}

async function api(path, body) {
  const response = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  let data = {};
  try { data = await response.json(); } catch (error) { data = {}; }
  if (!response.ok) {
    const detail = data.detail;
    throw new Error(typeof detail === 'string' ? detail : `请求失败（${response.status}）`);
  }
  return data;
}

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function pixelScale() {
  const width = canvas.getBoundingClientRect().width || 1;
  return canvas.width ? canvas.width / width : 1;
}

function defaultQuad(item) {
  const width = item.width || 1000;
  const height = item.height || 1000;
  const padX = Math.round(width * 0.15);
  const padY = Math.round(height * 0.15);
  return [
    [padX, padY],
    [width - padX, padY],
    [width - padX, height - padY],
    [padX, height - padY],
  ];
}

// ---------------------------------------------------------------- 列表渲染

function renderList() {
  el.fileList.innerHTML = '';
  el.countBadge.textContent = String(state.items.length);

  if (!state.items.length) {
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder';
    placeholder.textContent = '还没有导入图片';
    el.fileList.appendChild(placeholder);
    return;
  }

  state.items.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'file-item' + (item.id === state.selectedId ? ' active' : '');
    row.dataset.status = item.status;

    const thumb = document.createElement('img');
    thumb.className = 'file-thumb';
    thumb.alt = '';
    if (item.url) thumb.src = item.url;

    const main = document.createElement('div');
    main.className = 'file-main';

    const name = document.createElement('div');
    name.className = 'file-name';
    name.textContent = item.name;
    name.title = item.name;

    const status = document.createElement('div');
    status.className = 'file-state';
    status.textContent = STATUS_TEXT[item.status] || item.status;
    if (item.status === 'accepted' && item.cropW) {
      status.textContent += ` · ${item.cropW}×${item.cropH}`;
    }

    main.appendChild(name);
    main.appendChild(status);

    const dot = document.createElement('span');
    dot.className = 'dot';

    row.appendChild(thumb);
    row.appendChild(main);
    row.appendChild(dot);
    row.addEventListener('click', () => selectItem(item.id));
    el.fileList.appendChild(row);
  });
}

// ---------------------------------------------------------------- 选择与预览

function selectItem(id) {
  const item = state.items.find((entry) => entry.id === id);
  if (!item) return;
  state.selectedId = id;
  view.item = item;
  renderList();
  el.editorTitle.textContent = item.name;
  el.cropPreview.innerHTML = '';
  updateMeta();
  renderChips();

  if (!item.url) {
    view.img = null;
    canvas.style.display = 'none';
    el.emptyHint.hidden = item.status !== 'error';
    setStatus(item.error || '该图片导入失败', 'error');
    return;
  }

  if (item.cropUrl) showCropPreview(item);
  loadPreview(item);
}

function loadPreview(item) {
  const image = new Image();
  image.onload = () => {
    view.img = image;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.style.display = 'block';
    el.emptyHint.hidden = true;
    drawCanvas();
    updateMeta();
  };
  image.onerror = () => setStatus('预览图加载失败', 'error');
  image.src = item.url;
}

function showCropPreview(item) {
  el.cropPreview.innerHTML = '';
  const image = document.createElement('img');
  image.src = item.cropUrl;
  image.alt = '裁剪结果';
  const caption = document.createElement('span');
  caption.textContent = item.cropW ? `成品 ${item.cropW}×${item.cropH}` : '成品';
  el.cropPreview.appendChild(image);
  el.cropPreview.appendChild(caption);
}

function updateMeta() {
  const item = view.item;
  if (!item) {
    el.metaInfo.textContent = '';
    return;
  }
  const parts = [];
  if (item.width) parts.push(`${item.width}×${item.height}`);
  if (typeof item.confidence === 'number') parts.push(`置信度 ${(item.confidence * 100).toFixed(0)}%`);
  if (item.detectMs != null) parts.push(`检测 ${item.detectMs}ms`);
  if (item.reason === 'no_region') parts.push('未找到区域，请手动框选');
  if (item.reason === 'low_confidence') parts.push('置信度偏低，建议检查四角');
  if (item.edited) parts.push('已手动调整');
  if (item.rotate) parts.push(`输出旋转 ${item.rotate}°`);
  el.metaInfo.textContent = parts.join(' · ');
}

function renderChips() {
  el.regionChips.innerHTML = '';
  const item = view.item;
  if (!item || !item.regions || item.regions.length < 2) return;
  item.regions.forEach((region, index) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip' + (item.regionIndex === index ? ' active' : '');
    chip.textContent = `区域 ${index + 1}`;
    chip.addEventListener('click', () => {
      item.quad = region.quad.map(([x, y]) => [Number(x), Number(y)]);
      item.confidence = region.confidence;
      item.regionIndex = index;
      item.edited = true;
      resetAcceptedState(item);
      drawCanvas();
      renderChips();
      renderList();
      updateMeta();
    });
    el.regionChips.appendChild(chip);
  });
}

// ---------------------------------------------------------------- 画布绘制

function drawCanvas() {
  const item = view.item;
  if (view.img) {
    const width = view.img.naturalWidth;
    const height = view.img.naturalHeight;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(view.img, 0, 0, canvas.width, canvas.height);
  }

  if (!item || !item.quad || !view.img) return;

  const scale = canvas.width / item.width;
  const points = item.quad.map(([x, y]) => [x * scale, y * scale]);
  const dpr = pixelScale();

  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(37, 99, 235, 0.10)';
  ctx.fill();
  ctx.lineWidth = Math.max(1.5, 2 * dpr);
  ctx.strokeStyle = '#2563eb';
  ctx.stroke();

  const radius = Math.max(4, 7 * dpr);
  points.forEach(([x, y], index) => {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, 2 * dpr);
    ctx.strokeStyle = '#2563eb';
    ctx.stroke();

    ctx.fillStyle = '#2563eb';
    ctx.font = `${Math.max(10, 11 * dpr)}px "Segoe UI", "Microsoft YaHei", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(index + 1), x, y + radius * 1.9);
  });
}

// ---------------------------------------------------------------- 拖拽

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * (canvas.width / rect.width),
    (event.clientY - rect.top) * (canvas.height / rect.height),
  ];
}

function toOriginal(point) {
  const scale = canvas.width / view.item.width;
  return [point[0] / scale, point[1] / scale];
}

function nearestHandle(point) {
  const item = view.item;
  if (!item || !item.quad) return -1;
  const scale = canvas.width / item.width;
  const threshold = Math.max(6, 14 * pixelScale());
  let best = -1;
  let bestDistance = Infinity;
  item.quad.forEach(([x, y], index) => {
    const distance = Math.hypot(x * scale - point[0], y * scale - point[1]);
    if (distance < threshold && distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}

function pointInQuad(point) {
  const item = view.item;
  if (!item || !item.quad) return false;
  const scale = canvas.width / item.width;
  const polygon = item.quad.map(([x, y]) => [x * scale, y * scale]);
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersects = (yi > point[1]) !== (yj > point[1])
      && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi + 1e-9) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

function resetAcceptedState(item) {
  if (!item || item.status !== 'accepted') return;
  item.status = 'ok';
  item.cropUrl = null;
  item.cropW = 0;
  item.cropH = 0;
  el.cropPreview.innerHTML = '';
  renderResults();
}

canvas.addEventListener('pointerdown', (event) => {
  const item = view.item;
  if (!item || !item.quad) return;
  const point = canvasPoint(event);
  const handle = nearestHandle(point);

  if (handle >= 0) {
    view.dragMode = 'corner';
    view.dragging = handle;
  } else if (pointInQuad(point)) {
    view.dragMode = 'move';
    view.moveStart = {
      point: toOriginal(point),
      quad: item.quad.map(([x, y]) => [x, y]),
    };
  } else {
    return;
  }

  resetAcceptedState(item);
  canvas.setPointerCapture(event.pointerId);
  event.preventDefault();
});

canvas.addEventListener('pointermove', (event) => {
  const item = view.item;
  if (!item || !view.dragMode) return;
  const point = canvasPoint(event);
  const [x, y] = toOriginal(point);

  if (view.dragMode === 'corner') {
    item.quad[view.dragging] = [
      clamp(x, 0, item.width - 1),
      clamp(y, 0, item.height - 1),
    ];
  } else if (view.dragMode === 'move' && view.moveStart) {
    const start = view.moveStart;
    const xs = start.quad.map(([px]) => px);
    const ys = start.quad.map(([, py]) => py);
    let dx = x - start.point[0];
    let dy = y - start.point[1];
    dx = clamp(dx, -Math.min(...xs), item.width - 1 - Math.max(...xs));
    dy = clamp(dy, -Math.min(...ys), item.height - 1 - Math.max(...ys));
    item.quad = start.quad.map(([px, py]) => [px + dx, py + dy]);
  } else {
    return;
  }

  item.edited = true;
  drawCanvas();
  updateMeta();
});

function endDrag(event) {
  if (!view.dragMode) return;
  view.dragMode = null;
  view.dragging = null;
  view.moveStart = null;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  renderList();
}

canvas.addEventListener('pointerup', endDrag);
canvas.addEventListener('pointercancel', endDrag);

// ---------------------------------------------------------------- 上传与检测

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []).filter(Boolean);
  if (!files.length) return;

  setStatus(`正在导入 ${files.length} 张图片…`);
  const form = new FormData();
  files.forEach((file) => form.append('files', file));

  let data;
  try {
    const response = await fetch('/api/upload', { method: 'POST', body: form });
    data = await response.json();
    if (!response.ok) throw new Error(data.detail || '导入失败');
  } catch (error) {
    setStatus(`导入失败：${error.message}`, 'error');
    return;
  }

  const newItems = [];
  (data.items || []).forEach((entry) => {
    const item = {
      id: entry.id || `invalid-${Math.random().toString(36).slice(2, 8)}`,
      name: entry.name,
      url: entry.url || null,
      width: entry.width || 0,
      height: entry.height || 0,
      status: entry.error ? 'error' : 'pending',
      error: entry.error || null,
      quad: null,
      regions: [],
      regionIndex: 0,
      confidence: null,
      detectMs: null,
      reason: null,
      rotate: 0,
      cropUrl: null,
      cropW: 0,
      cropH: 0,
      edited: false,
    };
    state.items.push(item);
    if (!entry.error) newItems.push(item);
  });

  renderList();
  if (!state.selectedId && state.items.length) selectItem(state.items[0].id);

  for (let index = 0; index < newItems.length; index += 1) {
    setStatus(`正在检测 ${index + 1}/${newItems.length}…`);
    await detectItem(newItems[index], true);
  }
  setStatus(newItems.length ? `自动检测完成（${newItems.length} 张）` : '没有可检测的图片',
    newItems.length ? 'ok' : 'error');
}

async function detectItem(item, quiet) {
  if (!item || !item.id || item.status === 'error') return;
  item.status = 'detecting';
  item.reason = null;
  item.cropUrl = null;
  renderList();

  try {
    const data = await api('/api/detect', { image_id: item.id });
    item.regions = data.regions || [];
    item.reason = data.reason;
    item.detectMs = data.detect_ms;
    item.regionIndex = 0;

    if (item.regions.length) {
      item.quad = item.regions[0].quad.map(([x, y]) => [Number(x), Number(y)]);
      item.confidence = item.regions[0].confidence;
      item.status = data.status === 'ok' ? 'ok' : 'failed';
    } else {
      item.quad = defaultQuad(item);
      item.confidence = null;
      item.status = 'failed';
    }
    item.edited = false;
    if (!quiet) {
      setStatus(item.status === 'ok' ? '检测成功，可拖拽四角微调' : '未能可靠检测，已给出初始框，请手动调整',
        item.status === 'ok' ? 'ok' : 'error');
    }
  } catch (error) {
    item.status = 'failed';
    item.reason = 'error';
    item.quad = defaultQuad(item);
    setStatus(`检测失败：${error.message}`, 'error');
  }

  renderList();
  renderResults();
  if (state.selectedId === item.id) {
    drawCanvas();
    updateMeta();
    renderChips();
  }
}

// ---------------------------------------------------------------- 接受 / 丢弃

async function acceptItem(item, quiet) {
  if (!item || !item.quad || item.status === 'error') return false;
  try {
    const data = await api('/api/crop', {
      image_id: item.id,
      quad: item.quad,
      rotate: item.rotate || 0,
    });
    item.cropUrl = data.url;
    item.cropW = data.width;
    item.cropH = data.height;
    item.status = 'accepted';
    renderList();
    renderResults();
    if (state.selectedId === item.id) showCropPreview(item);
    if (!quiet) setStatus(`已接受：${item.name}`, 'ok');
    return true;
  } catch (error) {
    setStatus(`裁剪失败：${error.message}`, 'error');
    return false;
  }
}

function renderResults() {
  const accepted = state.items.filter((item) => item.status === 'accepted' && item.cropUrl);
  el.resultCount.textContent = String(accepted.length);
  el.resultGrid.innerHTML = '';

  if (!accepted.length) {
    const placeholder = document.createElement('p');
    placeholder.className = 'placeholder';
    placeholder.textContent = '点击“接受本图”后，成品会出现在这里';
    el.resultGrid.appendChild(placeholder);
    return;
  }

  accepted.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'result-card';
    const image = document.createElement('img');
    image.src = item.cropUrl;
    image.alt = item.name;
    image.title = '点击回到该图继续调整';
    image.addEventListener('click', () => selectItem(item.id));

    const caption = document.createElement('div');
    caption.className = 'cap';
    const title = document.createElement('strong');
    title.textContent = item.name;
    title.title = item.name;
    const size = document.createElement('span');
    size.textContent = `${item.cropW}×${item.cropH}`;
    caption.appendChild(title);
    caption.appendChild(size);

    card.appendChild(image);
    card.appendChild(caption);
    el.resultGrid.appendChild(card);
  });
}

// ---------------------------------------------------------------- 导出

function readExportOptions() {
  const maxEdgeRaw = el.optMaxEdge.value.trim();
  return {
    fmt: el.optFormat.value,
    quality: Number(el.optQuality.value),
    max_long_edge: maxEdgeRaw ? Number(maxEdgeRaw) : null,
    watermark_text: el.optWmText.value,
    watermark_position: el.optWmPos.value,
    watermark_opacity: Number(el.optWmOpacity.value) / 100,
    watermark_size_ratio: 0.04,
    label: el.optLabel.value,
    filename_template: el.optTemplate.value.trim() || '{name}_{index}',
    output_dir: el.optOutDir.value.trim() || null,
    write_sidecar: el.optSidecar.checked,
  };
}

function renderExportResult(data) {
  el.exportResult.innerHTML = '';

  const pathLine = document.createElement('div');
  pathLine.className = 'out-path';
  pathLine.textContent = `输出目录：${data.output_dir}`;
  el.exportResult.appendChild(pathLine);

  if (data.files.length) {
    const list = document.createElement('ul');
    data.files.forEach((file) => {
      const row = document.createElement('li');
      row.textContent = `${file.name}（${file.width}×${file.height}）`;
      list.appendChild(row);
    });
    el.exportResult.appendChild(list);
  }

  if (data.errors.length) {
    const list = document.createElement('ul');
    data.errors.forEach((error) => {
      const row = document.createElement('li');
      row.textContent = `失败：${error.image_id} - ${error.error}`;
      list.appendChild(row);
    });
    el.exportResult.appendChild(list);
  }
}

async function exportAll() {
  const accepted = state.items.filter((item) => item.status === 'accepted' && item.quad);
  if (!accepted.length) {
    setExportStatus('还没有已接受的图片，请先对每张图点“接受本图”', 'error');
    return;
  }

  el.btnExport.disabled = true;
  setExportStatus(`正在导出 ${accepted.length} 张…`);
  try {
    const data = await api('/api/export', {
      items: accepted.map((item) => ({
        image_id: item.id,
        quad: item.quad,
        rotate: item.rotate || 0,
      })),
      options: readExportOptions(),
    });
    renderExportResult(data);
    setExportStatus(
      `导出完成：${data.count} 张${data.errors.length ? `，失败 ${data.errors.length} 张` : ''}`,
      data.errors.length ? 'error' : 'ok'
    );
  } catch (error) {
    setExportStatus(`导出失败：${error.message}`, 'error');
  } finally {
    el.btnExport.disabled = false;
  }
}

// ---------------------------------------------------------------- 事件绑定

el.fileInput.addEventListener('change', () => {
  uploadFiles(el.fileInput.files);
  el.fileInput.value = '';
});

el.btnRedetect.addEventListener('click', async () => {
  if (!view.item) return;
  setStatus('正在重新检测…');
  await detectItem(view.item, false);
});

el.btnRotate.addEventListener('click', async () => {
  if (!view.item || view.item.status === 'error') return;
  const item = view.item;
  item.rotate = (item.rotate + 90) % 360;
  item.edited = true;
  updateMeta();
  setStatus(`输出将顺时针旋转 ${item.rotate}°`);
  if (item.status === 'accepted') await acceptItem(item, true);
  renderList();
});

el.btnAccept.addEventListener('click', () => {
  if (view.item) acceptItem(view.item, false);
});

el.btnReject.addEventListener('click', () => {
  const item = view.item;
  if (!item || item.status === 'error') return;
  item.status = 'rejected';
  item.cropUrl = null;
  renderList();
  renderResults();
  setStatus(`已丢弃：${item.name}`);
});

el.btnAcceptAll.addEventListener('click', async () => {
  const targets = state.items.filter((item) => item.quad && item.status !== 'rejected' && item.status !== 'error');
  if (!targets.length) {
    setStatus('没有可接受的图片', 'error');
    return;
  }
  for (let index = 0; index < targets.length; index += 1) {
    setStatus(`正在接受 ${index + 1}/${targets.length}…`);
    await acceptItem(targets[index], true);
  }
  setStatus(`已接受 ${targets.filter((item) => item.status === 'accepted').length} 张`, 'ok');
});

el.btnClear.addEventListener('click', () => {
  state.items = [];
  state.selectedId = null;
  view.item = null;
  view.img = null;
  view.dragMode = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas.removeAttribute('width');
  canvas.removeAttribute('height');
  canvas.style.display = 'none';
  el.emptyHint.hidden = false;
  el.editorTitle.textContent = '未选择图片';
  el.metaInfo.textContent = '';
  el.cropPreview.innerHTML = '';
  el.regionChips.innerHTML = '';
  el.exportResult.innerHTML = '';
  setExportStatus('');
  setStatus('列表已清空（已导出的文件不会被删除）');
  renderList();
  renderResults();
});

el.btnExport.addEventListener('click', exportAll);

el.optQuality.addEventListener('input', () => {
  el.qualityValue.textContent = el.optQuality.value;
});
el.optWmOpacity.addEventListener('input', () => {
  el.opacityValue.textContent = `${el.optWmOpacity.value}%`;
});

document.addEventListener('dragover', (event) => {
  event.preventDefault();
  el.canvasWrap.classList.add('dragging');
});
document.addEventListener('dragleave', (event) => {
  if (event.relatedTarget === null) el.canvasWrap.classList.remove('dragging');
});
document.addEventListener('drop', (event) => {
  event.preventDefault();
  el.canvasWrap.classList.remove('dragging');
  if (event.dataTransfer && event.dataTransfer.files.length) {
    uploadFiles(event.dataTransfer.files);
  }
});

window.addEventListener('resize', () => {
  if (view.img) drawCanvas();
});

// ---------------------------------------------------------------- 初始化

renderList();
renderResults();
setStatus('就绪：导入图片后会自动检测');

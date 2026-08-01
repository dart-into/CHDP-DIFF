const MODEL_URL = "models/densenet169.onnx";
const SAMPLES_URL = "assets/samples.json";
const LABEL_MAP_URL = "models/label_map.json";
const ORT_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const STABLE_SAMPLE_IDS = new Set([
  "danshen/group000003",
  "shengdihuang/409143",
  "wenshanyao/409101",
]);

const state = {
  samples: [],
  selected: null,
  labels: [],
  session: null,
  busy: false,
};

const el = {
  runBtn: document.getElementById("runBtn"),
  status: document.getElementById("status"),
  sampleGrid: document.getElementById("sampleGrid"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  confidence: document.getElementById("confidence"),
  boxCount: document.getElementById("boxCount"),
  runtime: document.getElementById("runtime"),
  resultCanvas: document.getElementById("resultCanvas"),
  emptyResult: document.getElementById("emptyResult"),
  heatmapImage: document.getElementById("heatmapImage"),
  panelImage: document.getElementById("panelImage"),
  flowImage: document.getElementById("flowImage"),
};

function setStatus(text) {
  el.status.textContent = text;
}

function labelDisplay(label) {
  const map = {
    chenpi: "Chenpi",
    danggui: "Danggui",
    danshen: "Danshen",
    fubaishao: "Fubaishao",
    fuchaobaizhu: "Fuchaobaizhu",
    fuling: "Fuling",
    huangqi: "Huangqi",
    shengdihuang: "Shengdihuang",
    wenshanyao: "Wenshanyao",
    yiyiren: "Yiyiren",
  };
  return map[label] || label || "Unknown";
}

function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
    img.src = src;
  });
}

async function ensureSession() {
  if (state.session) return state.session;
  if (!window.ort) {
    throw new Error("ONNX Runtime Web failed to load. Check network access to jsdelivr.");
  }
  ort.env.wasm.wasmPaths = ORT_WASM_BASE;
  setStatus("Loading ONNX model...");
  state.session = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  return state.session;
}

function preprocess(img, box) {
  const sourceW = img.naturalWidth;
  const sourceH = img.naturalHeight;
  const x = Math.max(0, Math.min(sourceW - 1, Math.round(box.x || 0)));
  const y = Math.max(0, Math.min(sourceH - 1, Math.round(box.y || 0)));
  const w = Math.max(1, Math.min(sourceW - x, Math.round(box.w || sourceW)));
  const h = Math.max(1, Math.min(sourceH - y, Math.round(box.h || sourceH)));

  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, x, y, w, h, 0, 0, 224, 224);
  const pixels = ctx.getImageData(0, 0, 224, 224).data;

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  const data = new Float32Array(1 * 3 * 224 * 224);
  for (let i = 0; i < 224 * 224; i++) {
    const r = pixels[i * 4] / 255;
    const g = pixels[i * 4 + 1] / 255;
    const b = pixels[i * 4 + 2] / 255;
    data[i] = (r - mean[0]) / std[0];
    data[224 * 224 + i] = (g - mean[1]) / std[1];
    data[2 * 224 * 224 + i] = (b - mean[2]) / std[2];
  }
  return new ort.Tensor("float32", data, [1, 3, 224, 224]);
}

async function classifyBox(session, img, box) {
  const input = preprocess(img, box);
  const feeds = {};
  feeds[session.inputNames[0]] = input;
  const output = await session.run(feeds);
  const logits = output[session.outputNames[0]].data;
  const probs = softmax(logits);
  let bestIndex = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIndex]) bestIndex = i;
  }
  const label = state.labels[bestIndex] || String(bestIndex);
  return {
    label,
    labelDisplay: labelDisplay(label),
    confidence: probs[bestIndex],
    probs,
  };
}

function drawResult(img, sample, results) {
  const canvas = el.resultCanvas;
  const wrapper = canvas.parentElement.getBoundingClientRect();
  const scale = Math.min(wrapper.width / img.naturalWidth, wrapper.height / img.naturalHeight);
  const drawW = Math.max(1, Math.round(img.naturalWidth * scale));
  const drawH = Math.max(1, Math.round(img.naturalHeight * scale));
  canvas.width = Math.round(wrapper.width);
  canvas.height = Math.round(wrapper.height);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#16202a";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  const ox = Math.round((canvas.width - drawW) / 2);
  const oy = Math.round((canvas.height - drawH) / 2);
  ctx.drawImage(img, ox, oy, drawW, drawH);

  ctx.lineWidth = 2;
  ctx.font = "13px system-ui, sans-serif";
  results.forEach((item) => {
    const box = item.box;
    const x = ox + box.x * scale;
    const y = oy + box.y * scale;
    const w = box.w * scale;
    const h = box.h * scale;
    const text = `${item.labelDisplay} ${(item.confidence * 100).toFixed(1)}%`;
    const labelW = ctx.measureText(text).width + 8;
    ctx.strokeStyle = "#00dc50";
    ctx.fillStyle = "#00dc50";
    ctx.strokeRect(x, y, w, h);
    ctx.fillRect(x, Math.max(0, y - 22), labelW, 20);
    ctx.fillStyle = "#000";
    ctx.fillText(text, x + 4, Math.max(14, y - 7));
  });
}

function renderSamples() {
  el.sampleGrid.innerHTML = "";
  state.samples.forEach((sample) => {
    const card = document.createElement("button");
    card.className = `sample-card${sample.id === state.selected?.id ? " active" : ""}`;
    card.type = "button";
    card.innerHTML = `
      <div class="thumbs">
        <img src="${sample.beforeUrl}" alt="Before ${sample.name}">
        <img src="${sample.afterUrl}" alt="After ${sample.name}">
      </div>
      <div>
        <strong>${sample.labelDisplay}</strong>
        <span>${sample.name}</span>
        <em>${sample.note || "curated sample pair"}</em>
      </div>
    `;
    card.addEventListener("click", () => {
      state.selected = sample;
      renderSamples();
      updateSampleView(sample);
      runInference();
    });
    el.sampleGrid.appendChild(card);
  });
}

function updateSampleView(sample) {
  el.heatmapImage.src = sample.heatmapUrl;
  el.panelImage.src = sample.panelUrl;
  el.boxCount.textContent = String(sample.boxes?.length || 0);
  el.resultTitle.textContent = `${sample.labelDisplay} sample selected`;
  el.resultMeta.textContent = "Click Run ONNX inference or choose another pair.";
}

async function runInference() {
  if (state.busy) return;
  if (!state.selected) {
    setStatus("Select a sample first.");
    return;
  }
  try {
    state.busy = true;
    el.runBtn.disabled = true;
    setStatus("Running ONNX/WASM inference in browser...");
    const started = performance.now();
    const session = await ensureSession();
    const img = await loadImage(state.selected.afterUrl);
    const boxes = (state.selected.boxes || []).slice(0, 4);
    if (!boxes.length) throw new Error("No detection boxes are available for this sample.");

    const results = [];
    for (const box of boxes) {
      const pred = await classifyBox(session, img, box);
      results.push({ ...pred, box });
    }
    const elapsed = performance.now() - started;
    const top = results.reduce((best, item) => (item.confidence > best.confidence ? item : best), results[0]);

    drawResult(img, state.selected, results);
    el.emptyResult.style.display = "none";
    el.resultTitle.textContent = `${top.labelDisplay} detected`;
    el.resultMeta.textContent = "Top label uses the highest-confidence region. Classification is computed in the browser.";
    el.confidence.textContent = `${(top.confidence * 100).toFixed(1)}%`;
    el.boxCount.textContent = String(results.length);
    el.runtime.textContent = `${Math.round(elapsed)} ms`;
    setStatus("ONNX/WASM inference complete.");
  } catch (err) {
    setStatus(err.message);
  } finally {
    state.busy = false;
    el.runBtn.disabled = false;
  }
}

async function init() {
  const [sampleData, labelMap] = await Promise.all([
    fetch(SAMPLES_URL).then((r) => r.json()),
    fetch(LABEL_MAP_URL).then((r) => r.json()),
  ]);
  state.samples = (sampleData.samples || []).filter((sample) => STABLE_SAMPLE_IDS.has(sample.id));
  state.labels = Object.keys(labelMap.idx2name)
    .sort((a, b) => Number(a) - Number(b))
    .map((key) => labelMap.idx2name[key]);
  state.selected = state.samples[0] || null;
  renderSamples();
  if (state.selected) updateSampleView(state.selected);
  setStatus("Ready. ONNX model loads on first run.");
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.ref) {
      document.querySelectorAll("[data-ref]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      el.flowImage.src = button.dataset.ref;
    }
    if (button.dataset.visual) {
      document.querySelectorAll("[data-visual]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".visual-img").forEach((img) => img.classList.remove("active"));
      document.getElementById(`${button.dataset.visual}Image`).classList.add("active");
    }
  });
});

el.runBtn.addEventListener("click", runInference);
init().catch((err) => setStatus(err.message));

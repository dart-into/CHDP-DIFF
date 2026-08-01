const SAMPLES_URL = "assets/samples.json";

const state = {
  samples: [],
  selected: null,
};

const el = {
  sampleGrid: document.getElementById("sampleGrid"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  confidence: document.getElementById("confidence"),
  boxCount: document.getElementById("boxCount"),
  resultImage: document.getElementById("resultImage"),
  heatmapImage: document.getElementById("heatmapImage"),
  panelImage: document.getElementById("panelImage"),
};

function renderSamples() {
  el.sampleGrid.innerHTML = "";
  state.samples.forEach((sample) => {
    const card = document.createElement("button");
    card.className = `sample-card${sample.id === state.selected?.id ? " active" : ""}`;
    card.type = "button";
    card.setAttribute("aria-label", sample.labelDisplay);
    card.innerHTML = `
      <img src="${sample.beforeUrl}" alt="Before">
      <img src="${sample.afterUrl}" alt="After">
    `;
    card.addEventListener("click", () => {
      state.selected = sample;
      renderSamples();
      updateSampleView(sample);
    });
    el.sampleGrid.appendChild(card);
  });
}

function updateSampleView(sample) {
  el.resultImage.src = sample.resultUrl;
  el.resultImage.alt = `Precomputed result for ${sample.labelDisplay}`;
  el.heatmapImage.src = sample.heatmapUrl;
  el.panelImage.src = sample.panelUrl;
  el.resultTitle.textContent = `${sample.predictedDisplay || sample.labelDisplay} detected`;
  el.resultMeta.textContent = "Local Python pipeline output";
  el.confidence.textContent = `${((sample.confidence || 0) * 100).toFixed(1)}%`;
  el.boxCount.textContent = String(sample.boxCount || 0);
}

async function init() {
  const sampleData = await fetch(SAMPLES_URL).then((r) => r.json());
  state.samples = sampleData.samples || [];
  state.selected = state.samples[0] || null;
  renderSamples();
  if (state.selected) updateSampleView(state.selected);
}

init().catch((err) => {
  el.sampleGrid.textContent = err.message;
});

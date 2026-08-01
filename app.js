const SAMPLES_URL = "assets/samples.json";

const state = {
  samples: [],
  selected: null,
};

const el = {
  status: document.getElementById("status"),
  sampleGrid: document.getElementById("sampleGrid"),
  resultTitle: document.getElementById("resultTitle"),
  resultMeta: document.getElementById("resultMeta"),
  confidence: document.getElementById("confidence"),
  boxCount: document.getElementById("boxCount"),
  resultImage: document.getElementById("resultImage"),
  heatmapImage: document.getElementById("heatmapImage"),
  panelImage: document.getElementById("panelImage"),
  differenceImage: document.getElementById("differenceImage"),
  registrationImage: document.getElementById("registrationImage"),
  notice: document.getElementById("notice"),
};

function setStatus(text) {
  el.status.textContent = text;
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
        <em>${sample.note || "precomputed sample pair"}</em>
      </div>
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
  el.resultImage.alt = `Precomputed result for ${sample.id}`;
  el.heatmapImage.src = sample.heatmapUrl;
  el.panelImage.src = sample.panelUrl;
  el.differenceImage.src = sample.differenceUrl;
  el.registrationImage.src = sample.registrationUrl;
  el.resultTitle.textContent = `${sample.predictedDisplay || sample.labelDisplay} detected`;
  el.resultMeta.textContent = `${sample.labelDisplay} / ${sample.name}. ${sample.note || "Local Python result."}`;
  el.confidence.textContent = `${((sample.confidence || 0) * 100).toFixed(1)}%`;
  el.boxCount.textContent = String(sample.boxCount || 0);
  setStatus(`${state.samples.length} examples`);
}

async function init() {
  const sampleData = await fetch(SAMPLES_URL).then((r) => r.json());
  state.samples = sampleData.samples || [];
  state.selected = state.samples[0] || null;
  if (el.notice) {
    el.notice.textContent = sampleData.notice || "";
  }
  renderSamples();
  if (state.selected) updateSampleView(state.selected);
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    if (button.dataset.visual) {
      document.querySelectorAll("[data-visual]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      document.querySelectorAll(".visual-img").forEach((img) => img.classList.remove("active"));
      document.getElementById(`${button.dataset.visual}Image`).classList.add("active");
    }
  });
});

init().catch((err) => setStatus(err.message));

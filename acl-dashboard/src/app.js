/**
 * TriPlane Health — Knee MRI Diagnostic Workstation Frontend Logic
 * Connects to FastAPI Inference API over HTTP JSON
 */

let API_BASE = localStorage.getItem("triplane_api_base") || "http://localhost:8000";

// Application State
const state = {
  activeExamId: "1172",
  activeGroundTruth: 1,
  activeGroundTruthLabel: "Tear (Positive)",
  currentFilter: "all",
  availableCases: [],
  // Volume storage for active exam (base64 strings)
  volumes: {
    axial: null,
    coronal: null,
    sagittal: null,
  },
  // Comparison results from /predict/compare
  comparisonResults: [],
  // GradCAM parameters
  camModel: "proposed",
  camPlane: "axial",
  camSliceIdx: 12,
  isAnalyzing: false,
  isComputingCam: false,
};

// Paper benchmark AUC reference values (from Table 11 of Research Dissertation)
const PAPER_BENCHMARKS = {
  proposed: 0.954,
  transfer_learning: 0.892,
  custom_cnn: 0.827,
};

const MODEL_DISPLAY_NAMES = {
  proposed: "Proposed Hybrid (DenseNet121 + Custom CNN + CBAM)",
  transfer_learning: "Transfer Learning (DenseNet121 Only)",
  custom_cnn: "Custom CNN (Trained from Scratch)",
};

// ============================================================================
// 1. INITIALIZATION & HEALTH CHECK
// ============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  initEventListeners();
  checkApiHealth();
  // Periodically check health every 10 seconds
  setInterval(checkApiHealth, 10000);

  // Load available cases from server
  await loadAvailableCases();
  // Automatically analyze default exam (1172 - ACL Tear)
  await selectCase("1172", 1, "Tear (Positive)");
  updateAuditBadge();
});

async function checkApiHealth() {
  const badge = document.getElementById("api-status-badge");
  const dot = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  const latency = document.getElementById("latency-pill");

  const t0 = performance.now();
  try {
    const res = await fetch(`${API_BASE}/health`, { method: "GET" });
    const elapsed = Math.round(performance.now() - t0);

    if (res.ok) {
      const data = await res.json();
      dot.className = "pulse-dot online";
      text.textContent = `API Online (${data.models_loaded.length} models)`;
      latency.textContent = `${elapsed} ms`;
    } else {
      throw new Error(`HTTP ${res.status}`);
    }
  } catch (err) {
    dot.className = "pulse-dot error";
    text.textContent = "API Disconnected";
    latency.textContent = "Offline";
  }
}

// ============================================================================
// 2. CASE QUEUE MANAGEMENT
// ============================================================================

async function loadAvailableCases() {
  const container = document.getElementById("case-list-container");
  const countBadge = document.getElementById("queue-case-count");

  try {
    const res = await fetch(`${API_BASE}/cases`);
    if (!res.ok) throw new Error("Failed to load cases");

    const data = await res.json();
    state.availableCases = data.cases || [];
    countBadge.textContent = `${state.availableCases.length} Held-Out Exams`;
    renderCaseList();
  } catch (err) {
    console.warn("Could not fetch case list from API, using default presets", err);
    // Fallback preset list of confirmed held-out exams
    state.availableCases = [
      { exam_id: "1172", ground_truth: 1, ground_truth_label: "Tear (Positive)" },
      { exam_id: "1174", ground_truth: 1, ground_truth_label: "Tear (Positive)" },
      { exam_id: "1130", ground_truth: 0, ground_truth_label: "Normal (Intact)" },
      { exam_id: "1131", ground_truth: 0, ground_truth_label: "Normal (Intact)" },
      { exam_id: "1181", ground_truth: 1, ground_truth_label: "Tear (Positive)" },
    ];
    countBadge.textContent = `${state.availableCases.length} Presets`;
    renderCaseList();
  }
}

function renderCaseList() {
  const container = document.getElementById("case-list-container");
  container.innerHTML = "";

  const filtered = state.availableCases.filter((c) => {
    if (state.currentFilter === "tear") return c.ground_truth === 1;
    if (state.currentFilter === "normal") return c.ground_truth === 0;
    return true;
  });

  if (filtered.length === 0) {
    container.innerHTML = `<div style="text-align:center; padding:1.5rem; color:var(--text-muted);">No cases match filter</div>`;
    return;
  }

  filtered.forEach((c) => {
    const isTear = c.ground_truth === 1;
    const card = document.createElement("div");
    card.className = `case-card ${state.activeExamId === c.exam_id ? "active" : ""}`;
    card.id = `case-card-${c.exam_id}`;
    card.innerHTML = `
      <div class="case-card-header">
        <span class="case-card-title">Exam #${c.exam_id}</span>
        <span class="gt-pill ${isTear ? "gt-tear" : "gt-normal"}">${isTear ? "TEAR" : "NORMAL"}</span>
      </div>
      <div class="case-card-sub">Held-out validation volume (3 planes)</div>
    `;
    card.addEventListener("click", () => selectCase(c.exam_id, c.ground_truth, c.ground_truth_label));
    container.appendChild(card);
  });
}

async function selectCase(examId, groundTruth, gtLabel) {
  state.activeExamId = examId;
  state.activeGroundTruth = groundTruth;
  state.activeGroundTruthLabel = gtLabel;

  // Update UI headers
  document.getElementById("display-exam-id").textContent = `Exam #${examId}`;
  const gtBadge = document.getElementById("display-gt-badge");
  gtBadge.textContent = `GT: ${gtLabel}`;
  gtBadge.className = `ground-truth-badge ${groundTruth === 1 ? "gt-tear" : "gt-normal"}`;

  // Update active state in queue list
  document.querySelectorAll(".case-card").forEach((el) => el.classList.remove("active"));
  const activeEl = document.getElementById(`case-card-${examId}`);
  if (activeEl) activeEl.classList.add("active");

  // Fetch 3 plane volumes for this case from /cases/{exam_id}
  setLoadingState(true);
  try {
    const res = await fetch(`${API_BASE}/cases/${examId}`);
    if (!res.ok) throw new Error("Could not fetch exam volumes");
    const data = await res.json();

    state.volumes.axial = data.axial;
    state.volumes.coronal = data.coronal;
    state.volumes.sagittal = data.sagittal;

    // Run prediction comparison across all 3 models
    await runMultiplaneComparison();
  } catch (err) {
    console.error("Error loading case volumes:", err);
    alert(`Failed to load case #${examId} from server: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

// ============================================================================
// 3. MULTIPLANE INFERENCE & COMPARISON
// ============================================================================

async function runMultiplaneComparison() {
  if (!state.volumes.axial || !state.volumes.coronal || !state.volumes.sagittal) {
    alert("Please ensure all 3 MRI planes (Axial, Coronal, Sagittal) are loaded.");
    return;
  }

  setLoadingState(true);
  const t0 = performance.now();

  try {
    const payload = {
      model: "proposed",
      axial: state.volumes.axial,
      coronal: state.volumes.coronal,
      sagittal: state.volumes.sagittal,
      threshold: 0.5,
      exam_id: state.activeExamId,
    };

    const res = await fetch(`${API_BASE}/predict/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }

    const results = await res.json();
    state.comparisonResults = results;

    const elapsed = Math.round(performance.now() - t0);
    renderDiagnosticResults(results, elapsed);
    updateAuditBadge();

    // Reset GradCAM visual container when new case is loaded
    resetGradCamDisplay();
  } catch (err) {
    console.error("Prediction failed:", err);
    alert(`Inference failed: ${err.message}`);
  } finally {
    setLoadingState(false);
  }
}

function renderDiagnosticResults(results, elapsedMs) {
  // Find Proposed Hybrid result
  const proposed = results.find((r) => r.model === "proposed") || results[0];
  if (!proposed) return;

  const prob = proposed.probability;
  const probPercent = (prob * 100).toFixed(1);
  const isTear = proposed.predicted_label === "tear";

  // 1. Update Hero Radial Gauge
  const gaugeFill = document.getElementById("gauge-bar-fill");
  const gaugeVal = document.getElementById("primary-prob-value");
  gaugeVal.textContent = `${probPercent}%`;

  // Circumference = 2 * PI * 50 = 314.159
  const offset = 314.159 * (1 - prob);
  gaugeFill.style.strokeDashoffset = offset;
  gaugeFill.style.stroke = isTear ? "var(--status-danger)" : "var(--status-success)";

  // 2. Verdict headline & triage badge
  const headline = document.getElementById("verdict-headline");
  const triageBadge = document.getElementById("display-triage-badge");

  if (prob >= 0.75) {
    headline.textContent = "HIGH CONFIDENCE ACL TEAR";
    headline.style.color = "var(--status-danger)";
    triageBadge.className = "triage-badge badge-high";
    triageBadge.textContent = "HIGH RISK (TEAR)";
  } else if (prob >= 0.40) {
    headline.textContent = "BORDERLINE / RADIOLOGIST REVIEW REQUIRED";
    headline.style.color = "var(--status-warning)";
    triageBadge.className = "triage-badge badge-review";
    triageBadge.textContent = "REVIEW REQUIRED";
  } else {
    headline.textContent = "ACL INTACT / NORMAL KNEE";
    headline.style.color = "var(--status-success)";
    triageBadge.className = "triage-badge badge-clear";
    triageBadge.textContent = "CLEAR / NORMAL";
  }

  document.getElementById("execution-time-badge").textContent = `${proposed.execution_time_ms || elapsedMs} ms`;

  // 3. Render Benchmark Comparison Table
  renderBenchmarkTable(results);

  // 4. Render Stage-2 Fusion Weights
  renderFusionWeights(proposed.fusion_weights);
}

function renderBenchmarkTable(results) {
  const tbody = document.getElementById("benchmark-table-body");
  tbody.innerHTML = "";

  const gt = state.activeGroundTruth;
  const hasGroundTruth = gt === 0 || gt === 1;
  const actualIsTear = gt === 1;
  const actualLabelText = hasGroundTruth ? (actualIsTear ? "TEAR" : "NORMAL") : "UNLABELED";

  results.forEach((r) => {
    const isProposed = r.model === "proposed";
    const isTear = r.predicted_label === "tear";
    const probPct = (r.probability * 100).toFixed(1);
    const paperAuc = PAPER_BENCHMARKS[r.model] ? PAPER_BENCHMARKS[r.model].toFixed(3) : "--";

    const tr = document.createElement("tr");
    if (isProposed) tr.className = "row-proposed";

    tr.innerHTML = `
      <td>
        <strong>${MODEL_DISPLAY_NAMES[r.model] || r.model}</strong>
        ${isProposed ? '<span class="model-badge-table">PROPOSED</span>' : ""}
      </td>
      <td>
        ${hasGroundTruth 
          ? `<span class="gt-pill ${actualIsTear ? "gt-tear" : "gt-normal"}">${actualLabelText}</span>` 
          : `<span class="badge">N/A</span>`}
      </td>
      <td>
        <span class="gt-pill ${isTear ? "gt-tear" : "gt-normal"}">
          ${isTear ? "TEAR" : "NORMAL"}
        </span>
      </td>
      <td class="mono-cell" style="font-weight:700; color:${isTear ? "var(--status-danger)" : "var(--status-success)"}">
        ${probPct}%
      </td>
      <td class="mono-cell">${r.plane_logits.axial.toFixed(2)}</td>
      <td class="mono-cell">${r.plane_logits.coronal.toFixed(2)}</td>
      <td class="mono-cell">${r.plane_logits.sagittal.toFixed(2)}</td>
      <td class="mono-cell" style="color:var(--accent-green); font-weight:600;">${paperAuc}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderFusionWeights(weights) {
  if (!weights) return;

  const axialW = weights.axial;
  const coronalW = weights.coronal;
  const sagittalW = weights.sagittal;
  const biasW = weights.bias;

  document.getElementById("weight-axial").textContent = `Weight: ${axialW >= 0 ? "+" : ""}${axialW.toFixed(4)}`;
  document.getElementById("weight-coronal").textContent = `Weight: ${coronalW >= 0 ? "+" : ""}${coronalW.toFixed(4)}`;
  document.getElementById("weight-sagittal").textContent = `Weight: ${sagittalW >= 0 ? "+" : ""}${sagittalW.toFixed(4)}`;
  document.getElementById("val-intercept").textContent = biasW.toFixed(4);

  // Normalize bar widths relative to max absolute weight
  const maxAbs = Math.max(Math.abs(axialW), Math.abs(coronalW), Math.abs(sagittalW), 1e-4);
  const wAxial = Math.min(100, Math.round((Math.abs(axialW) / maxAbs) * 100));
  const wCoronal = Math.min(100, Math.round((Math.abs(coronalW) / maxAbs) * 100));
  const wSagittal = Math.min(100, Math.round((Math.abs(sagittalW) / maxAbs) * 100));

  document.getElementById("bar-axial").style.width = `${wAxial}%`;
  document.getElementById("bar-coronal").style.width = `${wCoronal}%`;
  document.getElementById("bar-sagittal").style.width = `${wSagittal}%`;
}

// ============================================================================
// 4. GRAD-CAM ON-DEMAND EXPLAINABILITY
// ============================================================================

async function computeGradCam() {
  if (state.camModel === "custom_cnn") {
    alert("Grad-CAM is not available for Custom CNN (no attention mechanism). Please select Proposed Hybrid.");
    return;
  }

  const volumeB64 = state.volumes[state.camPlane];
  if (!volumeB64) {
    alert(`No volume loaded for ${state.camPlane} plane.`);
    return;
  }

  const spinner = document.getElementById("cam-spinner");
  const placeholder = document.getElementById("cam-placeholder");
  spinner.classList.remove("hidden");
  placeholder.classList.add("hidden");

  try {
    const payload = {
      model: state.camModel,
      plane: state.camPlane,
      slice_idx: state.camSliceIdx,
      target_layer: state.camModel === "proposed" ? "cbam" : "backbone",
      volume: volumeB64,
    };

    const res = await fetch(`${API_BASE}/gradcam`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errJson = await res.json();
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    renderGradCamImages(data);
  } catch (err) {
    console.error("GradCAM failed:", err);
    alert(`Failed to compute Grad-CAM: ${err.message}`);
    placeholder.classList.remove("hidden");
  } finally {
    spinner.classList.add("hidden");
  }
}

function renderGradCamImages(camData) {
  const imgOverlay = document.getElementById("img-cam-overlay");
  const imgOriginal = document.getElementById("img-cam-original");
  const imgHeatmap = document.getElementById("img-cam-heatmap");
  const hotspotTag = document.getElementById("hotspot-tag");
  const placeholder = document.getElementById("cam-placeholder");

  placeholder.classList.add("hidden");

  imgOverlay.src = camData.overlay_image;
  imgOverlay.classList.remove("hidden");

  imgOriginal.src = camData.original_image;
  imgOriginal.classList.remove("hidden");

  imgHeatmap.src = camData.heatmap_image;
  imgHeatmap.classList.remove("hidden");

  hotspotTag.textContent = `Peak Hotspot: (${camData.hotspot.x}, ${camData.hotspot.y}) • Confidence: ${(camData.confidence * 100).toFixed(1)}%`;
}

function resetGradCamDisplay() {
  const imgOverlay = document.getElementById("img-cam-overlay");
  const imgOriginal = document.getElementById("img-cam-original");
  const imgHeatmap = document.getElementById("img-cam-heatmap");
  const hotspotTag = document.getElementById("hotspot-tag");
  const placeholder = document.getElementById("cam-placeholder");

  imgOverlay.classList.add("hidden");
  imgOriginal.classList.add("hidden");
  imgHeatmap.classList.add("hidden");
  placeholder.classList.remove("hidden");
  hotspotTag.textContent = "Peak Hotspot: (--, --)";
}

// ============================================================================
// 5. MANUAL FILE UPLOAD HANDLING
// ============================================================================

function handleFileUpload(plane, file) {
  const nameEl = document.getElementById(`name-${plane}`);
  const iconEl = document.getElementById(`icon-${plane}`);
  const dropzone = document.getElementById(`drop-${plane}`);

  if (!file) return;

  nameEl.textContent = file.name;
  iconEl.textContent = "⏳";

  const reader = new FileReader();
  reader.onload = (e) => {
    // Convert ArrayBuffer to base64
    const bytes = new Uint8Array(e.target.result);
    let binary = "";
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    state.volumes[plane] = btoa(binary);

    iconEl.textContent = "✅";
    dropzone.classList.add("ready");
    checkAllUploadsReady();
  };
  reader.readAsArrayBuffer(file);
}

function checkAllUploadsReady() {
  const btnRun = document.getElementById("btn-run-analysis");
  if (state.volumes.axial && state.volumes.coronal && state.volumes.sagittal) {
    btnRun.disabled = false;
  } else {
    btnRun.disabled = true;
  }
}

// ============================================================================
// 6. AUDIT TRAIL MODAL
// ============================================================================

async function openAuditModal() {
  const modal = document.getElementById("modal-audit");
  const tbody = document.getElementById("audit-table-body");
  tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;">Loading audit trail from SQLite...</td></tr>`;
  modal.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/audit-logs?limit=50`);
    if (!res.ok) throw new Error("Failed to fetch audit logs");
    const data = await res.json();

    document.getElementById("audit-count").textContent = data.total_records;

    if (data.logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align:center; color:var(--text-muted);">No predictions logged yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    data.logs.forEach((log) => {
      const isTear = log.predicted_label === "tear";
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>#${log.id}</td>
        <td>${log.timestamp.slice(0, 19).replace("T", " ")}</td>
        <td><strong>${log.exam_id || "Anonymous"}</strong></td>
        <td>${log.model}</td>
        <td><span class="gt-pill ${isTear ? "gt-tear" : "gt-normal"}">${log.predicted_label.toUpperCase()}</span></td>
        <td style="font-weight:700;">${(log.probability * 100).toFixed(1)}%</td>
        <td>${log.plane_logits.axial.toFixed(2)}</td>
        <td>${log.plane_logits.coronal.toFixed(2)}</td>
        <td>${log.plane_logits.sagittal.toFixed(2)}</td>
        <td>${log.execution_time_ms.toFixed(1)}ms</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" style="color:var(--status-danger); text-align:center;">Failed to load audit logs: ${err.message}</td></tr>`;
  }
}

async function updateAuditBadge() {
  try {
    const res = await fetch(`${API_BASE}/audit-logs?limit=1`);
    if (res.ok) {
      const data = await res.json();
      document.getElementById("audit-count").textContent = data.total_records;
    }
  } catch (_) {}
}

// ============================================================================
// 7. EVENT LISTENERS
// ============================================================================

function initEventListeners() {
  // Filter pills
  document.getElementById("filter-all").addEventListener("click", () => setFilter("all"));
  document.getElementById("filter-tear").addEventListener("click", () => setFilter("tear"));
  document.getElementById("filter-normal").addEventListener("click", () => setFilter("normal"));

  // Manual file uploads
  ["axial", "coronal", "sagittal"].forEach((plane) => {
    const input = document.getElementById(`file-${plane}`);
    input.addEventListener("change", (e) => handleFileUpload(plane, e.target.files[0]));
  });

  // Run analysis button
  document.getElementById("btn-run-analysis").addEventListener("click", () => {
    state.activeExamId = "Custom Upload";
    state.activeGroundTruth = null;
    state.activeGroundTruthLabel = "Unlabeled";
    document.getElementById("display-exam-id").textContent = "Custom Exam";
    document.getElementById("display-gt-badge").textContent = "GT: Unlabeled";
    document.getElementById("display-gt-badge").className = "ground-truth-badge";
    runMultiplaneComparison();
  });

  // Grad-CAM Controls
  document.getElementById("select-cam-model").addEventListener("change", (e) => {
    state.camModel = e.target.value;
    document.getElementById("cam-model-badge").textContent =
      state.camModel === "proposed" ? "Proposed Hybrid" : "Transfer Learning";
  });

  ["axial", "coronal", "sagittal"].forEach((plane) => {
    document.getElementById(`btn-plane-${plane}`).addEventListener("click", () => {
      document.querySelectorAll(".plane-btn").forEach((b) => b.classList.remove("active"));
      document.getElementById(`btn-plane-${plane}`).classList.add("active");
      state.camPlane = plane;
    });
  });

  const slider = document.getElementById("slice-slider");
  slider.addEventListener("input", (e) => {
    state.camSliceIdx = parseInt(e.target.value, 10);
    document.getElementById("slice-index-display").textContent = `${state.camSliceIdx} / 24`;
  });

  document.getElementById("btn-generate-cam").addEventListener("click", computeGradCam);

  // Modals
  document.getElementById("btn-open-audit").addEventListener("click", openAuditModal);
  document.getElementById("btn-close-audit").addEventListener("click", () => {
    document.getElementById("modal-audit").classList.add("hidden");
  });

  document.getElementById("btn-open-settings").addEventListener("click", () => {
    document.getElementById("input-api-base").value = API_BASE;
    document.getElementById("modal-settings").classList.remove("hidden");
  });

  document.getElementById("btn-close-settings").addEventListener("click", () => {
    document.getElementById("modal-settings").classList.add("hidden");
  });

  document.getElementById("btn-save-settings").addEventListener("click", () => {
    const val = document.getElementById("input-api-base").value.trim().replace(/\/$/, "");
    if (val) {
      API_BASE = val;
      localStorage.setItem("triplane_api_base", API_BASE);
      document.getElementById("modal-settings").classList.add("hidden");
      checkApiHealth();
    }
  });
}

function setFilter(filter) {
  state.currentFilter = filter;
  document.querySelectorAll(".filter-pills .pill").forEach((p) => p.classList.remove("active"));
  document.getElementById(`filter-${filter}`).classList.add("active");
  renderCaseList();
}

function setLoadingState(loading) {
  state.isAnalyzing = loading;
  const btn = document.getElementById("btn-run-analysis");
  if (loading) {
    document.getElementById("verdict-headline").textContent = "Computing Multi-Plane Prediction...";
    document.getElementById("primary-prob-value").textContent = "...";
  }
}

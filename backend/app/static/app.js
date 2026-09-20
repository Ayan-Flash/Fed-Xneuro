/**
 * Fed-XNeuro Platform - Frontend Controller & Telemetry Streaming
 */

let currentSocket = null;
let activeSimId = null;
let telemetryData = {
  rounds: [],
  accuracies: [],
  losses: [],
};

// --- Initialization ---
document.addEventListener("DOMContentLoaded", () => {
  loadSimulations();
  drawBrainSlice("mriCanvas", 0.78);
  setInterval(loadSimulations, 4000);
});

// --- Tab Switching ---
function switchTab(tabId, btn) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

  const target = document.getElementById(tabId);
  if (target) target.classList.add("active");
  if (btn) btn.classList.add("active");
}

// --- Simulation Management ---
async function loadSimulations() {
  try {
    const res = await fetch("/api/v1/simulations?limit=50");
    if (!res.ok) return;
    const sims = await res.json();

    document.getElementById("statTotalSims").textContent = sims.length;
    const activeCount = sims.filter(s => s.status === "running" || s.status === "starting").length;
    document.getElementById("statActiveSims").textContent = activeCount;

    const tbody = document.getElementById("simulationsTableBody");
    if (sims.length === 0) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted);">No simulations found. Create one above!</td></tr>`;
      return;
    }

    tbody.innerHTML = sims.map(sim => {
      const statusBadgeClass = `badge-${sim.status || "created"}`;
      const accText = sim.final_accuracy !== null && sim.final_accuracy !== undefined 
        ? `${sim.final_accuracy.toFixed(1)}%` 
        : "-";

      return `
        <tr>
          <td><strong>#${sim.id}</strong></td>
          <td style="font-family: monospace; font-size: 0.8rem; color: var(--text-secondary);">${sim.run_id.slice(0, 16)}...</td>
          <td><strong>${sim.name}</strong></td>
          <td><span class="badge" style="background: rgba(99, 102, 241, 0.15); color: #818cf8;">${sim.algorithm.toUpperCase()}</span></td>
          <td>${sim.dataset}</td>
          <td>${sim.current_round} / ${sim.num_rounds}</td>
          <td><span class="badge ${statusBadgeClass}">${sim.status}</span></td>
          <td><strong>${accText}</strong></td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="btn btn-secondary btn-sm" onclick="viewTelemetry(${sim.id}, '${sim.name}', ${sim.num_rounds})">
                Live
              </button>
              <button class="btn btn-secondary btn-sm" onclick="viewClinicianReport(${sim.id})">
                Report
              </button>
              <button class="btn btn-danger btn-sm" onclick="deleteSimulation(${sim.id})">
                ✕
              </button>
            </div>
          </td>
        </tr>
      `;
    }).join("");

  } catch (err) {
    console.error("Failed to load simulations:", err);
  }
}

async function handleCreateSimulation(e) {
  e.preventDefault();

  const payload = {
    name: document.getElementById("simName").value,
    dataset: document.getElementById("simDataset").value,
    model: document.getElementById("simModel").value,
    algorithm: document.getElementById("simAlgorithm").value,
    num_clients: parseInt(document.getElementById("simClients").value),
    num_rounds: parseInt(document.getElementById("simRounds").value),
    local_epochs: parseInt(document.getElementById("simEpochs").value),
    batch_size: parseInt(document.getElementById("simBatch").value),
    learning_rate: 0.01,
  };

  try {
    // 1. Create Simulation
    const createRes = await fetch("/api/v1/simulations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!createRes.ok) throw new Error("Failed to create simulation");
    const sim = await createRes.json();

    // 2. Start Simulation
    await fetch(`/api/v1/simulations/${sim.id}/start`, { method: "POST" });

    // Switch to telemetry tab and listen
    loadSimulations();
    viewTelemetry(sim.id, sim.name, sim.num_rounds);
  } catch (err) {
    alert("Error starting simulation: " + err.message);
  }
}

async function deleteSimulation(id) {
  if (!confirm(`Delete simulation #${id}?`)) return;
  try {
    await fetch(`/api/v1/simulations/${id}`, { method: "DELETE" });
    loadSimulations();
  } catch (err) {
    console.error("Failed to delete simulation:", err);
  }
}

// --- WebSocket Live Telemetry ---
function viewTelemetry(simId, simName, totalRounds) {
  activeSimId = simId;
  telemetryData = { rounds: [], accuracies: [], losses: [] };

  document.getElementById("telemetryTitle").textContent = `📡 Live Telemetry: ${simName} (#${simId})`;
  document.getElementById("telemetryDesc").textContent = `Streaming round metrics in real-time over WebSocket /ws/simulations/${simId}`;
  document.getElementById("liveRound").textContent = `0 / ${totalRounds}`;

  // Switch tab
  const telBtn = document.querySelectorAll(".tab-btn")[1];
  switchTab("telemetryTab", telBtn);

  // Connect WebSocket
  connectWebSocket(simId);
}

function connectWebSocket(simId) {
  if (currentSocket) {
    currentSocket.close();
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/simulations/${simId}`;
  currentSocket = new WebSocket(wsUrl);

  const statusBadge = document.getElementById("wsStatusBadge");
  const statusText = document.getElementById("wsStatusText");

  currentSocket.onopen = () => {
    statusBadge.className = "status-badge";
    statusText.textContent = `Connected to #${simId}`;
  };

  currentSocket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.event === "round_complete") {
        handleRoundUpdate(msg);
      } else if (msg.event === "simulation_completed") {
        statusText.textContent = `Simulation Completed`;
        loadSimulations();
      }
    } catch (e) {
      console.error("WS Parse error:", e);
    }
  };

  currentSocket.onclose = () => {
    statusText.textContent = "Standby";
  };
}

function handleRoundUpdate(msg) {
  const round = msg.round;
  const acc = msg.accuracy;
  const loss = msg.loss;
  const eps = msg.privacy_budget_spent;

  document.getElementById("liveRound").textContent = `Round ${round}`;
  document.getElementById("liveAccuracy").textContent = `${acc.toFixed(1)}%`;
  document.getElementById("liveLoss").textContent = loss.toFixed(4);
  if (eps !== undefined && eps !== null) {
    document.getElementById("liveEpsilon").textContent = `ε = ${eps.toFixed(2)}`;
  }

  telemetryData.rounds.push(round);
  telemetryData.accuracies.push(acc);
  telemetryData.losses.push(loss);

  drawChart("accuracyChart", telemetryData.rounds, telemetryData.accuracies, "Accuracy (%)", "#06b6d4");
  drawChart("lossChart", telemetryData.rounds, telemetryData.losses, "Loss", "#f43f5e");
}

// --- Clinician View ---
async function viewClinicianReport(simId) {
  try {
    const res = await fetch(`/api/v1/results/${simId}/clinician-dashboard`);
    if (!res.ok) throw new Error("Failed to fetch report");
    const data = await res.json();
    const report = data.report;

    const prob = report.risk_probability || 0.65;
    const pct = (prob * 100).toFixed(1);
    const category = report.risk_category || "MODERATE";

    const riskCard = document.getElementById("riskCardContainer");
    riskCard.className = `risk-card ${category.toLowerCase()}`;
    document.getElementById("riskScore").textContent = `${pct}%`;
    document.getElementById("riskCategory").textContent = `${category} RISK`;
    document.getElementById("patientIdLabel").textContent = `Patient: ${report.patient_id || "PAT_0001"}`;

    // Render SHAP Bars
    const barsContainer = document.getElementById("shapBarsContainer");
    const items = report.clinical_importance || [];
    barsContainer.innerHTML = items.map(item => `
      <div class="feature-bar-row">
        <div class="feature-name">${item.feature}</div>
        <div class="feature-bar-container">
          <div class="feature-bar-fill" style="width: ${Math.min(100, item.relative_pct * 2.5)}%;"></div>
        </div>
        <div class="feature-pct">${item.relative_pct.toFixed(1)}%</div>
      </div>
    `).join("");

    // Render Brain Heatmap
    drawBrainSlice("mriCanvas", prob);

    // ASCII report
    document.getElementById("asciiReport").textContent = data.ascii || "";

    // Switch Tab
    const clinBtn = document.querySelectorAll(".tab-btn")[2];
    switchTab("clinicianTab", clinBtn);

  } catch (err) {
    alert("Could not load clinician report: " + err.message);
  }
}

// --- Canvas Chart Drawing Utility ---
function drawChart(canvasId, xVals, yVals, label, color) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width = canvas.parentElement.clientWidth;
  const h = canvas.height = canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, w, h);
  if (xVals.length < 2) return;

  const padLeft = 45;
  const padBottom = 30;
  const padTop = 20;
  const padRight = 20;

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const minY = Math.min(...yVals) * 0.95;
  const maxY = Math.max(...yVals) * 1.05 || 1.0;

  // Grid lines
  ctx.strokeStyle = "rgba(51, 65, 85, 0.4)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH / 4) * i;
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
  }
  ctx.stroke();

  // Plot line
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();

  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw points
  ctx.fillStyle = color;
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

// --- 3D MRI Brain Slice Simulation Canvas ---
function drawBrainSlice(canvasId, riskLevel) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;

  ctx.fillStyle = "#030712";
  ctx.fillRect(0, 0, w, h);

  // Outer brain parenchymal contour
  ctx.fillStyle = "#1e293b";
  ctx.beginPath();
  ctx.ellipse(cx, cy, w * 0.40, h * 0.44, 0, 0, Math.PI * 2);
  ctx.fill();

  // Cortical gyri / sulci simulation lines
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(cx - 30, cy - 20, 45, 0.2, Math.PI * 0.9);
  ctx.arc(cx + 30, cy - 20, 45, 0.1, Math.PI * 0.8);
  ctx.stroke();

  // Ventricles (hypointense CSF center)
  ctx.fillStyle = "#0b0f19";
  ctx.beginPath();
  const vScale = 1.0 + riskLevel * 0.6;
  ctx.ellipse(cx, cy - 8, 14 * vScale, 28 * vScale, 0, 0, Math.PI * 2);
  ctx.fill();

  // Bilateral Hippocampal Atrophy ROI Heatmap Overlay
  const hippoColor = riskLevel > 0.7 ? "rgba(239, 68, 68, 0.85)" : (riskLevel > 0.3 ? "rgba(245, 158, 11, 0.85)" : "rgba(16, 185, 129, 0.85)");
  
  // Left Hippocampus
  const gradL = ctx.createRadialGradient(cx - 42, cy + 18, 2, cx - 42, cy + 18, 22);
  gradL.addColorStop(0, hippoColor);
  gradL.addColorStop(1, "transparent");
  ctx.fillStyle = gradL;
  ctx.beginPath();
  ctx.arc(cx - 42, cy + 18, 22, 0, Math.PI * 2);
  ctx.fill();

  // Right Hippocampus
  const gradR = ctx.createRadialGradient(cx + 42, cy + 18, 2, cx + 42, cy + 18, 22);
  gradR.addColorStop(0, hippoColor);
  gradR.addColorStop(1, "transparent");
  ctx.fillStyle = gradR;
  ctx.beginPath();
  ctx.arc(cx + 42, cy + 18, 22, 0, Math.PI * 2);
  ctx.fill();

  // Labels
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px sans-serif";
  ctx.fillText("L Hippocampus", cx - 75, cy + 46);
  ctx.fillText("R Hippocampus", cx + 18, cy + 46);
}

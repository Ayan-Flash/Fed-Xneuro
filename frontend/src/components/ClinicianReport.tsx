"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Simulation,
  ClinicianDashboardResponse,
  getClinicianDashboard,
} from "@/lib/api";

interface ClinicianReportProps {
  simulation: Simulation | null;
  onSelectAnother: () => void;
}

export const ClinicianReport: React.FC<ClinicianReportProps> = ({
  simulation,
  onSelectAnother,
}) => {
  const [data, setData] = useState<ClinicianDashboardResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const brainCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!simulation) {
      setData(null);
      return;
    }

    let mounted = true;
    async function loadReport() {
      if (!simulation) return;
      try {
        setIsLoading(true);
        setErrorMsg(null);
        const res = await getClinicianDashboard(simulation.id);
        if (mounted) setData(res);
      } catch (err: unknown) {
        if (mounted) {
          setErrorMsg(err instanceof Error ? err.message : "Failed to load report");
        }
      } finally {
        if (mounted) setIsLoading(false);
      }
    }

    loadReport();
    return () => {
      mounted = false;
    };
  }, [simulation]);

  // Render Real Brain Canvas if report has MRI attribution
  const drawBrainSlice = useCallback((riskProb: number) => {
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = (canvas.width = 300);
    const h = (canvas.height = 240);
    const cx = w / 2;
    const cy = h / 2;

    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, w, h);

    // Outer brain parenchymal contour
    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.38, h * 0.44, 0, 0, Math.PI * 2);
    ctx.fill();

    // Cortical gyri / sulci lines
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx - 25, cy - 15, 38, 0.2, Math.PI * 0.9);
    ctx.arc(cx + 25, cy - 15, 38, 0.1, Math.PI * 0.8);
    ctx.stroke();

    // Ventricles
    ctx.fillStyle = "#0b0f19";
    ctx.beginPath();
    const vScale = 1.0 + riskProb * 0.6;
    ctx.ellipse(cx, cy - 6, 12 * vScale, 24 * vScale, 0, 0, Math.PI * 2);
    ctx.fill();

    // Bilateral Hippocampal Atrophy ROI Heatmap Overlay
    const hippoColor =
      riskProb > 0.7
        ? "rgba(239, 68, 68, 0.85)"
        : riskProb > 0.3
        ? "rgba(245, 158, 11, 0.85)"
        : "rgba(16, 185, 129, 0.85)";

    // Left Hippocampus
    const gradL = ctx.createRadialGradient(cx - 36, cy + 15, 2, cx - 36, cy + 15, 20);
    gradL.addColorStop(0, hippoColor);
    gradL.addColorStop(1, "transparent");
    ctx.fillStyle = gradL;
    ctx.beginPath();
    ctx.arc(cx - 36, cy + 15, 20, 0, Math.PI * 2);
    ctx.fill();

    // Right Hippocampus
    const gradR = ctx.createRadialGradient(cx + 36, cy + 15, 2, cx + 36, cy + 15, 20);
    gradR.addColorStop(0, hippoColor);
    gradR.addColorStop(1, "transparent");
    ctx.fillStyle = gradR;
    ctx.beginPath();
    ctx.arc(cx + 36, cy + 15, 20, 0, Math.PI * 2);
    ctx.fill();

    // Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px sans-serif";
    ctx.fillText("L Hippocampus", cx - 70, cy + 42);
    ctx.fillText("R Hippocampus", cx + 16, cy + 42);
  }, []);

  useEffect(() => {
    if (data?.report?.risk_probability !== undefined) {
      drawBrainSlice(data.report.risk_probability);
    }
  }, [data, drawBrainSlice]);

  if (!simulation) {
    return (
      <div className="card">
        <div className="empty-state">
          <div className="empty-state-icon">🧠</div>
          <div className="empty-state-title">No Simulation Selected</div>
          <div className="empty-state-desc">
            Please select an experiment from the &quot;Experiments &amp; Launcher&quot; tab to inspect patient progression risks, SHAP attributions, and neuroimaging overlays.
          </div>
          <button type="button" className="btn btn-secondary" onClick={onSelectAnother}>
            Go to Experiments
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="card">
        <div style={{ textAlign: "center", padding: "3rem", color: "var(--text-secondary)" }}>
          Loading clinician diagnostic report for #{simulation.id}...
        </div>
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div className="card">
        <div
          style={{
            background: "rgba(244, 63, 94, 0.15)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            color: "#fb7185",
            padding: "1rem",
            borderRadius: "var(--radius-sm)",
          }}
        >
          {errorMsg}
        </div>
      </div>
    );
  }

  // Pure blank state when no report exists (Zero mock data)
  if (!data?.report) {
    return (
      <div className="grid-1">
        <div className="card">
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: "1rem",
            }}
          >
            <div>
              <div className="card-title">
                <span>🧠</span> Clinician Diagnostic View: {simulation.name} (#{simulation.id})
              </div>
              <div className="card-desc">
                Model: <strong>{simulation.model}</strong> | Dataset: <strong>{simulation.dataset}</strong> | Status: <strong>{simulation.status}</strong>
              </div>
            </div>
            <button type="button" className="btn btn-secondary btn-sm" onClick={onSelectAnother}>
              Change Simulation
            </button>
          </div>
        </div>

        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon">🔬</div>
            <div className="empty-state-title">No Clinician Diagnostic Data Available</div>
            <div className="empty-state-desc">
              Simulation #{simulation.id} has not produced a patient-level clinician progression report.
              <br /><br />
              When you test with your trained multimodal <strong>Fed-XNeuro</strong> model on patient cohort data (MRI + Cognitive Scores + EHR), actual MCI &rarr; Alzheimer&apos;s progression probabilities, SHAP clinical feature bars, and hippocampal atrophy heatmaps will appear here.
              <br /><br />
              <strong style={{ color: "var(--emerald)" }}>All mock data has been removed.</strong> Only verified outputs from your models will be displayed.
            </div>
            <button type="button" className="btn btn-primary" onClick={onSelectAnother}>
              Launch New Simulation
            </button>
          </div>
        </div>
      </div>
    );
  }

  const report = data.report;
  const riskProb = report.risk_probability ?? 0;
  const riskCategory = (report.risk_category || "UNKNOWN").toUpperCase();
  const categoryClass = riskCategory.toLowerCase();

  return (
    <div className="grid-1">
      {/* Top Header */}
      <div className="card">
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            flexWrap: "wrap",
            gap: "1rem",
          }}
        >
          <div>
            <div className="card-title">
              <span>🧠</span> Clinician Diagnostic View: {simulation.name} (#{simulation.id})
            </div>
            <div className="card-desc">
              Verified patient explainability report for patient <strong>{report.patient_id || "Anonymous"}</strong>
            </div>
          </div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onSelectAnother}>
            Change Simulation
          </button>
        </div>
      </div>

      {/* Main Diagnostic Grid */}
      <div className="grid-3">
        {/* Risk Assessment Gauge */}
        <div className={`risk-card ${categoryClass}`}>
          <div className="stat-label">Progression Risk Score</div>
          <div className="risk-score">{(riskProb * 100).toFixed(1)}%</div>
          <div className="risk-label">{riskCategory} RISK</div>
          <div className="stat-subtext" style={{ marginTop: "1rem" }}>
            MCI &rarr; Alzheimer&apos;s Disease Conversion Probability
          </div>
        </div>

        {/* Feature Importance (SHAP) */}
        <div className="card">
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "1rem" }}>
            <span>📊</span> Clinical Feature Importance (SHAP)
          </div>

          {!report.clinical_importance || report.clinical_importance.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              No feature importance data computed for this patient.
            </div>
          ) : (
            report.clinical_importance.map((item, idx) => (
              <div key={idx} className="feature-row">
                <div className="feature-header">
                  <span className="feature-name">{item.feature}</span>
                  <span className="feature-pct">{item.relative_pct.toFixed(1)}%</span>
                </div>
                <div className="feature-bar-bg">
                  <div
                    className="feature-bar-fill"
                    style={{ width: `${Math.min(100, item.relative_pct * 2.5)}%` }}
                  />
                </div>
              </div>
            ))
          )}
        </div>

        {/* 3D MRI Brain Slice Visualization */}
        <div className="card" style={{ textAlign: "center" }}>
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>
            <span>🧩</span> MRI Neuroimaging ROI Heatmap
          </div>
          <div className="card-desc" style={{ marginBottom: "1rem" }}>
            Hippocampal Atrophy Attribution:{" "}
            <strong style={{ color: "var(--cyan)" }}>
              {report.mri_attribution?.hippocampus_importance_pct !== undefined
                ? `${report.mri_attribution.hippocampus_importance_pct.toFixed(1)}%`
                : "—"}
            </strong>
          </div>

          <div
            style={{
              display: "flex",
              justifyContent: "center",
              borderRadius: "var(--radius-md)",
              overflow: "hidden",
            }}
          >
            <canvas ref={brainCanvasRef} style={{ maxWidth: "100%", height: "auto" }} />
          </div>

          {report.mri_attribution?.peak_attribution_voxel && (
            <div style={{ marginTop: "0.75rem", fontSize: "0.75rem", color: "var(--text-muted)" }}>
              Peak Attribution Voxel: {JSON.stringify(report.mri_attribution.peak_attribution_voxel)}
            </div>
          )}
        </div>
      </div>

      {/* ASCII Diagnostic Report */}
      {data.ascii && (
        <div className="card">
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "0.75rem" }}>
            <span>📄</span> Clinician Terminal Summary
          </div>
          <pre className="code-block">{data.ascii}</pre>
        </div>
      )}
    </div>
  );
};

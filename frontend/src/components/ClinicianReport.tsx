"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Simulation,
  ClinicianDashboardResponse,
  getClinicianDashboard,
} from "@/lib/api";
import { IconBrain, IconBarChart, IconScan, IconMicroscope } from "@/components/Icons";

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
  const [barsVisible, setBarsVisible] = useState(false);
  const brainCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!simulation) {
      setData(null);
      setBarsVisible(false);
      return;
    }

    let mounted = true;
    async function loadReport() {
      if (!simulation) return;
      try {
        setIsLoading(true);
        setErrorMsg(null);
        setBarsVisible(false);
        const res = await getClinicianDashboard(simulation.id);
        if (mounted) {
          setData(res);
          // Trigger bar animation after mount
          setTimeout(() => setBarsVisible(true), 100);
        }
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

  // Render Brain Canvas if report has MRI attribution
  const drawBrainSlice = useCallback((riskProb: number) => {
    const canvas = brainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = (canvas.width = 300);
    const h = (canvas.height = 240);
    const cx = w / 2;
    const cy = h / 2;

    ctx.fillStyle = "#1E2F3B";
    ctx.fillRect(0, 0, w, h);

    // Outer brain parenchymal contour
    ctx.fillStyle = "#2B4050";
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.38, h * 0.44, 0, 0, Math.PI * 2);
    ctx.fill();

    // Cortical gyri / sulci lines
    ctx.strokeStyle = "#435C6E";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx - 25, cy - 15, 38, 0.2, Math.PI * 0.9);
    ctx.arc(cx + 25, cy - 15, 38, 0.1, Math.PI * 0.8);
    ctx.stroke();

    // Ventricles
    ctx.fillStyle = "#16232D";
    ctx.beginPath();
    const vScale = 1.0 + riskProb * 0.6;
    ctx.ellipse(cx, cy - 6, 12 * vScale, 24 * vScale, 0, 0, Math.PI * 2);
    ctx.fill();

    // Bilateral Hippocampal Atrophy ROI Heatmap Overlay
    const hippoColor =
      riskProb > 0.7
        ? "rgba(201, 130, 130, 0.9)"
        : riskProb > 0.3
        ? "rgba(213, 182, 106, 0.9)"
        : "rgba(127, 166, 138, 0.9)";

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
    ctx.fillStyle = "#E4EEE5";
    ctx.font = "10px 'Inter', sans-serif";
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
          <div className="empty-state-icon"><IconBrain size={42} color="var(--text-muted)" /></div>
          <div className="empty-state-title">No Simulation Selected</div>
          <div className="empty-state-desc">
            Select an experiment from the Experiments tab to view patient progression risks, SHAP feature attributions, and neuroimaging heatmaps.
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
      <div className="card animate-tab-slide">
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem", padding: "2rem" }}>
          <div className="shimmer" style={{ height: "24px", width: "60%" }} />
          <div className="shimmer" style={{ height: "120px", width: "100%" }} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
            <div className="shimmer" style={{ height: "160px" }} />
            <div className="shimmer" style={{ height: "160px" }} />
            <div className="shimmer" style={{ height: "160px" }} />
          </div>
        </div>
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div className="card">
        <div className="alert-error">{errorMsg}</div>
      </div>
    );
  }

  // Blank state when no report exists
  if (!data?.report) {
    return (
      <div className="grid-1 animate-tab-slide">
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
                <IconBrain size={20} /> Clinician Diagnostic View: {simulation.name}
              </div>
              <div className="card-desc">
                Model: <strong>{simulation.model}</strong> · Dataset: <strong>{simulation.dataset}</strong> · Status: <strong>{simulation.status}</strong>
              </div>
            </div>
            <button type="button" className="btn btn-secondary btn-sm" onClick={onSelectAnother}>
              Change
            </button>
          </div>
        </div>

        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon"><IconMicroscope size={42} color="var(--text-muted)" /></div>
            <div className="empty-state-title">No Diagnostic Data Available</div>
            <div className="empty-state-desc">
              This simulation has not produced a patient-level clinician report yet.
              When your trained <strong>Fed-XNeuro</strong> model processes patient cohort data (MRI + Cognitive Scores + EHR), progression probabilities, SHAP attributions, and hippocampal atrophy heatmaps will appear here.
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
    <div className="grid-1 animate-tab-slide">
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
              <IconBrain size={20} /> Clinician Diagnostic View: {simulation.name}
            </div>
            <div className="card-desc">
              Patient explainability report for <strong>{report.patient_id || "Anonymous"}</strong>
            </div>
          </div>
          <button type="button" className="btn btn-secondary btn-sm" onClick={onSelectAnother}>
            Change
          </button>
        </div>
      </div>

      {/* Main Diagnostic Grid */}
      <div className="grid-3">
        {/* Risk Assessment */}
        <div className={`risk-card ${categoryClass}`}>
          <div className="stat-label">Progression Risk</div>
          <div className="risk-score">{(riskProb * 100).toFixed(1)}%</div>
          <div className="risk-label">{riskCategory} RISK</div>
          <div className="stat-subtext" style={{ marginTop: "1rem" }}>
            MCI → Alzheimer&apos;s Conversion
          </div>
        </div>

        {/* Feature Importance (SHAP) */}
        <div className="card">
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "1rem" }}>
            <IconBarChart size={16} /> Clinical Feature Importance
          </div>

          {!report.clinical_importance || report.clinical_importance.length === 0 ? (
            <div style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
              No feature importance data computed for this patient.
            </div>
          ) : (
            report.clinical_importance.map((item, idx) => (
              <div
                key={idx}
                className="feature-row"
                style={{ animationDelay: `${idx * 0.08}s` }}
              >
                <div className="feature-header">
                  <span className="feature-name">{item.feature}</span>
                  <span className="feature-pct">{item.relative_pct.toFixed(1)}%</span>
                </div>
                <div className="feature-bar-bg">
                  <div
                    className="feature-bar-fill"
                    style={{
                      width: barsVisible
                        ? `${Math.min(100, item.relative_pct * 2.5)}%`
                        : "0%",
                      animationDelay: `${idx * 0.1}s`,
                    }}
                  />
                </div>
              </div>
            ))
          )}
        </div>

        {/* 3D MRI Brain Slice Visualization */}
        <div className="card" style={{ textAlign: "center" }}>
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "0.5rem" }}>
            <IconScan size={16} /> MRI Neuroimaging Heatmap
          </div>
          <div className="card-desc" style={{ marginBottom: "1rem" }}>
            Hippocampal Atrophy:{" "}
            <strong style={{ color: "var(--primary)" }}>
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
        </div>
      </div>
    </div>
  );
};

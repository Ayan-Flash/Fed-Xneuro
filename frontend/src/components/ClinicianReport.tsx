"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { BrainCircuit, SearchX, Activity, ChartBar, Brain, FileText, AlertTriangle } from "lucide-react";
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

    ctx.fillStyle = "#1e293b";
    ctx.beginPath();
    ctx.ellipse(cx, cy, w * 0.38, h * 0.44, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx - 25, cy - 15, 38, 0.2, Math.PI * 0.9);
    ctx.arc(cx + 25, cy - 15, 38, 0.1, Math.PI * 0.8);
    ctx.stroke();

    ctx.fillStyle = "#0b0f19";
    ctx.beginPath();
    const vScale = 1.0 + riskProb * 0.6;
    ctx.ellipse(cx, cy - 6, 12 * vScale, 24 * vScale, 0, 0, Math.PI * 2);
    ctx.fill();

    const hippoColor =
      riskProb > 0.7
        ? "rgba(217, 83, 79, 0.85)" // Soft Red
        : riskProb > 0.3
        ? "rgba(232, 163, 23, 0.85)" // Amber
        : "rgba(25, 124, 130, 0.85)"; // Teal

    const gradL = ctx.createRadialGradient(cx - 36, cy + 15, 2, cx - 36, cy + 15, 20);
    gradL.addColorStop(0, hippoColor);
    gradL.addColorStop(1, "transparent");
    ctx.fillStyle = gradL;
    ctx.beginPath();
    ctx.arc(cx - 36, cy + 15, 20, 0, Math.PI * 2);
    ctx.fill();

    const gradR = ctx.createRadialGradient(cx + 36, cy + 15, 2, cx + 36, cy + 15, 20);
    gradR.addColorStop(0, hippoColor);
    gradR.addColorStop(1, "transparent");
    ctx.fillStyle = gradR;
    ctx.beginPath();
    ctx.arc(cx + 36, cy + 15, 20, 0, Math.PI * 2);
    ctx.fill();

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
      <div className="card border-[var(--border-color)]">
        <div className="empty-state">
          <div className="w-16 h-16 rounded-full bg-[var(--primary-glow)] flex items-center justify-center text-[var(--primary)] mx-auto mb-4">
            <BrainCircuit size={32} />
          </div>
          <div className="empty-state-title text-[var(--text-primary)]">No Patient Analysis Selected</div>
          <div className="empty-state-desc text-[var(--text-secondary)]">
            Please select an experiment from the "Experiments & Launcher" tab to inspect patient progression risks, SHAP attributions, and neuroimaging overlays.
          </div>
          <button type="button" className="btn bg-white border border-[var(--border-color)] text-[var(--text-primary)] hover:bg-[var(--bg-primary)] hover:border-[var(--primary)] shadow-sm font-semibold py-2 px-4 rounded-[var(--radius-md)] transition-all mt-4" onClick={onSelectAnother}>
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
          <Activity size={32} className="animate-pulse mx-auto mb-2 text-[var(--primary)]" />
          Loading clinical diagnostic report for #{simulation.id}...
        </div>
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div className="card">
        <div
          style={{
            background: "var(--crimson-glow)",
            border: "1px solid var(--crimson)",
            color: "var(--crimson)",
            padding: "1rem",
            borderRadius: "var(--radius-sm)",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem"
          }}
        >
          <AlertTriangle size={20} />
          {errorMsg}
        </div>
      </div>
    );
  }

  if (!data?.report) {
    return (
      <div className="grid-1">
        <div className="card">
          <div className="flex justify-between items-center flex-wrap gap-4">
            <div>
              <div className="card-title flex items-center gap-2">
                <BrainCircuit className="text-[var(--primary)]" size={24} /> Clinical Diagnostic View: {simulation.name} (#{simulation.id})
              </div>
              <div className="card-desc">
                Model: <strong>{simulation.model}</strong> | Dataset: <strong>{simulation.dataset}</strong> | Status: <strong>{simulation.status}</strong>
              </div>
            </div>
            <button type="button" className="btn bg-white border border-[var(--border-color)] text-[var(--text-primary)] hover:bg-[var(--bg-primary)] hover:border-[var(--primary)] shadow-sm font-semibold py-2 px-4 rounded-[var(--radius-md)] transition-all" onClick={onSelectAnother}>
              Change Analysis
            </button>
          </div>
        </div>

        <div className="card">
          <div className="empty-state">
            <div className="w-16 h-16 rounded-full bg-[var(--primary-glow)] flex items-center justify-center text-[var(--primary)] mx-auto mb-4">
              <SearchX size={32} />
            </div>
            <div className="empty-state-title">No Clinical Diagnostic Data Available</div>
            <div className="empty-state-desc">
              Simulation #{simulation.id} has not produced a patient-level clinical progression report.
              <br /><br />
              When you test with your trained multimodal <strong>Fed-XNeuro</strong> model on patient cohort data, actual MCI &rarr; Alzheimer's progression probabilities, SHAP clinical feature bars, and hippocampal atrophy heatmaps will appear here.
            </div>
            <button type="button" className="btn bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white shadow-md font-semibold py-2 px-4 rounded-[var(--radius-md)] transition-all mt-4" onClick={onSelectAnother}>
              Select Another Experiment
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
      {/* Disclaimer Alert */}
      <div className="bg-[var(--amber-glow)] border-l-4 border-[var(--amber)] p-4 rounded-r-md flex items-start gap-3">
        <AlertTriangle className="text-[var(--amber)] mt-0.5 shrink-0" size={20} />
        <div>
          <h3 className="text-[var(--text-primary)] font-bold text-sm">Medical Disclaimer</h3>
          <p className="text-[var(--text-secondary)] text-xs mt-1">
            This prediction is a screening aid and is not a medical diagnosis. The output provided by the Fed-XNeuro AI model must be reviewed and interpreted by a qualified healthcare professional in conjunction with other clinical evidence.
          </p>
        </div>
      </div>

      {/* Top Header */}
      <div className="card">
        <div className="flex justify-between items-center flex-wrap gap-4">
          <div>
            <div className="card-title flex items-center gap-2">
              <BrainCircuit className="text-[var(--primary)]" size={24} /> Patient Diagnostic Analysis: {simulation.name} (#{simulation.id})
            </div>
            <div className="card-desc mt-1">
              Verified patient explainability report for patient <strong>{report.patient_id || "Anonymous"}</strong>
            </div>
          </div>
          <button type="button" className="btn bg-white border border-[var(--border-color)] text-[var(--text-primary)] hover:bg-[var(--bg-primary)] hover:border-[var(--primary)] shadow-sm font-semibold py-2 px-4 rounded-[var(--radius-md)] transition-all" onClick={onSelectAnother}>
            Change Analysis
          </button>
        </div>
      </div>

      {/* Main Diagnostic Grid */}
      <div className="grid-3">
        {/* Risk Assessment Gauge */}
        <div className={`risk-card ${categoryClass} relative overflow-hidden bg-white`}>
          <div className="absolute top-0 right-0 w-32 h-32 bg-current opacity-5 rounded-bl-full" />
          <div className="stat-label text-[var(--text-secondary)] font-semibold flex items-center gap-2">
            <Activity size={18} /> Progression Risk Score
          </div>
          <div className={`risk-score my-4 font-black ${
            riskProb > 0.7 ? 'text-[var(--crimson)]' : riskProb > 0.3 ? 'text-[var(--amber)]' : 'text-[var(--primary)]'
          }`}>
            {(riskProb * 100).toFixed(1)}%
          </div>
          <div className={`text-sm font-bold tracking-widest px-3 py-1 rounded-full inline-block ${
             riskProb > 0.7 ? 'bg-[var(--crimson-glow)] text-[var(--crimson)]' : riskProb > 0.3 ? 'bg-[var(--amber-glow)] text-[var(--amber)]' : 'bg-[var(--primary-glow)] text-[var(--primary)]'
          }`}>
            {riskCategory} RISK
          </div>
          <div className="stat-subtext mt-4 text-[var(--text-muted)] text-xs font-medium">
            MCI &rarr; Alzheimer's Disease Conversion Probability
          </div>
        </div>

        {/* Feature Importance (SHAP) */}
        <div className="card flex flex-col">
          <div className="card-title flex items-center gap-2 text-base mb-4 border-b border-[var(--border-color)] pb-3">
            <ChartBar className="text-[var(--primary)]" size={20} /> Clinical Feature Importance (SHAP)
          </div>

          {!report.clinical_importance || report.clinical_importance.length === 0 ? (
            <div className="text-[var(--text-muted)] text-sm flex-1 flex items-center justify-center">
              No feature importance data computed for this patient.
            </div>
          ) : (
            <div className="space-y-3">
              {report.clinical_importance.map((item, idx) => (
                <div key={idx} className="feature-row">
                  <div className="feature-header flex justify-between text-sm mb-1 font-medium text-[var(--text-primary)]">
                    <span className="feature-name">{item.feature}</span>
                    <span className="feature-pct">{item.relative_pct.toFixed(1)}%</span>
                  </div>
                  <div className="h-2 w-full bg-[var(--bg-card-hover)] rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[var(--primary)] rounded-full"
                      style={{ width: `${Math.min(100, item.relative_pct * 2.5)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 3D MRI Brain Slice Visualization */}
        <div className="card text-center flex flex-col">
          <div className="card-title flex items-center justify-center gap-2 text-base mb-2">
            <Brain className="text-[var(--primary)]" size={20} /> MRI Neuroimaging ROI Heatmap
          </div>
          <div className="card-desc mb-4 text-sm border-b border-[var(--border-color)] pb-3">
            Hippocampal Atrophy Attribution:{" "}
            <strong className="text-[var(--primary)]">
              {report.mri_attribution?.hippocampus_importance_pct !== undefined
                ? `${report.mri_attribution.hippocampus_importance_pct.toFixed(1)}%`
                : "—"}
            </strong>
          </div>

          <div className="flex justify-center rounded-[var(--radius-md)] overflow-hidden bg-[#030712] p-2 flex-1 items-center">
            <canvas ref={brainCanvasRef} style={{ maxWidth: "100%", height: "auto" }} />
          </div>

          {report.mri_attribution?.peak_attribution_voxel && (
            <div className="mt-3 text-xs text-[var(--text-muted)] font-mono">
              Peak Voxel: {JSON.stringify(report.mri_attribution.peak_attribution_voxel)}
            </div>
          )}
        </div>
      </div>

      {/* ASCII Diagnostic Report */}
      {data.ascii && (
        <div className="card">
          <div className="card-title flex items-center gap-2 text-base mb-3">
            <FileText className="text-[var(--primary)]" size={20} /> Clinician Terminal Summary
          </div>
          <pre className="code-block text-xs text-[var(--text-primary)] bg-[var(--bg-card-hover)] border border-[var(--border-color)] p-4 rounded-[var(--radius-sm)] overflow-x-auto">
            {data.ascii}
          </pre>
        </div>
      )}
    </div>
  );
};

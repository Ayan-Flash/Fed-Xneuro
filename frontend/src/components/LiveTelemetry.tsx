"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import { Activity, LineChart, TrendingDown, Zap, SearchX } from "lucide-react";
import {
  Simulation,
  MetricItem,
  getMetrics,
  getSimulationWebSocketUrl,
} from "@/lib/api";

interface LiveTelemetryProps {
  simulation: Simulation | null;
  onSelectAnother: () => void;
}

interface RoundEvent {
  round: number;
  accuracy: number;
  loss: number;
  privacy_budget_spent?: number;
  timestamp: string;
}

export const LiveTelemetry: React.FC<LiveTelemetryProps> = ({
  simulation,
  onSelectAnother,
}) => {
  const [metrics, setMetrics] = useState<MetricItem[]>([]);
  const [events, setEvents] = useState<RoundEvent[]>([]);
  const [wsStatus, setWsStatus] = useState<"connecting" | "connected" | "disconnected" | "completed">("disconnected");
  const [latestAccuracy, setLatestAccuracy] = useState<number | null>(null);
  const [latestLoss, setLatestLoss] = useState<number | null>(null);
  const [latestEpsilon, setLatestEpsilon] = useState<number | null>(null);
  const [currentRound, setCurrentRound] = useState<number>(0);

  const accCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const lossCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Load existing metrics from database
  useEffect(() => {
    if (!simulation) {
      setMetrics([]);
      setEvents([]);
      setLatestAccuracy(null);
      setLatestLoss(null);
      setLatestEpsilon(null);
      setCurrentRound(0);
      return;
    }

    setCurrentRound(simulation.current_round || 0);
    setLatestAccuracy(simulation.final_accuracy);
    setLatestLoss(simulation.final_loss);

    async function loadPastMetrics() {
      if (!simulation) return;
      try {
        const data = await getMetrics(simulation.id);
        setMetrics(data);
        if (data.length > 0) {
          const last = data[data.length - 1];
          setLatestAccuracy(last.accuracy);
          setLatestLoss(last.loss);
          setCurrentRound(last.round_number);
          if (last.privacy_budget_spent !== undefined) {
            setLatestEpsilon(last.privacy_budget_spent);
          }
        }
      } catch (err) {
        console.error("Failed to load past metrics:", err);
      }
    }

    loadPastMetrics();
  }, [simulation]);

  // WebSocket Live Stream
  useEffect(() => {
    if (!simulation) return;

    const wsUrl = getSimulationWebSocketUrl(simulation.id);
    setWsStatus("connecting");
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setWsStatus("connected");
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.event === "round_complete") {
          const r = msg.round;
          const acc = msg.accuracy;
          const l = msg.loss;
          const eps = msg.privacy_budget_spent;

          setCurrentRound(r);
          setLatestAccuracy(acc);
          setLatestLoss(l);
          if (eps !== undefined && eps !== null) setLatestEpsilon(eps);

          setMetrics((prev) => [
            ...prev,
            {
              id: Date.now(),
              simulation_id: simulation.id,
              round_number: r,
              loss: l,
              accuracy: acc,
              privacy_budget_spent: eps,
              created_at: new Date().toISOString(),
            },
          ]);

          setEvents((prev) => [
            {
              round: r,
              accuracy: acc,
              loss: l,
              privacy_budget_spent: eps,
              timestamp: new Date().toLocaleTimeString(),
            },
            ...prev,
          ]);
        } else if (msg.event === "simulation_completed") {
          setWsStatus("completed");
        }
      } catch (e) {
        console.error("WebSocket message parsing error:", e);
      }
    };

    ws.onclose = () => {
      setWsStatus((prev) => (prev === "completed" ? "completed" : "disconnected"));
    };

    return () => {
      ws.close();
    };
  }, [simulation]);

  // Draw Canvas Charts
  const renderCharts = useCallback(() => {
    if (!simulation || metrics.length === 0) return;

    const rounds = metrics.map((m) => m.round_number);
    const accuracies = metrics.map((m) => m.accuracy);
    const losses = metrics.map((m) => m.loss);

    drawChart(accCanvasRef.current, rounds, accuracies, "Accuracy (%)", "var(--secondary)");
    drawChart(lossCanvasRef.current, rounds, losses, "Loss", "var(--crimson)");
  }, [simulation, metrics]);

  useEffect(() => {
    renderCharts();
    window.addEventListener("resize", renderCharts);
    return () => window.removeEventListener("resize", renderCharts);
  }, [renderCharts]);

  if (!simulation) {
    return (
      <div className="card">
        <div className="empty-state text-center py-12 px-4 rounded-[var(--radius-md)] border border-dashed border-[var(--border-color)] bg-white">
          <div className="empty-state-icon flex justify-center mb-4">
             <Activity size={48} className="text-[var(--border-color)]" />
          </div>
          <div className="empty-state-title text-xl font-bold text-[var(--text-primary)] mb-2">No Active Telemetry Stream</div>
          <div className="empty-state-desc text-[var(--text-secondary)] max-w-md mx-auto mb-6">
            No simulation is currently selected for live telemetry streaming. Select an experiment from the &quot;Experiments &amp; Launcher&quot; tab to view real-time training progress, accuracy curves, and privacy accounting.
          </div>
          <button type="button" className="btn btn-secondary px-4 py-2 bg-white border border-[var(--border-color)] rounded-[var(--radius-sm)] text-[var(--text-primary)] hover:bg-[var(--bg-secondary)] transition-colors" onClick={onSelectAnother}>
            Go to Experiments
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="grid-1">
      {/* Telemetry Header */}
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
            <div className="card-title flex items-center gap-2">
              <Activity className="text-[var(--primary)]" size={24} /> Live Telemetry: {simulation.name} (#{simulation.id})
            </div>
            <div className="card-desc mt-1">
              Streaming round-by-round global evaluation metrics via WebSocket for run{" "}
              <code style={{ color: "var(--primary)", fontFamily: "var(--font-mono)" }}>
                {simulation.run_id}
              </code>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div className="status-badge flex items-center gap-2 bg-[var(--bg-secondary)] px-3 py-1.5 rounded-full border border-[var(--border-color)] text-sm font-medium">
              <span
                className={`status-dot w-2.5 h-2.5 rounded-full ${
                  wsStatus === "connected"
                    ? "bg-[var(--secondary)] animate-pulse"
                    : wsStatus === "completed"
                    ? "bg-[var(--primary)]"
                    : wsStatus === "connecting"
                    ? "bg-[var(--amber)]"
                    : "bg-[var(--text-muted)]"
                }`}
              />
              <span style={{ textTransform: "capitalize" }} className="text-[var(--text-primary)]">
                {wsStatus === "connected"
                  ? "Streaming Live"
                  : wsStatus === "completed"
                  ? "Simulation Finished"
                  : wsStatus === "connecting"
                  ? "Connecting WS..."
                  : "Standby / Closed"}
              </span>
            </div>
            <button type="button" className="btn btn-secondary btn-sm bg-white border border-[var(--border-color)] text-[var(--text-primary)] px-3 py-1.5 rounded-[var(--radius-sm)] hover:bg-[var(--bg-secondary)] transition-colors text-sm" onClick={onSelectAnother}>
              Change Simulation
            </button>
          </div>
        </div>

        {/* Real-time Metric Stat Cards */}
        <div className="grid-4 mt-6 gap-4">
          <div className="stat-card border border-[var(--border-color)] rounded-[var(--radius-md)] p-4 bg-white shadow-sm">
            <div className="stat-label text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-1">Communication Round</div>
            <div className="stat-value text-2xl font-bold text-[var(--text-primary)]">
              {currentRound} / {simulation.num_rounds}
            </div>
            <div className="stat-subtext text-xs text-[var(--text-secondary)] mt-1">
              {simulation.num_rounds > 0
                ? `${Math.round((currentRound / simulation.num_rounds) * 100)}% progress`
                : "—"}
            </div>
          </div>

          <div className="stat-card border border-[var(--border-color)] rounded-[var(--radius-md)] p-4 bg-white shadow-sm">
            <div className="stat-label text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-1">Global Accuracy</div>
            <div className="stat-value text-2xl font-bold" style={{ color: "var(--secondary)" }}>
              {latestAccuracy !== null && latestAccuracy !== undefined
                ? `${latestAccuracy.toFixed(2)}%`
                : "—"}
            </div>
            <div className="stat-subtext text-xs text-[var(--text-secondary)] mt-1">Verified on held-out test split</div>
          </div>

          <div className="stat-card border border-[var(--border-color)] rounded-[var(--radius-md)] p-4 bg-white shadow-sm">
            <div className="stat-label text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-1">Global Loss</div>
            <div className="stat-value text-2xl font-bold" style={{ color: "var(--crimson)" }}>
              {latestLoss !== null && latestLoss !== undefined
                ? latestLoss.toFixed(4)
                : "—"}
            </div>
            <div className="stat-subtext text-xs text-[var(--text-secondary)] mt-1">Sample-weighted objective</div>
          </div>

          <div className="stat-card border border-[var(--border-color)] rounded-[var(--radius-md)] p-4 bg-white shadow-sm">
            <div className="stat-label text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide mb-1">Privacy Budget Spent (&epsilon;)</div>
            <div className="stat-value text-2xl font-bold" style={{ color: "var(--amber)" }}>
              {latestEpsilon !== null && latestEpsilon !== undefined
                ? latestEpsilon.toFixed(2)
                : "None"}
            </div>
            <div className="stat-subtext text-xs text-[var(--text-secondary)] mt-1">Rényi Differential Privacy</div>
          </div>
        </div>
      </div>

      {/* Real Charts or Blank State */}
      {metrics.length === 0 ? (
        <div className="card">
          <div className="empty-state text-center py-12 px-4 rounded-[var(--radius-md)] border border-dashed border-[var(--border-color)] bg-white">
            <div className="empty-state-icon flex justify-center mb-4">
              <SearchX size={48} className="text-[var(--border-color)]" />
            </div>
            <div className="empty-state-title text-xl font-bold text-[var(--text-primary)] mb-2">Awaiting First Round Telemetry</div>
            <div className="empty-state-desc text-[var(--text-secondary)] max-w-md mx-auto">
              Simulation #{simulation.id} is preparing client nodes and data partitions. As soon as Round 1 completes, live accuracy and loss curves will appear here in real time.
            </div>
          </div>
        </div>
      ) : (
        <div className="grid-2 gap-4">
          <div className="card">
            <div className="card-title flex items-center gap-2" style={{ fontSize: "1rem", color: "var(--secondary)" }}>
              <LineChart size={18} /> Global Accuracy Curve
            </div>
            <div style={{ width: "100%", height: "260px", marginTop: "1rem" }}>
              <canvas ref={accCanvasRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>

          <div className="card">
            <div className="card-title flex items-center gap-2" style={{ fontSize: "1rem", color: "var(--crimson)" }}>
              <TrendingDown size={18} /> Global Loss Curve
            </div>
            <div style={{ width: "100%", height: "260px", marginTop: "1rem" }}>
              <canvas ref={lossCanvasRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>
        </div>
      )}

      {/* Live Event Stream */}
      {events.length > 0 && (
        <div className="card">
          <div className="card-title flex items-center gap-2 mb-4" style={{ fontSize: "1rem" }}>
            <Zap className="text-[var(--amber)]" size={18} /> Real-Time Communication Round Stream
          </div>
          <div className="table-container overflow-x-auto">
            <table className="data-table w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-[var(--border-color)]">
                  <th className="py-2 px-4 font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wide">Round</th>
                  <th className="py-2 px-4 font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wide">Accuracy</th>
                  <th className="py-2 px-4 font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wide">Loss</th>
                  <th className="py-2 px-4 font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wide">Privacy Spent</th>
                  <th className="py-2 px-4 font-semibold text-xs text-[var(--text-muted)] uppercase tracking-wide">Time</th>
                </tr>
              </thead>
              <tbody>
                {events.map((evt, idx) => (
                  <tr key={idx} className="border-b border-[var(--border-color)] hover:bg-[var(--bg-secondary)] transition-colors">
                    <td className="py-2 px-4">
                      <strong className="text-[var(--text-primary)]">Round {evt.round}</strong>
                    </td>
                    <td className="py-2 px-4" style={{ color: "var(--secondary)", fontWeight: 700 }}>
                      {evt.accuracy.toFixed(2)}%
                    </td>
                    <td className="py-2 px-4" style={{ color: "var(--crimson)", fontFamily: "var(--font-mono)" }}>
                      {evt.loss.toFixed(4)}
                    </td>
                    <td className="py-2 px-4 text-[var(--text-primary)]">
                      {evt.privacy_budget_spent !== undefined
                        ? `ε = ${evt.privacy_budget_spent.toFixed(2)}`
                        : "—"}
                    </td>
                    <td className="py-2 px-4 text-[var(--text-muted)]" style={{ fontSize: "0.8rem" }}>
                      {evt.timestamp}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

// Canvas Chart Drawing Helper
function drawChart(
  canvas: HTMLCanvasElement | null,
  xVals: number[],
  yVals: number[],
  label: string,
  color: string
) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const w = (canvas.width = canvas.parentElement?.clientWidth || 500);
  const h = (canvas.height = canvas.parentElement?.clientHeight || 260);

  ctx.clearRect(0, 0, w, h);
  if (xVals.length < 1) return;

  const padLeft = 50;
  const padBottom = 30;
  const padTop = 20;
  const padRight = 20;

  const plotW = w - padLeft - padRight;
  const plotH = h - padTop - padBottom;

  const minY = Math.min(...yVals) * 0.95;
  const maxY = Math.max(...yVals) * 1.05 || 1.0;

  // Grid Lines
  ctx.strokeStyle = "rgba(0, 0, 0, 0.05)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH / 4) * i;
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
  }
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = maxY - ((maxY - minY) / 4) * i;
    const y = padTop + (plotH / 4) * i;
    ctx.fillText(val.toFixed(2), padLeft - 8, y + 3);
  }

  // Draw Path
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();

  if (xVals.length === 1) {
    const x = padLeft + plotW / 2;
    const y = padTop + plotH / 2;
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    return;
  }

  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Dots
  ctx.fillStyle = color;
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
}

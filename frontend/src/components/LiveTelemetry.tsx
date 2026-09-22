"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Simulation,
  MetricItem,
  getMetrics,
  getSimulationWebSocketUrl,
} from "@/lib/api";
import { IconSatellite, IconHourglass, IconLightning } from "@/components/Icons";

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

/* Circular progress ring component */
function ProgressRing({
  current,
  total,
  size = 80,
}: {
  current: number;
  total: number;
  size?: number;
}) {
  const strokeWidth = 6;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const pct = total > 0 ? current / total : 0;
  const offset = circumference - pct * circumference;

  return (
    <div className="progress-ring-container" style={{ width: size, height: size }}>
      <svg className="progress-ring-svg" width={size} height={size}>
        <circle className="progress-ring-bg" cx={size / 2} cy={size / 2} r={radius} />
        <circle
          className="progress-ring-fill"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={offset}
        />
      </svg>
      <div className="progress-ring-text">
        <span className="ring-value">
          {current}/{total}
        </span>
        <span className="ring-label">Rounds</span>
      </div>
    </div>
  );
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
  const [showAllEvents, setShowAllEvents] = useState(false);

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
    setLatestAccuracy(simulation.final_accuracy ?? null);
    setLatestLoss(simulation.final_loss ?? null);

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

    drawChart(accCanvasRef.current, rounds, accuracies, "Accuracy (%)", "#7FA68A");
    drawChart(lossCanvasRef.current, rounds, losses, "Loss", "#C98282");
  }, [simulation, metrics]);

  useEffect(() => {
    renderCharts();
    window.addEventListener("resize", renderCharts);
    return () => window.removeEventListener("resize", renderCharts);
  }, [renderCharts]);

  if (!simulation) {
    return (
      <div className="card">
        <div className="empty-state">
          <div className="empty-state-icon"><IconSatellite size={42} color="var(--text-muted)" /></div>
          <div className="empty-state-title">No Active Telemetry</div>
          <div className="empty-state-desc">
            Select an experiment from the Experiments tab to view real-time training progress, accuracy curves, and privacy budget tracking.
          </div>
          <button type="button" className="btn btn-secondary" onClick={onSelectAnother}>
            Go to Experiments
          </button>
        </div>
      </div>
    );
  }

  const visibleEvents = showAllEvents ? events : events.slice(0, 5);

  return (
    <div className="grid-1 animate-tab-slide">
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
            <div className="card-title">
              <IconSatellite size={20} /> Live Telemetry: {simulation.name}
            </div>
            <div className="card-desc">
              Streaming real-time global evaluation metrics via WebSocket
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <div className="status-badge">
              <span
                className={`status-dot ${
                  wsStatus === "connected"
                    ? ""
                    : wsStatus === "completed"
                    ? ""
                    : wsStatus === "connecting"
                    ? "warning"
                    : "offline"
                }`}
              />
              <span>
                {wsStatus === "connected"
                  ? "Streaming"
                  : wsStatus === "completed"
                  ? "Finished"
                  : wsStatus === "connecting"
                  ? "Connecting…"
                  : "Standby"}
              </span>
            </div>
            <button type="button" className="btn btn-secondary btn-sm" onClick={onSelectAnother}>
              Change
            </button>
          </div>
        </div>

        {/* Stat Cards with Progress Ring */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto 1fr 1fr 1fr",
            gap: "1.25rem",
            marginTop: "1.5rem",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", justifyContent: "center" }}>
            <ProgressRing current={currentRound} total={simulation.num_rounds} size={86} />
          </div>

          <div className="stat-card">
            <div className="stat-label">Global Accuracy</div>
            <div className="stat-value" style={{ color: "var(--primary)" }}>
              {latestAccuracy !== null && latestAccuracy !== undefined
                ? `${latestAccuracy.toFixed(2)}%`
                : "—"}
            </div>
            <div className="stat-subtext">Held-out test split</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Global Loss</div>
            <div className="stat-value" style={{ color: "var(--alert-color)" }}>
              {latestLoss !== null && latestLoss !== undefined
                ? latestLoss.toFixed(4)
                : "—"}
            </div>
            <div className="stat-subtext">Weighted objective</div>
          </div>

          <div className="stat-card">
            <div className="stat-label">Privacy ε</div>
            <div className="stat-value" style={{ color: "var(--gold)" }}>
              {latestEpsilon !== null && latestEpsilon !== undefined
                ? latestEpsilon.toFixed(2)
                : "None"}
            </div>
            <div className="stat-subtext">Differential Privacy</div>
          </div>
        </div>
      </div>

      {/* Charts or Blank State */}
      {metrics.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <div className="empty-state-icon"><IconHourglass size={42} color="var(--text-muted)" /></div>
            <div className="empty-state-title">Awaiting First Round</div>
            <div className="empty-state-desc">
              Preparing client nodes and data partitions. Live accuracy and loss curves will appear once Round 1 completes.
            </div>
          </div>
        </div>
      ) : (
        <div className="grid-2">
          <div className="card">
            <div className="card-title" style={{ fontSize: "1rem", color: "var(--primary)" }}>
              Accuracy Curve
            </div>
            <div style={{ width: "100%", height: "260px", marginTop: "1rem" }}>
              <canvas ref={accCanvasRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>

          <div className="card">
            <div className="card-title" style={{ fontSize: "1rem", color: "var(--alert-color)" }}>
              Loss Curve
            </div>
            <div style={{ width: "100%", height: "260px", marginTop: "1rem" }}>
              <canvas ref={lossCanvasRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>
        </div>
      )}

      {/* Compact Event Feed */}
      {events.length > 0 && (
        <div className="card">
          <div className="card-title" style={{ fontSize: "1rem", marginBottom: "0.75rem" }}>
            <IconLightning size={16} /> Round Event Feed
          </div>

          <div className="event-feed">
            {visibleEvents.map((evt, idx) => (
              <div
                key={idx}
                className="event-item"
                style={{ animationDelay: `${idx * 0.04}s` }}
              >
                <div className="event-icon">R{evt.round}</div>
                <div className="event-details">
                  <span className="event-round">Round {evt.round}</span>
                  <span className="event-metric" style={{ color: "var(--primary)" }}>
                    Acc: {evt.accuracy.toFixed(2)}%
                  </span>
                  <span className="event-metric" style={{ color: "var(--alert-color)" }}>
                    Loss: {evt.loss.toFixed(4)}
                  </span>
                  {evt.privacy_budget_spent !== undefined && (
                    <span className="event-metric" style={{ color: "var(--gold)" }}>
                      ε={evt.privacy_budget_spent.toFixed(2)}
                    </span>
                  )}
                  <span className="event-time">{evt.timestamp}</span>
                </div>
              </div>
            ))}
          </div>

          {events.length > 5 && (
            <div style={{ textAlign: "center", marginTop: "0.75rem" }}>
              <button
                type="button"
                className="toggle-btn"
                onClick={() => setShowAllEvents(!showAllEvents)}
              >
                {showAllEvents
                  ? `Show Less`
                  : `Show All ${events.length} Events`}
              </button>
            </div>
          )}
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
  ctx.strokeStyle = "rgba(38, 59, 74, 0.10)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (plotH / 4) * i;
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
  }
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = "rgba(38, 59, 74, 0.55)";
  ctx.font = "11px 'Inter', sans-serif";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const val = maxY - ((maxY - minY) / 4) * i;
    const y = padTop + (plotH / 4) * i;
    ctx.fillText(val.toFixed(2), padLeft - 8, y + 4);
  }

  // X-axis labels
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(38, 59, 74, 0.45)";
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (xVals.length === 1 ? plotW / 2 : (i / (xVals.length - 1)) * plotW);
    ctx.fillText(`R${xVals[i]}`, x, h - 6);
  }

  if (xVals.length === 1) {
    const x = padLeft + plotW / 2;
    const y = padTop + plotH / 2;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    return;
  }

  // Area fill (gradient)
  const gradient = ctx.createLinearGradient(0, padTop, 0, padTop + plotH);
  gradient.addColorStop(0, color + "30");
  gradient.addColorStop(1, color + "05");

  ctx.beginPath();
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  // Close path for area fill
  ctx.lineTo(padLeft + plotW, padTop + plotH);
  ctx.lineTo(padLeft, padTop + plotH);
  ctx.closePath();
  ctx.fillStyle = gradient;
  ctx.fill();

  // Draw Line
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Draw Dots with hover-style glow
  for (let i = 0; i < xVals.length; i++) {
    const x = padLeft + (i / (xVals.length - 1)) * plotW;
    const y = padTop + plotH - ((yVals[i] - minY) / (maxY - minY || 1)) * plotH;

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = color + "22";
    ctx.fill();

    // Inner dot
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    // White center for last point
    if (i === xVals.length - 1) {
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fillStyle = "#FFFFFF";
      ctx.fill();
    }
  }
}

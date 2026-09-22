"use client";

import React, { useState } from "react";
import { Simulation, deleteSimulation } from "@/lib/api";
import { IconClipboard, IconMicroscope, IconChartUp, IconBrain, IconClose } from "@/components/Icons";

interface SimulationsTableProps {
  simulations: Simulation[];
  isLoading: boolean;
  onRefresh: () => void;
  onSelectTelemetry: (sim: Simulation) => void;
  onSelectClinician: (sim: Simulation) => void;
}

export const SimulationsTable: React.FC<SimulationsTableProps> = ({
  simulations,
  isLoading,
  onRefresh,
  onSelectTelemetry,
  onSelectClinician,
}) => {
  const [deletingId, setDeletingId] = useState<number | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  async function handleDelete(id: number, e: React.MouseEvent) {
    e.stopPropagation();
    if (!confirm(`Delete simulation #${id}?`)) return;
    try {
      setDeletingId(id);
      await deleteSimulation(id);
      onRefresh();
    } catch (err) {
      alert("Failed to delete: " + err);
    } finally {
      setDeletingId(null);
    }
  }

  function toggleExpand(id: number) {
    setExpandedId((prev) => (prev === id ? null : id));
  }

  const activeCount = simulations.filter(
    (s) => s.status === "running" || s.status === "starting"
  ).length;

  return (
    <div className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: "1.25rem",
        }}
      >
        <div>
          <div className="card-title">
            <IconClipboard size={20} /> Experiment History
          </div>
          <div className="card-desc">
            Federated training runs and global accuracy results.
          </div>
        </div>

        <div style={{ display: "flex", gap: "1rem" }}>
          <div className="stat-card" style={{ padding: "0.5rem 1rem", textAlign: "center" }}>
            <div className="stat-label">Runs</div>
            <div className="stat-value" style={{ fontSize: "1.2rem" }}>
              {simulations.length}
            </div>
          </div>
          {activeCount > 0 && (
            <div className="stat-card" style={{ padding: "0.5rem 1rem", textAlign: "center" }}>
              <div className="stat-label">Active</div>
              <div
                className="stat-value"
                style={{ fontSize: "1.2rem", color: "var(--primary)" }}
              >
                {activeCount}
              </div>
            </div>
          )}
        </div>
      </div>

      {isLoading && simulations.length === 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
          {[1, 2, 3].map((i) => (
            <div key={i} className="shimmer" style={{ height: "48px", width: "100%" }} />
          ))}
        </div>
      ) : simulations.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon"><IconMicroscope size={42} color="var(--text-muted)" /></div>
          <div className="empty-state-title">No Experiments Yet</div>
          <div className="empty-state-desc">
            Configure your model, dataset, and clients above and launch your first federated simulation.
          </div>
        </div>
      ) : (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Experiment</th>
                <th>Algorithm</th>
                <th>Progress</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th style={{ textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {simulations.map((sim, idx) => {
                const statusBadgeClass = `badge-${sim.status || "created"}`;
                const accText =
                  sim.final_accuracy !== null && sim.final_accuracy !== undefined
                    ? `${sim.final_accuracy.toFixed(1)}%`
                    : "—";
                const isExpanded = expandedId === sim.id;
                const isRunning = sim.status === "running" || sim.status === "starting";
                const progressPct = sim.num_rounds > 0
                  ? Math.round((sim.current_round / sim.num_rounds) * 100)
                  : 0;

                return (
                  <React.Fragment key={sim.id}>
                    <tr
                      className={`expandable-row animate-fade-up stagger-${Math.min(idx + 1, 8)}`}
                      onClick={() => toggleExpand(sim.id)}
                      style={{ cursor: "pointer" }}
                    >
                      <td>
                        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                          <span className="badge badge-id">#{sim.id}</span>
                          <strong>{sim.name}</strong>
                        </div>
                      </td>
                      <td>
                        <span
                          className="badge"
                          style={{
                            background: "var(--primary-subtle)",
                            color: "var(--primary-hover)",
                            border: "1px solid rgba(127, 166, 138, 0.35)",
                          }}
                        >
                          {sim.algorithm.toUpperCase()}
                        </span>
                      </td>
                      <td>
                        <div>
                          <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600, fontSize: "0.85rem" }}>
                            {sim.current_round}/{sim.num_rounds}
                          </span>
                          {isRunning && (
                            <div className="progress-bar-mini">
                              <div
                                className="progress-bar-mini-fill"
                                style={{ width: `${progressPct}%` }}
                              />
                            </div>
                          )}
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${statusBadgeClass}`}>{sim.status}</span>
                      </td>
                      <td>
                        <strong
                          style={{
                            color:
                              sim.final_accuracy !== null && sim.final_accuracy !== undefined
                                ? "var(--primary)"
                                : "var(--text-muted)",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {accText}
                        </strong>
                      </td>
                      <td style={{ textAlign: "right" }}>
                        <div
                          style={{
                            display: "inline-flex",
                            gap: "6px",
                            justifyContent: "flex-end",
                          }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => onSelectTelemetry(sim)}
                            title="View live training telemetry"
                          >
                            <IconChartUp size={14} /> Telemetry
                          </button>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => onSelectClinician(sim)}
                            title="View clinician report"
                          >
                            <IconBrain size={14} /> XAI
                          </button>
                          <button
                            type="button"
                            className="btn btn-danger btn-sm"
                            disabled={deletingId === sim.id}
                            onClick={(e) => handleDelete(sim.id, e)}
                            title="Delete"
                          >
                            {deletingId === sim.id ? "…" : <IconClose size={12} />}
                          </button>
                        </div>
                      </td>
                    </tr>

                    {/* Expanded Detail Row */}
                    {isExpanded && (
                      <tr className="expanded-details">
                        <td colSpan={6}>
                          <div className="detail-grid">
                            <div className="detail-item">
                              <div className="detail-item-label">Dataset</div>
                              <div className="detail-item-value">{sim.dataset}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">Model</div>
                              <div className="detail-item-value">{sim.model}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">Clients</div>
                              <div className="detail-item-value">{sim.num_clients}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">Epochs</div>
                              <div className="detail-item-value">{sim.local_epochs}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">Batch Size</div>
                              <div className="detail-item-value">{sim.batch_size}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">LR</div>
                              <div className="detail-item-value">{sim.learning_rate}</div>
                            </div>
                            <div className="detail-item">
                              <div className="detail-item-label">Partition</div>
                              <div className="detail-item-value">{sim.partition_type}</div>
                            </div>
                            {sim.final_loss !== null && sim.final_loss !== undefined && (
                              <div className="detail-item">
                                <div className="detail-item-label">Final Loss</div>
                                <div className="detail-item-value" style={{ color: "var(--alert-color)" }}>
                                  {sim.final_loss.toFixed(4)}
                                </div>
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

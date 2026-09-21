"use client";

import React, { useState } from "react";
import { Simulation, deleteSimulation } from "@/lib/api";

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

  async function handleDelete(id: number) {
    if (!confirm(`Are you sure you want to delete simulation #${id}?`)) return;
    try {
      setDeletingId(id);
      await deleteSimulation(id);
      onRefresh();
    } catch (err) {
      alert("Failed to delete simulation: " + err);
    } finally {
      setDeletingId(null);
    }
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
            <span>📋</span> Experiment History &amp; Model Registry
          </div>
          <div className="card-desc">
            Decentralized federated training runs, communication rounds, and verified global accuracy.
          </div>
        </div>

        <div style={{ display: "flex", gap: "1rem" }}>
          <div className="stat-card" style={{ padding: "0.5rem 1rem", textAlign: "center" }}>
            <div className="stat-label">Total Runs</div>
            <div className="stat-value" style={{ fontSize: "1.2rem" }}>
              {simulations.length}
            </div>
          </div>
          <div className="stat-card" style={{ padding: "0.5rem 1rem", textAlign: "center" }}>
            <div className="stat-label">Active</div>
            <div
              className="stat-value"
              style={{ fontSize: "1.2rem", color: activeCount > 0 ? "var(--cyan)" : "inherit" }}
            >
              {activeCount}
            </div>
          </div>
        </div>
      </div>

      {isLoading && simulations.length === 0 ? (
        <div style={{ textAlign: "center", padding: "2rem", color: "var(--text-secondary)" }}>
          Loading simulations...
        </div>
      ) : simulations.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🔬</div>
          <div className="empty-state-title">No Simulations Found</div>
          <div className="empty-state-desc">
            No federated experiments have been executed yet. Configure your model, dataset, and clients in the form above and click &quot;Launch Federated Simulation&quot; to begin.
          </div>
        </div>
      ) : (
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Run ID</th>
                <th>Name</th>
                <th>Algorithm</th>
                <th>Dataset</th>
                <th>Rounds</th>
                <th>Status</th>
                <th>Accuracy</th>
                <th style={{ textAlign: "right" }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {simulations.map((sim) => {
                const statusBadgeClass = `badge-${sim.status || "created"}`;
                const accText =
                  sim.final_accuracy !== null && sim.final_accuracy !== undefined
                    ? `${sim.final_accuracy.toFixed(1)}%`
                    : "—";

                return (
                  <tr key={sim.id}>
                    <td>
                      <strong>#{sim.id}</strong>
                    </td>
                    <td
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      {sim.run_id ? `${sim.run_id.slice(0, 12)}...` : "—"}
                    </td>
                    <td>
                      <strong>{sim.name}</strong>
                    </td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background: "rgba(99, 102, 241, 0.15)",
                          color: "#818cf8",
                        }}
                      >
                        {sim.algorithm.toUpperCase()}
                      </span>
                    </td>
                    <td style={{ color: "var(--text-secondary)" }}>{sim.dataset}</td>
                    <td>
                      {sim.current_round} / {sim.num_rounds}
                    </td>
                    <td>
                      <span className={`badge ${statusBadgeClass}`}>{sim.status}</span>
                    </td>
                    <td>
                      <strong
                        style={{
                          color:
                            sim.final_accuracy !== null ? "var(--emerald)" : "var(--text-muted)",
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
                      >
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => onSelectTelemetry(sim)}
                          title="View live training telemetry"
                        >
                          Telemetry
                        </button>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => onSelectClinician(sim)}
                          title="View clinician explainability report"
                        >
                          XAI Report
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger btn-sm"
                          disabled={deletingId === sim.id}
                          onClick={() => handleDelete(sim.id)}
                          title="Delete simulation"
                        >
                          {deletingId === sim.id ? "..." : "✕"}
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

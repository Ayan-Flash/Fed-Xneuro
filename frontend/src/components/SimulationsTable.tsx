"use client";

import React, { useState } from "react";
import { Simulation, deleteSimulation } from "@/lib/api";
import { ListFilter, Activity, Play, LayoutDashboard, Search, Trash2 } from "lucide-react";

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
      <div className="flex justify-between items-start mb-5 border-b border-[var(--border-color)] pb-4">
        <div>
          <div className="card-title flex items-center gap-2">
            <ListFilter className="text-[var(--primary)]" size={24} /> Experiment History &amp; Model Registry
          </div>
          <div className="card-desc mt-1">
            Decentralized federated training runs, communication rounds, and verified global accuracy.
          </div>
        </div>

        <div className="flex gap-4">
          <div className="stat-card px-4 py-2 text-center rounded-[var(--radius-md)] border border-[var(--border-color)] bg-white shadow-sm">
            <div className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Total Runs</div>
            <div className="text-lg font-bold text-[var(--text-primary)]">
              {simulations.length}
            </div>
          </div>
          <div className="stat-card px-4 py-2 text-center rounded-[var(--radius-md)] border border-[var(--border-color)] bg-white shadow-sm">
            <div className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wide">Active</div>
            <div
              className="text-lg font-bold"
              style={{ color: activeCount > 0 ? "var(--primary)" : "inherit" }}
            >
              {activeCount}
            </div>
          </div>
        </div>
      </div>

      {isLoading && simulations.length === 0 ? (
        <div className="text-center py-8 text-[var(--text-secondary)]">
          Loading simulations...
        </div>
      ) : simulations.length === 0 ? (
        <div className="empty-state text-center py-12 px-4 rounded-[var(--radius-md)] border border-dashed border-[var(--border-color)] bg-white">
          <div className="empty-state-icon flex justify-center mb-4">
             <Search size={48} className="text-[var(--border-color)]" />
          </div>
          <div className="empty-state-title text-xl font-bold text-[var(--text-primary)] mb-2">No Simulations Found</div>
          <div className="empty-state-desc text-[var(--text-secondary)] max-w-md mx-auto">
            No federated experiments have been executed yet. Configure your model, dataset, and clients in the form above and click &quot;Launch Federated Simulation&quot; to begin.
          </div>
        </div>
      ) : (
        <div className="table-container overflow-x-auto">
          <table className="data-table w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-[var(--border-color)]">
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">ID</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Run ID</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Name</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Algorithm</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Dataset</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Rounds</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Status</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide">Accuracy</th>
                <th className="py-3 px-4 font-semibold text-sm text-[var(--text-muted)] uppercase tracking-wide text-right">Actions</th>
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
                  <tr key={sim.id} className="border-b border-[var(--border-color)] hover:bg-[var(--bg-secondary)] transition-colors">
                    <td className="py-3 px-4">
                      <strong className="text-[var(--text-primary)]">#{sim.id}</strong>
                    </td>
                    <td
                      className="py-3 px-4"
                      style={{
                        fontFamily: "var(--font-mono)",
                        fontSize: "0.75rem",
                        color: "var(--text-muted)",
                      }}
                    >
                      {sim.run_id ? `${sim.run_id.slice(0, 12)}...` : "—"}
                    </td>
                    <td className="py-3 px-4">
                      <strong className="text-[var(--text-primary)]">{sim.name}</strong>
                    </td>
                    <td className="py-3 px-4">
                      <span
                        className="badge inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                        style={{
                          background: "var(--primary-glow)",
                          color: "var(--primary-dark)",
                        }}
                      >
                        {sim.algorithm.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-[var(--text-secondary)]">{sim.dataset}</td>
                    <td className="py-3 px-4 text-[var(--text-primary)]">
                      {sim.current_round} / {sim.num_rounds}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`badge ${statusBadgeClass} inline-flex items-center px-2 py-0.5 rounded text-xs font-medium`}>{sim.status}</span>
                    </td>
                    <td className="py-3 px-4">
                      <strong
                        style={{
                          color:
                            sim.final_accuracy !== null ? "var(--secondary)" : "var(--text-muted)",
                        }}
                      >
                        {accText}
                      </strong>
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex gap-2 justify-end">
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm flex items-center gap-1 bg-white border border-[var(--border-color)] hover:bg-[var(--bg-secondary)] text-[var(--text-primary)] px-2 py-1 rounded text-sm transition-colors"
                          onClick={() => onSelectTelemetry(sim)}
                          title="View live training telemetry"
                        >
                          <Activity size={14} /> Telemetry
                        </button>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm flex items-center gap-1 bg-white border border-[var(--border-color)] hover:bg-[var(--bg-secondary)] text-[var(--text-primary)] px-2 py-1 rounded text-sm transition-colors"
                          onClick={() => onSelectClinician(sim)}
                          title="View clinician explainability report"
                        >
                          <LayoutDashboard size={14} /> XAI Report
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger btn-sm flex items-center gap-1 bg-white border border-[var(--crimson)] text-[var(--crimson)] hover:bg-[var(--crimson-glow)] px-2 py-1 rounded text-sm transition-colors disabled:opacity-50"
                          disabled={deletingId === sim.id}
                          onClick={() => handleDelete(sim.id)}
                          title="Delete simulation"
                        >
                          {deletingId === sim.id ? "..." : <Trash2 size={14} />}
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

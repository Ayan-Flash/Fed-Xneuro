"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Navbar } from "@/components/Navbar";
import { SimulationLauncher } from "@/components/SimulationLauncher";
import { SimulationsTable } from "@/components/SimulationsTable";
import { LiveTelemetry } from "@/components/LiveTelemetry";
import { ClinicianReport } from "@/components/ClinicianReport";
import { Simulation, getSimulations } from "@/lib/api";

export default function AdminDashboard() {
  const [activeTab, setActiveTab] = useState<"simulations" | "telemetry" | "clinician">("simulations");
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [selectedSim, setSelectedSim] = useState<Simulation | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchSims = useCallback(async () => {
    try {
      const list = await getSimulations();
      setSimulations(list);

      // Keep selectedSim updated with fresh data if present
      setSelectedSim((prev) => {
        if (!prev) return prev;
        const found = list.find((s) => s.id === prev.id);
        return found || prev;
      });
    } catch (err) {
      console.error("Failed to fetch simulations:", err);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchSims();
    const timer = setInterval(fetchSims, 4000);
    return () => clearInterval(timer);
  }, [fetchSims]);

  function handleLaunched(newSim: Simulation) {
    fetchSims();
    setSelectedSim(newSim);
    setActiveTab("telemetry");
  }

  function handleSelectTelemetry(sim: Simulation) {
    setSelectedSim(sim);
    setActiveTab("telemetry");
  }

  function handleSelectClinician(sim: Simulation) {
    setSelectedSim(sim);
    setActiveTab("clinician");
  }

  return (
    <div>
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} role="admin" />

      <main className="main-container">
        {activeTab === "simulations" && (
          <div className="grid-1">
            <SimulationLauncher onLaunched={handleLaunched} />
            <SimulationsTable
              simulations={simulations}
              isLoading={isLoading}
              onRefresh={fetchSims}
              onSelectTelemetry={handleSelectTelemetry}
              onSelectClinician={handleSelectClinician}
            />
          </div>
        )}

        {activeTab === "telemetry" && (
          <LiveTelemetry
            simulation={selectedSim}
            onSelectAnother={() => setActiveTab("simulations")}
          />
        )}

        {activeTab === "clinician" && (
          <ClinicianReport
            simulation={selectedSim}
            onSelectAnother={() => setActiveTab("simulations")}
          />
        )}
      </main>
    </div>
  );
}

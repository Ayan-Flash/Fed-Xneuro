"use client";

import React, { useState, useEffect, useCallback } from "react";
import { Navbar } from "@/components/Navbar";
import { SimulationLauncher } from "@/components/SimulationLauncher";
import { SimulationsTable } from "@/components/SimulationsTable";
import { LiveTelemetry } from "@/components/LiveTelemetry";
import { ClinicianReport } from "@/components/ClinicianReport";
import { Simulation, getSimulations } from "@/lib/api";

import { useSearchParams } from "next/navigation";

function DashboardContent() {
  const searchParams = useSearchParams();
  const role = searchParams.get("role") || "clinician";

  const [activeTab, setActiveTab] = useState<"simulations" | "telemetry" | "clinician">("simulations");
  const [simulations, setSimulations] = useState<Simulation[]>([]);
  const [selectedSim, setSelectedSim] = useState<Simulation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [tabKey, setTabKey] = useState(0);

  const fetchSims = useCallback(async () => {
    try {
      const list = await getSimulations(role);
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

  function handleTabChange(tab: "simulations" | "telemetry" | "clinician") {
    setActiveTab(tab);
    setTabKey((k) => k + 1); // force re-mount for animation
  }

  function handleLaunched(newSim: Simulation) {
    fetchSims();
    setSelectedSim(newSim);
    setActiveTab("telemetry");
    setTabKey((k) => k + 1);
  }

  function handleSelectTelemetry(sim: Simulation) {
    setSelectedSim(sim);
    setActiveTab("telemetry");
    setTabKey((k) => k + 1);
  }

  function handleSelectClinician(sim: Simulation) {
    setSelectedSim(sim);
    setActiveTab("clinician");
    setTabKey((k) => k + 1);
  }

  const isResearcher = role === "researcher";

  return (
    <div>
      <Navbar activeTab={activeTab} onTabChange={handleTabChange} role={role} />

      <main className="main-container" key={tabKey}>
        {activeTab === "simulations" && (
          <div className="grid-1 animate-tab-slide">
            {isResearcher && <SimulationLauncher onLaunched={handleLaunched} />}
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
            onSelectAnother={() => handleTabChange("simulations")}
          />
        )}

        {activeTab === "clinician" && (
          <ClinicianReport
            simulation={selectedSim}
            onSelectAnother={() => handleTabChange("simulations")}
          />
        )}
      </main>
    </div>
  );
}

export default function Dashboard() {
  return (
    <React.Suspense fallback={<div style={{ padding: "2rem", textAlign: "center" }}>Loading Dashboard...</div>}>
      <DashboardContent />
    </React.Suspense>
  );
}

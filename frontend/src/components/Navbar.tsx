"use client";

import React, { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";
import { IconMicroscope, IconChartUp, IconBrain } from "@/components/Icons";

interface NavbarProps {
  activeTab: "simulations" | "telemetry" | "clinician";
  onTabChange: (tab: "simulations" | "telemetry" | "clinician") => void;
  role?: string;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, role = "clinician" }) => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;

    async function verifyHealth() {
      const ok = await checkBackendHealth();
      if (mounted) setIsOnline(ok);
    }

    verifyHealth();
    const interval = setInterval(verifyHealth, 8000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const isClinician = role === "clinician";
  const isResearcher = role === "researcher";
  const isAdmin = role === "admin";

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-icon">X</div>
          <div>
            <div className="brand-title">FED-XNEURO</div>
            <div className="brand-subtitle">
              Federated Clinical AI for Alzheimer&apos;s Progression
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
          <div className="badge badge-created" style={{ background: "rgba(25, 124, 130, 0.1)", color: "var(--primary)" }}>
            {isClinician ? "Clinician Mode" : isResearcher ? "Researcher Mode" : "Admin Mode"}
          </div>
          <div className="status-badge">
            <span
              className={`status-dot ${
                isOnline === null ? "warning" : isOnline ? "" : "offline"
              }`}
            />
            <span>
              {isOnline === null
                ? "Connecting…"
                : isOnline
                ? "System Online"
                : "System Offline"}
            </span>
          </div>
        </div>
      </header>

      <nav className="nav-tabs">
        <button
          type="button"
          className={`tab-btn ${activeTab === "simulations" ? "active" : ""}`}
          onClick={() => onTabChange("simulations")}
        >
          <IconMicroscope size={16} /> {isClinician ? "Case History" : "Experiments"}
        </button>

        {(isResearcher || isAdmin) && (
          <button
            type="button"
            className={`tab-btn ${activeTab === "telemetry" ? "active" : ""}`}
            onClick={() => onTabChange("telemetry")}
          >
            <IconChartUp size={16} /> Live Telemetry
          </button>
        )}

        {isClinician && (
          <button
            type="button"
            className={`tab-btn ${activeTab === "clinician" ? "active" : ""}`}
            onClick={() => onTabChange("clinician")}
          >
            <IconBrain size={16} /> Clinician XAI
          </button>
        )}
      </nav>
    </>
  );
};

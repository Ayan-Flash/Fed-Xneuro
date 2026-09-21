"use client";

import React, { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";

interface NavbarProps {
  activeTab: "simulations" | "telemetry" | "clinician";
  onTabChange: (tab: "simulations" | "telemetry" | "clinician") => void;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange }) => {
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

  return (
    <>
      <header className="header">
        <div className="brand">
          <div className="brand-icon">X</div>
          <div>
            <div className="brand-title">FED-XNEURO PLATFORM</div>
            <div className="brand-subtitle">
              Next.js Clinician &amp; Research Federated Learning Dashboard
            </div>
          </div>
        </div>

        <div className="header-status">
          <div className="status-badge">
            <span
              className={`status-dot ${
                isOnline === null ? "warning" : isOnline ? "" : "offline"
              }`}
            />
            <span>
              {isOnline === null
                ? "Connecting to API..."
                : isOnline
                ? "Backend Online (:8000)"
                : "Backend Offline"}
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
          <span>🔬</span> Experiments &amp; Launcher
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === "telemetry" ? "active" : ""}`}
          onClick={() => onTabChange("telemetry")}
        >
          <span>📈</span> Live FL Telemetry
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === "clinician" ? "active" : ""}`}
          onClick={() => onTabChange("clinician")}
        >
          <span>🧠</span> Clinician Diagnostic XAI
        </button>
      </nav>
    </>
  );
};

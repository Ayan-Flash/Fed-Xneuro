"use client";

import React, { useEffect, useState } from "react";
import { checkBackendHealth } from "@/lib/api";
import { useRouter } from "next/navigation";
import { BrainCircuit, Activity, TestTube2, Building2, ShieldAlert, LogOut } from "lucide-react";

interface NavbarProps {
  activeTab: "simulations" | "telemetry" | "clinician";
  onTabChange: (tab: "simulations" | "telemetry" | "clinician") => void;
  role?: "admin" | "hospital";
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, onTabChange, role = "hospital" }) => {
  const [isOnline, setIsOnline] = useState<boolean | null>(null);
  const router = useRouter();

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
      <header className="header flex justify-between items-center px-6 py-4 bg-white border-b border-[var(--border-color)]">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-full bg-[var(--primary-glow)] flex items-center justify-center text-[var(--primary)]">
            <BrainCircuit size={24} />
          </div>
          <div>
            <div className="font-bold text-[var(--text-primary)] text-xl tracking-tight">FED-XNEURO</div>
            <div className="text-xs text-[var(--text-secondary)] font-medium uppercase tracking-wider">
              {role === 'admin' ? 'System Administration' : 'Clinical Diagnostics'}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-[var(--bg-primary)] rounded-full border border-[var(--border-color)] text-sm">
            {role === 'admin' ? <ShieldAlert size={16} className="text-[var(--primary)]" /> : <Building2 size={16} className="text-[var(--primary)]" />}
            <span className="font-semibold text-[var(--text-primary)] capitalize">{role} Access</span>
          </div>

          <div className="header-status flex items-center gap-2 px-3 py-1.5">
            <span className={`w-2.5 h-2.5 rounded-full ${isOnline === null ? "bg-amber-400" : isOnline ? "bg-emerald-500" : "bg-red-500"}`} />
            <span className="text-sm font-medium text-[var(--text-secondary)]">
              {isOnline === null ? "Connecting..." : isOnline ? "System Online" : "System Offline"}
            </span>
          </div>
          
          <button 
            onClick={() => router.push('/login')}
            className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--primary)] hover:bg-[var(--primary-glow)] rounded-md transition-colors"
          >
            <LogOut size={16} /> Logout
          </button>
        </div>
      </header>

      <nav className="nav-tabs flex bg-[var(--bg-primary)] border-b border-[var(--border-color)] px-6">
        <button
          type="button"
          className={`flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-2 transition-colors ${activeTab === "simulations" ? "border-[var(--primary)] text-[var(--primary)] bg-white" : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/50"}`}
          onClick={() => onTabChange("simulations")}
        >
          <TestTube2 size={18} /> Experiments &amp; Launcher
        </button>

        <button
          type="button"
          className={`flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-2 transition-colors ${activeTab === "telemetry" ? "border-[var(--primary)] text-[var(--primary)] bg-white" : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/50"}`}
          onClick={() => onTabChange("telemetry")}
        >
          <Activity size={18} /> Live FL Telemetry
        </button>

        <button
          type="button"
          className={`flex items-center gap-2 px-6 py-4 text-sm font-semibold border-b-2 transition-colors ${activeTab === "clinician" ? "border-[var(--primary)] text-[var(--primary)] bg-white" : "border-transparent text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/50"}`}
          onClick={() => onTabChange("clinician")}
        >
          <BrainCircuit size={18} /> Clinician Diagnostic XAI
        </button>
      </nav>
    </>
  );
};

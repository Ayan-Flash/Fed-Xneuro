"use client";

import React from "react";
import { useRouter } from "next/navigation";
import { Brain, Activity, Shield, ArrowRight } from "lucide-react"; // I'll use the existing Icons or build my own since lucide-react might not be installed. Wait, no new npm dependencies permitted!

// Let me use the custom Icons from @/components/Icons
import { 
  IconSatellite, 
  IconScan, 
  IconBarChart 
} from "@/components/Icons";

export default function LandingPage() {
  const router = useRouter();

  return (
    <div style={{ minHeight: "100vh", display: "flex", flexDirection: "column" }}>
      {/* Navigation */}
      <header className="header">
        <div className="brand">
          <div className="brand-icon">F</div>
          <div>
            <h1 className="brand-title">FedX Nuro</h1>
            <div className="brand-subtitle">Federated Neuro-Simulation</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: "1rem" }}>
          <button 
            className="btn btn-secondary" 
            onClick={() => router.push("/login")}
          >
            Sign In
          </button>
          <button 
            className="btn btn-primary"
            onClick={() => router.push("/login")}
          >
            Get Started
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "4rem 2rem", textAlign: "center" }}>
        <div className="animate-fade-up" style={{ maxWidth: "800px" }}>
          <h1 style={{ fontSize: "3.5rem", fontWeight: 800, color: "var(--text-primary)", marginBottom: "1.5rem", lineHeight: 1.1 }}>
            Secure, Federated AI for Neuroimaging
          </h1>
          <p style={{ fontSize: "1.25rem", color: "var(--text-secondary)", marginBottom: "2.5rem", lineHeight: 1.6 }}>
            Run advanced neurological simulations across decentralized hospital nodes without ever moving sensitive patient data. 
            Powered by privacy-preserving federated learning.
          </p>
          <div style={{ display: "flex", gap: "1rem", justifyContent: "center" }}>
            <button 
              className="btn btn-primary" 
              style={{ fontSize: "1.1rem", padding: "0.8rem 1.5rem" }}
              onClick={() => router.push("/login")}
            >
              Start Simulating
            </button>
            <button 
              className="btn btn-secondary" 
              style={{ fontSize: "1.1rem", padding: "0.8rem 1.5rem" }}
              onClick={() => router.push("/dashboard")}
            >
              View Demo Dashboard
            </button>
          </div>
        </div>

        {/* Feature Cards */}
        <div className="grid-3 animate-fade-up stagger-2" style={{ marginTop: "5rem", maxWidth: "1200px" }}>
          <div className="card">
            <div style={{ color: "var(--primary)", marginBottom: "1rem" }}>
              <IconSatellite />
            </div>
            <h3 className="card-title">Federated Learning</h3>
            <p className="card-desc">
              Train on global neuroimaging datasets collaboratively while keeping patient data strictly local.
            </p>
          </div>
          <div className="card">
            <div style={{ color: "var(--primary)", marginBottom: "1rem" }}>
              <IconScan />
            </div>
            <h3 className="card-title">Privacy Preserving</h3>
            <p className="card-desc">
              Enterprise-grade encryption and decentralized model aggregation ensures zero data leakage.
            </p>
          </div>
          <div className="card">
            <div style={{ color: "var(--primary)", marginBottom: "1rem" }}>
              <IconBarChart />
            </div>
            <h3 className="card-title">Real-Time Telemetry</h3>
            <p className="card-desc">
              Monitor node participation, accuracy metrics, and loss curves live as the simulation converges.
            </p>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{ 
        padding: "2rem", 
        borderTop: "1px solid var(--border-color)", 
        background: "rgba(248, 245, 237, 0.4)",
        display: "flex", 
        justifyContent: "space-between",
        alignItems: "center",
        fontSize: "0.85rem",
        color: "var(--text-secondary)"
      }}>
        <div>&copy; {new Date().getFullYear()} FedX Nuro. All rights reserved.</div>
        <div style={{ display: "flex", gap: "1.5rem" }}>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>Privacy Policy</a>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>Terms of Service</a>
          <a href="#" style={{ color: "var(--text-secondary)", textDecoration: "none" }}>Contact Support</a>
        </div>
      </footer>
    </div>
  );
}

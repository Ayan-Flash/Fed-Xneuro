"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [role, setRole] = useState("clinician");
  const [email, setEmail] = useState("");

  const handleRoleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newRole = e.target.value;
    setRole(newRole);
    // Auto-fill a demo email based on role
    if (newRole === "clinician") setEmail("dr.smith@hospital.edu");
    else if (newRole === "researcher") setEmail("research@fedx.ai");
    else if (newRole === "admin") setEmail("it-admin@hospital.edu");
    else setEmail("");
  };

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    
    // Simulate authentication delay
    setTimeout(() => {
      router.push(`/dashboard?role=${role}`);
    }, 1000);
  };

  return (
    <div style={{ 
      minHeight: "100vh", 
      display: "flex", 
      alignItems: "center", 
      justifyContent: "center",
      background: "var(--bg-primary)" 
    }}>
      <div className="card animate-fade-up" style={{ width: "100%", maxWidth: "420px", padding: "2.5rem" }}>
        <div style={{ textAlign: "center", marginBottom: "2rem" }}>
          <div className="brand-icon" style={{ margin: "0 auto 1rem auto", width: "50px", height: "50px", fontSize: "1.5rem" }}>
            F
          </div>
          <h2 style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--text-primary)" }}>Welcome Back</h2>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.9rem", marginTop: "0.5rem" }}>
            Sign in to FedX Nuro
          </p>
        </div>

        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label className="form-label" htmlFor="role">Select Role</label>
            <select 
              id="role" 
              className="form-control" 
              value={role} 
              onChange={handleRoleChange}
              style={{ padding: "0.75rem", fontSize: "0.95rem" }}
            >
              <option value="clinician">Clinician / Neurologist</option>
              <option value="researcher">AI Researcher</option>
              <option value="admin">Hospital IT Admin</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="email">Email Address</label>
            <input 
              type="email" 
              id="email"
              className="form-control" 
              placeholder="clinician@hospital.edu" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="form-group" style={{ marginBottom: "0.5rem" }}>
            <label className="form-label" htmlFor="password">Password</label>
            <input 
              type="password" 
              id="password"
              className="form-control" 
              placeholder="••••••••" 
              defaultValue="password123"
              required
            />
          </div>
          
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1.5rem" }}>
            <a href="#" style={{ fontSize: "0.8rem", color: "var(--primary)", textDecoration: "none", fontWeight: 600 }}>
              Forgot Password?
            </a>
          </div>

          <button 
            type="submit" 
            className={`btn btn-primary ${isLoading ? "launching" : ""}`} 
            style={{ width: "100%", padding: "0.85rem", fontSize: "1rem" }}
            disabled={isLoading}
          >
            {isLoading ? "Authenticating..." : "Sign In"}
          </button>
        </form>

        <div style={{ marginTop: "1.5rem", textAlign: "center" }}>
          <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
            Don't have an account?{" "}
            <a href="#" style={{ color: "var(--primary)", textDecoration: "none", fontWeight: 600 }}>
              Request Access
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

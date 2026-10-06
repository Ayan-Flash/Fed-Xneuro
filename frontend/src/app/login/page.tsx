"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import { 
  IconAdmin, 
  IconHospital, 
  IconBrain, 
  IconError, 
  IconCheck, 
  IconSecurity, 
  IconLightning, 
  IconRocket,
  IconScan,
  IconAssessment
} from "@/components/Icons";
import { loginUser, checkBackendHealth, UserProfile } from "@/lib/api";

type Role = "hospital" | "admin";

interface DemoAccount {
  role: Role;
  name: string;
  title: string;
  email: string;
  password: string;
  organization: string;
  nodeType: string;
  features: string[];
  redirectPath: string;
}

const DEMO_ACCOUNTS: DemoAccount[] = [
  {
    role: "hospital",
    name: "Dr. Sarah Chen, MD",
    title: "Senior Neurologist & Clinical Investigator",
    email: "doctor@memorial.org",
    password: "demo123",
    organization: "Memorial Neuroscience Center (Hospital Node A)",
    nodeType: "Hospital Edge Client",
    features: [
      "Patient MRI upload & 3D Hippocampal heatmap review",
      "Multimodal cognitive score assessment (MMSE, CDR-SB)",
      "Longitudinal disease progression risk forecasting",
      "Local on-device Differential Privacy (DP-FL) guard",
    ],
    redirectPath: "/dashboard/hospital",
  },
  {
    role: "admin",
    name: "Alex Rivera",
    title: "Lead FL Architect & Network Coordinator",
    email: "admin@platform.com",
    password: "admin123",
    organization: "Fed-XNeuro Central Aggregation Server",
    nodeType: "Global Orchestration Node",
    features: [
      "Federated round management (FedAvg, FedProx, SCAFFOLD)",
      "Multi-hospital node telemetry & communication latency",
      "Cumulative Differential Privacy budget accounting (ε, δ)",
      "Comparative benchmark suites & global model checkpointing",
    ],
    redirectPath: "/dashboard/admin",
  },
];

export default function LoginPage() {
  const router = useRouter();
  const [role, setRole] = useState<Role>("hospital");
  const [email, setEmail] = useState("doctor@memorial.org");
  const [password, setPassword] = useState("demo123");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [backendLatency, setBackendLatency] = useState<number | null>(null);

  // Check backend health on mount
  useEffect(() => {
    let isMounted = true;
    const checkHealth = async () => {
      const startTime = performance.now();
      try {
        await checkBackendHealth();
        if (isMounted) {
          setBackendLatency(Math.round(performance.now() - startTime));
          setBackendStatus("online");
        }
      } catch {
        if (isMounted) {
          setBackendStatus("offline");
        }
      }
    };
    checkHealth();
    return () => {
      isMounted = false;
    };
  }, []);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage((current) => (current === msg ? null : current));
    }, 3000);
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    showToast(`Copied ${text} to clipboard`);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const applyDemoAccount = (account: DemoAccount) => {
    setRole(account.role);
    setEmail(account.email);
    setPassword(account.password);
    setError("");
    showToast(`Loaded ${account.role === "admin" ? "Admin" : "Clinician"} demo credentials`);
  };

  const handleQuickLaunch = async (account: DemoAccount) => {
    setRole(account.role);
    setEmail(account.email);
    setPassword(account.password);
    setError("");
    setIsLoading(true);
    showToast(`Launching ${account.name} session...`);

    try {
      // Attempt actual authentication
      const auth = await loginUser(account.email, account.password);
      if (typeof window !== "undefined") {
        localStorage.setItem("fedx_token", auth.access_token);
        const userObj: UserProfile = auth.user || {
          id: account.role === "admin" ? 2 : 1,
          email: account.email,
          full_name: account.name,
          is_active: true,
          is_superuser: account.role === "admin",
        };
        localStorage.setItem("fedx_user", JSON.stringify(userObj));
        localStorage.setItem("fedx_role", account.role);
      }
      router.push(account.redirectPath);
    } catch {
      // Seamless offline demo fallback
      if (typeof window !== "undefined") {
        localStorage.setItem("fedx_token", `demo-token-${account.role}-${Date.now()}`);
        localStorage.setItem(
          "fedx_user",
          JSON.stringify({
            id: account.role === "admin" ? 2 : 1,
            email: account.email,
            full_name: account.name,
            is_active: true,
            is_superuser: account.role === "admin",
          })
        );
        localStorage.setItem("fedx_role", account.role);
      }
      showToast("Offline Demo Mode activated — redirecting...");
      setTimeout(() => {
        router.push(account.redirectPath);
      }, 500);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");

    if (!email || !password) {
      setError("Please provide both email and password.");
      return;
    }

    setIsLoading(true);

    try {
      // 1. Try real backend login
      const auth = await loginUser(email, password);
      if (typeof window !== "undefined") {
        localStorage.setItem("fedx_token", auth.access_token);
        const resolvedRole = auth.user?.is_superuser ? "admin" : role;
        const userObj = auth.user || {
          id: resolvedRole === "admin" ? 2 : 1,
          email,
          full_name: resolvedRole === "admin" ? "Alex Rivera" : "Dr. Sarah Chen, MD",
          is_active: true,
          is_superuser: resolvedRole === "admin",
        };
        localStorage.setItem("fedx_user", JSON.stringify(userObj));
        localStorage.setItem("fedx_role", resolvedRole);
        
        router.push(resolvedRole === "admin" ? "/dashboard/admin" : "/dashboard/hospital");
      }
    } catch (err: unknown) {
      // 2. Check if matching demo credentials were used in offline mode
      const isClinicianMatch = email === "doctor@memorial.org" && password === "demo123";
      const isAdminMatch = email === "admin@platform.com" && password === "admin123";

      if (isClinicianMatch || isAdminMatch) {
        const fallbackRole = isAdminMatch ? "admin" : "hospital";
        const fallbackName = isAdminMatch ? "Alex Rivera" : "Dr. Sarah Chen, MD";
        if (typeof window !== "undefined") {
          localStorage.setItem("fedx_token", `demo-offline-${Date.now()}`);
          localStorage.setItem(
            "fedx_user",
            JSON.stringify({
              id: fallbackRole === "admin" ? 2 : 1,
              email,
              full_name: fallbackName,
              is_active: true,
              is_superuser: fallbackRole === "admin",
            })
          );
          localStorage.setItem("fedx_role", fallbackRole);
        }
        showToast("Backend offline: Verified via Local Demo Registry");
        setTimeout(() => {
          router.push(fallbackRole === "admin" ? "/dashboard/admin" : "/dashboard/hospital");
        }, 500);
      } else {
        const msg = err instanceof Error ? err.message : "Invalid email or password.";
        setError(msg);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)] flex flex-col font-sans">
      <Navbar />

      {/* Floating Action Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 animate-fade-in-up bg-gray-900 text-white text-xs sm:text-sm font-medium px-4 py-3 rounded-xl shadow-xl flex items-center gap-3 border border-gray-700">
          <div className="w-2 h-2 rounded-full bg-teal-400 animate-ping" />
          <span>{toastMessage}</span>
        </div>
      )}

      <div className="flex-1 flex flex-col items-center justify-center py-10 px-4 sm:px-6 lg:px-8 mt-14 relative">
        {/* Ambient background glow elements */}
        <div className="absolute top-1/4 left-1/3 -translate-x-1/2 w-96 h-96 bg-[var(--color-primary)]/10 rounded-full blur-3xl -z-10 pointer-events-none" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-[var(--color-secondary)]/10 rounded-full blur-3xl -z-10 pointer-events-none" />

        {/* Top Header & System Telemetry Status Banner */}
        <div className="max-w-5xl w-full text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200/80 text-[11px] font-semibold text-[var(--color-primary)] mb-3 shadow-xs">
            <span className="w-2 h-2 rounded-full bg-[var(--color-primary)] animate-pulse" />
            <span>Interactive Demo & Testing Environment</span>
            <span className="text-gray-300">|</span>
            <span className="text-gray-500 font-mono">v2.4 Live Multi-Modal</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold text-[var(--color-text-main)] tracking-tight">
            Fed-XNeuro Access Portal
          </h1>
          <p className="mt-2 text-sm text-gray-500 max-w-xl mx-auto">
            Federated learning diagnostic gateway for privacy-preserving MCI-to-AD progression prediction.
          </p>

          {/* Real-time Environment Status Bar */}
          <div className="mt-5 max-w-3xl mx-auto grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
            <div className="bg-white/90 backdrop-blur-xs border border-gray-200/80 rounded-xl px-3.5 py-2 flex items-center justify-between shadow-2xs">
              <span className="text-gray-500">Backend API</span>
              <span className="inline-flex items-center gap-1.5 font-medium">
                {backendStatus === "checking" && (
                  <span className="text-gray-400">Pinging...</span>
                )}
                {backendStatus === "online" && (
                  <>
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-emerald-700 font-semibold">Online {backendLatency ? `(${backendLatency}ms)` : ""}</span>
                  </>
                )}
                {backendStatus === "offline" && (
                  <>
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="text-amber-700 font-semibold">Offline (Demo Mode)</span>
                  </>
                )}
              </span>
            </div>

            <div className="bg-white/90 backdrop-blur-xs border border-gray-200/80 rounded-xl px-3.5 py-2 flex items-center justify-between shadow-2xs">
              <span className="text-gray-500">Active Model</span>
              <span className="font-semibold text-gray-700 flex items-center gap-1">
                <IconBrain size={14} className="text-[var(--color-primary)]" />
                Fed-XNeuro 3D ResNet
              </span>
            </div>

            <div className="bg-white/90 backdrop-blur-xs border border-gray-200/80 rounded-xl px-3.5 py-2 flex items-center justify-between shadow-2xs">
              <span className="text-gray-500">DP-FL Guard</span>
              <span className="font-semibold text-emerald-700 flex items-center gap-1">
                <IconSecurity size={14} className="text-emerald-600" />
                ε = 2.45, δ = 10⁻⁵
              </span>
            </div>
          </div>
        </div>

        {/* Main Content Layout: Form Card + Demo Personas */}
        <div className="max-w-5xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* LEFT COLUMN: Login Form (5 cols on lg) */}
          <div className="lg:col-span-6 bg-white py-7 px-6 sm:px-8 shadow-xl shadow-teal-900/5 rounded-2xl border border-gray-100 relative">
            <div className="flex items-center justify-between pb-4 mb-5 border-b border-gray-100">
              <div>
                <h2 className="text-lg font-bold text-gray-800">Sign In with Credentials</h2>
                <p className="text-xs text-gray-500">Enter your credentials or choose a quick role</p>
              </div>
              <div className="w-10 h-10 bg-teal-50 rounded-xl flex items-center justify-center text-[var(--color-primary)] border border-teal-100">
                <IconBrain size={22} />
              </div>
            </div>

            {/* Role Switcher Tabs */}
            <div className="flex p-1 bg-gray-50/90 rounded-xl mb-5 border border-gray-200/60">
              <button
                type="button"
                onClick={() => {
                  setRole("hospital");
                  setEmail("doctor@memorial.org");
                  setPassword("demo123");
                  setError("");
                }}
                className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-all ${
                  role === "hospital"
                    ? "bg-white text-[var(--color-primary)] shadow-xs border border-gray-200/80"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconHospital size={16} />
                <span>Hospital Clinician</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setRole("admin");
                  setEmail("admin@platform.com");
                  setPassword("admin123");
                  setError("");
                }}
                className={`flex-1 flex items-center justify-center gap-2 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-all ${
                  role === "admin"
                    ? "bg-white text-[var(--color-primary)] shadow-xs border border-gray-200/80"
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconAdmin size={16} />
                <span>System Admin</span>
              </button>
            </div>

            {error && (
              <div className="mb-4 bg-red-50/90 border border-red-200 text-red-700 px-3.5 py-2.5 rounded-xl flex items-center gap-2.5 text-xs sm:text-sm">
                <IconError size={16} className="shrink-0" />
                <span>{error}</span>
              </div>
            )}

            <form className="space-y-4" onSubmit={handleSubmit}>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5" htmlFor="email">
                  {role === "admin" ? "Administrator Email" : "Hospital Clinician Email"}
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-3.5 py-2.5 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] transition-all text-sm"
                  placeholder={role === "admin" ? "admin@platform.com" : "doctor@memorial.org"}
                />
              </div>

              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700" htmlFor="password">
                    Password
                  </label>
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="text-xs font-medium text-gray-500 hover:text-[var(--color-primary)] transition-colors"
                  >
                    {showPassword ? "Hide" : "Show"} password
                  </button>
                </div>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="appearance-none block w-full px-3.5 py-2.5 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] transition-all text-sm pr-12 font-mono"
                    placeholder="••••••••"
                  />
                </div>
              </div>

              <div className="flex items-center justify-between text-xs text-gray-500 pt-1">
                <label className="flex items-center gap-2 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    defaultChecked
                    className="rounded border-gray-300 text-[var(--color-primary)] focus:ring-[var(--color-primary)] h-4 w-4"
                  />
                  <span>Save session on device</span>
                </label>
                <span className="text-[11px] text-teal-700 font-medium">Demo Auth Bypass Enabled</span>
              </div>

              <div className="pt-2">
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full flex justify-center items-center py-3 px-4 rounded-xl shadow-md shadow-[var(--color-primary)]/20 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)] transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed hover:-translate-y-0.5 cursor-pointer"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      <span>Authenticating Session...</span>
                    </div>
                  ) : (
                    <span className="flex items-center gap-2">
                      <IconRocket size={16} />
                      Sign In as {role === "admin" ? "Administrator" : "Hospital Clinician"}
                    </span>
                  )}
                </button>
              </div>
            </form>

            {/* Instant Guest Sandbox Links */}
            <div className="mt-5 pt-4 border-t border-gray-100">
              <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2 text-center">
                Fast Sandbox Direct Links (No Sign In Required)
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Link
                  href="/dashboard/hospital"
                  className="flex items-center justify-center gap-1.5 py-2 px-3 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-teal-50 hover:text-[var(--color-primary)] rounded-lg border border-gray-200 transition-colors text-center"
                >
                  <IconHospital size={14} />
                  <span>Hospital View →</span>
                </Link>
                <Link
                  href="/dashboard/admin"
                  className="flex items-center justify-center gap-1.5 py-2 px-3 text-xs font-semibold text-gray-700 bg-gray-50 hover:bg-teal-50 hover:text-[var(--color-primary)] rounded-lg border border-gray-200 transition-colors text-center"
                >
                  <IconAdmin size={14} />
                  <span>Admin View →</span>
                </Link>
              </div>
            </div>
          </div>

          {/* RIGHT COLUMN: Quick-Test Demo Personas (6 cols on lg) */}
          <div className="lg:col-span-6 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-base font-bold text-gray-900 flex items-center gap-2">
                  <IconLightning size={18} className="text-amber-500" />
                  Pre-Configured Test Personas
                </h2>
                <p className="text-xs text-gray-500">
                  Click <strong className="text-gray-700">Launch</strong> for instant test login or auto-fill credentials.
                </p>
              </div>
              <span className="text-[10px] font-bold uppercase tracking-wider bg-gray-100 text-gray-600 px-2 py-1 rounded-md">
                2 Ready Accounts
              </span>
            </div>

            {DEMO_ACCOUNTS.map((account) => {
              const isSelected = role === account.role;
              return (
                <div
                  key={account.role}
                  className={`bg-white rounded-2xl p-5 border transition-all duration-200 ${
                    isSelected
                      ? "border-[var(--color-primary)] shadow-md shadow-teal-900/5 ring-1 ring-[var(--color-primary)]/20"
                      : "border-gray-200/80 hover:border-gray-300 shadow-xs"
                  }`}
                >
                  <div className="flex items-start justify-between gap-3 mb-2">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-xs ${
                        account.role === "admin"
                          ? "bg-indigo-50 text-indigo-700 border border-indigo-100"
                          : "bg-teal-50 text-[var(--color-primary)] border border-teal-100"
                      }`}>
                        {account.role === "admin" ? <IconAdmin size={20} /> : <IconHospital size={20} />}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h3 className="text-sm font-bold text-gray-900">{account.name}</h3>
                          <span className={`text-[10px] font-semibold px-2 py-0.5 rounded-full ${
                            account.role === "admin"
                              ? "bg-indigo-50 text-indigo-700 border border-indigo-200/60"
                              : "bg-teal-50 text-teal-700 border border-teal-200/60"
                          }`}>
                            {account.role === "admin" ? "Admin" : "Clinician"}
                          </span>
                        </div>
                        <p className="text-xs text-gray-500">{account.title}</p>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-50/80 rounded-xl p-3 my-3 border border-gray-100 text-xs font-mono space-y-1.5">
                    <div className="flex items-center justify-between text-gray-600">
                      <span className="text-gray-400 font-sans">Email:</span>
                      <div className="flex items-center gap-1.5">
                        <span>{account.email}</span>
                        <button
                          type="button"
                          onClick={() => copyToClipboard(account.email, `${account.role}-email`)}
                          className="text-gray-400 hover:text-[var(--color-primary)] p-0.5 rounded hover:bg-gray-200 transition-colors"
                          title="Copy email"
                        >
                          {copiedKey === `${account.role}-email` ? (
                            <IconCheck size={12} className="text-emerald-600" />
                          ) : (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                            </svg>
                          )}
                        </button>
                      </div>
                    </div>
                    <div className="flex items-center justify-between text-gray-600">
                      <span className="text-gray-400 font-sans">Password:</span>
                      <div className="flex items-center gap-1.5">
                        <span>{account.password}</span>
                        <button
                          type="button"
                          onClick={() => copyToClipboard(account.password, `${account.role}-pwd`)}
                          className="text-gray-400 hover:text-[var(--color-primary)] p-0.5 rounded hover:bg-gray-200 transition-colors"
                          title="Copy password"
                        >
                          {copiedKey === `${account.role}-pwd` ? (
                            <IconCheck size={12} className="text-emerald-600" />
                          ) : (
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
                              <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
                            </svg>
                          )}
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* Capabilities Bullet points */}
                  <ul className="text-xs text-gray-600 space-y-1 mb-4 pl-1">
                    {account.features.slice(0, 2).map((feat, idx) => (
                      <li key={idx} className="flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] shrink-0" />
                        <span className="truncate">{feat}</span>
                      </li>
                    ))}
                  </ul>

                  {/* Dual Action Buttons */}
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <button
                      type="button"
                      onClick={() => applyDemoAccount(account)}
                      className="w-full py-2 px-3 text-xs font-semibold text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 rounded-xl transition-all hover:border-gray-300 cursor-pointer"
                    >
                      Fill Form Credentials
                    </button>
                    <button
                      type="button"
                      disabled={isLoading}
                      onClick={() => handleQuickLaunch(account)}
                      className={`w-full py-2 px-3 text-xs font-semibold text-white rounded-xl shadow-xs transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                        account.role === "admin"
                          ? "bg-indigo-600 hover:bg-indigo-700 shadow-indigo-600/20"
                          : "bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] shadow-teal-600/20"
                      }`}
                    >
                      <IconLightning size={14} />
                      <span>Instant Launch ⚡</span>
                    </button>
                  </div>
                </div>
              );
            })}

            {/* Quick Testing Tips Card */}
            <div className="bg-emerald-50/60 border border-emerald-200/70 rounded-2xl p-4 text-xs text-emerald-900">
              <div className="flex items-center gap-2 font-bold mb-1 text-emerald-800">
                <IconCheck size={16} className="text-emerald-600" />
                <span>Zero-Friction Testing Guarantee</span>
              </div>
              <p className="text-emerald-700 leading-relaxed text-[11px]">
                Both accounts are seeded in SQLite database. Clicking <strong>Instant Launch</strong> will authenticate with the live FastAPI backend, store the JWT bearer token in localStorage, and load the dashboard.
              </p>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

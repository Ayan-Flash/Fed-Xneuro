"use client";

import React, { useState } from "react";
import {
  IconSettings,
  IconSecurity,
  IconHospital,
  IconBrain,
  IconCheck,
} from "@/components/Icons";

export default function AdminSettingsPage() {
  const [activeTab, setActiveTab] = useState<"general" | "security" | "federation" | "notifications">("general");
  const [saved, setSaved] = useState(false);

  // General settings state
  const [platformName, setPlatformName] = useState("Fed-XNeuro Platform");
  const [sessionTimeout, setSessionTimeout] = useState("30");
  const [maintenanceMode, setMaintenanceMode] = useState(false);
  const [autoBackup, setAutoBackup] = useState(true);

  // Security settings
  const [mfa, setMfa] = useState(true);
  const [ipWhitelist, setIpWhitelist] = useState(false);
  const [passwordPolicy, setPasswordPolicy] = useState("strong");
  const [auditRetention, setAuditRetention] = useState("90");

  // Federation settings
  const [dpEnabled, setDpEnabled] = useState(true);
  const [epsilonBudget, setEpsilonBudget] = useState("1.0");
  const [noiseMultiplier, setNoiseMultiplier] = useState("0.5");
  const [clipNorm, setClipNorm] = useState("1.0");
  const [minClients, setMinClients] = useState("3");
  const [aggregationAlgo, setAggregationAlgo] = useState("fedavg");

  // Notification settings
  const [emailAlerts, setEmailAlerts] = useState(true);
  const [slackIntegration, setSlackIntegration] = useState(false);
  const [alertOnNewNode, setAlertOnNewNode] = useState(true);
  const [alertOnHighRisk, setAlertOnHighRisk] = useState(true);
  const [alertOnDPBudget, setAlertOnDPBudget] = useState(true);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const tabs = [
    { id: "general" as const, label: "General", icon: IconSettings },
    { id: "security" as const, label: "Security", icon: IconSecurity },
    { id: "federation" as const, label: "Federation", icon: IconBrain },
    { id: "notifications" as const, label: "Notifications", icon: IconHospital },
  ];

  const Toggle = ({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) => (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200 focus:outline-none ${
        checked ? "bg-[var(--color-primary)]" : "bg-gray-300"
      }`}
    >
      <span className={`inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform duration-200 ${checked ? "translate-x-6" : "translate-x-1"}`} />
    </button>
  );

  return (
    <div className="p-6 lg:p-8 max-w-5xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Platform Settings</h1>
          <p className="text-gray-500 text-sm mt-1">Configure system-wide parameters, security policies, and federation settings.</p>
        </div>
        {saved && (
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-xl text-sm font-semibold text-emerald-700 animate-fade-in-up">
            <IconCheck size={16} />
            Settings saved successfully
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-gray-100/80 rounded-xl mb-6 border border-gray-200/60">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-lg flex-1 justify-center transition-all ${
              activeTab === tab.id
                ? "bg-white text-[var(--color-primary)] shadow-xs border border-gray-200/80"
                : "text-gray-500 hover:text-gray-700"
            }`}
          >
            <tab.icon size={15} />
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Settings Content */}
      <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sm:p-8">
        {activeTab === "general" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">General Configuration</h3>
              <p className="text-xs text-gray-500">Basic platform identity and operational settings.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Platform Name</label>
                <input type="text" value={platformName} onChange={(e) => setPlatformName(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Session Timeout (minutes)</label>
                <input type="number" value={sessionTimeout} onChange={(e) => setSessionTimeout(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)]">Maintenance Mode</div>
                  <div className="text-xs text-gray-500 mt-0.5">Temporarily restrict platform access for system updates.</div>
                </div>
                <Toggle checked={maintenanceMode} onChange={setMaintenanceMode} />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)]">Automatic Backups</div>
                  <div className="text-xs text-gray-500 mt-0.5">Daily encrypted backups of all platform data.</div>
                </div>
                <Toggle checked={autoBackup} onChange={setAutoBackup} />
              </div>
            </div>
          </div>
        )}

        {activeTab === "security" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Security Policies</h3>
              <p className="text-xs text-gray-500">Authentication, access control, and compliance settings.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)]">Multi-Factor Authentication</div>
                  <div className="text-xs text-gray-500 mt-0.5">Require MFA for all admin and hospital accounts.</div>
                </div>
                <Toggle checked={mfa} onChange={setMfa} />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)]">IP Whitelisting</div>
                  <div className="text-xs text-gray-500 mt-0.5">Restrict access to approved IP ranges only.</div>
                </div>
                <Toggle checked={ipWhitelist} onChange={setIpWhitelist} />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Password Policy</label>
                <select value={passwordPolicy} onChange={(e) => setPasswordPolicy(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white">
                  <option value="basic">Basic (8+ characters)</option>
                  <option value="moderate">Moderate (12+ chars, mixed case + numbers)</option>
                  <option value="strong">Strong (16+ chars, mixed case, numbers, symbols)</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Audit Log Retention (days)</label>
                <input type="number" value={auditRetention} onChange={(e) => setAuditRetention(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
            </div>
          </div>
        )}

        {activeTab === "federation" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Federated Learning Configuration</h3>
              <p className="text-xs text-gray-500">Differential privacy, aggregation, and model distribution parameters.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)]">Differential Privacy (DP-FL)</div>
                  <div className="text-xs text-gray-500 mt-0.5">Enable (ε, δ)-DP noise injection during aggregation.</div>
                </div>
                <Toggle checked={dpEnabled} onChange={setDpEnabled} />
              </div>
              {dpEnabled && (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 p-4 bg-[var(--color-light-teal)]/30 rounded-xl border border-[var(--color-primary)]/10">
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Privacy Budget (ε)</label>
                    <input type="number" step="0.1" value={epsilonBudget} onChange={(e) => setEpsilonBudget(e.target.value)} className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Noise Multiplier</label>
                    <input type="number" step="0.1" value={noiseMultiplier} onChange={(e) => setNoiseMultiplier(e.target.value)} className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Clip Norm (C)</label>
                    <input type="number" step="0.1" value={clipNorm} onChange={(e) => setClipNorm(e.target.value)} className="w-full px-3 py-2 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                  </div>
                </div>
              )}
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Minimum Participating Clients</label>
                <input type="number" value={minClients} onChange={(e) => setMinClients(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Aggregation Algorithm</label>
                <select value={aggregationAlgo} onChange={(e) => setAggregationAlgo(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white">
                  <option value="fedavg">FedAvg (Weighted Averaging)</option>
                  <option value="fedprox">FedProx (Proximal Regularization)</option>
                  <option value="fedavgm">FedAvgM (Server Momentum)</option>
                  <option value="scaffold">SCAFFOLD (Control Variates)</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {activeTab === "notifications" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Notification Preferences</h3>
              <p className="text-xs text-gray-500">Configure alert channels and notification triggers.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div><div className="text-sm font-bold text-[var(--color-text-main)]">Email Alerts</div><div className="text-xs text-gray-500 mt-0.5">Receive critical system notifications via email.</div></div>
                <Toggle checked={emailAlerts} onChange={setEmailAlerts} />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div><div className="text-sm font-bold text-[var(--color-text-main)]">Slack Integration</div><div className="text-xs text-gray-500 mt-0.5">Post alerts to your team&apos;s Slack channel.</div></div>
                <Toggle checked={slackIntegration} onChange={setSlackIntegration} />
              </div>
              <div className="pt-2 border-t border-gray-100">
                <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Alert Triggers</div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-3 bg-gray-50/60 rounded-xl">
                    <span className="text-sm text-[var(--color-text-main)] font-medium">New hospital node registration</span>
                    <Toggle checked={alertOnNewNode} onChange={setAlertOnNewNode} />
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50/60 rounded-xl">
                    <span className="text-sm text-[var(--color-text-main)] font-medium">High-risk patient detected</span>
                    <Toggle checked={alertOnHighRisk} onChange={setAlertOnHighRisk} />
                  </div>
                  <div className="flex items-center justify-between p-3 bg-gray-50/60 rounded-xl">
                    <span className="text-sm text-[var(--color-text-main)] font-medium">DP budget threshold reached</span>
                    <Toggle checked={alertOnDPBudget} onChange={setAlertOnDPBudget} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Save Button */}
        <div className="flex justify-end pt-6 mt-6 border-t border-gray-100">
          <button
            onClick={handleSave}
            className="px-8 py-3 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-sm font-bold rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all flex items-center gap-2"
          >
            <IconCheck size={16} />
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}

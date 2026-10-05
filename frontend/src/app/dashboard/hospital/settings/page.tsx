"use client";

import React, { useState } from "react";
import {
  IconSettings,
  IconHospital,
  IconSecurity,
  IconCheck,
  IconPatients,
} from "@/components/Icons";
import { FileUpload } from "@/components/FileUpload";

export default function HospitalSettingsPage() {
  const [activeTab, setActiveTab] = useState<"profile" | "team" | "data" | "preferences">("profile");
  const [saved, setSaved] = useState(false);

  // Profile
  const [hospitalName, setHospitalName] = useState("St. Jude Neuroscience Center");
  const [address, setAddress] = useState("350 W. Thomas Rd, New York, NY 10001");
  const [phone, setPhone] = useState("+1 (212) 555-0142");
  const [department, setDepartment] = useState("Neuroscience & Cognitive Medicine");

  // Team
  const [teamMembers, setTeamMembers] = useState([
    { name: "Dr. Sarah Jenkins", role: "Lead Clinician", email: "s.jenkins@stjude.org", status: "active" },
    { name: "Dr. Michael Torres", role: "Neurologist", email: "m.torres@stjude.org", status: "active" },
    { name: "Nurse Rachel Green", role: "Clinical Nurse", email: "r.green@stjude.org", status: "active" },
    { name: "Dr. Patricia Davis", role: "Psychologist", email: "p.davis@stjude.org", status: "invited" },
  ]);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("clinician");
  const [inviteSuccess, setInviteSuccess] = useState(false);
  const [uploadedDataFiles, setUploadedDataFiles] = useState<File[]>([]);
  const [uploadedClinicalDocs, setUploadedClinicalDocs] = useState<File[]>([]);

  // Preferences
  const [autoSave, setAutoSave] = useState(true);
  const [darkMode, setDarkMode] = useState(false);
  const [compactView, setCompactView] = useState(false);
  const [defaultCDR, setDefaultCDR] = useState("0.5 - Very Mild Dementia");
  const [timezone, setTimezone] = useState("America/New_York");

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleInvite = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inviteEmail) return;
    const roleMap: Record<string, string> = {
      clinician: "Clinician",
      nurse: "Clinical Nurse",
      researcher: "Researcher",
      admin: "Hospital Admin",
    };
    const newMember = {
      name: inviteEmail.split("@")[0].replace(".", " ").replace(/\b\w/g, (c) => c.toUpperCase()),
      role: roleMap[inviteRole] || "Clinician",
      email: inviteEmail,
      status: "invited",
    };
    setTeamMembers((prev) => [...prev, newMember]);
    setInviteSuccess(true);
    setTimeout(() => {
      setInviteSuccess(false);
      setInviteEmail("");
    }, 2000);
  };

  const handleRemoveMember = (email: string) => {
    setTeamMembers((prev) => prev.filter((m) => m.email !== email));
  };

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

  const tabs = [
    { id: "profile" as const, label: "Hospital Profile", icon: IconHospital },
    { id: "team" as const, label: "Team Members", icon: IconPatients },
    { id: "data" as const, label: "Data Management", icon: IconSecurity },
    { id: "preferences" as const, label: "Preferences", icon: IconSettings },
  ];

  return (
    <div className="p-6 lg:p-8 max-w-5xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Hospital Settings</h1>
          <p className="text-gray-500 text-sm mt-1">Manage your hospital profile, team, and clinical preferences.</p>
        </div>
        {saved && (
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-50 border border-emerald-200 rounded-xl text-sm font-semibold text-emerald-700 animate-fade-in-up">
            <IconCheck size={16} /> Settings saved
          </div>
        )}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 p-1 bg-gray-100/80 rounded-xl mb-6 border border-gray-200/60 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-lg flex-1 justify-center transition-all whitespace-nowrap ${
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

      {/* Content */}
      <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sm:p-8">
        {activeTab === "profile" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Hospital Profile</h3>
              <p className="text-xs text-gray-500">Your institution&apos;s identity within the federated network.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Hospital Name</label>
                <input type="text" value={hospitalName} onChange={(e) => setHospitalName(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Department</label>
                <input type="text" value={department} onChange={(e) => setDepartment(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Address</label>
                <input type="text" value={address} onChange={(e) => setAddress(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Phone</label>
                <input type="tel" value={phone} onChange={(e) => setPhone(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
              </div>
              <div className="p-4 bg-[var(--color-light-teal)]/30 border border-[var(--color-primary)]/10 rounded-xl">
                <div className="flex items-center gap-2 mb-1">
                  <IconSecurity size={14} className="text-[var(--color-primary)]" />
                  <span className="text-xs font-bold text-[var(--color-primary)]">Federated Node Status</span>
                </div>
                <div className="flex items-center gap-4 text-xs text-gray-600">
                  <span>Node ID: <strong className="font-mono">HSP-001</strong></span>
                  <span>Status: <strong className="text-emerald-600">● Active</strong></span>
                  <span>Uptime: <strong className="font-mono">99.9%</strong></span>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === "team" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Team Members</h3>
              <p className="text-xs text-gray-500">Manage clinicians and staff who can access this hospital&apos;s portal.</p>
            </div>

            {/* Team Table */}
            <div className="overflow-x-auto border border-gray-100 rounded-xl">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-xs text-gray-500 uppercase tracking-wider font-semibold bg-gray-50/60">
                    <th className="text-left px-4 py-3">Member</th>
                    <th className="text-left px-4 py-3">Role</th>
                    <th className="text-center px-4 py-3">Status</th>
                    <th className="text-right px-4 py-3">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {teamMembers.map((m, i) => (
                    <tr key={i} className="hover:bg-gray-50/70 transition-colors">
                      <td className="px-4 py-3">
                        <div className="font-bold text-[var(--color-text-main)]">{m.name}</div>
                        <div className="text-[11px] text-gray-400">{m.email}</div>
                      </td>
                      <td className="px-4 py-3 text-gray-600">{m.role}</td>
                      <td className="px-4 py-3 text-center">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold border ${
                          m.status === "active"
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-amber-50 text-amber-700 border-amber-200"
                        }`}>
                          {m.status === "active" && <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />}
                          {m.status.charAt(0).toUpperCase() + m.status.slice(1)}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={() => handleRemoveMember(m.email)}
                          className="text-xs text-gray-400 hover:text-rose-600 font-medium transition-colors"
                          title="Remove member"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Invite */}
            <div className="pt-4 border-t border-gray-100">
              <h4 className="text-sm font-bold text-[var(--color-text-main)] mb-3">Invite Team Member</h4>
              {inviteSuccess ? (
                <div className="bg-emerald-50 border border-emerald-200 rounded-xl p-4 flex items-center gap-3 animate-fade-in-up">
                  <IconCheck size={18} className="text-emerald-600" />
                  <span className="text-sm font-semibold text-emerald-700">Invitation sent successfully!</span>
                </div>
              ) : (
                <form onSubmit={handleInvite} className="flex flex-col sm:flex-row gap-3">
                  <input
                    type="email"
                    required
                    value={inviteEmail}
                    onChange={(e) => setInviteEmail(e.target.value)}
                    placeholder="colleague@hospital.org"
                    className="flex-1 px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]"
                  />
                  <select
                    value={inviteRole}
                    onChange={(e) => setInviteRole(e.target.value)}
                    className="px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white"
                  >
                    <option value="clinician">Clinician</option>
                    <option value="nurse">Clinical Nurse</option>
                    <option value="researcher">Researcher</option>
                    <option value="admin">Hospital Admin</option>
                  </select>
                  <button type="submit" className="px-6 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-bold rounded-xl shadow-xs transition-all">
                    Send Invite
                  </button>
                </form>
              )}
            </div>
          </div>
        )}

        {activeTab === "data" && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Data Management</h3>
                <p className="text-xs text-gray-500">Import patient data, upload clinical documents, and manage local datasets.</p>
              </div>
              {(uploadedDataFiles.length > 0 || uploadedClinicalDocs.length > 0) && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-emerald-50 text-emerald-700 border border-emerald-200 rounded-full text-xs font-semibold">
                  <IconCheck size={14} />
                  {uploadedDataFiles.length + uploadedClinicalDocs.length} files secured in node vault
                </span>
              )}
            </div>

            {/* Bulk Patient Import */}
            <div className="pt-2">
              <h4 className="text-sm font-bold text-[var(--color-text-main)] mb-3">Bulk Patient Data Import</h4>
              <FileUpload
                accept=".csv,.xlsx,.json"
                maxFiles={3}
                maxSizeMB={50}
                label="Upload patient data files (CSV, Excel, JSON)"
                sublabel="Data is encrypted locally and never leaves your hospital network"
                icon="document"
                onFilesChange={setUploadedDataFiles}
              />
            </div>

            {/* Clinical Document Upload */}
            <div className="pt-4 border-t border-gray-100">
              <h4 className="text-sm font-bold text-[var(--color-text-main)] mb-3">Clinical Document Archive</h4>
              <FileUpload
                accept=".pdf,.doc,.docx,.txt"
                maxFiles={10}
                maxSizeMB={25}
                label="Upload clinical reports, consent forms, or study protocols"
                sublabel="Documents are stored in your hospital's encrypted local vault"
                icon="document"
                onFilesChange={setUploadedClinicalDocs}
              />
            </div>

            <div className="p-4 bg-amber-50/60 border border-amber-200/60 rounded-xl">
              <p className="text-xs text-amber-800 font-medium">
                <strong>Privacy Notice:</strong> All uploaded data is processed and stored entirely within your hospital&apos;s local federated node. No raw data is transmitted to the central server. Only encrypted model gradient updates are shared during federated training.
              </p>
            </div>
          </div>
        )}

        {activeTab === "preferences" && (
          <div className="space-y-6">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1">Interface Preferences</h3>
              <p className="text-xs text-gray-500">Customize your clinical portal experience.</p>
            </div>
            <div className="space-y-5 pt-2">
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div><div className="text-sm font-bold text-[var(--color-text-main)]">Auto-Save Assessments</div><div className="text-xs text-gray-500 mt-0.5">Automatically save assessment forms as you type.</div></div>
                <Toggle checked={autoSave} onChange={setAutoSave} />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div><div className="text-sm font-bold text-[var(--color-text-main)]">Dark Mode</div><div className="text-xs text-gray-500 mt-0.5">Switch to a dark interface theme (coming soon).</div></div>
                <Toggle checked={darkMode} onChange={setDarkMode} />
              </div>
              <div className="flex items-center justify-between p-4 bg-gray-50/80 rounded-xl border border-gray-100">
                <div><div className="text-sm font-bold text-[var(--color-text-main)]">Compact Patient List</div><div className="text-xs text-gray-500 mt-0.5">Show more patients per page with reduced spacing.</div></div>
                <Toggle checked={compactView} onChange={setCompactView} />
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Default CDR Scale</label>
                <select value={defaultCDR} onChange={(e) => setDefaultCDR(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white">
                  <option>0 - Normal</option>
                  <option>0.5 - Very Mild Dementia</option>
                  <option>1 - Mild Dementia</option>
                  <option>2 - Moderate Dementia</option>
                  <option>3 - Severe Dementia</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Timezone</label>
                <select value={timezone} onChange={(e) => setTimezone(e.target.value)} className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white">
                  <option value="America/New_York">Eastern Time (ET)</option>
                  <option value="America/Chicago">Central Time (CT)</option>
                  <option value="America/Denver">Mountain Time (MT)</option>
                  <option value="America/Los_Angeles">Pacific Time (PT)</option>
                  <option value="UTC">UTC</option>
                </select>
              </div>
            </div>
          </div>
        )}

        {/* Save */}
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

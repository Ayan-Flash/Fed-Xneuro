"use client";

import React, { useState } from "react";
import {
  IconHospital,
  IconCheck,
  IconRiskLow,
  IconRiskModerate,
  IconRiskHigh,
  IconSettings,
  IconChartBar,
} from "@/components/Icons";

interface Hospital {
  id: string;
  name: string;
  location: string;
  status: "active" | "pending" | "inactive";
  patients: number;
  assessments: number;
  lastSync: string;
  nodeHealth: number;
}

const HOSPITALS: Hospital[] = [
  { id: "HSP-001", name: "St. Jude Neuroscience Center", location: "New York, NY", status: "active", patients: 342, assessments: 1240, lastSync: "2 mins ago", nodeHealth: 99 },
  { id: "HSP-002", name: "Memorial Healthcare Research", location: "Los Angeles, CA", status: "active", patients: 289, assessments: 980, lastSync: "5 mins ago", nodeHealth: 97 },
  { id: "HSP-003", name: "Cleveland Clinic Neurolab", location: "Cleveland, OH", status: "active", patients: 451, assessments: 1890, lastSync: "1 min ago", nodeHealth: 100 },
  { id: "HSP-004", name: "Johns Hopkins Brain Center", location: "Baltimore, MD", status: "active", patients: 567, assessments: 2340, lastSync: "8 mins ago", nodeHealth: 95 },
  { id: "HSP-005", name: "Mayo Clinic Alzheimer's Unit", location: "Rochester, MN", status: "pending", patients: 0, assessments: 0, lastSync: "Awaiting setup", nodeHealth: 0 },
  { id: "HSP-006", name: "UCSF Memory Clinic", location: "San Francisco, CA", status: "active", patients: 198, assessments: 670, lastSync: "12 mins ago", nodeHealth: 92 },
  { id: "HSP-007", name: "Massachusetts General Hospital", location: "Boston, MA", status: "inactive", patients: 156, assessments: 420, lastSync: "3 days ago", nodeHealth: 0 },
  { id: "HSP-008", name: "Duke University Cognitive Lab", location: "Durham, NC", status: "active", patients: 223, assessments: 890, lastSync: "4 mins ago", nodeHealth: 98 },
];

export default function AdminHospitalsPage() {
  const [hospitals, setHospitals] = useState<Hospital[]>(HOSPITALS);
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState<"all" | "active" | "pending" | "inactive">("all");
  const [showAddModal, setShowAddModal] = useState(false);
  const [addForm, setAddForm] = useState({ name: "", location: "", contact: "", email: "" });
  const [addSuccess, setAddSuccess] = useState(false);

  const filtered = hospitals.filter((h) => {
    const matchSearch =
      h.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      h.location.toLowerCase().includes(searchTerm.toLowerCase()) ||
      h.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchStatus = statusFilter === "all" || h.status === statusFilter;
    return matchSearch && matchStatus;
  });

  const totalActive = hospitals.filter((h) => h.status === "active").length;
  const totalPatients = hospitals.reduce((sum, h) => sum + h.patients, 0);
  const totalAssessments = hospitals.reduce((sum, h) => sum + h.assessments, 0);

  const handleAddHospital = (e: React.FormEvent) => {
    e.preventDefault();
    if (!addForm.name || !addForm.location) return;

    const newHosp: Hospital = {
      id: `HSP-00${hospitals.length + 1}`,
      name: addForm.name,
      location: addForm.location,
      status: "active",
      patients: 0,
      assessments: 0,
      lastSync: "Just now",
      nodeHealth: 100,
    };

    setHospitals((prev) => [newHosp, ...prev]);
    setAddSuccess(true);
    setTimeout(() => {
      setShowAddModal(false);
      setAddSuccess(false);
      setAddForm({ name: "", location: "", contact: "", email: "" });
    }, 1200);
  };

  const toggleHospitalStatus = (id: string) => {
    setHospitals((prev) =>
      prev.map((h) => {
        if (h.id !== id) return h;
        const nextStatus: Record<Hospital["status"], Hospital["status"]> = {
          active: "inactive",
          inactive: "pending",
          pending: "active",
        };
        return { ...h, status: nextStatus[h.status] };
      })
    );
  };

  const statusBadge = (status: Hospital["status"]) => {
    const map = {
      active: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", dot: "bg-emerald-500" },
      pending: { bg: "bg-amber-50", text: "text-amber-700", border: "border-amber-200", dot: "bg-amber-500" },
      inactive: { bg: "bg-gray-100", text: "text-gray-500", border: "border-gray-200", dot: "bg-gray-400" },
    };
    const s = map[status];
    return (
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold ${s.bg} ${s.text} border ${s.border}`}>
        <span className={`w-1.5 h-1.5 rounded-full ${s.dot} ${status === "active" ? "animate-pulse" : ""}`} />
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </span>
    );
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Hospital Network</h1>
          <p className="text-gray-500 text-sm mt-1">Manage federated hospital nodes across the network.</p>
        </div>
        <button
          onClick={() => setShowAddModal(true)}
          className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all duration-200"
        >
          <IconHospital size={18} />
          <span>Register Hospital</span>
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex items-center gap-4 hover:shadow-md transition-shadow">
          <div className="w-11 h-11 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center"><IconHospital size={20} /></div>
          <div><div className="text-xs text-gray-500 font-medium">Active Nodes</div><div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono">{totalActive}</div></div>
        </div>
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex items-center gap-4 hover:shadow-md transition-shadow">
          <div className="w-11 h-11 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center"><IconChartBar size={20} /></div>
          <div><div className="text-xs text-gray-500 font-medium">Total Patients</div><div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono">{totalPatients.toLocaleString()}</div></div>
        </div>
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex items-center gap-4 hover:shadow-md transition-shadow">
          <div className="w-11 h-11 rounded-xl bg-blue-50 text-blue-600 flex items-center justify-center"><IconSettings size={20} /></div>
          <div><div className="text-xs text-gray-500 font-medium">Total Assessments</div><div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono">{totalAssessments.toLocaleString()}</div></div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-2xl shadow-xs border border-gray-100 mb-6">
        <div className="p-4 flex flex-col sm:flex-row gap-3 border-b border-gray-100">
          <input
            type="text"
            placeholder="Search hospitals..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="flex-1 px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]"
          />
          <div className="flex gap-2">
            {(["all", "active", "pending", "inactive"] as const).map((s) => (
              <button
                key={s}
                onClick={() => setStatusFilter(s)}
                className={`px-3.5 py-2 text-xs font-semibold rounded-xl transition-all ${
                  statusFilter === s
                    ? "bg-[var(--color-primary)] text-white shadow-xs"
                    : "bg-gray-50 text-gray-600 hover:bg-gray-100"
                }`}
              >
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 uppercase tracking-wider font-semibold bg-gray-50/60">
                <th className="text-left px-5 py-3">Hospital</th>
                <th className="text-left px-5 py-3 hidden md:table-cell">Location</th>
                <th className="text-center px-5 py-3">Status</th>
                <th className="text-center px-5 py-3 hidden lg:table-cell">Patients</th>
                <th className="text-center px-5 py-3 hidden lg:table-cell">Assessments</th>
                <th className="text-center px-5 py-3 hidden xl:table-cell">Node Health</th>
                <th className="text-right px-5 py-3">Last Sync</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {filtered.map((h) => (
                <tr key={h.id} className="hover:bg-gray-50/70 transition-colors group">
                  <td className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="w-9 h-9 rounded-lg bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center shrink-0 border border-[var(--color-primary)]/10">
                        <IconHospital size={16} />
                      </div>
                      <div>
                        <div className="font-bold text-[var(--color-text-main)] group-hover:text-[var(--color-primary)] transition-colors">{h.name}</div>
                        <div className="text-[11px] text-gray-400 font-mono">{h.id}</div>
                      </div>
                    </div>
                  </td>
                  <td className="px-5 py-4 text-gray-600 hidden md:table-cell">{h.location}</td>
                  <td className="px-5 py-4 text-center">
                    <button
                      onClick={() => toggleHospitalStatus(h.id)}
                      className="cursor-pointer hover:opacity-80 transition-opacity"
                      title="Click to cycle status (active / pending / inactive)"
                    >
                      {statusBadge(h.status)}
                    </button>
                  </td>
                  <td className="px-5 py-4 text-center font-mono font-bold text-[var(--color-text-main)] hidden lg:table-cell">{h.patients.toLocaleString()}</td>
                  <td className="px-5 py-4 text-center font-mono font-bold text-[var(--color-text-main)] hidden lg:table-cell">{h.assessments.toLocaleString()}</td>
                  <td className="px-5 py-4 hidden xl:table-cell">
                    {h.nodeHealth > 0 ? (
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-16 h-1.5 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${h.nodeHealth > 90 ? "bg-emerald-500" : h.nodeHealth > 70 ? "bg-amber-500" : "bg-red-500"}`}
                            style={{ width: `${h.nodeHealth}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono font-bold text-gray-600">{h.nodeHealth}%</span>
                      </div>
                    ) : (
                      <span className="text-xs text-gray-400">—</span>
                    )}
                  </td>
                  <td className="px-5 py-4 text-right text-xs text-gray-500 font-mono">{h.lastSync}</td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-5 py-12 text-center text-gray-400 text-sm">
                    No hospitals found matching your criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Add Hospital Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={() => setShowAddModal(false)}>
          <div className="bg-white rounded-2xl shadow-xl border border-gray-100 w-full max-w-lg animate-fade-in-up" onClick={(e) => e.stopPropagation()}>
            {addSuccess ? (
              <div className="p-12 text-center">
                <div className="w-14 h-14 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center mx-auto mb-4 border border-emerald-200">
                  <IconCheck size={28} />
                </div>
                <h3 className="text-lg font-bold text-[var(--color-text-main)]">Hospital Registered</h3>
                <p className="text-sm text-gray-500 mt-1">Node enrollment has been initiated. Setup instructions sent.</p>
              </div>
            ) : (
              <>
                <div className="px-6 py-5 border-b border-gray-100 flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-bold text-[var(--color-text-main)]">Register New Hospital</h3>
                    <p className="text-xs text-gray-500 mt-0.5">Add a new federated node to the network</p>
                  </div>
                  <button onClick={() => setShowAddModal(false)} className="p-2 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6 6 18M6 6l12 12" /></svg>
                  </button>
                </div>
                <form onSubmit={handleAddHospital} className="p-6 space-y-4">
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Hospital Name</label>
                    <input type="text" required value={addForm.name} onChange={(e) => setAddForm({ ...addForm, name: e.target.value })} placeholder="e.g., Memorial Healthcare" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Location</label>
                    <input type="text" required value={addForm.location} onChange={(e) => setAddForm({ ...addForm, location: e.target.value })} placeholder="e.g., New York, NY" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Contact Person</label>
                      <input type="text" required value={addForm.contact} onChange={(e) => setAddForm({ ...addForm, contact: e.target.value })} placeholder="Dr. Full Name" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                    </div>
                    <div>
                      <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Email</label>
                      <input type="email" required value={addForm.email} onChange={(e) => setAddForm({ ...addForm, email: e.target.value })} placeholder="admin@hospital.org" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" />
                    </div>
                  </div>
                  <div className="flex justify-end gap-3 pt-3 border-t border-gray-100">
                    <button type="button" onClick={() => setShowAddModal(false)} className="px-5 py-2.5 text-xs font-semibold text-gray-600 hover:bg-gray-100 rounded-xl transition-colors">Cancel</button>
                    <button type="submit" className="px-6 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-bold rounded-xl shadow-md transition-all flex items-center gap-2">
                      <IconHospital size={15} /> Register Node
                    </button>
                  </div>
                </form>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

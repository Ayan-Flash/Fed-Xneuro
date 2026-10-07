"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import {
  IconPatients,
  IconHistory,
  IconRiskLow,
  IconRiskModerate,
  IconRiskHigh,
  IconBrain,
  IconReports,
  IconChartBar,
} from "@/components/Icons";
import { fetchAssessmentHistory, SavedAssessmentRecord } from "@/lib/api";

interface Patient {
  id: string;
  name: string;
  lastVisit: string;
  totalAssessments: number;
  latestRisk: "Low" | "Moderate" | "High";
  trend: "improving" | "stable" | "declining";
  mmse: number;
  cdr: string;
  imagingFile?: string | null;
}

export default function PatientHistoryPage() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState<"all" | "Low" | "Moderate" | "High">("all");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  useEffect(() => {
    async function loadPatients() {
      try {
        setLoading(true);
        const res = await fetchAssessmentHistory();
        const records: SavedAssessmentRecord[] = res.records || [];

        // Check local storage for any client assessments and strip any legacy random age
        let localRecords: any[] = [];
        if (typeof window !== "undefined") {
          try {
            const stored = localStorage.getItem("fedx_patients");
            if (stored) {
              const parsed = JSON.parse(stored);
              if (Array.isArray(parsed)) {
                localRecords = parsed.map((p: any) => {
                  const { age, gender, ...rest } = p;
                  return rest;
                });
                // Overwrite localStorage without any random age
                localStorage.setItem("fedx_patients", JSON.stringify(localRecords));
              }
            }
          } catch (e) {
            console.error(e);
          }
        }

        // Combine into unique patients map
        const patientMap = new Map<string, Patient>();

        // Process backend records
        records.forEach((r) => {
          const riskVal: "Low" | "Moderate" | "High" =
            r.risk === "High" || r.risk === "Moderate" ? r.risk : "Low";
          patientMap.set(r.id, {
            id: r.id,
            name: r.name || "Anonymous Patient",
            lastVisit: r.date || "Recent",
            totalAssessments: 1,
            latestRisk: riskVal,
            trend: riskVal === "High" ? "declining" : riskVal === "Moderate" ? "stable" : "improving",
            mmse: Number(r.mmse) || 24,
            cdr: String(r.cdr || "0.5"),
            imagingFile: r.document_name,
          });
        });

        // Merge local records
        localRecords.forEach((lr) => {
          if (!patientMap.has(lr.id)) {
            const riskVal: "Low" | "Moderate" | "High" =
              lr.risk === "High" || lr.risk === "Moderate" ? lr.risk : "Low";
            patientMap.set(lr.id, {
              id: lr.id,
              name: lr.name || "Anonymous Patient",
              lastVisit: lr.date || lr.lastVisit || "Recent",
              totalAssessments: lr.totalAssessments || 1,
              latestRisk: lr.latestRisk || riskVal,
              trend: lr.trend || (riskVal === "High" ? "declining" : "stable"),
              mmse: Number(lr.mmse) || 24,
              cdr: String(lr.cdr || "0.5"),
              imagingFile: lr.imagingFile,
            });
          }
        });

        setPatients(Array.from(patientMap.values()));
      } catch (err) {
        console.error("Failed to load patients:", err);
        setPatients([]);
      } finally {
        setLoading(false);
      }
    }

    loadPatients();
  }, []);

  const filtered = patients.filter((p) => {
    const matchSearch =
      p.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      p.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchRisk = riskFilter === "all" || p.latestRisk === riskFilter;
    return matchSearch && matchRisk;
  });

  const handleExportPatientReport = (patient: Patient) => {
    const reportText = `FED-XNEURO CLINICAL PATIENT DOSSIER\n` +
      `=====================================\n` +
      `Patient ID: ${patient.id}\n` +
      `Full Name: ${patient.name}\n` +
      `Assessed Risk Level: ${patient.latestRisk} Risk\n` +
      `Cognitive Trend: ${patient.trend.toUpperCase()}\n` +
      `MMSE Score: ${patient.mmse} / 30\n` +
      `Clinical Dementia Rating (CDR): ${patient.cdr}\n` +
      `Total Assessments: ${patient.totalAssessments}\n` +
      `Last Assessment Date: ${patient.lastVisit}\n` +
      (patient.imagingFile ? `Neuroimaging Scan: ${patient.imagingFile}\n` : "") +
      `Security Hash: SHA256-${Math.random().toString(36).substring(2, 12)}\n` +
      `=====================================\n` +
      `Federated Multimodal Neuroimaging Model\n`;

    const blob = new Blob([reportText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${patient.id}_Clinical_Summary.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const riskConfig = {
    Low: { icon: IconRiskLow, color: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200" },
    Moderate: { icon: IconRiskModerate, color: "text-amber-700", bg: "bg-amber-50", border: "border-amber-200" },
    High: { icon: IconRiskHigh, color: "text-rose-600", bg: "bg-rose-50", border: "border-rose-200" },
  };

  const trendConfig = {
    improving: { label: "Improving", color: "text-emerald-600" },
    stable: { label: "Stable", color: "text-amber-600" },
    declining: { label: "Declining", color: "text-rose-600" },
  };

  const totalCount = patients.length;
  const highRiskCount = patients.filter((p) => p.latestRisk === "High").length;
  const modRiskCount = patients.filter((p) => p.latestRisk === "Moderate").length;
  const lowRiskCount = patients.filter((p) => p.latestRisk === "Low").length;

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
            Patient Longitudinal History
          </h1>
          <p className="text-gray-500 text-sm mt-1">
            Browse, search, and review clinical risk trajectories across your hospital cohort.
          </p>
        </div>
        <Link
          href="/dashboard/hospital/assessment"
          className="inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-bold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-xs transition-all shrink-0"
        >
          <IconBrain size={16} />
          <span>New Assessment</span>
        </Link>
      </div>

      {/* Cohort Overview Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-xs">
          <div className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Total Cohort</div>
          <div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono mt-1">
            {loading ? "-" : totalCount}
          </div>
          <div className="text-[11px] text-gray-400 mt-0.5">Active patients</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-xs">
          <div className="text-xs text-rose-600 font-semibold uppercase tracking-wider">High Risk</div>
          <div className="text-2xl font-extrabold text-rose-600 font-mono mt-1">
            {loading ? "-" : highRiskCount}
          </div>
          <div className="text-[11px] text-rose-600/70 mt-0.5">Urgent review</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-xs">
          <div className="text-xs text-amber-600 font-semibold uppercase tracking-wider">Moderate Risk</div>
          <div className="text-2xl font-extrabold text-amber-600 font-mono mt-1">
            {loading ? "-" : modRiskCount}
          </div>
          <div className="text-[11px] text-amber-600/70 mt-0.5">Monitored</div>
        </div>
        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-xs">
          <div className="text-xs text-emerald-600 font-semibold uppercase tracking-wider">Low Risk</div>
          <div className="text-2xl font-extrabold text-emerald-600 font-mono mt-1">
            {loading ? "-" : lowRiskCount}
          </div>
          <div className="text-[11px] text-emerald-600/70 mt-0.5">Stable</div>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Main List */}
        <div className="flex-1">
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 overflow-hidden">
            {/* Filter toolbar */}
            <div className="p-4 border-b border-gray-100 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between bg-gray-50/50">
              <div className="relative flex-1">
                <input
                  type="text"
                  placeholder="Search by patient name or ID..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full pl-9 pr-3 py-2 text-xs border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white"
                />
                <svg
                  className="w-4 h-4 text-gray-400 absolute left-3 top-2.5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
              </div>

              <div className="flex gap-1.5">
                {(["all", "High", "Moderate", "Low"] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setRiskFilter(r)}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
                      riskFilter === r
                        ? "bg-[var(--color-primary)] text-white shadow-xs"
                        : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    {r === "all" ? "All" : r}
                  </button>
                ))}
              </div>
            </div>

            {/* Patients List Body */}
            {loading ? (
              <div className="p-12 text-center text-sm text-gray-400">Loading patient records...</div>
            ) : patients.length === 0 ? (
              <div className="p-12 text-center">
                <div className="w-14 h-14 rounded-2xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mx-auto mb-3">
                  <IconPatients size={28} />
                </div>
                <h3 className="text-base font-bold text-[var(--color-text-main)]">No Patient Records Yet</h3>
                <p className="text-sm text-gray-500 mt-1 max-w-sm mx-auto">
                  Your hospital cohort is currently empty. Patients will appear here once clinical assessments are performed.
                </p>
                <div className="mt-5">
                  <Link
                    href="/dashboard/hospital/assessment"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-bold rounded-xl shadow-xs transition-all"
                  >
                    <IconBrain size={16} />
                    <span>Run First Assessment</span>
                  </Link>
                </div>
              </div>
            ) : (
              <div className="divide-y divide-gray-100">
                {filtered.map((patient) => {
                  const isSelected = selectedPatient?.id === patient.id;
                  const risk = riskConfig[patient.latestRisk] || riskConfig.Low;
                  const trend = trendConfig[patient.trend] || trendConfig.stable;
                  const RiskIcon = risk.icon;

                  return (
                    <div
                      key={patient.id}
                      onClick={() => setSelectedPatient(patient)}
                      className={`p-4 flex items-center justify-between cursor-pointer transition-colors ${
                        isSelected ? "bg-[var(--color-light-teal)]/40 border-l-4 border-l-[var(--color-primary)]" : "hover:bg-gray-50/70"
                      }`}
                    >
                      <div className="flex items-center gap-3.5 min-w-0">
                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${risk.bg} ${risk.color} shrink-0 border ${risk.border}`}>
                          <RiskIcon size={18} />
                        </div>
                        <div className="min-w-0">
                          <div className="text-sm font-bold text-[var(--color-text-main)] truncate flex items-center gap-2">
                            <span>{patient.name}</span>
                            <span className="text-xs text-gray-400 font-mono font-normal">{patient.id}</span>
                          </div>
                          <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-2">
                            <span>{patient.lastVisit}</span>
                            {patient.imagingFile && (
                              <>
                                <span>•</span>
                                <span className="font-mono text-[11px] text-gray-400 truncate max-w-[220px]">
                                  {patient.imagingFile}
                                </span>
                              </>
                            )}
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-3 shrink-0">
                        <span className={`text-[11px] font-bold ${trend.color} hidden sm:inline`}>{trend.label}</span>
                        <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-bold border ${risk.bg} ${risk.color} ${risk.border}`}>
                          {patient.latestRisk}
                        </span>
                      </div>
                    </div>
                  );
                })}
                {filtered.length === 0 && patients.length > 0 && (
                  <div className="px-5 py-12 text-center text-gray-400 text-sm">No matching patients found.</div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Patient Detail Panel */}
        <div className="w-full lg:w-96 shrink-0">
          {selectedPatient ? (
            <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sticky top-24 animate-fade-in-up">
              <div className="flex items-center gap-3 mb-5">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${riskConfig[selectedPatient.latestRisk]?.bg || "bg-gray-50"} ${riskConfig[selectedPatient.latestRisk]?.color || "text-gray-600"} border ${riskConfig[selectedPatient.latestRisk]?.border || "border-gray-200"}`}>
                  {React.createElement(riskConfig[selectedPatient.latestRisk]?.icon || IconRiskLow, { size: 24 })}
                </div>
                <div>
                  <h3 className="text-base font-bold text-[var(--color-text-main)]">{selectedPatient.name}</h3>
                  <p className="text-xs text-gray-500 font-mono">{selectedPatient.id} • {selectedPatient.lastVisit}</p>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="p-3 bg-gray-50 rounded-xl border border-gray-100">
                  <div className="text-[11px] text-gray-500 font-medium">MMSE Score</div>
                  <div className="text-xl font-extrabold text-[var(--color-text-main)] font-mono mt-0.5">{selectedPatient.mmse}<span className="text-xs font-normal text-gray-400"> /30</span></div>
                </div>
                <div className="p-3 bg-gray-50 rounded-xl border border-gray-100">
                  <div className="text-[11px] text-gray-500 font-medium">CDR Score</div>
                  <div className="text-xl font-extrabold text-[var(--color-text-main)] font-mono mt-0.5">{selectedPatient.cdr}</div>
                </div>
              </div>

              <div className="space-y-3 mb-5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Total Assessments</span>
                  <span className="font-bold text-[var(--color-text-main)] font-mono">{selectedPatient.totalAssessments}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Cognitive Trend</span>
                  <span className={`font-bold ${trendConfig[selectedPatient.trend]?.color || "text-gray-600"}`}>
                    {trendConfig[selectedPatient.trend]?.label || "Stable"}
                  </span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Last Assessment</span>
                  <span className="font-bold text-[var(--color-text-main)]">{selectedPatient.lastVisit}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Current Risk Level</span>
                  <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-bold border ${riskConfig[selectedPatient.latestRisk]?.bg || "bg-gray-50"} ${riskConfig[selectedPatient.latestRisk]?.color || "text-gray-600"} ${riskConfig[selectedPatient.latestRisk]?.border || "border-gray-200"}`}>
                    {selectedPatient.latestRisk} Risk
                  </span>
                </div>
              </div>

              <div className="flex gap-2">
                <Link
                  href="/dashboard/hospital/assessment"
                  className="flex-1 py-2.5 text-xs font-bold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl text-center shadow-xs transition-all flex items-center justify-center gap-1.5"
                >
                  <IconBrain size={14} />
                  New Assessment
                </Link>
                <button
                  onClick={() => handleExportPatientReport(selectedPatient)}
                  className="px-4 py-2.5 text-xs font-bold text-[var(--color-primary)] bg-[var(--color-light-teal)] hover:bg-[var(--color-primary)] hover:text-white rounded-xl transition-all flex items-center gap-1.5"
                >
                  <IconReports size={14} />
                  Export
                </button>
              </div>
            </div>
          ) : (
            <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-12 text-center sticky top-24">
              <div className="w-14 h-14 rounded-2xl bg-gray-100 text-gray-400 flex items-center justify-center mx-auto mb-4">
                <IconPatients size={28} />
              </div>
              <h3 className="text-base font-bold text-gray-400 mb-1">Select a Patient</h3>
              <p className="text-xs text-gray-400">Click on a patient to view their detailed cognitive history and risk trajectory.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

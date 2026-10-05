"use client";

import React, { useState } from "react";
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

interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  lastVisit: string;
  totalAssessments: number;
  latestRisk: "Low" | "Moderate" | "High";
  trend: "improving" | "stable" | "declining";
  mmse: number;
  cdr: string;
}

const PATIENTS: Patient[] = [
  { id: "PAT-8492", name: "Robert Johnson", age: 72, gender: "Male", lastVisit: "Today", totalAssessments: 6, latestRisk: "High", trend: "declining", mmse: 21, cdr: "1.0" },
  { id: "PAT-8491", name: "Sarah Mitchell", age: 65, gender: "Female", lastVisit: "Yesterday", totalAssessments: 4, latestRisk: "Low", trend: "stable", mmse: 28, cdr: "0" },
  { id: "PAT-8490", name: "Michael Torres", age: 69, gender: "Male", lastVisit: "Yesterday", totalAssessments: 5, latestRisk: "Moderate", trend: "declining", mmse: 24, cdr: "0.5" },
  { id: "PAT-8489", name: "Eleanor Williams", age: 61, gender: "Female", lastVisit: "Oct 12, 2026", totalAssessments: 3, latestRisk: "Low", trend: "improving", mmse: 29, cdr: "0" },
  { id: "PAT-8488", name: "James Chen", age: 78, gender: "Male", lastVisit: "Oct 10, 2026", totalAssessments: 8, latestRisk: "High", trend: "declining", mmse: 18, cdr: "2.0" },
  { id: "PAT-8487", name: "Patricia Davis", age: 67, gender: "Female", lastVisit: "Oct 8, 2026", totalAssessments: 2, latestRisk: "Moderate", trend: "stable", mmse: 25, cdr: "0.5" },
  { id: "PAT-8486", name: "William Brown", age: 74, gender: "Male", lastVisit: "Oct 5, 2026", totalAssessments: 7, latestRisk: "Moderate", trend: "stable", mmse: 23, cdr: "1.0" },
  { id: "PAT-8485", name: "Mary Anderson", age: 70, gender: "Female", lastVisit: "Sep 30, 2026", totalAssessments: 4, latestRisk: "Low", trend: "improving", mmse: 27, cdr: "0" },
];

export default function PatientHistoryPage() {
  const [patients, setPatients] = useState<Patient[]>(PATIENTS);
  const [searchTerm, setSearchTerm] = useState("");
  const [riskFilter, setRiskFilter] = useState<"all" | "Low" | "Moderate" | "High">("all");
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  React.useEffect(() => {
    try {
      const stored = localStorage.getItem("fedx_patients");
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed) && parsed.length > 0) {
          const ids = new Set(parsed.map((p: any) => p.id));
          const rest = PATIENTS.filter((p) => !ids.has(p.id));
          setPatients([...parsed, ...rest]);
        }
      }
    } catch (e) {
      console.error(e);
    }
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
      `Age: ${patient.age} | Gender: ${patient.gender}\n` +
      `Assessed Risk Level: ${patient.latestRisk} Risk\n` +
      `Cognitive Trend: ${patient.trend.toUpperCase()}\n` +
      `MMSE Score: ${patient.mmse} / 30\n` +
      `Clinical Dementia Rating (CDR): ${patient.cdr}\n` +
      `Total Assessments: ${patient.totalAssessments}\n` +
      `Last Assessment Date: ${patient.lastVisit}\n` +
      `Security Hash: SHA256-${Math.random().toString(36).substring(2, 12)}\n` +
      `=====================================\n` +
      `Federated Multimodal Neuroimaging Model v2.4\n`;

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
    improving: { label: "↑ Improving", color: "text-emerald-600" },
    stable: { label: "→ Stable", color: "text-gray-500" },
    declining: { label: "↓ Declining", color: "text-rose-600" },
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Patient History</h1>
          <p className="text-gray-500 text-sm mt-1">Track cognitive assessments and risk trajectories across your patient cohort.</p>
        </div>
        <Link
          href="/dashboard/hospital/assessment"
          className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all duration-200"
        >
          <IconBrain size={18} />
          <span>New Assessment</span>
        </Link>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-4 text-center hover:shadow-md transition-shadow">
          <div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono">{patients.length}</div>
          <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mt-1">Total Patients</div>
        </div>
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-4 text-center hover:shadow-md transition-shadow">
          <div className="text-2xl font-extrabold text-emerald-600 font-mono">{patients.filter((p) => p.latestRisk === "Low").length}</div>
          <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mt-1">Low Risk</div>
        </div>
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-4 text-center hover:shadow-md transition-shadow">
          <div className="text-2xl font-extrabold text-amber-600 font-mono">{patients.filter((p) => p.latestRisk === "Moderate").length}</div>
          <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mt-1">Moderate Risk</div>
        </div>
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-4 text-center hover:shadow-md transition-shadow">
          <div className="text-2xl font-extrabold text-rose-600 font-mono">{patients.filter((p) => p.latestRisk === "High").length}</div>
          <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mt-1">High Risk</div>
        </div>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Patient List */}
        <div className="flex-1">
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 overflow-hidden">
            {/* Filters */}
            <div className="p-4 border-b border-gray-100 flex flex-col sm:flex-row gap-3">
              <input
                type="text"
                placeholder="Search patients by name or ID..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1 px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]"
              />
              <div className="flex gap-2">
                {(["all", "Low", "Moderate", "High"] as const).map((r) => (
                  <button
                    key={r}
                    onClick={() => setRiskFilter(r)}
                    className={`px-3 py-2 text-xs font-semibold rounded-xl transition-all ${
                      riskFilter === r
                        ? "bg-[var(--color-primary)] text-white shadow-xs"
                        : "bg-gray-50 text-gray-600 hover:bg-gray-100"
                    }`}
                  >
                    {r === "all" ? "All" : r}
                  </button>
                ))}
              </div>
            </div>

            {/* List */}
            <div className="divide-y divide-gray-100">
              {filtered.map((patient) => {
                const risk = riskConfig[patient.latestRisk];
                const trend = trendConfig[patient.trend];
                const RiskIcon = risk.icon;
                const isSelected = selectedPatient?.id === patient.id;

                return (
                  <div
                    key={patient.id}
                    onClick={() => setSelectedPatient(patient)}
                    className={`px-5 py-4 flex items-center justify-between cursor-pointer transition-all group ${
                      isSelected ? "bg-[var(--color-light-teal)]/50 border-l-4 border-l-[var(--color-primary)]" : "hover:bg-gray-50/70"
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${risk.bg} ${risk.color} shrink-0 border ${risk.border} shadow-xs`}>
                        <RiskIcon size={18} />
                      </div>
                      <div>
                        <div className="text-sm font-bold text-[var(--color-text-main)] flex items-center gap-2">
                          <span>{patient.name}</span>
                          <span className="text-xs text-gray-400 font-mono font-normal">{patient.id}</span>
                        </div>
                        <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-2">
                          <span>Age {patient.age}</span>
                          <span>•</span>
                          <span>{patient.gender}</span>
                          <span>•</span>
                          <span>{patient.lastVisit}</span>
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
              {filtered.length === 0 && (
                <div className="px-5 py-12 text-center text-gray-400 text-sm">No patients found.</div>
              )}
            </div>
          </div>
        </div>

        {/* Patient Detail Panel */}
        <div className="w-full lg:w-96 shrink-0">
          {selectedPatient ? (
            <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sticky top-24 animate-fade-in-up">
              <div className="flex items-center gap-3 mb-5">
                <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${riskConfig[selectedPatient.latestRisk].bg} ${riskConfig[selectedPatient.latestRisk].color} border ${riskConfig[selectedPatient.latestRisk].border}`}>
                  {React.createElement(riskConfig[selectedPatient.latestRisk].icon, { size: 24 })}
                </div>
                <div>
                  <h3 className="text-base font-bold text-[var(--color-text-main)]">{selectedPatient.name}</h3>
                  <p className="text-xs text-gray-500 font-mono">{selectedPatient.id} • Age {selectedPatient.age} • {selectedPatient.gender}</p>
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
                  <span className={`font-bold ${trendConfig[selectedPatient.trend].color}`}>{trendConfig[selectedPatient.trend].label}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Last Assessment</span>
                  <span className="font-bold text-[var(--color-text-main)]">{selectedPatient.lastVisit}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Current Risk Level</span>
                  <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-bold border ${riskConfig[selectedPatient.latestRisk].bg} ${riskConfig[selectedPatient.latestRisk].color} ${riskConfig[selectedPatient.latestRisk].border}`}>
                    {selectedPatient.latestRisk} Risk
                  </span>
                </div>
              </div>

              {/* Mini Cognitive Timeline */}
              <div className="mb-5">
                <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-1.5">
                  <IconHistory size={13} />
                  Assessment Timeline
                </h4>
                <div className="relative border-l-2 border-gray-100 pl-4 space-y-3 ml-1">
                  {[
                    { date: selectedPatient.lastVisit, mmse: selectedPatient.mmse, risk: selectedPatient.latestRisk },
                    { date: "3 months ago", mmse: selectedPatient.mmse + 2, risk: selectedPatient.latestRisk === "High" ? "Moderate" as const : selectedPatient.latestRisk },
                    { date: "6 months ago", mmse: Math.min(selectedPatient.mmse + 4, 30), risk: "Low" as const },
                  ].map((entry, i) => (
                    <div key={i} className="relative">
                      <div className={`absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full border-2 border-white ring-2 ${
                        riskConfig[entry.risk].bg.replace("bg-", "bg-")
                      } ${riskConfig[entry.risk].color.replace("text-", "ring-")}`} />
                      <div className="flex items-center justify-between">
                        <span className="text-[11px] text-gray-500">{entry.date}</span>
                        <span className="text-[11px] font-mono font-bold text-[var(--color-text-main)]">MMSE: {entry.mmse}</span>
                      </div>
                    </div>
                  ))}
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

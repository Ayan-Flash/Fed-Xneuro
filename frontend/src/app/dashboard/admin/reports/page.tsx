"use client";

import React, { useState } from "react";
import {
  IconReports,
  IconChartBar,
  IconHospital,
  IconSecurity,
  IconCheck,
} from "@/components/Icons";

interface ReportEntry {
  id: string;
  title: string;
  type: "audit" | "analytics" | "compliance" | "model";
  date: string;
  status: "ready" | "generating" | "scheduled";
  size?: string;
}

const REPORTS: ReportEntry[] = [
  { id: "RPT-2024-001", title: "Q3 2026 Federated Model Performance Summary", type: "model", date: "Oct 1, 2026", status: "ready", size: "2.4 MB" },
  { id: "RPT-2024-002", title: "Monthly Security Audit — September 2026", type: "audit", date: "Sep 30, 2026", status: "ready", size: "1.8 MB" },
  { id: "RPT-2024-003", title: "HIPAA Compliance Report — All Nodes", type: "compliance", date: "Sep 28, 2026", status: "ready", size: "3.1 MB" },
  { id: "RPT-2024-004", title: "Cohort Risk Distribution Analytics", type: "analytics", date: "Sep 25, 2026", status: "ready", size: "1.2 MB" },
  { id: "RPT-2024-005", title: "Node Health & Uptime Report — Q3", type: "model", date: "Sep 22, 2026", status: "ready", size: "890 KB" },
  { id: "RPT-2024-006", title: "Differential Privacy Budget Consumption", type: "compliance", date: "Sep 18, 2026", status: "ready", size: "540 KB" },
  { id: "RPT-2024-007", title: "October Analytics — In Progress", type: "analytics", date: "Oct 4, 2026", status: "generating" },
  { id: "RPT-2024-008", title: "Scheduled: Q4 Compliance Pre-Audit", type: "compliance", date: "Oct 15, 2026", status: "scheduled" },
];

export default function AdminReportsPage() {
  const [reports, setReports] = useState<ReportEntry[]>(REPORTS);
  const [typeFilter, setTypeFilter] = useState<"all" | "audit" | "analytics" | "compliance" | "model">("all");
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [generatingReport, setGeneratingReport] = useState(false);
  const [generateSuccess, setGenerateSuccess] = useState(false);

  const filtered = reports.filter((r) => typeFilter === "all" || r.type === typeFilter);

  const typeIcon = (type: ReportEntry["type"]) => {
    const map = {
      audit: { icon: IconSecurity, bg: "bg-rose-50", text: "text-rose-600", border: "border-rose-200" },
      analytics: { icon: IconChartBar, bg: "bg-blue-50", text: "text-blue-600", border: "border-blue-200" },
      compliance: { icon: IconCheck, bg: "bg-emerald-50", text: "text-emerald-600", border: "border-emerald-200" },
      model: { icon: IconReports, bg: "bg-amber-50", text: "text-amber-600", border: "border-amber-200" },
    };
    const m = map[type];
    return (
      <div className={`w-10 h-10 rounded-xl ${m.bg} ${m.text} flex items-center justify-center border ${m.border} shrink-0`}>
        <m.icon size={18} />
      </div>
    );
  };

  const statusBadge = (status: ReportEntry["status"]) => {
    const map = {
      ready: { bg: "bg-emerald-50", text: "text-emerald-700", border: "border-emerald-200", label: "Ready" },
      generating: { bg: "bg-blue-50", text: "text-blue-700", border: "border-blue-200", label: "Generating..." },
      scheduled: { bg: "bg-gray-100", text: "text-gray-600", border: "border-gray-200", label: "Scheduled" },
    };
    const s = map[status];
    return (
      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-bold ${s.bg} ${s.text} border ${s.border}`}>
        {status === "generating" && <span className="w-1.5 h-1.5 rounded-full bg-blue-500 animate-pulse mr-1.5" />}
        {s.label}
      </span>
    );
  };

  const handleDownload = (report: ReportEntry) => {
    setDownloadingId(report.id);
    
    // Generate real downloadable text/csv report
    const content = `FedX-Neuro Platform Report\nID: ${report.id}\nTitle: ${report.title}\nCategory: ${report.type}\nGenerated: ${report.date}\nStatus: ${report.status}\nIntegrity Hash: SHA256-${Math.random().toString(36).substring(2, 15)}\n\n[CONFIDENTIAL CLINICAL & FEDERATED AUDIT DATA]\n`;
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${report.id}_FedXNeuro_Report.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    setTimeout(() => setDownloadingId(null), 1200);
  };

  const handleGenerateReport = () => {
    setGeneratingReport(true);
    setTimeout(() => {
      const newId = `RPT-2026-00${reports.length + 1}`;
      const newReport: ReportEntry = {
        id: newId,
        title: `Comprehensive Federated Audit & Telemetry — ${new Date().toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}`,
        type: "audit",
        date: "Today",
        status: "ready",
        size: "1.6 MB",
      };
      setReports((prev) => [newReport, ...prev]);
      setGeneratingReport(false);
      setGenerateSuccess(true);
      setTimeout(() => setGenerateSuccess(false), 3000);
    }, 1500);
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Platform Reports</h1>
          <p className="text-gray-500 text-sm mt-1">Generate, view, and export audit logs, analytics, and compliance reports.</p>
        </div>
        <button
          onClick={handleGenerateReport}
          disabled={generatingReport}
          className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed"
        >
          {generatingReport ? (
            <><svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" /></svg> Generating...</>
          ) : (
            <><IconReports size={18} /> <span>Generate New Report</span></>
          )}
        </button>
      </div>

      {generateSuccess && (
        <div className="mb-6 bg-emerald-50 border border-emerald-200 rounded-2xl p-4 flex items-center gap-3 animate-fade-in-up">
          <div className="w-8 h-8 rounded-full bg-emerald-100 text-emerald-600 flex items-center justify-center"><IconCheck size={18} /></div>
          <div>
            <p className="text-sm font-bold text-emerald-800">Report Generated Successfully</p>
            <p className="text-xs text-emerald-600">Your new report is ready for download.</p>
          </div>
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Total Reports", value: REPORTS.length.toString(), color: "text-[var(--color-primary)]" },
          { label: "Ready", value: REPORTS.filter((r) => r.status === "ready").length.toString(), color: "text-emerald-600" },
          { label: "In Progress", value: REPORTS.filter((r) => r.status === "generating").length.toString(), color: "text-blue-600" },
          { label: "Scheduled", value: REPORTS.filter((r) => r.status === "scheduled").length.toString(), color: "text-gray-500" },
        ].map((stat, i) => (
          <div key={i} className="bg-white rounded-2xl shadow-xs border border-gray-100 p-4 text-center hover:shadow-md transition-shadow">
            <div className={`text-2xl font-extrabold font-mono ${stat.color}`}>{stat.value}</div>
            <div className="text-[11px] font-semibold text-gray-500 uppercase tracking-wider mt-1">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-2 mb-5">
        {(["all", "audit", "analytics", "compliance", "model"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTypeFilter(t)}
            className={`px-3.5 py-2 text-xs font-semibold rounded-xl transition-all ${
              typeFilter === t
                ? "bg-[var(--color-primary)] text-white shadow-xs"
                : "bg-white text-gray-600 hover:bg-gray-50 border border-gray-200"
            }`}
          >
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Report List */}
      <div className="bg-white rounded-2xl shadow-xs border border-gray-100 overflow-hidden">
        <div className="divide-y divide-gray-100">
          {filtered.map((report) => (
            <div key={report.id} className="px-5 py-4 flex items-center gap-4 hover:bg-gray-50/70 transition-colors group">
              {typeIcon(report.type)}
              <div className="flex-1 min-w-0">
                <div className="text-sm font-bold text-[var(--color-text-main)] group-hover:text-[var(--color-primary)] transition-colors truncate">{report.title}</div>
                <div className="text-xs text-gray-500 mt-0.5 flex items-center gap-2">
                  <span className="font-mono">{report.id}</span>
                  <span>•</span>
                  <span>{report.date}</span>
                  {report.size && <><span>•</span><span>{report.size}</span></>}
                </div>
              </div>
              <div className="flex items-center gap-3 shrink-0">
                {statusBadge(report.status)}
                {report.status === "ready" && (
                  <button
                    onClick={() => handleDownload(report)}
                    className={`px-3.5 py-2 text-xs font-semibold rounded-xl transition-all flex items-center gap-1.5 ${
                      downloadingId === report.id
                        ? "bg-emerald-50 text-emerald-600 border border-emerald-200"
                        : "bg-gray-50 text-gray-600 hover:bg-[var(--color-light-teal)] hover:text-[var(--color-primary)] border border-gray-200 hover:border-[var(--color-primary)]/30"
                    }`}
                  >
                    {downloadingId === report.id ? (
                      <><IconCheck size={14} /> Downloaded</>
                    ) : (
                      <>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" x2="12" y1="15" y2="3" /></svg>
                        Download
                      </>
                    )}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

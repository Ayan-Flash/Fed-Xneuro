"use client";

import React from "react";
import { 
  IconHospital, 
  IconPatients, 
  IconAssessment, 
  IconChartBar,
  IconCheck,
  IconSecurity
} from "@/components/Icons";

export default function AdminDashboard() {
  const stats = [
    { name: "Total Hospitals", value: "120", change: "+4 this month", icon: IconHospital },
    { name: "Total Patients", value: "8,432", change: "+12.5%", icon: IconPatients },
    { name: "Total Assessments", value: "24,890", change: "+18.2%", icon: IconAssessment },
    { name: "Model Accuracy", value: "96.4%", change: "Validated AUC: 0.98", icon: IconCheck },
  ];

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-[var(--color-text-main)]">System Overview</h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              All Nodes Operational
            </span>
          </div>
          <p className="text-gray-500 text-sm mt-1">Platform-wide federated network statistics, risk telemetry, and audit logs.</p>
        </div>
        <div className="text-xs text-gray-400 font-mono bg-white px-3 py-1.5 rounded-xl border border-gray-200/80 shadow-xs self-start sm:self-auto">
          Last Sync: <span className="text-gray-700 font-medium">Just now</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 mb-8">
        {stats.map((stat, i) => (
          <div key={i} className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex items-center gap-4 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
            <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center shrink-0 shadow-xs">
              <stat.icon size={22} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="text-xs font-medium text-gray-500 truncate">{stat.name}</div>
              <div className="text-2xl font-extrabold text-[var(--color-text-main)] tracking-tight mt-0.5">{stat.value}</div>
              <div className="text-[11px] font-medium text-emerald-600 mt-1">{stat.change}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 sm:gap-8">
        {/* Risk Distribution Chart */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-xs border border-gray-100 p-6">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
            <div>
              <h2 className="text-base font-bold text-[var(--color-text-main)] flex items-center gap-2">
                <IconChartBar size={18} className="text-[var(--color-primary)]" />
                Global Cohort Risk Distribution
              </h2>
              <p className="text-xs text-gray-500 mt-0.5">Aggregated predictions across 120 federated hospital nodes</p>
            </div>
            <select className="bg-gray-50/90 border border-gray-200 text-xs rounded-xl focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] px-3 py-1.5 font-medium text-gray-700 self-start sm:self-auto">
              <option>Last 30 Days</option>
              <option>Last 3 Months</option>
              <option>This Year</option>
            </select>
          </div>
          
          <div className="h-64 flex items-end justify-between gap-2 px-2 pb-6 pt-4 border-b border-gray-100 relative">
            {/* Y-axis labels */}
            <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-[11px] text-gray-400 pb-6 font-mono">
              <span>100%</span>
              <span>50%</span>
              <span>0%</span>
            </div>
            
            <div className="w-full flex items-end justify-around pl-10 h-full relative">
              {/* Grid lines */}
              <div className="absolute left-10 right-0 top-0 border-t border-dashed border-gray-100"></div>
              <div className="absolute left-10 right-0 top-1/2 border-t border-dashed border-gray-100"></div>
              
              {/* Bars */}
              <div className="w-16 md:w-24 bg-gradient-to-t from-[var(--color-secondary)] to-emerald-400 rounded-t-xl h-[25%] relative group transition-all duration-300 hover:brightness-110 shadow-xs">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 text-white text-[11px] font-medium py-1 px-2.5 rounded-lg shadow-lg pointer-events-none">25% (6,222)</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-600 font-semibold translate-y-6">Low Risk</div>
              </div>
              <div className="w-16 md:w-24 bg-gradient-to-t from-amber-500 to-amber-400 rounded-t-xl h-[45%] relative group transition-all duration-300 hover:brightness-110 shadow-xs">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 text-white text-[11px] font-medium py-1 px-2.5 rounded-lg shadow-lg pointer-events-none">45% (11,200)</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-600 font-semibold translate-y-6">Moderate</div>
              </div>
              <div className="w-16 md:w-24 bg-gradient-to-t from-rose-600 to-rose-400 rounded-t-xl h-[30%] relative group transition-all duration-300 hover:brightness-110 shadow-xs">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-900 text-white text-[11px] font-medium py-1 px-2.5 rounded-lg shadow-lg pointer-events-none">30% (7,468)</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-600 font-semibold translate-y-6">High Risk</div>
              </div>
            </div>
          </div>

          <div className="pt-4 flex items-center justify-between text-xs text-gray-500">
            <span>Aggregated via Differential Privacy (&epsilon;=1.0)</span>
            <span className="font-semibold text-[var(--color-primary)]">Total: 24,890 evaluations</span>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-[var(--color-text-main)] mb-5 flex items-center gap-2">
              <IconSecurity size={18} className="text-[var(--color-primary)]" />
              Security & Node Audits
            </h2>
            <div className="relative border-l-2 border-gray-100 pl-4 space-y-4 ml-2">
              {[
                { title: "New Hospital Registered", desc: "Memorial Healthcare joined the network.", time: "10 mins ago" },
                { title: "Federated Model Update", desc: "Global model v2.4 aggregated across nodes.", time: "2 hours ago" },
                { title: "Differential Privacy Verified", desc: "Local noise calibration verified on Node 4.", time: "5 hours ago" },
                { title: "Audit Log Exported", desc: "Encrypted export by admin_jsmith.", time: "1 day ago" },
                { title: "Access Attempt Warning", desc: "Unusual authentication pattern blocked.", time: "1 day ago", alert: true },
              ].map((log, i) => (
                <div key={i} className="relative">
                  <div className={`absolute -left-[21px] top-1 w-2.5 h-2.5 rounded-full border-2 border-white ring-2 ${log.alert ? 'bg-rose-500 ring-rose-200' : 'bg-[var(--color-primary)] ring-[var(--color-light-teal)]'}`}></div>
                  <div className={`text-xs font-bold ${log.alert ? 'text-rose-600' : 'text-[var(--color-text-main)]'}`}>{log.title}</div>
                  <div className="text-[11px] text-gray-500 mt-0.5 leading-snug">{log.desc}</div>
                  <div className="text-[10px] text-gray-400 mt-1 font-mono">{log.time}</div>
                </div>
              ))}
            </div>
          </div>
          <button className="w-full mt-6 py-2.5 text-xs font-bold text-[var(--color-primary)] bg-[var(--color-light-teal)] rounded-xl hover:bg-[var(--color-primary)] hover:text-white transition-all shadow-xs">
            View All Audit Logs
          </button>
        </div>
      </div>
    </div>
  );
}

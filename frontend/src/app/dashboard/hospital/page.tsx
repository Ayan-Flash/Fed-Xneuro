"use client";

import React from "react";
import Link from "next/link";
import { 
  IconPatients, 
  IconBrain,
  IconRiskLow,
  IconRiskModerate,
  IconRiskHigh
} from "@/components/Icons";

export default function HospitalDashboard() {
  const recentAssessments = [
    { id: "PAT-8492", date: "Today, 09:42 AM", name: "Robert J.", risk: "High", icon: IconRiskHigh, color: "text-[var(--color-danger)]", bg: "bg-red-50" },
    { id: "PAT-8491", date: "Yesterday", name: "Sarah M.", risk: "Low", icon: IconRiskLow, color: "text-[var(--color-primary)]", bg: "bg-[var(--color-light-teal)]" },
    { id: "PAT-8490", date: "Yesterday", name: "Michael T.", risk: "Moderate", icon: IconRiskModerate, color: "text-[var(--color-warning)]", bg: "bg-yellow-50" },
    { id: "PAT-8489", date: "Oct 12, 2026", name: "Eleanor W.", risk: "Low", icon: IconRiskLow, color: "text-[var(--color-primary)]", bg: "bg-[var(--color-light-teal)]" },
  ];

  return (
    <div className="p-6 lg:p-10 max-w-7xl mx-auto animate-fade-in-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Clinical Dashboard</h1>
          <p className="text-gray-600 mt-1">Welcome back. Here is your daily patient overview.</p>
        </div>
        <Link 
          href="/dashboard/hospital/assessment"
          className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-sm transition-all duration-200"
        >
          <IconBrain size={18} />
          New Cognitive Assessment
        </Link>
      </div>

      {/* Quick Actions & Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10">
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col items-center justify-center text-center hover:border-[var(--color-primary)]/30 transition-colors cursor-pointer">
          <div className="w-12 h-12 rounded-full bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-3">
            <IconPatients size={24} />
          </div>
          <h3 className="text-lg font-bold text-[var(--color-text-main)]">Manage Patients</h3>
          <p className="text-sm text-gray-500 mt-1">View or edit patient records.</p>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex items-center gap-4">
           <div>
             <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">Assessments This Week</div>
             <div className="text-3xl font-bold text-[var(--color-text-main)] mt-2">24</div>
             <div className="text-xs font-medium text-[var(--color-primary)] mt-1">+12% from last week</div>
           </div>
        </div>
        
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex items-center gap-4">
           <div>
             <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">High Risk Identified</div>
             <div className="text-3xl font-bold text-[var(--color-danger)] mt-2">3</div>
             <div className="text-xs font-medium text-gray-500 mt-1">Requires clinical review</div>
           </div>
        </div>
      </div>

      {/* Recent Assessments List */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
        <div className="px-6 py-5 border-b border-gray-100 flex justify-between items-center bg-gray-50/50">
          <h2 className="text-lg font-bold text-[var(--color-text-main)]">Recent Assessments</h2>
          <button className="text-sm font-medium text-[var(--color-primary)] hover:text-[var(--color-secondary)]">View All</button>
        </div>
        
        <div className="divide-y divide-gray-100">
          {recentAssessments.map((assessment, i) => (
            <div key={i} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${assessment.bg} ${assessment.color}`}>
                  <assessment.icon size={20} />
                </div>
                <div>
                  <div className="font-semibold text-[var(--color-text-main)]">{assessment.name} <span className="text-xs text-gray-400 font-normal ml-2">{assessment.id}</span></div>
                  <div className="text-xs text-gray-500">{assessment.date}</div>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${assessment.bg} ${assessment.color} border-current/20`}>
                  {assessment.risk} Risk
                </span>
                <button className="text-sm font-medium text-gray-500 hover:text-[var(--color-primary)]">
                  View Report
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

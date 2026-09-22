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
  const [greeting, setGreeting] = React.useState("Welcome back");

  React.useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("Good morning");
    else if (hour < 17) setGreeting("Good afternoon");
    else setGreeting("Good evening");
  }, []);

  const recentAssessments = [
    { id: "PAT-8492", date: "Today, 09:42 AM", name: "Robert J.", age: 72, risk: "High", icon: IconRiskHigh, color: "text-rose-600", bg: "bg-rose-50", border: "border-rose-200" },
    { id: "PAT-8491", date: "Yesterday", name: "Sarah M.", age: 65, risk: "Low", icon: IconRiskLow, color: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200" },
    { id: "PAT-8490", date: "Yesterday", name: "Michael T.", age: 69, risk: "Moderate", icon: IconRiskModerate, color: "text-amber-700", bg: "bg-amber-50", border: "border-amber-200" },
    { id: "PAT-8489", date: "Oct 12, 2026", name: "Eleanor W.", age: 61, risk: "Low", icon: IconRiskLow, color: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-200" },
  ];

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
              {greeting}, Dr. Jenkins
            </h1>
            <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-semibold bg-[var(--color-light-teal)] text-[var(--color-primary)]">
              St. Jude Node
            </span>
          </div>
          <p className="text-gray-500 text-sm mt-1">Here is your clinical patient cohort overview for today.</p>
        </div>
        <Link 
          href="/dashboard/hospital/assessment"
          className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all duration-200"
        >
          <IconBrain size={18} />
          <span>New Cognitive Assessment</span>
        </Link>
      </div>

      {/* Quick Actions & Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center gap-4 hover:shadow-md hover:border-[var(--color-primary)]/30 hover:-translate-y-0.5 transition-all duration-200 cursor-pointer group">
          <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center shrink-0 group-hover:scale-105 group-hover:bg-[var(--color-primary)] group-hover:text-white transition-all">
            <IconPatients size={22} />
          </div>
          <div>
            <h3 className="text-base font-bold text-[var(--color-text-main)] group-hover:text-[var(--color-primary)] transition-colors">Manage Patients</h3>
            <p className="text-xs text-gray-500 mt-0.5">View or edit 142 patient cohort records.</p>
          </div>
        </div>
        
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center justify-between hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Assessments This Week</div>
            <div className="text-3xl font-extrabold text-[var(--color-text-main)] mt-1.5 tracking-tight font-mono">24</div>
            <div className="text-xs font-semibold text-emerald-600 mt-1">+12% from last week</div>
          </div>
          <div className="w-10 h-10 rounded-xl bg-emerald-50 text-emerald-600 flex items-center justify-center text-sm font-bold">
            &uarr;
          </div>
        </div>
        
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center justify-between hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">High Risk Cohort</div>
            <div className="text-3xl font-extrabold text-rose-600 mt-1.5 tracking-tight font-mono">3</div>
            <div className="text-xs font-semibold text-rose-600/80 mt-1">Requires clinical follow-up</div>
          </div>
          <div className="w-10 h-10 rounded-xl bg-rose-50 text-rose-600 flex items-center justify-center text-sm font-bold">
            !
          </div>
        </div>
      </div>

      {/* Recent Assessments List */}
      <div className="bg-white rounded-2xl shadow-xs border border-gray-100 overflow-hidden">
        <div className="px-6 py-4.5 border-b border-gray-100 flex justify-between items-center bg-gray-50/40">
          <div>
            <h2 className="text-base font-bold text-[var(--color-text-main)]">Recent Assessments</h2>
            <p className="text-xs text-gray-500">Evaluations processed via federated explainable AI</p>
          </div>
          <button className="text-xs font-semibold text-[var(--color-primary)] hover:text-[var(--color-secondary)] px-3 py-1.5 rounded-lg hover:bg-[var(--color-light-teal)] transition-colors">
            View All (24)
          </button>
        </div>
        
        <div className="divide-y divide-gray-100">
          {recentAssessments.map((assessment, i) => (
            <div key={i} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50/70 transition-colors group">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${assessment.bg} ${assessment.color} shrink-0 border ${assessment.border} shadow-xs`}>
                  <assessment.icon size={20} />
                </div>
                <div>
                  <div className="text-sm font-bold text-[var(--color-text-main)] flex items-center gap-2">
                    <span>{assessment.name}</span>
                    <span className="text-xs text-gray-400 font-mono font-normal">{assessment.id}</span>
                    <span className="text-xs text-gray-400 font-normal">Age {assessment.age}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-0.5">{assessment.date}</div>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${assessment.bg} ${assessment.color} ${assessment.border}`}>
                  {assessment.risk} Risk
                </span>
                <Link 
                  href="/dashboard/hospital/assessment"
                  className="text-xs font-semibold text-gray-600 hover:text-[var(--color-primary)] px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors group-hover:translate-x-0.5"
                >
                  View Report &rarr;
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

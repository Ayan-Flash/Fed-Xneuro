"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { 
  IconPatients, 
  IconBrain,
  IconRiskLow,
  IconRiskModerate,
  IconRiskHigh
} from "@/components/Icons";
import { fetchAssessmentHistory, SavedAssessmentRecord } from "@/lib/api";

export default function HospitalDashboard() {
  const [greeting, setGreeting] = useState("Welcome back");
  const [assessments, setAssessments] = useState<SavedAssessmentRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const hour = new Date().getHours();
    if (hour < 12) setGreeting("Good morning");
    else if (hour < 17) setGreeting("Good afternoon");
    else setGreeting("Good evening");
  }, []);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const res = await fetchAssessmentHistory();
        let records = res.records || [];

        // Check local storage for any newly created client assessments
        if (typeof window !== "undefined") {
          try {
            const stored = localStorage.getItem("fedx_patients");
            if (stored) {
              const parsed = JSON.parse(stored);
              if (Array.isArray(parsed) && parsed.length > 0) {
                const apiIds = new Set(records.map((r) => r.id));
                const localOnly = parsed.filter((p: any) => !apiIds.has(p.id));
                records = [...localOnly, ...records];
              }
            }
          } catch (e) {
            console.error("Error reading localStorage:", e);
          }
        }

        setAssessments(records);
      } catch (err) {
        console.error("Failed to fetch assessment records:", err);
        setAssessments([]);
      } finally {
        setLoading(false);
      }
    }

    loadData();
  }, []);

  // Compute dynamic clinical metrics
  const uniquePatients = new Set(assessments.map((a) => a.id)).size;
  const totalAssessments = assessments.length;
  const highRiskCohort = assessments.filter(
    (a) => a.risk === "High" || (a.progression_probability && a.progression_probability >= 65)
  ).length;

  const getRiskBadge = (risk: string) => {
    switch (risk) {
      case "High":
        return {
          icon: IconRiskHigh,
          color: "text-rose-600",
          bg: "bg-rose-50",
          border: "border-rose-200"
        };
      case "Moderate":
        return {
          icon: IconRiskModerate,
          color: "text-amber-700",
          bg: "bg-amber-50",
          border: "border-amber-200"
        };
      default:
        return {
          icon: IconRiskLow,
          color: "text-emerald-700",
          bg: "bg-emerald-50",
          border: "border-emerald-200"
        };
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-7xl mx-auto animate-fade-in-up">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
            {greeting}, Dr. Jenkins
          </h1>
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

      {/* Dynamic Actions & Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        <Link 
          href="/dashboard/hospital/patients"
          className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center gap-4 hover:shadow-md hover:border-[var(--color-primary)]/30 hover:-translate-y-0.5 transition-all duration-200 group"
        >
          <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center shrink-0 group-hover:scale-105 group-hover:bg-[var(--color-primary)] group-hover:text-white transition-all">
            <IconPatients size={22} />
          </div>
          <div>
            <h3 className="text-base font-bold text-[var(--color-text-main)] group-hover:text-[var(--color-primary)] transition-colors">Manage Patients</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              {loading ? "Loading records..." : `View or edit ${uniquePatients} patient cohort records.`}
            </p>
          </div>
        </Link>
        
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center justify-between hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Total Assessments</div>
            <div className="text-3xl font-extrabold text-[var(--color-text-main)] mt-1.5 tracking-tight font-mono">
              {loading ? "-" : totalAssessments}
            </div>
            <div className="text-xs font-semibold text-gray-500 mt-1">Recorded cohort evaluations</div>
          </div>
          <div className="w-10 h-10 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center text-sm font-bold">
            <IconBrain size={20} />
          </div>
        </div>
        
        <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex items-center justify-between hover:shadow-md hover:-translate-y-0.5 transition-all duration-200">
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">High Risk Cohort</div>
            <div className="text-3xl font-extrabold text-rose-600 mt-1.5 tracking-tight font-mono">
              {loading ? "-" : highRiskCohort}
            </div>
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
          <Link
            href="/dashboard/hospital/patients"
            className="text-xs font-semibold text-[var(--color-primary)] hover:text-[var(--color-secondary)] px-3 py-1.5 rounded-lg hover:bg-[var(--color-light-teal)] transition-colors"
          >
            View All ({totalAssessments})
          </Link>
        </div>
        
        {loading ? (
          <div className="p-8 text-center text-sm text-gray-400">Loading assessments...</div>
        ) : assessments.length === 0 ? (
          <div className="py-12 px-6 text-center">
            <div className="w-14 h-14 mx-auto mb-3 rounded-2xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center">
              <IconBrain size={28} />
            </div>
            <h3 className="text-base font-bold text-[var(--color-text-main)]">No Clinical Assessments Yet</h3>
            <p className="text-sm text-gray-500 mt-1 max-w-md mx-auto">
              Get started by creating a cognitive health assessment for a patient to view real-time risk predictions and explanations here.
            </p>
            <div className="mt-5">
              <Link
                href="/dashboard/hospital/assessment"
                className="inline-flex items-center gap-2 px-5 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-bold rounded-xl shadow-xs hover:bg-[var(--color-secondary)] transition-all"
              >
                <IconBrain size={16} />
                <span>Start First Assessment</span>
              </Link>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {assessments.slice(0, 5).map((assessment, i) => {
              const badge = getRiskBadge(assessment.risk);
              const BadgeIcon = badge.icon;
              return (
                <div key={assessment.id || i} className="px-6 py-4 flex items-center justify-between hover:bg-gray-50/70 transition-colors group">
                  <div className="flex items-center gap-4">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${badge.bg} ${badge.color} shrink-0 border ${badge.border} shadow-xs`}>
                      <BadgeIcon size={20} />
                    </div>
                    <div>
                      <div className="text-sm font-bold text-[var(--color-text-main)] flex items-center gap-2">
                        <span>{assessment.name}</span>
                        <span className="text-xs text-gray-400 font-mono font-normal">{assessment.id}</span>
                        {assessment.age && <span className="text-xs text-gray-400 font-normal">Age {assessment.age}</span>}
                      </div>
                      <div className="text-xs text-gray-500 mt-0.5">{assessment.date || "Recent"}</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-bold border ${badge.bg} ${badge.color} ${badge.border}`}>
                      {assessment.risk} Risk
                    </span>
                    <Link 
                      href="/dashboard/hospital/patients"
                      className="text-xs font-semibold text-gray-600 hover:text-[var(--color-primary)] px-3 py-1.5 rounded-lg hover:bg-gray-100 transition-colors group-hover:translate-x-0.5"
                    >
                      View Report &rarr;
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

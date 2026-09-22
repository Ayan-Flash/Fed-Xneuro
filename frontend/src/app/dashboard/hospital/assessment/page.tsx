"use client";

import React, { useState } from "react";
import Link from "next/link";
import { 
  IconBrain, 
  IconRiskHigh,
  IconRiskModerate,
  IconChartBar
} from "@/components/Icons";

type AssessmentState = "input" | "analyzing" | "result";

export default function AssessmentPage() {
  const [state, setState] = useState<AssessmentState>("input");
  const [analyzingPhase, setAnalyzingPhase] = useState("Extracting patient metrics...");

  const handleSimulate = (e: React.FormEvent) => {
    e.preventDefault();
    setState("analyzing");
    setAnalyzingPhase("Extracting cognitive & demographic features...");
    
    setTimeout(() => {
      setAnalyzingPhase("Running federated multimodal neuro-model...");
    }, 900);

    setTimeout(() => {
      setAnalyzingPhase("Computing explainability attributions...");
    }, 1700);

    setTimeout(() => {
      setState("result");
    }, 2400);
  };

  const handleReset = () => {
    setState("input");
  };

  const handlePrint = () => {
    if (typeof window !== "undefined") {
      window.print();
    }
  };

  return (
    <div className="p-6 lg:p-8 max-w-5xl mx-auto animate-fade-in-up">
      {/* Breadcrumb Navigation */}
      <div className="mb-5 flex items-center gap-2.5 text-xs font-semibold text-gray-500">
        <Link href="/dashboard/hospital" className="hover:text-[var(--color-primary)] transition-colors">Hospital Dashboard</Link>
        <span>/</span>
        <span className="text-[var(--color-text-main)] font-bold">New Assessment</span>
      </div>

      {/* Progress Stepper */}
      <div className="bg-white rounded-2xl border border-gray-100 p-4 mb-6 shadow-xs">
        <div className="grid grid-cols-3 gap-2 text-center text-xs">
          <div className={`flex items-center justify-center gap-2 py-2 px-3 rounded-xl transition-all ${
            state === "input" 
              ? "bg-[var(--color-light-teal)] text-[var(--color-primary)] font-bold shadow-xs" 
              : "text-gray-400 font-medium"
          }`}>
            <span className="w-5 h-5 rounded-full bg-white text-[var(--color-primary)] flex items-center justify-center text-[10px] font-bold border border-[var(--color-primary)]/20">1</span>
            <span className="hidden sm:inline">Patient Input</span>
          </div>
          <div className={`flex items-center justify-center gap-2 py-2 px-3 rounded-xl transition-all ${
            state === "analyzing" 
              ? "bg-[var(--color-light-teal)] text-[var(--color-primary)] font-bold shadow-xs" 
              : "text-gray-400 font-medium"
          }`}>
            <span className="w-5 h-5 rounded-full bg-white text-[var(--color-primary)] flex items-center justify-center text-[10px] font-bold border border-[var(--color-primary)]/20">2</span>
            <span className="hidden sm:inline">AI Analysis</span>
          </div>
          <div className={`flex items-center justify-center gap-2 py-2 px-3 rounded-xl transition-all ${
            state === "result" 
              ? "bg-emerald-50 text-emerald-800 font-bold shadow-xs" 
              : "text-gray-400 font-medium"
          }`}>
            <span className="w-5 h-5 rounded-full bg-white text-emerald-700 flex items-center justify-center text-[10px] font-bold border border-emerald-200">3</span>
            <span className="hidden sm:inline">Clinical Report</span>
          </div>
        </div>
      </div>

      {state === "input" && (
        <form onSubmit={handleSimulate} className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sm:p-8">
          <div className="flex items-center justify-between pb-4 mb-6 border-b border-gray-100">
            <div>
              <h2 className="text-lg font-bold text-[var(--color-text-main)]">Patient Clinical Parameters</h2>
              <p className="text-xs text-gray-500 mt-0.5">Input standard neuropsychological and demographic metrics</p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-gray-100 text-gray-600">
              Protocol: ADNI-3
            </span>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5 mb-8">
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Patient ID</label>
              <input type="text" defaultValue="PAT-8495" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" placeholder="e.g., PAT-8495" required />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Age</label>
              <input type="number" defaultValue="72" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" placeholder="e.g., 72" required />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">MMSE Score (0-30)</label>
              <input type="number" max="30" min="0" defaultValue="21" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" placeholder="Mini-Mental State Exam" required />
            </div>
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Clinical Dementia Rating (CDR)</label>
              <select defaultValue="1 - Mild Dementia" className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white">
                <option>0 - Normal</option>
                <option>0.5 - Very Mild Dementia</option>
                <option>1 - Mild Dementia</option>
                <option>2 - Moderate Dementia</option>
                <option>3 - Severe Dementia</option>
              </select>
            </div>
            <div className="sm:col-span-2">
              <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5">Neuroimaging (MRI / FDG-PET) [Optional]</label>
              <div className="border-2 border-dashed border-gray-200 rounded-2xl p-6 flex flex-col items-center justify-center text-gray-500 hover:bg-gray-50/70 hover:border-[var(--color-primary)]/40 transition-all cursor-pointer group">
                <div className="w-12 h-12 rounded-2xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-2 group-hover:scale-105 transition-transform">
                  <IconBrain size={24} />
                </div>
                <span className="text-xs font-semibold text-gray-700">Click to upload DICOM / NIfTI volumetric data</span>
                <span className="text-[11px] text-gray-400 mt-0.5">Encrypted locally before feature extraction</span>
              </div>
            </div>
          </div>

          <div className="flex justify-end gap-3 pt-4 border-t border-gray-100">
            <Link href="/dashboard/hospital" className="px-5 py-2.5 text-xs font-semibold text-gray-600 hover:bg-gray-100 rounded-xl transition-colors">
              Cancel
            </Link>
            <button type="submit" className="px-6 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-bold rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all flex items-center gap-2">
              <IconBrain size={16} />
              <span>Run AI Prediction</span>
            </button>
          </div>
        </form>
      )}

      {state === "analyzing" && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-16 flex flex-col items-center justify-center text-center animate-fade-in-up">
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-[var(--color-light-teal)] rounded-full animate-ping opacity-60"></div>
            <div className="relative w-20 h-20 bg-gradient-to-tr from-[var(--color-primary)] to-[var(--color-secondary)] rounded-2xl flex items-center justify-center text-white shadow-lg shadow-[var(--color-primary)]/25 animate-pulse">
              <IconBrain size={38} />
            </div>
          </div>
          <h2 className="text-xl font-bold text-[var(--color-text-main)] mb-2">Analyzing Clinical Cohort Data</h2>
          <p className="text-xs sm:text-sm text-[var(--color-primary)] font-semibold font-mono animate-fade-in-up">{analyzingPhase}</p>
          <div className="w-64 h-1.5 bg-gray-100 rounded-full mt-5 overflow-hidden">
            <div className="h-full bg-[var(--color-primary)] rounded-full animate-pulse w-3/4"></div>
          </div>
        </div>
      )}

      {state === "result" && (
        <div className="animate-fade-in-up space-y-6">
          {/* Medical Disclaimer Alert */}
          <div className="bg-amber-50 border border-amber-200/80 p-4 rounded-2xl flex gap-3 shadow-xs">
            <div className="text-amber-600 shrink-0 mt-0.5">
              <IconRiskModerate size={20} />
            </div>
            <div>
              <p className="text-xs font-bold text-amber-900 uppercase tracking-wider">Clinical Decision Support Aid</p>
              <p className="text-xs text-amber-800/90 mt-0.5 leading-relaxed">This evaluation is an explainable probabilistic screening aid, not an autonomous medical diagnosis. Findings must be correlated with clinical judgment and history.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Risk Card */}
            <div className="md:col-span-1 bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex flex-col items-center justify-center text-center">
              <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-6">Assessed Risk Level</h3>
              
              <div className="w-36 h-36 rounded-full border-8 border-rose-50 flex items-center justify-center relative mb-6">
                <svg className="absolute inset-0 w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="50" fill="transparent" stroke="#FEE2E2" strokeWidth="8" />
                  <circle cx="60" cy="60" r="50" fill="transparent" stroke="#E11D48" strokeWidth="8" strokeDasharray="314" strokeDashoffset="75" strokeLinecap="round" className="transition-all duration-1000 ease-out" />
                </svg>
                <div className="flex flex-col items-center z-10">
                  <IconRiskHigh size={30} className="text-rose-600 mb-1" />
                  <span className="text-lg font-extrabold text-rose-600">High Risk</span>
                </div>
              </div>

              <div className="w-full space-y-2.5 text-xs">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-500">Progression Probability</span>
                  <span className="font-bold text-[var(--color-text-main)] font-mono text-sm">78.4%</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-500">Model Confidence</span>
                  <span className="font-bold text-emerald-700 font-mono text-sm">92.1%</span>
                </div>
              </div>
            </div>

            {/* Metrics & Factors */}
            <div className="md:col-span-2 bg-white rounded-2xl shadow-xs border border-gray-100 p-6">
              <div className="flex items-center justify-between pb-4 mb-6 border-b border-gray-100">
                <h3 className="text-base font-bold text-[var(--color-text-main)]">Cognitive Metrics & Feature Attributions</h3>
                <span className="text-xs text-gray-400 font-mono">SHAP Attributions</span>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mb-6">
                <div className="p-3.5 bg-gray-50/70 rounded-xl border border-gray-100">
                  <div className="text-xs text-gray-500 font-medium">MMSE Score</div>
                  <div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono mt-0.5">21 <span className="text-xs font-normal text-gray-400">/ 30</span></div>
                  <div className="text-[11px] font-semibold text-rose-600 mt-1">Below clinical threshold (&lt;24)</div>
                </div>
                <div className="p-3.5 bg-gray-50/70 rounded-xl border border-gray-100">
                  <div className="text-xs text-gray-500 font-medium">CDR Score</div>
                  <div className="text-2xl font-extrabold text-[var(--color-text-main)] font-mono mt-0.5">1.0</div>
                  <div className="text-[11px] font-semibold text-amber-700 mt-1">Mild Dementia Indication</div>
                </div>
              </div>

              <div>
                <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3.5 flex items-center gap-2">
                  <IconChartBar size={15} className="text-[var(--color-primary)]" />
                  Primary Predictive Factors
                </h4>
                <div className="space-y-3">
                  <div className="flex items-center gap-3 text-xs">
                    <div className="w-1/3 text-gray-700 font-medium truncate">Memory Recall Decline</div>
                    <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-rose-500 rounded-full w-[85%] transition-all duration-700"></div>
                    </div>
                    <div className="font-mono font-bold text-rose-600 w-10 text-right">85%</div>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <div className="w-1/3 text-gray-700 font-medium truncate">Executive Function Test</div>
                    <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-amber-500 rounded-full w-[60%] transition-all duration-700"></div>
                    </div>
                    <div className="font-mono font-bold text-amber-700 w-10 text-right">60%</div>
                  </div>
                  <div className="flex items-center gap-3 text-xs">
                    <div className="w-1/3 text-gray-700 font-medium truncate">Age & Demographic Risk</div>
                    <div className="flex-1 h-2.5 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--color-primary)] rounded-full w-[40%] transition-all duration-700"></div>
                    </div>
                    <div className="font-mono font-bold text-[var(--color-primary)] w-10 text-right">40%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-base font-bold text-[var(--color-text-main)]">Clinical Protocol Recommendation</h3>
              <p className="text-xs text-gray-500 mt-0.5">High-risk profile indicates scheduling 6-month cognitive monitoring and biomarker review.</p>
            </div>
            <div className="flex flex-wrap gap-2.5 w-full sm:w-auto">
              <button 
                type="button"
                onClick={handleReset}
                className="px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold rounded-xl transition-colors"
              >
                New Assessment
              </button>
              <button 
                type="button"
                onClick={handlePrint}
                className="px-4 py-2.5 bg-white border border-[var(--color-primary)] text-[var(--color-primary)] text-xs font-semibold rounded-xl hover:bg-[var(--color-light-teal)] transition-colors flex items-center gap-1.5"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 6 2 18 2 18 9"></polyline>
                  <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                  <rect x="6" y="14" width="12" height="8"></rect>
                </svg>
                Export Report
              </button>
              <Link 
                href="/dashboard/hospital"
                className="px-5 py-2.5 bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white text-xs font-semibold rounded-xl shadow-xs transition-colors text-center"
              >
                Return to Cohort
              </Link>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

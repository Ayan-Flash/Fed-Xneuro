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

  const handleSimulate = (e: React.FormEvent) => {
    e.preventDefault();
    setState("analyzing");
    setTimeout(() => {
      setState("result");
    }, 2500);
  };

  return (
    <div className="p-6 lg:p-10 max-w-5xl mx-auto animate-fade-in-up">
      <div className="mb-6 flex items-center gap-3 text-sm font-medium text-gray-500">
        <Link href="/dashboard/hospital" className="hover:text-[var(--color-primary)]">Dashboard</Link>
        <span>/</span>
        <span className="text-[var(--color-text-main)]">New Cognitive Assessment</span>
      </div>

      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">Cognitive Health Assessment</h1>
          <p className="text-gray-600 mt-1">Run AI-assisted analysis for risk prediction.</p>
        </div>
      </div>

      {state === "input" && (
        <form onSubmit={handleSimulate} className="bg-white rounded-xl shadow-sm border border-gray-100 p-8">
          <h2 className="text-lg font-bold text-[var(--color-text-main)] mb-6 border-b border-gray-100 pb-4">Patient Data Input</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Patient ID</label>
              <input type="text" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)]" placeholder="e.g., PAT-8495" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
              <input type="number" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)]" placeholder="e.g., 72" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">MMSE Score (0-30)</label>
              <input type="number" max="30" min="0" className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)]" placeholder="Mini-Mental State Exam" required />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">CDR Score</label>
              <select className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)]">
                <option>0 - Normal</option>
                <option>0.5 - Very Mild Dementia</option>
                <option>1 - Mild Dementia</option>
                <option>2 - Moderate Dementia</option>
                <option>3 - Severe Dementia</option>
              </select>
            </div>
            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">Attach Neuroimaging (MRI/PET) [Optional]</label>
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 flex flex-col items-center justify-center text-gray-500 hover:bg-gray-50 hover:border-[var(--color-primary)] transition-colors cursor-pointer">
                <IconBrain size={32} className="mb-2 text-gray-400" />
                <span className="text-sm font-medium">Click to upload DICOM / NIfTI files</span>
                <span className="text-xs mt-1">or drag and drop</span>
              </div>
            </div>
          </div>

          <div className="flex justify-end">
            <button type="submit" className="px-8 py-3 bg-[var(--color-primary)] text-white font-semibold rounded-lg shadow-sm hover:bg-[var(--color-secondary)] transition-colors flex items-center gap-2">
              <IconBrain size={18} />
              Run AI Analysis
            </button>
          </div>
        </form>
      )}

      {state === "analyzing" && (
        <div className="bg-white rounded-xl shadow-sm border border-[var(--color-primary)]/20 p-16 flex flex-col items-center justify-center text-center animate-fade-in-up">
          <div className="relative mb-8">
            <div className="absolute inset-0 bg-[var(--color-light-teal)] rounded-full animate-ping opacity-75"></div>
            <div className="relative w-20 h-20 bg-[var(--color-primary)] rounded-full flex items-center justify-center text-white shadow-lg">
              <IconBrain size={40} />
            </div>
          </div>
          <h2 className="text-2xl font-bold text-[var(--color-text-main)] mb-2">Analyzing Patient Data</h2>
          <p className="text-gray-500 max-w-md">The AI model is processing the cognitive metrics and comparing against the clinical baseline dataset...</p>
        </div>
      )}

      {state === "result" && (
        <div className="animate-fade-in-up space-y-6">
          {/* Medical Disclaimer */}
          <div className="bg-yellow-50 border-l-4 border-[var(--color-warning)] p-4 rounded-r-lg flex gap-3">
            <div className="text-[var(--color-warning)] shrink-0">
              <IconRiskModerate size={20} />
            </div>
            <div>
              <p className="text-sm text-yellow-800 font-medium">Clinical Disclaimer</p>
              <p className="text-xs text-yellow-700 mt-1">This prediction is a screening aid and is not a medical diagnosis. All AI-assisted assessments must be reviewed by a qualified healthcare professional.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Risk Card */}
            <div className="md:col-span-1 bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col items-center justify-center text-center">
              <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Assessed Risk Level</h3>
              
              <div className="w-32 h-32 rounded-full border-8 border-red-100 flex items-center justify-center relative mb-6">
                <svg className="absolute inset-0 w-full h-full transform -rotate-90">
                  <circle cx="60" cy="60" r="56" fill="transparent" stroke="var(--color-danger)" strokeWidth="8" strokeDasharray="351" strokeDashoffset="87" className="transition-all duration-1000 ease-out" />
                </svg>
                <div className="flex flex-col items-center">
                  <IconRiskHigh size={32} className="text-[var(--color-danger)] mb-1" />
                  <span className="text-xl font-bold text-[var(--color-danger)]">High Risk</span>
                </div>
              </div>

              <div className="w-full space-y-3 text-sm">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-500">Prediction Probability</span>
                  <span className="font-bold text-[var(--color-text-main)]">78.4%</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-gray-500">Model Confidence</span>
                  <span className="font-bold text-[var(--color-text-main)]">92.1%</span>
                </div>
              </div>
            </div>

            {/* Metrics & Factors */}
            <div className="md:col-span-2 bg-white rounded-xl shadow-sm border border-gray-100 p-6">
              <h3 className="text-lg font-bold text-[var(--color-text-main)] mb-6 border-b border-gray-100 pb-4">Cognitive Metrics & Factors</h3>
              
              <div className="grid grid-cols-2 gap-6 mb-8">
                <div>
                  <div className="text-sm text-gray-500 mb-1">MMSE Score</div>
                  <div className="text-2xl font-bold text-[var(--color-text-main)]">21 <span className="text-sm font-normal text-gray-400">/ 30</span></div>
                  <div className="text-xs font-medium text-[var(--color-danger)] mt-1">Below normal threshold (24)</div>
                </div>
                <div>
                  <div className="text-sm text-gray-500 mb-1">CDR Score</div>
                  <div className="text-2xl font-bold text-[var(--color-text-main)]">1.0</div>
                  <div className="text-xs font-medium text-[var(--color-warning)] mt-1">Mild Dementia Range</div>
                </div>
              </div>

              <div className="mb-6">
                <h4 className="text-sm font-bold text-[var(--color-text-main)] mb-3 flex items-center gap-2">
                  <IconChartBar size={16} className="text-[var(--color-primary)]" />
                  Primary Contributing Factors
                </h4>
                <div className="space-y-3">
                  <div className="flex items-center gap-4">
                    <div className="w-1/3 text-sm text-gray-600">Memory Recall Decline</div>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--color-danger)] w-[85%] rounded-full"></div>
                    </div>
                    <div className="text-xs font-medium text-gray-500 w-8">85%</div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-1/3 text-sm text-gray-600">Executive Function</div>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--color-warning)] w-[60%] rounded-full"></div>
                    </div>
                    <div className="text-xs font-medium text-gray-500 w-8">60%</div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="w-1/3 text-sm text-gray-600">Age Factor</div>
                    <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div className="h-full bg-[var(--color-primary)] w-[40%] rounded-full"></div>
                    </div>
                    <div className="text-xs font-medium text-gray-500 w-8">40%</div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div>
              <h3 className="text-lg font-bold text-[var(--color-text-main)]">Recommended Next Steps</h3>
              <p className="text-sm text-gray-600 mt-1">Based on the High Risk assessment, clinical follow-up is advised.</p>
            </div>
            <div className="flex gap-3 w-full sm:w-auto">
              <button className="flex-1 sm:flex-none px-4 py-2 bg-white border border-[var(--color-primary)] text-[var(--color-primary)] text-sm font-semibold rounded-lg hover:bg-[var(--color-light-teal)] transition-colors">
                Export PDF Report
              </button>
              <button className="flex-1 sm:flex-none px-4 py-2 bg-[var(--color-primary)] text-white text-sm font-semibold rounded-lg hover:bg-[var(--color-secondary)] transition-colors">
                Schedule Specialist
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

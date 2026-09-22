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
    { name: "Total Hospitals", value: "120", icon: IconHospital },
    { name: "Total Patients", value: "8,432", icon: IconPatients },
    { name: "Total Assessments", value: "24,890", icon: IconAssessment },
    { name: "Model Accuracy", value: "96.4%", icon: IconCheck },
  ];

  return (
    <div className="p-6 lg:p-10 max-w-7xl mx-auto animate-fade-in-up">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-[var(--color-text-main)]">System Overview</h1>
        <p className="text-gray-600 mt-1">Platform-wide statistics and management.</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-10">
        {stats.map((stat, i) => (
          <div key={i} className="bg-white rounded-xl shadow-sm border border-gray-100 p-6 flex items-center gap-4">
            <div className="w-12 h-12 rounded-lg bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center shrink-0">
              <stat.icon size={24} />
            </div>
            <div>
              <div className="text-sm font-medium text-gray-500">{stat.name}</div>
              <div className="text-2xl font-bold text-[var(--color-text-main)] mt-1">{stat.value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Risk Distribution Chart (Placeholder UI) */}
        <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-bold text-[var(--color-text-main)] flex items-center gap-2">
              <IconChartBar size={20} className="text-[var(--color-primary)]" />
              Global Risk Distribution
            </h2>
            <select className="bg-gray-50 border border-gray-200 text-sm rounded-lg focus:ring-[var(--color-primary)] focus:border-[var(--color-primary)] px-3 py-1.5">
              <option>Last 30 Days</option>
              <option>Last 3 Months</option>
              <option>This Year</option>
            </select>
          </div>
          
          <div className="h-64 flex items-end justify-between gap-2 px-2 pb-6 pt-4 border-b border-gray-100 relative">
            {/* Y-axis labels */}
            <div className="absolute left-0 top-0 h-full flex flex-col justify-between text-xs text-gray-400 pb-6">
              <span>100%</span>
              <span>50%</span>
              <span>0%</span>
            </div>
            
            <div className="w-full flex items-end justify-around pl-10 h-full relative">
              {/* Grid lines */}
              <div className="absolute left-10 right-0 top-0 border-t border-dashed border-gray-200"></div>
              <div className="absolute left-10 right-0 top-1/2 border-t border-dashed border-gray-200"></div>
              
              {/* Bars */}
              <div className="w-16 md:w-24 bg-gray-100 rounded-t-md h-[25%] relative group">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-800 text-white text-xs py-1 px-2 rounded">25%</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-500 font-medium translate-y-6">Low</div>
              </div>
              <div className="w-16 md:w-24 bg-[var(--color-warning)] rounded-t-md h-[45%] relative group">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-800 text-white text-xs py-1 px-2 rounded">45%</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-500 font-medium translate-y-6">Moderate</div>
              </div>
              <div className="w-16 md:w-24 bg-[var(--color-danger)] rounded-t-md h-[30%] relative group">
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-gray-800 text-white text-xs py-1 px-2 rounded">30%</div>
                <div className="absolute bottom-0 w-full text-center text-xs text-gray-500 font-medium translate-y-6">High</div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
          <h2 className="text-lg font-bold text-[var(--color-text-main)] mb-6 flex items-center gap-2">
            <IconSecurity size={20} className="text-[var(--color-primary)]" />
            Security & Activity Logs
          </h2>
          <div className="space-y-4">
            {[
              { title: "New Hospital Registered", desc: "Memorial Healthcare joined the network.", time: "10 mins ago" },
              { title: "System Update", desc: "AI model v2.1.4 deployed successfully.", time: "2 hours ago" },
              { title: "API Key Rotated", desc: "For Central General Hospital.", time: "5 hours ago" },
              { title: "Audit Log Exported", desc: "By admin_jsmith.", time: "1 day ago" },
              { title: "Failed Login Attempt", desc: "Multiple attempts from IP 192.168.1.5", time: "1 day ago", alert: true },
            ].map((log, i) => (
              <div key={i} className="flex gap-4">
                <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${log.alert ? 'bg-[var(--color-danger)]' : 'bg-[var(--color-primary)]'}`}></div>
                <div>
                  <div className={`text-sm font-medium ${log.alert ? 'text-[var(--color-danger)]' : 'text-[var(--color-text-main)]'}`}>{log.title}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{log.desc}</div>
                  <div className="text-[10px] text-gray-400 mt-1">{log.time}</div>
                </div>
              </div>
            ))}
          </div>
          <button className="w-full mt-6 py-2 text-sm font-medium text-[var(--color-primary)] bg-[var(--color-light-teal)] rounded-lg hover:bg-[var(--color-secondary)] hover:text-white transition-colors">
            View All Logs
          </button>
        </div>
      </div>
    </div>
  );
}

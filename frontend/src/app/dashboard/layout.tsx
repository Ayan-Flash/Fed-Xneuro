"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { 
  IconBrain, 
  IconAdmin, 
  IconHospital, 
  IconReports, 
  IconSettings,
  IconHistory
} from "@/components/Icons";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  const role = pathname?.includes("/admin") ? "admin" : "hospital";

  const adminLinks = [
    { name: "Overview", href: "/dashboard/admin", icon: IconAdmin },
    { name: "Hospitals", href: "#", icon: IconHospital },
    { name: "Reports", href: "#", icon: IconReports },
    { name: "Settings", href: "#", icon: IconSettings },
  ];

  const hospitalLinks = [
    { name: "Dashboard", href: "/dashboard/hospital", icon: IconHospital },
    { name: "New Assessment", href: "/dashboard/hospital/assessment", icon: IconBrain },
    { name: "Patient History", href: "#", icon: IconHistory },
    { name: "Settings", href: "#", icon: IconSettings },
  ];

  const links = role === "admin" ? adminLinks : hospitalLinks;

  const handleLogout = () => {
    router.push("/login");
  };

  return (
    <div className="flex h-screen bg-[var(--color-bg)] overflow-hidden font-sans">
      
      {/* Sidebar (Desktop) */}
      <div className="hidden md:flex flex-col w-64 bg-white border-r border-gray-200">
        <div className="flex items-center h-20 px-6 border-b border-gray-100">
          <Link href="/" className="flex items-center gap-2 group">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white shadow-sm">
              <IconBrain size={18} />
            </div>
            <div>
              <span className="font-bold text-lg tracking-tight text-[var(--color-text-main)]">FedX Nuro</span>
            </div>
          </Link>
        </div>

        <div className="px-5 py-4">
          <div className="flex items-center justify-between mb-3 px-1">
            <div className="text-[11px] font-bold text-gray-400 uppercase tracking-wider">
              {role === "admin" ? "Admin Console" : "Clinical Portal"}
            </div>
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse"></span>
              Live
            </span>
          </div>
          <nav className="space-y-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-all text-sm font-medium relative ${
                    isActive 
                      ? "bg-[var(--color-light-teal)] text-[var(--color-primary)] font-semibold shadow-xs" 
                      : "text-gray-600 hover:bg-gray-50 hover:text-[var(--color-primary)]"
                  }`}
                >
                  {isActive && (
                    <span className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-5 bg-[var(--color-primary)] rounded-r-full"></span>
                  )}
                  <link.icon size={18} />
                  {link.name}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* User Profile Card */}
        <div className="mt-auto px-5 py-4 border-t border-gray-100 bg-gray-50/50">
          <div className="flex items-center gap-3 mb-3 px-1">
            <div className="w-9 h-9 rounded-full bg-[var(--color-light-teal)] text-[var(--color-primary)] font-bold text-xs flex items-center justify-center border border-[var(--color-primary)]/20 shadow-xs">
              {role === "admin" ? "AD" : "SJ"}
            </div>
            <div className="min-w-0 flex-1">
              <p className="text-xs font-bold text-gray-800 truncate">
                {role === "admin" ? "Admin Operator" : "Dr. Sarah Jenkins"}
              </p>
              <p className="text-[11px] text-gray-500 truncate">
                {role === "admin" ? "Core Node #01" : "St. Jude Neuroscience"}
              </p>
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="flex items-center justify-center gap-2 w-full px-3 py-2 text-xs font-semibold text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-xl transition-all border border-gray-200/80 hover:border-red-200"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            Sign Out
          </button>
        </div>
      </div>

      {/* Mobile Header & Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Mobile Header */}
        <div className="md:hidden flex items-center justify-between h-16 px-4 bg-white border-b border-gray-200 z-30">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white">
              <IconBrain size={18} />
            </div>
            <span className="font-bold text-lg tracking-tight text-[var(--color-text-main)]">FedX Nuro</span>
          </Link>
          <button 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 rounded-lg text-gray-600 hover:bg-gray-100 focus:outline-none"
            aria-label="Toggle Navigation Menu"
          >
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              {isMobileMenuOpen ? (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Backdrop & Dropdown */}
        {isMobileMenuOpen && (
          <>
            <div 
              className="md:hidden fixed inset-0 bg-black/30 backdrop-blur-xs z-30 top-16"
              onClick={() => setIsMobileMenuOpen(false)}
            />
            <div className="md:hidden bg-white border-b border-gray-200 absolute w-full z-40 top-16 shadow-xl animate-fade-in-up">
              <div className="px-4 py-3 space-y-1">
                <div className="flex items-center justify-between text-[11px] font-bold text-gray-400 uppercase tracking-widest px-3 py-1">
                  <span>{role === "admin" ? "Admin Console" : "Clinical Portal"}</span>
                  <span className="text-emerald-600">● Live Node</span>
                </div>
                {links.map((link) => {
                  const isActive = pathname === link.href;
                  return (
                    <Link
                      key={link.name}
                      href={link.href}
                      onClick={() => setIsMobileMenuOpen(false)}
                      className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl transition-colors text-sm font-medium ${
                        isActive 
                          ? "bg-[var(--color-light-teal)] text-[var(--color-primary)] font-semibold" 
                          : "text-gray-600 hover:bg-gray-50"
                      }`}
                    >
                      <link.icon size={18} />
                      {link.name}
                    </Link>
                  );
                })}
                <button
                  onClick={handleLogout}
                  className="flex items-center gap-3 w-full px-3.5 py-2.5 mt-2 border-t border-gray-100 text-sm font-semibold text-red-600 hover:bg-red-50 rounded-lg"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                    <polyline points="16 17 21 12 16 7" />
                    <line x1="21" y1="12" x2="9" y2="12" />
                  </svg>
                  Sign Out
                </button>
              </div>
            </div>
          </>
        )}

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto bg-[var(--color-bg)]">
          {children}
        </main>
      </div>
    </div>
  );
}

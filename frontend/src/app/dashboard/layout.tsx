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

        <div className="px-6 py-4">
          <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-4">
            {role === "admin" ? "Admin Portal" : "Clinical Portal"}
          </div>
          <nav className="space-y-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.name}
                  href={link.href}
                  className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors text-sm font-medium ${
                    isActive 
                      ? "bg-[var(--color-light-teal)] text-[var(--color-primary)]" 
                      : "text-gray-600 hover:bg-gray-50 hover:text-[var(--color-primary)]"
                  }`}
                >
                  <link.icon size={18} />
                  {link.name}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="mt-auto px-6 py-4 border-t border-gray-100">
          <button
            onClick={handleLogout}
            className="flex items-center gap-3 w-full px-3 py-2.5 text-sm font-medium text-gray-600 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
              <polyline points="16 17 21 12 16 7" />
              <line x1="21" y1="12" x2="9" y2="12" />
            </svg>
            Sign Out
          </button>
        </div>
      </div>

      {/* Mobile Header & Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Mobile Header */}
        <div className="md:hidden flex items-center justify-between h-16 px-4 bg-white border-b border-gray-200">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white">
              <IconBrain size={18} />
            </div>
            <span className="font-bold text-lg tracking-tight text-[var(--color-text-main)]">FedX Nuro</span>
          </Link>
          <button 
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="text-gray-500 focus:outline-none"
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

        {/* Mobile Menu Dropdown */}
        {isMobileMenuOpen && (
          <div className="md:hidden bg-white border-b border-gray-200 absolute w-full z-40 top-16 shadow-lg">
            <div className="px-4 py-2 space-y-1">
               <div className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2 px-3 pt-2">
                {role === "admin" ? "Admin Portal" : "Clinical Portal"}
              </div>
              {links.map((link) => {
                const isActive = pathname === link.href;
                return (
                  <Link
                    key={link.name}
                    href={link.href}
                    onClick={() => setIsMobileMenuOpen(false)}
                    className={`flex items-center gap-3 px-3 py-3 rounded-lg transition-colors text-base font-medium ${
                      isActive 
                        ? "bg-[var(--color-light-teal)] text-[var(--color-primary)]" 
                        : "text-gray-600"
                    }`}
                  >
                    <link.icon size={20} />
                    {link.name}
                  </Link>
                );
              })}
              <button
                onClick={handleLogout}
                className="flex items-center gap-3 w-full px-3 py-3 mt-4 border-t border-gray-100 text-base font-medium text-red-600"
              >
                Sign Out
              </button>
            </div>
          </div>
        )}

        {/* Main Content Area */}
        <main className="flex-1 overflow-auto bg-[var(--color-bg)]">
          {children}
        </main>
      </div>
    </div>
  );
}

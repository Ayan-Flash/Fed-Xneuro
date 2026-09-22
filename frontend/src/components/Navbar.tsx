"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { IconBrain } from "./Icons";

export const Navbar = () => {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // If we are in the dashboard, we don't show the public navbar
  if (pathname?.startsWith("/dashboard")) {
    return null;
  }

  return (
    <nav className="fixed w-full z-50 top-0 left-0 bg-white/90 backdrop-blur-md border-b border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          
          {/* Logo / Brand */}
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" className="flex items-center gap-2 group">
              <div className="w-10 h-10 rounded-lg bg-[var(--color-primary)] flex items-center justify-center text-white shadow-sm group-hover:bg-[var(--color-secondary)] transition-colors">
                <IconBrain size={22} />
              </div>
              <div>
                <span className="font-bold text-xl tracking-tight text-[var(--color-text-main)]">FedX Nuro</span>
                <span className="block text-[10px] font-semibold text-[var(--color-primary)] uppercase tracking-widest leading-none">Cognitive Health</span>
              </div>
            </Link>
          </div>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center space-x-8">
            <Link href="/" className="text-sm font-medium text-[var(--color-text-main)] hover:text-[var(--color-primary)] transition-colors">Home</Link>
            <Link href="#how-it-works" className="text-sm font-medium text-gray-500 hover:text-[var(--color-primary)] transition-colors">How It Works</Link>
            <Link href="#features" className="text-sm font-medium text-gray-500 hover:text-[var(--color-primary)] transition-colors">Features</Link>
            <Link href="#about" className="text-sm font-medium text-gray-500 hover:text-[var(--color-primary)] transition-colors">About</Link>
            <Link href="#contact" className="text-sm font-medium text-gray-500 hover:text-[var(--color-primary)] transition-colors">Contact</Link>
          </div>

          {/* Login / CTA */}
          <div className="hidden md:flex items-center space-x-4">
            <Link 
              href="/login" 
              className="px-5 py-2.5 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-sm transition-all duration-200"
            >
              Login
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="text-gray-500 hover:text-[var(--color-primary)] focus:outline-none"
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>

        </div>
      </div>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-white border-b border-gray-100 shadow-sm animate-fade-in-up">
          <div className="px-4 pt-2 pb-6 space-y-1">
            <Link href="/" onClick={() => setMobileMenuOpen(false)} className="block px-3 py-2 text-base font-medium text-[var(--color-text-main)] hover:bg-[var(--color-light-teal)] hover:text-[var(--color-primary)] rounded-md">Home</Link>
            <Link href="#how-it-works" onClick={() => setMobileMenuOpen(false)} className="block px-3 py-2 text-base font-medium text-gray-600 hover:bg-[var(--color-light-teal)] hover:text-[var(--color-primary)] rounded-md">How It Works</Link>
            <Link href="#features" onClick={() => setMobileMenuOpen(false)} className="block px-3 py-2 text-base font-medium text-gray-600 hover:bg-[var(--color-light-teal)] hover:text-[var(--color-primary)] rounded-md">Features</Link>
            <Link href="/login" onClick={() => setMobileMenuOpen(false)} className="block px-3 py-2 text-base font-semibold text-[var(--color-primary)] bg-[var(--color-light-teal)] rounded-md mt-4">Login</Link>
          </div>
        </div>
      )}
    </nav>
  );
};

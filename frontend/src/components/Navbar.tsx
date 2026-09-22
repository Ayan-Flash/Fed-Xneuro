"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { IconBrain } from "./Icons";

interface NavbarProps {
  activeTab?: string;
  onTabChange?: (tab: any) => void;
  role?: string;
}

export const Navbar: React.FC<NavbarProps> = () => {
  const pathname = usePathname();
  const router = useRouter();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("home");

  // Track active section on scroll
  useEffect(() => {
    if (pathname !== "/") return;

    const sections = ["home", "how-it-works", "features", "about", "contact"];
    const handleScroll = () => {
      const scrollPosition = window.scrollY + 120;
      for (const sectionId of [...sections].reverse()) {
        const el = document.getElementById(sectionId);
        if (el && el.offsetTop <= scrollPosition) {
          setActiveSection(sectionId);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, [pathname]);

  // Handle smooth scroll to section
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, targetId: string) => {
    e.preventDefault();
    setMobileMenuOpen(false);

    if (pathname === "/") {
      const el = document.getElementById(targetId);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
        window.history.pushState(null, "", `#${targetId}`);
        setActiveSection(targetId);
      } else if (targetId === "home") {
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    } else {
      router.push(`/#${targetId}`);
    }
  };

  // If we are in the dashboard, we don't show the public navbar
  if (pathname?.startsWith("/dashboard")) {
    return null;
  }

  const navLinks = [
    { label: "Home", id: "home" },
    { label: "How It Works", id: "how-it-works" },
    { label: "Features", id: "features" },
    { label: "About", id: "about" },
    { label: "Contact", id: "contact" },
  ];

  return (
    <nav className="fixed w-full z-50 top-0 left-0 bg-white/90 backdrop-blur-md border-b border-gray-100 shadow-sm transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-20">
          
          {/* Logo / Brand */}
          <div className="flex-shrink-0 flex items-center">
            <Link href="/" onClick={(e) => handleNavClick(e, "home")} className="flex items-center gap-2.5 group cursor-pointer">
              <div className="w-10 h-10 rounded-xl bg-[var(--color-primary)] flex items-center justify-center text-white shadow-sm group-hover:bg-[var(--color-secondary)] group-hover:scale-105 transition-all">
                <IconBrain size={22} />
              </div>
              <div>
                <span className="font-bold text-xl tracking-tight text-[var(--color-text-main)]">FedX Nuro</span>
                <span className="block text-[10px] font-semibold text-[var(--color-primary)] uppercase tracking-widest leading-none">Cognitive Health</span>
              </div>
            </Link>
          </div>

          {/* Desktop Links */}
          <div className="hidden md:flex items-center space-x-1 lg:space-x-2">
            {navLinks.map((link) => {
              const isActive = activeSection === link.id && pathname === "/";
              return (
                <a
                  key={link.id}
                  href={`#${link.id}`}
                  onClick={(e) => handleNavClick(e, link.id)}
                  className={`px-3.5 py-2 rounded-lg text-sm font-medium transition-all cursor-pointer ${
                    isActive
                      ? "text-[var(--color-primary)] bg-[var(--color-light-teal)]/70 font-semibold"
                      : "text-gray-600 hover:text-[var(--color-primary)] hover:bg-gray-50"
                  }`}
                >
                  {link.label}
                </a>
              );
            })}
          </div>

          {/* Login / CTA */}
          <div className="hidden md:flex items-center space-x-4">
            <Link 
              href="/login" 
              className="px-5 py-2.5 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-sm hover:shadow hover:-translate-y-0.5 transition-all duration-200"
            >
              Login
            </Link>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button 
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-gray-500 hover:text-[var(--color-primary)] hover:bg-gray-50 focus:outline-none transition-colors"
              aria-label="Toggle Navigation Menu"
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
        <div className="md:hidden bg-white border-b border-gray-100 shadow-lg animate-fade-in-up">
          <div className="px-4 pt-3 pb-6 space-y-1.5">
            {navLinks.map((link) => {
              const isActive = activeSection === link.id && pathname === "/";
              return (
                <a
                  key={link.id}
                  href={`#${link.id}`}
                  onClick={(e) => handleNavClick(e, link.id)}
                  className={`block px-3.5 py-2.5 text-base font-medium rounded-lg transition-colors ${
                    isActive
                      ? "text-[var(--color-primary)] bg-[var(--color-light-teal)] font-semibold"
                      : "text-gray-700 hover:bg-gray-50 hover:text-[var(--color-primary)]"
                  }`}
                >
                  {link.label}
                </a>
              );
            })}
            <div className="pt-2">
              <Link 
                href="/login" 
                onClick={() => setMobileMenuOpen(false)} 
                className="block w-full text-center px-4 py-2.5 text-base font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-sm"
              >
                Login
              </Link>
            </div>
          </div>
        </div>
      )}
    </nav>
  );
};

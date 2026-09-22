"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import { IconAdmin, IconHospital, IconBrain, IconError } from "@/components/Icons";

type Role = "admin" | "hospital";

export default function LoginPage() {
  const router = useRouter();
  const [role, setRole] = useState<Role>("hospital");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    
    if (!email || !password) {
      setError("Please fill in all required fields.");
      return;
    }
    
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      // Route based on role
      if (role === "admin") {
        router.push("/dashboard/admin");
      } else {
        router.push("/dashboard/hospital");
      }
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)] flex flex-col font-sans">
      <Navbar />
      
      <div className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 mt-16 relative">
        {/* Subtle ambient lighting */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-96 h-96 bg-[var(--color-primary)]/10 rounded-full blur-3xl -z-10 pointer-events-none"></div>

        <div className="max-w-md w-full animate-fade-in-up">
          
          <div className="text-center mb-6">
            <div className="mx-auto w-12 h-12 bg-gradient-to-tr from-[var(--color-primary)] to-[var(--color-secondary)] text-white rounded-2xl flex items-center justify-center shadow-md shadow-[var(--color-primary)]/20 mb-3 hover:scale-105 transition-transform">
              <IconBrain size={28} />
            </div>
            <h2 className="text-2xl sm:text-3xl font-extrabold text-[var(--color-text-main)] tracking-tight">
              Sign in to your account
            </h2>
            <p className="mt-1.5 text-xs sm:text-sm text-gray-500">
              Access the clinical cognitive health platform
            </p>
          </div>

          <div className="bg-white py-8 px-6 sm:px-10 shadow-lg shadow-[var(--color-primary)]/5 rounded-2xl border border-gray-100">
            
            {/* Role Selection */}
            <div className="flex p-1 bg-gray-50/80 rounded-xl mb-6 border border-gray-100">
              <button
                type="button"
                onClick={() => {
                  setRole("hospital");
                  setEmail("doctor@memorial.org");
                  setPassword("demo123");
                }}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs sm:text-sm font-semibold rounded-lg transition-all ${
                  role === "hospital" 
                    ? "bg-white text-[var(--color-primary)] shadow-xs border border-gray-200/80" 
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconHospital size={16} />
                Hospital Clinician
              </button>
              <button
                type="button"
                onClick={() => {
                  setRole("admin");
                  setEmail("admin@platform.com");
                  setPassword("admin123");
                }}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs sm:text-sm font-semibold rounded-lg transition-all ${
                  role === "admin" 
                    ? "bg-white text-[var(--color-primary)] shadow-xs border border-gray-200/80" 
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconAdmin size={16} />
                System Admin
              </button>
            </div>

            {error && (
              <div className="mb-5 bg-red-50/90 border border-red-200 text-red-700 px-3.5 py-2.5 rounded-xl flex items-center gap-2.5 text-xs sm:text-sm">
                <IconError size={16} />
                {error}
              </div>
            )}

            <form className="space-y-5" onSubmit={handleSubmit}>
              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5" htmlFor="email">
                  {role === "admin" ? "Admin Email" : "Hospital ID / Email"}
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="appearance-none block w-full px-3.5 py-2.5 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] transition-all text-sm"
                  placeholder={role === "admin" ? "admin@platform.com" : "hospital@domain.com"}
                />
              </div>

              <div>
                <label className="block text-xs font-semibold uppercase tracking-wider text-gray-700 mb-1.5" htmlFor="password">
                  Password
                </label>
                <div className="relative">
                  <input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    autoComplete="current-password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="appearance-none block w-full px-3.5 py-2.5 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] transition-all text-sm pr-12"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-xs font-medium text-gray-400 hover:text-[var(--color-primary)] focus:outline-none transition-colors"
                  >
                    {showPassword ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs sm:text-sm">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    defaultChecked
                    className="h-4 w-4 text-[var(--color-primary)] focus:ring-[var(--color-primary)] border-gray-300 rounded cursor-pointer"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-gray-600 cursor-pointer">
                    Remember credentials
                  </label>
                </div>

                <div>
                  <a href="#" className="font-medium text-[var(--color-primary)] hover:text-[var(--color-secondary)] transition-colors">
                    Forgot password?
                  </a>
                </div>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full flex justify-center items-center py-3 px-4 border border-transparent rounded-xl shadow-md shadow-[var(--color-primary)]/20 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)] transition-all duration-200 disabled:opacity-70 disabled:cursor-not-allowed hover:-translate-y-0.5"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Authenticating...
                    </div>
                  ) : (
                    `Sign in as ${role === "admin" ? "Administrator" : "Hospital Clinician"}`
                  )}
                </button>
              </div>
            </form>

            <div className="mt-6 pt-5 border-t border-gray-100 flex items-center justify-between text-xs text-gray-400">
              <span>Enterprise Grade Security</span>
              <span className="font-medium text-gray-500">Fed-XNeuro v2.4</span>
            </div>
          </div>
          
        </div>
      </div>
    </div>
  );
}

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
      
      <div className="flex-1 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 mt-16">
        <div className="max-w-md w-full animate-fade-in-up">
          
          <div className="text-center mb-8">
            <div className="mx-auto w-12 h-12 bg-[var(--color-primary)] text-white rounded-xl flex items-center justify-center shadow-sm mb-4">
              <IconBrain size={28} />
            </div>
            <h2 className="text-3xl font-extrabold text-[var(--color-text-main)]">
              Sign in to your account
            </h2>
            <p className="mt-2 text-sm text-gray-600">
              Access the clinical cognitive health platform
            </p>
          </div>

          <div className="bg-white py-8 px-4 shadow-md rounded-2xl sm:px-10 border border-gray-100">
            
            {/* Role Selection */}
            <div className="flex p-1 bg-gray-50 rounded-xl mb-8 border border-gray-100">
              <button
                type="button"
                onClick={() => setRole("hospital")}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-lg transition-all ${
                  role === "hospital" 
                    ? "bg-white text-[var(--color-primary)] shadow-sm border border-gray-200" 
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconHospital size={18} />
                Hospital
              </button>
              <button
                type="button"
                onClick={() => setRole("admin")}
                className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-sm font-medium rounded-lg transition-all ${
                  role === "admin" 
                    ? "bg-white text-[var(--color-primary)] shadow-sm border border-gray-200" 
                    : "text-gray-500 hover:text-gray-700"
                }`}
              >
                <IconAdmin size={18} />
                Admin
              </button>
            </div>

            {error && (
              <div className="mb-6 bg-red-50 border border-red-200 text-red-600 px-4 py-3 rounded-lg flex items-center gap-3 text-sm">
                <IconError size={18} />
                {error}
              </div>
            )}

            <form className="space-y-6" onSubmit={handleSubmit}>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="email">
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
                  className="appearance-none block w-full px-4 py-3 border border-gray-300 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-colors sm:text-sm"
                  placeholder={role === "admin" ? "admin@platform.com" : "hospital@domain.com"}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1" htmlFor="password">
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
                    className="appearance-none block w-full px-4 py-3 border border-gray-300 rounded-lg placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] focus:border-transparent transition-colors sm:text-sm pr-12"
                    placeholder="••••••••"
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute inset-y-0 right-0 pr-3 flex items-center text-sm font-medium text-[var(--color-primary)] hover:text-[var(--color-secondary)] focus:outline-none"
                  >
                    {showPassword ? "Hide" : "Show"}
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <input
                    id="remember-me"
                    name="remember-me"
                    type="checkbox"
                    className="h-4 w-4 text-[var(--color-primary)] focus:ring-[var(--color-primary)] border-gray-300 rounded"
                  />
                  <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
                    Remember me
                  </label>
                </div>

                <div className="text-sm">
                  <a href="#" className="font-medium text-[var(--color-primary)] hover:text-[var(--color-secondary)]">
                    Forgot your password?
                  </a>
                </div>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full flex justify-center py-3 px-4 border border-transparent rounded-lg shadow-sm text-sm font-medium text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[var(--color-primary)] transition-colors disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isLoading ? (
                    <div className="flex items-center gap-2">
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Authenticating...
                    </div>
                  ) : (
                    "Sign in"
                  )}
                </button>
              </div>
            </form>
          </div>
          
        </div>
      </div>
    </div>
  );
}

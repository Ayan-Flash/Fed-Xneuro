"use client";

import React, { useState } from 'react';
import { useRouter } from 'next/navigation';
import { ShieldAlert, Building2, UserCircle, Key, Eye, EyeOff, BrainCircuit, Activity } from 'lucide-react';

export default function LoginPage() {
  const router = useRouter();
  const [role, setRole] = useState<'admin' | 'hospital'>('hospital');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in all fields.');
      return;
    }
    
    setError('');
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      if (email.includes('error')) {
        setError('Invalid credentials. Please try again.');
        return;
      }
      
      // Redirect based on role
      if (role === 'admin') {
        router.push('/dashboard/admin');
      } else {
        router.push('/dashboard/hospital');
      }
    }, 1200);
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 bg-[var(--bg-primary)]">
      <div className="w-full max-w-md bg-[var(--bg-card)] rounded-[var(--radius-lg)] border border-[var(--border-color)] p-8 shadow-xl">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="w-16 h-16 rounded-full bg-[var(--primary-glow)] flex items-center justify-center text-[var(--primary)]">
              <BrainCircuit size={32} />
            </div>
          </div>
          <h1 className="text-2xl font-bold text-[var(--text-primary)]">Fed-XNeuro Platform</h1>
          <p className="text-[var(--text-secondary)] text-sm mt-2 flex items-center justify-center gap-1">
            <Activity size={16} /> Clinical AI Diagnostics
          </p>
        </div>

        {/* Role Selection */}
        <div className="grid grid-cols-2 gap-3 mb-8">
          <button
            type="button"
            onClick={() => { setRole('hospital'); setError(''); }}
            className={`flex flex-col items-center justify-center p-4 rounded-[var(--radius-md)] border transition-all ${
              role === 'hospital' 
                ? 'bg-[var(--primary-glow)] border-[var(--primary)] text-[var(--primary)] shadow-md' 
                : 'bg-white border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--primary)]'
            }`}
          >
            <Building2 size={24} className="mb-2" />
            <span className="font-semibold text-sm">Hospital</span>
          </button>
          
          <button
            type="button"
            onClick={() => { setRole('admin'); setError(''); }}
            className={`flex flex-col items-center justify-center p-4 rounded-[var(--radius-md)] border transition-all ${
              role === 'admin' 
                ? 'bg-[var(--primary-glow)] border-[var(--primary)] text-[var(--primary)] shadow-md' 
                : 'bg-white border-[var(--border-color)] text-[var(--text-secondary)] hover:border-[var(--primary)]'
            }`}
          >
            <ShieldAlert size={24} className="mb-2" />
            <span className="font-semibold text-sm">System Admin</span>
          </button>
        </div>

        {/* Login Form */}
        <form onSubmit={handleLogin} className="space-y-4">
          {error && (
            <div className="bg-[var(--crimson-glow)] text-[var(--crimson)] p-3 rounded-[var(--radius-sm)] text-sm font-medium border border-[var(--crimson)] flex items-center gap-2">
              <ShieldAlert size={16} />
              {error}
            </div>
          )}

          <div className="space-y-1">
            <label className="text-sm font-semibold text-[var(--text-secondary)]">Email / Username</label>
            <div className="relative">
              <UserCircle className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" size={18} />
              <input 
                type="text" 
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-white border border-[var(--border-color)] rounded-[var(--radius-sm)] focus:outline-none focus:border-[var(--primary)] focus:ring-2 focus:ring-[var(--primary-glow)] text-[var(--text-primary)] transition-all"
                placeholder={role === 'admin' ? "admin@system.local" : "clinician@hospital.org"}
              />
            </div>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between items-center">
              <label className="text-sm font-semibold text-[var(--text-secondary)]">Password</label>
              <button type="button" className="text-xs text-[var(--primary)] hover:underline">Forgot password?</button>
            </div>
            <div className="relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" size={18} />
              <input 
                type={showPassword ? "text" : "password"} 
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-10 py-2 bg-white border border-[var(--border-color)] rounded-[var(--radius-sm)] focus:outline-none focus:border-[var(--primary)] focus:ring-2 focus:ring-[var(--primary-glow)] text-[var(--text-primary)] transition-all"
                placeholder="••••••••"
              />
              <button 
                type="button" 
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>

          <div className="flex items-center pt-2">
            <input type="checkbox" id="remember" className="mr-2 rounded border-[var(--border-color)] text-[var(--primary)] focus:ring-[var(--primary)]" />
            <label htmlFor="remember" className="text-sm text-[var(--text-secondary)]">Remember me on this device</label>
          </div>

          <button 
            type="submit" 
            disabled={isLoading}
            className="w-full mt-6 py-3 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius-sm)] hover:bg-[var(--primary-hover)] transition-all flex justify-center items-center gap-2 shadow-lg disabled:opacity-70"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Authenticating...
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="mt-8 pt-6 border-t border-[var(--border-color)] text-center text-xs text-[var(--text-muted)]">
          &copy; {new Date().getFullYear()} Fed-XNeuro Platform. HIPAA Compliant.<br/>
          Unauthorized access is strictly prohibited.
        </div>
      </div>
    </div>
  );
}

import React from 'react';
import Link from 'next/link';
import { BrainCircuit, ShieldCheck, Activity, Network, Lock, FileText, ArrowRight } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--bg-primary)] flex flex-col">
      {/* Navigation */}
      <nav className="flex justify-between items-center p-6 border-b border-[var(--border-color)] bg-white/80 backdrop-blur-md sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-[var(--radius-md)] bg-gradient-to-br from-[var(--primary)] to-[var(--cyan)] flex items-center justify-center shadow-lg shadow-[var(--primary-glow)] text-white">
            <BrainCircuit size={24} />
          </div>
          <div>
            <h1 className="text-lg font-bold text-[var(--text-primary)] leading-none tracking-wide bg-clip-text text-transparent bg-gradient-to-r from-[var(--primary)] to-[var(--cyan)]">
              Fed-XNeuro
            </h1>
            <p className="text-xs text-[var(--text-secondary)] mt-1">Clinical AI Diagnostics</p>
          </div>
        </div>
        <div>
          <Link 
            href="/login" 
            className="px-6 py-2.5 bg-[var(--primary)] text-white font-semibold rounded-[var(--radius-sm)] shadow-[0_0_15px_var(--primary-glow)] hover:bg-[var(--primary-hover)] hover:shadow-[0_0_20px_var(--primary-glow)] transition-all flex items-center gap-2 text-sm"
          >
            Access Portal <ArrowRight size={16} />
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-grow flex flex-col items-center justify-center px-4 py-20 text-center">
        <div className="max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--emerald-glow)] border border-[var(--border-color)] text-[var(--primary)] text-sm font-semibold mb-8">
            <Activity size={16} className="animate-pulse" />
            <span>Next-Generation Diagnostic Screening</span>
          </div>
          
          <h2 className="text-5xl md:text-6xl font-extrabold text-[var(--text-primary)] mb-6 leading-tight">
            Collaborative AI for <br/> 
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--primary)] to-[var(--cyan)]">Neuro-Diagnostics</span>
          </h2>
          
          <p className="text-lg md:text-xl text-[var(--text-secondary)] mb-10 max-w-2xl mx-auto leading-relaxed">
            Empowering clinical institutions with privacy-preserving, federated machine learning to detect and analyze neurodegenerative patterns with unprecedented accuracy.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link 
              href="/login" 
              className="w-full sm:w-auto px-8 py-3.5 bg-[var(--primary)] text-white font-semibold text-lg rounded-[var(--radius-md)] shadow-[0_4px_20px_var(--primary-glow)] hover:bg-[var(--primary-hover)] hover:-translate-y-0.5 transition-all flex items-center justify-center gap-2"
            >
              Clinical Sign In
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="max-w-6xl mx-auto w-full grid grid-cols-1 md:grid-cols-3 gap-8 mt-24 text-left">
          
          <div className="bg-[var(--bg-card)] p-8 rounded-[var(--radius-lg)] border border-[var(--border-color)] shadow-sm hover:border-[var(--primary)] hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-[var(--radius-md)] bg-[var(--cyan-glow)] flex items-center justify-center text-[var(--cyan)] mb-6">
              <Network size={28} />
            </div>
            <h3 className="text-xl font-bold text-[var(--text-primary)] mb-3">Federated Learning</h3>
            <p className="text-[var(--text-secondary)] leading-relaxed text-sm">
              Train robust diagnostic models collaboratively across multiple clinical sites without ever pooling sensitive patient data.
            </p>
          </div>

          <div className="bg-[var(--bg-card)] p-8 rounded-[var(--radius-lg)] border border-[var(--border-color)] shadow-sm hover:border-[var(--primary)] hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-[var(--radius-md)] bg-[var(--primary-glow)] flex items-center justify-center text-[var(--primary)] mb-6">
              <Lock size={28} />
            </div>
            <h3 className="text-xl font-bold text-[var(--text-primary)] mb-3">HIPAA Compliant</h3>
            <p className="text-[var(--text-secondary)] leading-relaxed text-sm">
              Strict adherence to healthcare data privacy standards. All telemetry and weights are encrypted end-to-end.
            </p>
          </div>

          <div className="bg-[var(--bg-card)] p-8 rounded-[var(--radius-lg)] border border-[var(--border-color)] shadow-sm hover:border-[var(--primary)] hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-[var(--radius-md)] bg-[var(--emerald-glow)] flex items-center justify-center text-[var(--emerald)] mb-6">
              <FileText size={28} />
            </div>
            <h3 className="text-xl font-bold text-[var(--text-primary)] mb-3">Explainable AI (XAI)</h3>
            <p className="text-[var(--text-secondary)] leading-relaxed text-sm">
              Transparent reporting with SHAP feature importance analysis, ensuring clinicians understand the "why" behind every screening result.
            </p>
          </div>

        </div>
      </main>

      {/* Footer */}
      <footer className="py-8 border-t border-[var(--border-color)] text-center text-[var(--text-muted)] text-sm">
        <div className="flex items-center justify-center gap-2 mb-2 text-[var(--text-secondary)] font-medium">
          <ShieldCheck size={18} className="text-[var(--primary)]" />
          <span>Clinical Grade Infrastructure</span>
        </div>
        <p>&copy; {new Date().getFullYear()} Fed-XNeuro Platform. Designed for Clinical Research.</p>
        <p className="mt-1 text-xs opacity-75">Not intended for standalone diagnostic use.</p>
      </footer>
    </div>
  );
}

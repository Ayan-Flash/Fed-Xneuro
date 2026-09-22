"use client";

import React from "react";
import Link from "next/link";
import { Navbar } from "@/components/Navbar";
import { 
  IconBrain, 
  IconHospital, 
  IconAssessment, 
  IconPrediction, 
  IconReports, 
  IconSecurity, 
  IconHistory, 
  IconChartBar,
  IconCheck,
  IconAI
} from "@/components/Icons";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--color-bg)] font-sans">
      <Navbar />
      
      {/* Spacer for fixed navbar */}
      <div className="h-20"></div>

      {/* Hero Section */}
      <section className="relative pt-20 pb-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-light-teal)] to-[var(--color-bg)] -z-10" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center animate-fade-in-up">
          <div className="inline-flex items-center justify-center p-3 mb-8 bg-white rounded-full shadow-sm border border-[var(--color-primary)]/10 text-[var(--color-primary)]">
            <IconBrain size={32} />
          </div>
          <h1 className="text-4xl md:text-6xl font-extrabold text-[var(--color-text-main)] tracking-tight mb-6 leading-tight">
            AI-Powered Cognitive <br className="hidden md:block" /> Health Assessment
          </h1>
          <p className="max-w-3xl mx-auto text-lg md:text-xl text-gray-600 mb-10 leading-relaxed">
            Advanced AI-assisted cognitive assessment and risk prediction designed to support healthcare professionals in early identification and informed decision-making.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link 
              href="/login" 
              className="w-full sm:w-auto px-8 py-3.5 text-base font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-md transition-all duration-200"
            >
              Get Started
            </Link>
            <Link 
              href="#how-it-works" 
              className="w-full sm:w-auto px-8 py-3.5 text-base font-semibold text-[var(--color-primary)] bg-white border border-[var(--color-primary)]/20 hover:bg-[var(--color-light-teal)] rounded-lg shadow-sm transition-all duration-200"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Trust / Statistics Section */}
      <section className="py-16 bg-white border-y border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div className="flex flex-col items-center justify-center">
              <div className="text-4xl font-bold text-[var(--color-primary)] mb-2">10k+</div>
              <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">Assessments Completed</div>
            </div>
            <div className="flex flex-col items-center justify-center">
              <div className="text-4xl font-bold text-[var(--color-primary)] mb-2">500+</div>
              <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">Healthcare Professionals</div>
            </div>
            <div className="flex flex-col items-center justify-center">
              <div className="text-4xl font-bold text-[var(--color-primary)] mb-2">120</div>
              <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">Hospitals Connected</div>
            </div>
            <div className="flex flex-col items-center justify-center">
              <div className="text-4xl font-bold text-[var(--color-primary)] mb-2">95%</div>
              <div className="text-sm font-medium text-gray-500 uppercase tracking-wider">AI-Assisted Processing</div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-24 bg-[var(--color-bg)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-4">How It Works</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">A streamlined clinical workflow designed for efficiency and accuracy.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative">
            {/* Connecting Line */}
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-0.5 bg-gray-200 -z-10 -translate-y-1/2"></div>
            
            {[
              { title: "Patient Information", desc: "Securely input demographic and clinical history.", icon: IconHospital },
              { title: "Cognitive Assessment", desc: "Conduct standardized cognitive evaluation tests.", icon: IconAssessment },
              { title: "AI Analysis", desc: "Advanced models process multidimensional data.", icon: IconAI },
              { title: "Risk Assessment", desc: "Review comprehensive reports and risk predictions.", icon: IconReports }
            ].map((step, i) => (
              <div key={i} className="flex flex-col items-center text-center bg-white p-6 rounded-xl shadow-sm border border-gray-100 relative">
                <div className="w-12 h-12 rounded-full bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-4 border-4 border-white shadow-sm absolute -top-6">
                  {i + 1}
                </div>
                <div className="mt-6 mb-4 text-[var(--color-secondary)]">
                  <step.icon size={32} />
                </div>
                <h3 className="text-lg font-bold text-[var(--color-text-main)] mb-2">{step.title}</h3>
                <p className="text-sm text-gray-600">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section id="features" className="py-24 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-4">Key Features</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">Comprehensive tools for modern clinical environments.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { title: "AI Risk Prediction", desc: "Assist clinical decision-making with transparent risk modeling.", icon: IconPrediction },
              { title: "Historical Tracking", desc: "Monitor cognitive decline or stability over extended periods.", icon: IconHistory },
              { title: "Clinical Reports", desc: "Generate professional, ready-to-share medical reports.", icon: IconReports },
              { title: "Data Visualization", desc: "Interactive dashboards to analyze patient populations.", icon: IconChartBar },
              { title: "Patient Management", desc: "Organized securely with role-based access controls.", icon: IconHospital },
              { title: "Secure Access", desc: "Enterprise-grade security protecting sensitive health data.", icon: IconSecurity },
            ].map((feature, i) => (
              <div key={i} className="p-6 rounded-xl border border-gray-100 hover:border-[var(--color-primary)]/30 hover:shadow-md transition-all group bg-[var(--color-bg)]">
                <div className="w-12 h-12 rounded-lg bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-4 group-hover:bg-[var(--color-primary)] group-hover:text-white transition-colors">
                  <feature.icon size={24} />
                </div>
                <h3 className="text-xl font-bold text-[var(--color-text-main)] mb-2">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Security & Privacy */}
      <section className="py-24 bg-[var(--color-primary)] text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-16 items-center">
            <div>
              <h2 className="text-3xl font-bold mb-6">Uncompromising Security & Privacy</h2>
              <p className="text-[var(--color-light-teal)] mb-8 text-lg leading-relaxed">
                We prioritize the security of sensitive healthcare data. Our platform is built with strict access controls and robust audit trails to support your institutional compliance requirements.
              </p>
              <ul className="space-y-4">
                {[
                  "Secure Role-Based Authentication",
                  "Protected Patient Data Encryption",
                  "Comprehensive Audit & Activity Tracking",
                  "Isolated Hospital Environments"
                ].map((item, i) => (
                  <li key={i} className="flex items-center gap-3">
                    <div className="flex-shrink-0 w-6 h-6 rounded-full bg-white/20 flex items-center justify-center">
                      <IconCheck size={14} color="#FFF" />
                    </div>
                    <span className="font-medium text-[var(--color-bg)]">{item}</span>
                  </li>
                ))}
              </ul>
            </div>
            <div className="bg-white/10 p-8 rounded-2xl border border-white/20 backdrop-blur-sm">
               <div className="flex flex-col gap-6">
                 <div className="flex items-center gap-4 bg-white/5 p-4 rounded-lg">
                   <IconSecurity size={32} />
                   <div>
                     <h4 className="font-bold">End-to-End Encryption</h4>
                     <p className="text-sm text-white/70">Data is encrypted at rest and in transit.</p>
                   </div>
                 </div>
                 <div className="flex items-center gap-4 bg-white/5 p-4 rounded-lg">
                   <IconHospital size={32} />
                   <div>
                     <h4 className="font-bold">Strict Role Access</h4>
                     <p className="text-sm text-white/70">Administrators and Hospital staff have separated domains.</p>
                   </div>
                 </div>
               </div>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-24 bg-white text-center">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-6">
            Make Cognitive Health Assessment More Accessible.
          </h2>
          <p className="text-lg text-gray-600 mb-10">
            Join the growing network of healthcare professionals utilizing AI for cognitive screening.
          </p>
          <Link 
            href="/login" 
            className="inline-flex px-10 py-4 text-lg font-bold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-lg transition-all duration-200 hover:-translate-y-1"
          >
            Get Started Now
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[var(--color-text-main)] text-white/60 py-12 border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            <div className="col-span-1 md:col-span-2">
              <div className="flex items-center gap-2 mb-4 text-white">
                <IconBrain size={24} />
                <span className="font-bold text-xl tracking-tight">FedX Nuro</span>
              </div>
              <p className="text-sm max-w-sm mb-6">
                Professional clinical platform for AI-assisted cognitive health assessment and risk prediction.
              </p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-4">Platform</h4>
              <ul className="space-y-2 text-sm">
                <li><Link href="#features" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="#how-it-works" className="hover:text-white transition-colors">How It Works</Link></li>
                <li><Link href="/login" className="hover:text-white transition-colors">Login</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-bold mb-4">Legal</h4>
              <ul className="space-y-2 text-sm">
                <li><Link href="#" className="hover:text-white transition-colors">Privacy Policy</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Terms of Service</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Security</Link></li>
                <li><Link href="#" className="hover:text-white transition-colors">Contact</Link></li>
              </ul>
            </div>
          </div>
          <div className="pt-8 border-t border-white/10 text-sm flex flex-col md:flex-row justify-between items-center">
            <p>&copy; {new Date().getFullYear()} FedX Nuro. All rights reserved.</p>
            <p className="mt-2 md:mt-0">Not a diagnostic tool. Intended for professional screening aid.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

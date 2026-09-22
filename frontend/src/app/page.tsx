"use client";

import React, { useState } from "react";
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

function AnimatedNumber({ target, suffix = "" }: { target: number; suffix?: string }) {
  const [count, setCount] = useState(0);

  React.useEffect(() => {
    let start = 0;
    const duration = 1200;
    const stepTime = 25;
    const totalSteps = duration / stepTime;
    const increment = target / totalSteps;

    const timer = setInterval(() => {
      start += increment;
      if (start >= target) {
        setCount(target);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, stepTime);

    return () => clearInterval(timer);
  }, [target]);

  return <span>{count.toLocaleString()}{suffix}</span>;
}

export default function LandingPage() {
  const [contactSubmitted, setContactSubmitted] = useState(false);
  const [formData, setFormData] = useState({ name: "", email: "", institution: "", message: "" });

  const handleContactSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setContactSubmitted(true);
  };

  return (
    <div className="min-h-screen bg-[var(--color-bg)] font-sans">
      <Navbar />
      
      {/* Spacer for fixed navbar */}
      <div className="h-20"></div>

      {/* Hero Section */}
      <section id="home" className="relative pt-6 sm:pt-8 pb-8 sm:pb-12 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[var(--color-light-teal)] via-[var(--color-bg)] to-[var(--color-bg)] -z-10" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center animate-fade-in-up">
          <div className="inline-flex items-center justify-center p-3.5 mb-5 bg-white rounded-2xl shadow-sm border border-[var(--color-primary)]/15 text-[var(--color-primary)] ring-4 ring-[var(--color-primary)]/5 hover:scale-105 transition-transform duration-300">
            <IconBrain size={30} />
          </div>
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold text-[var(--color-text-main)] tracking-tight mb-4 sm:mb-5 leading-[1.18] max-w-4xl mx-auto">
            AI-Powered Cognitive <br className="hidden sm:block" />
            <span className="bg-gradient-to-r from-[var(--color-primary)] via-[#22848b] to-[var(--color-secondary)] bg-clip-text text-transparent">
              Health Assessment
            </span>
          </h1>
          <p className="max-w-2xl sm:max-w-3xl mx-auto text-base sm:text-lg lg:text-xl text-gray-600 mb-6 leading-relaxed font-normal">
            Advanced AI-assisted cognitive assessment and risk prediction designed to support healthcare professionals in early identification and informed decision-making.
          </p>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-3.5 sm:gap-4">
            <Link 
              href="/login" 
              className="w-full sm:w-auto px-8 py-3.5 text-base font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md hover:shadow-lg hover:-translate-y-0.5 transition-all duration-200 flex items-center justify-center gap-2 group"
            >
              <span>Get Started</span>
              <svg className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
            <Link 
              href="#how-it-works" 
              className="w-full sm:w-auto px-8 py-3.5 text-base font-semibold text-[var(--color-primary)] bg-white border border-[var(--color-primary)]/20 hover:bg-[var(--color-light-teal)] hover:border-[var(--color-primary)]/40 rounded-xl shadow-sm hover:shadow hover:-translate-y-0.5 transition-all duration-200"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>

      {/* Trust / Statistics Section */}
      <section className="py-8 bg-white border-y border-gray-100/80 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            <div className="flex flex-col items-center justify-center p-3 rounded-xl hover:bg-[var(--color-bg)]/50 transition-colors">
              <div className="text-4xl font-extrabold text-[var(--color-primary)] mb-1.5 tracking-tight font-mono">
                <AnimatedNumber target={10} suffix="k+" />
              </div>
              <div className="text-xs sm:text-sm font-semibold text-gray-500 uppercase tracking-wider">Assessments Completed</div>
            </div>
            <div className="flex flex-col items-center justify-center p-3 rounded-xl hover:bg-[var(--color-bg)]/50 transition-colors">
              <div className="text-4xl font-extrabold text-[var(--color-primary)] mb-1.5 tracking-tight font-mono">
                <AnimatedNumber target={500} suffix="+" />
              </div>
              <div className="text-xs sm:text-sm font-semibold text-gray-500 uppercase tracking-wider">Healthcare Professionals</div>
            </div>
            <div className="flex flex-col items-center justify-center p-3 rounded-xl hover:bg-[var(--color-bg)]/50 transition-colors">
              <div className="text-4xl font-extrabold text-[var(--color-primary)] mb-1.5 tracking-tight font-mono">
                <AnimatedNumber target={120} />
              </div>
              <div className="text-xs sm:text-sm font-semibold text-gray-500 uppercase tracking-wider">Hospitals Connected</div>
            </div>
            <div className="flex flex-col items-center justify-center p-3 rounded-xl hover:bg-[var(--color-bg)]/50 transition-colors">
              <div className="text-4xl font-extrabold text-[var(--color-primary)] mb-1.5 tracking-tight font-mono">
                <AnimatedNumber target={95} suffix="%" />
              </div>
              <div className="text-xs sm:text-sm font-semibold text-gray-500 uppercase tracking-wider">AI-Assisted Processing</div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-10 sm:py-12 bg-[var(--color-bg)]">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-3">How It Works</h2>
            <p className="text-base sm:text-lg text-gray-600 max-w-2xl mx-auto">A streamlined clinical workflow designed for efficiency, clarity, and precision.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-5 relative">
            {/* Connecting Line */}
            <div className="hidden md:block absolute top-1/2 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-[var(--color-secondary)]/30 to-transparent -z-10 -translate-y-1/2"></div>
            
            {[
              { title: "Patient Information", desc: "Securely input demographic and clinical history.", icon: IconHospital },
              { title: "Cognitive Assessment", desc: "Conduct standardized cognitive evaluation tests.", icon: IconAssessment },
              { title: "AI Analysis", desc: "Advanced models process multidimensional data.", icon: IconAI },
              { title: "Risk Assessment", desc: "Review comprehensive reports and risk predictions.", icon: IconReports }
            ].map((step, i) => (
              <div key={i} className="flex flex-col items-center text-center bg-white p-5 rounded-2xl shadow-xs border border-gray-100 hover:border-[var(--color-primary)]/20 hover:shadow-md hover:-translate-y-1 transition-all duration-300 relative group">
                <div className="w-11 h-11 rounded-full bg-[var(--color-light-teal)] text-[var(--color-primary)] font-bold text-sm flex items-center justify-center mb-4 border-4 border-white shadow-xs group-hover:scale-110 group-hover:bg-[var(--color-primary)] group-hover:text-white transition-all duration-300 absolute -top-5.5">
                  {i + 1}
                </div>
                <div className="mt-4 mb-2.5 text-[var(--color-secondary)] group-hover:text-[var(--color-primary)] group-hover:scale-105 transition-all duration-300">
                  <step.icon size={32} />
                </div>
                <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1.5">{step.title}</h3>
                <p className="text-xs sm:text-sm text-gray-600 leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Key Features Section */}
      <section id="features" className="py-10 sm:py-12 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-3">Key Features</h2>
            <p className="text-base sm:text-lg text-gray-600 max-w-2xl mx-auto">Comprehensive tools built specifically for modern neurological and clinical environments.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-5">
            {[
              { title: "AI Risk Prediction", desc: "Assist clinical decision-making with transparent risk modeling.", icon: IconPrediction },
              { title: "Historical Tracking", desc: "Monitor cognitive decline or stability over extended periods.", icon: IconHistory },
              { title: "Clinical Reports", desc: "Generate professional, ready-to-share medical reports.", icon: IconReports },
              { title: "Data Visualization", desc: "Interactive dashboards to analyze patient populations.", icon: IconChartBar },
              { title: "Patient Management", desc: "Organized securely with role-based access controls.", icon: IconHospital },
              { title: "Secure Access", desc: "Enterprise-grade security protecting sensitive health data.", icon: IconSecurity },
            ].map((feature, i) => (
              <div key={i} className="p-4 sm:p-5 rounded-2xl border border-gray-100/90 hover:border-[var(--color-primary)]/30 hover:shadow-md hover:-translate-y-0.5 transition-all duration-200 group bg-[var(--color-bg)]/60 hover:bg-white">
                <div className="w-10 h-10 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-3 group-hover:bg-[var(--color-primary)] group-hover:text-white group-hover:scale-105 transition-all duration-200 shadow-xs">
                  <feature.icon size={20} />
                </div>
                <h3 className="text-base font-bold text-[var(--color-text-main)] mb-1.5 group-hover:text-[var(--color-primary)] transition-colors">{feature.title}</h3>
                <p className="text-gray-600 leading-relaxed text-xs sm:text-sm">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section id="about" className="py-10 sm:py-12 bg-[var(--color-bg)] border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center mb-8">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-[var(--color-text-main)] mb-2">
              Pioneering Privacy-First Federated Neuroscience
            </h2>
            <p className="text-base text-gray-600 leading-relaxed">
              Fed-XNeuro bridges decentralized artificial intelligence and clinical cognitive diagnostics, allowing research hospitals and clinics to collaborate globally without ever exposing sensitive patient data.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            <div className="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-3">
                <IconBrain size={26} />
              </div>
              <h3 className="text-xl font-bold text-[var(--color-text-main)] mb-1.5">Multimodal Biomarkers</h3>
              <p className="text-gray-600 leading-relaxed text-sm">
                Integrates standardized cognitive assessments (MMSE, CDR) with deep learning neuroimaging feature extractors for balanced, holistic patient evaluation.
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-3">
                <IconSecurity size={26} />
              </div>
              <h3 className="text-xl font-bold text-[var(--color-text-main)] mb-1.5">Differential Privacy Guard</h3>
              <p className="text-gray-600 leading-relaxed text-sm">
                Built on rigorous (ε, δ)-Differential Privacy and secure gradient aggregation, ensuring that patient health records strictly stay within local hospital boundaries.
              </p>
            </div>

            <div className="bg-white p-5 rounded-2xl border border-gray-100 shadow-sm hover:shadow-md transition-shadow">
              <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-3">
                <IconHospital size={26} />
              </div>
              <h3 className="text-xl font-bold text-[var(--color-text-main)] mb-1.5">Clinician-in-the-Loop</h3>
              <p className="text-gray-600 leading-relaxed text-sm">
                Designed to empower medical professionals rather than replace them, providing transparent confidence intervals and interpretable risk trajectories.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Security & Privacy */}
      <section className="py-10 sm:py-12 bg-[var(--color-primary)] text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-2 gap-10 items-center">
            <div>
              <h2 className="text-2xl font-bold mb-3">Uncompromising Security & Privacy</h2>
              <p className="text-[var(--color-light-teal)] mb-4 text-sm leading-relaxed">
                We prioritize the security of sensitive healthcare data. Our platform is built with strict access controls and robust audit trails to support your institutional compliance requirements.
              </p>
              <ul className="space-y-2.5">
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
            <div className="bg-white/10 p-5 rounded-2xl border border-white/20 backdrop-blur-sm">
               <div className="flex flex-col gap-4">
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
      <section className="py-10 sm:py-12 bg-white text-center">
        <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-[var(--color-text-main)] mb-6">
            Make Cognitive Health Assessment More Accessible.
          </h2>
          <p className="text-base text-gray-600 mb-6">
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

      {/* Contact Section */}
      <section id="contact" className="py-10 sm:py-12 bg-white border-t border-gray-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="max-w-3xl mx-auto text-center mb-8">
            <h2 className="text-2xl sm:text-3xl font-extrabold text-[var(--color-text-main)] mb-2">
              Connect With Our Clinical Team
            </h2>
            <p className="text-base text-gray-600 leading-relaxed">
              Have questions about integrating Fed-XNeuro with your medical center, hospital network, or research cohort? We are here to help.
            </p>
          </div>

          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-5 gap-6 bg-[var(--color-bg)] p-6 sm:p-8 rounded-2xl border border-gray-100 shadow-sm">
            <div className="md:col-span-2 space-y-6 flex flex-col justify-between">
              <div>
                <h3 className="text-xl font-bold text-[var(--color-text-main)] mb-2">Hospital Partnerships</h3>
                <p className="text-sm text-gray-600 mb-4">
                  Join our decentralized federated learning network for Alzheimer's and cognitive impairment research.
                </p>
                <div className="space-y-3 text-sm text-gray-700">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-white border border-gray-200 flex items-center justify-center text-[var(--color-primary)]">
                      <IconHospital size={16} />
                    </div>
                    <span>Clinical Research Division</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-white border border-gray-200 flex items-center justify-center text-[var(--color-primary)]">
                      <IconReports size={16} />
                    </div>
                    <span>support@fedxneuro.health</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-white border border-gray-200 flex items-center justify-center text-[var(--color-primary)]">
                      <IconSecurity size={16} />
                    </div>
                    <span>HIPAA & GDPR Compliant Infrastructure</span>
                  </div>
                </div>
              </div>
              <div className="p-4 rounded-xl bg-white border border-gray-100 text-xs text-gray-500">
                Response time: Within 24 business hours for clinical inquiries.
              </div>
            </div>

            <div className="md:col-span-3 bg-white p-5 sm:p-6 rounded-xl border border-gray-100 shadow-sm">
              {contactSubmitted ? (
                <div className="py-12 text-center flex flex-col items-center justify-center">
                  <div className="w-14 h-14 rounded-full bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center mb-4">
                    <IconCheck size={28} />
                  </div>
                  <h4 className="text-xl font-bold text-[var(--color-text-main)] mb-2">Inquiry Received</h4>
                  <p className="text-sm text-gray-600 max-w-sm">
                    Thank you, {formData.name || "Doctor"}. Our clinical collaboration team has received your request and will reach out to <span className="font-semibold">{formData.email}</span> shortly.
                  </p>
                  <button
                    onClick={() => {
                      setContactSubmitted(false);
                      setFormData({ name: "", email: "", institution: "", message: "" });
                    }}
                    className="mt-6 text-xs text-[var(--color-primary)] font-semibold hover:underline"
                  >
                    Send another inquiry
                  </button>
                </div>
              ) : (
                <form onSubmit={handleContactSubmit} className="space-y-3">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">Full Name</label>
                      <input 
                        type="text" 
                        required 
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                        placeholder="Dr. Sarah Jenkins" 
                        className="w-full px-3.5 py-2.5 text-sm rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" 
                      />
                    </div>
                    <div>
                      <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">Work Email</label>
                      <input 
                        type="email" 
                        required 
                        value={formData.email}
                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                        placeholder="s.jenkins@hospital.org" 
                        className="w-full px-3.5 py-2.5 text-sm rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" 
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">Institution / Hospital</label>
                    <input 
                      type="text" 
                      required 
                      value={formData.institution}
                      onChange={(e) => setFormData({ ...formData, institution: e.target.value })}
                      placeholder="Memorial Health Neuroscience Center" 
                      className="w-full px-3.5 py-2.5 text-sm rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]" 
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 uppercase tracking-wider mb-1.5">Inquiry Details</label>
                    <textarea 
                      rows={3} 
                      required 
                      value={formData.message}
                      onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                      placeholder="Tell us about your cohort size or clinical deployment inquiry..." 
                      className="w-full px-3.5 py-2.5 text-sm rounded-lg border border-gray-200 focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] resize-none" 
                    />
                  </div>
                  <button 
                    type="submit" 
                    className="w-full py-3 text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-lg shadow-sm transition-all duration-200"
                  >
                    Send Clinical Inquiry
                  </button>
                </form>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-[var(--color-text-main)] text-white/60 py-8 border-t border-white/10">
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
                <li><Link href="#home" className="hover:text-white transition-colors">Home</Link></li>
                <li><Link href="#how-it-works" className="hover:text-white transition-colors">How It Works</Link></li>
                <li><Link href="#features" className="hover:text-white transition-colors">Features</Link></li>
                <li><Link href="#about" className="hover:text-white transition-colors">About</Link></li>
                <li><Link href="#contact" className="hover:text-white transition-colors">Contact</Link></li>
              </ul>
            </div>
            <div>
              <h4 className="text-white font-bold mb-4">Portals</h4>
              <ul className="space-y-2 text-sm">
                <li><Link href="/login" className="hover:text-white transition-colors">Clinician Login</Link></li>
                <li><Link href="/dashboard" className="hover:text-white transition-colors">Hospital Dashboard</Link></li>
                <li><Link href="/dashboard/admin" className="hover:text-white transition-colors">Admin Console</Link></li>
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

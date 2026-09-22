import Link from "next/link";
import { IconBrain } from "@/components/Icons";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[var(--color-bg)] flex flex-col items-center justify-center p-6 text-center font-sans relative overflow-hidden">
      {/* Background ambient lighting */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-96 h-96 bg-[var(--color-primary)]/10 rounded-full blur-3xl -z-10 pointer-events-none" />

      <div className="max-w-md w-full animate-fade-in-up bg-white p-8 sm:p-10 rounded-3xl shadow-lg shadow-[var(--color-primary)]/5 border border-gray-100">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[var(--color-primary)] to-[var(--color-secondary)] text-white mx-auto flex items-center justify-center shadow-md shadow-[var(--color-primary)]/25 mb-6">
          <IconBrain size={34} />
        </div>

        <h1 className="text-2xl sm:text-3xl font-extrabold text-[var(--color-text-main)] tracking-tight mb-2">
          404 — Resource Not Found
        </h1>

        <p className="text-sm text-gray-500 mb-8 leading-relaxed">
          The requested clinical node, assessment protocol, or page could not be located on the Fed-XNeuro federated network.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          <Link
            href="/"
            className="w-full sm:w-auto px-6 py-3 text-xs sm:text-sm font-semibold text-white bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] rounded-xl shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 transition-all"
          >
            Return to Home
          </Link>
          <Link
            href="/dashboard/hospital"
            className="w-full sm:w-auto px-6 py-3 text-xs sm:text-sm font-semibold text-gray-700 bg-gray-50 border border-gray-200 hover:bg-gray-100 rounded-xl transition-all"
          >
            Clinical Dashboard
          </Link>
        </div>
      </div>

      <p className="mt-8 text-xs text-gray-400">
        Fed-XNeuro Federated Healthcare Network
      </p>
    </div>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fed-XNeuro — Federated Clinical AI Platform",
  description: "Next.js Clinician & Researcher Dashboard for Multimodal Explainable Federated Learning",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

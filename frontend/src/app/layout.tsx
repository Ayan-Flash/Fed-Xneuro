import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fed-XNeuro — Federated Clinical AI Platform",
  description: "Clinician & Researcher Dashboard for Multimodal Explainable Federated Learning in Alzheimer's Progression Prediction",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body className="font-sans antialiased selection:bg-[#197C82]/20 selection:text-[#197C82]">{children}</body>
    </html>
  );
}

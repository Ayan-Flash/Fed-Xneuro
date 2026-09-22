"use client";

import React from "react";

interface IconProps {
  size?: number | string;
  color?: string;
  className?: string;
}

const defaultSize = 24;
const defaultColor = "currentColor";

// Basic wrapper for consistent SVG props
const IconWrapper = ({ size = defaultSize, color = defaultColor, className, children, viewBox = "0 0 24 24" }: IconProps & { children: React.ReactNode, viewBox?: string }) => (
  <svg width={size} height={size} viewBox={viewBox} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
    {children}
  </svg>
);

export const IconBrain = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z" />
    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z" />
    <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
    <path d="M12 18v4" />
  </IconWrapper>
);

export const IconAI = (props: IconProps) => (
  <IconWrapper {...props}>
    <rect width="18" height="18" x="3" y="3" rx="2" />
    <path d="M8 9h1a2 2 0 0 1 2 2v4" />
    <path d="M14 9v6" />
    <path d="M7 15h4" />
  </IconWrapper>
);

export const IconAdmin = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </IconWrapper>
);

export const IconHospital = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M12 6v4" />
    <path d="M10 8h4" />
    <rect width="20" height="14" x="2" y="6" rx="2" />
    <path d="M2 10h20" />
    <path d="M12 20v-4" />
  </IconWrapper>
);

export const IconPatients = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <line x1="19" x2="19" y1="8" y2="14" />
    <line x1="22" x2="16" y1="11" y2="11" />
  </IconWrapper>
);

export const IconAssessment = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
    <path d="M8 13h2" />
    <path d="M8 17h2" />
    <path d="M14 13h2" />
    <path d="M14 17h2" />
  </IconWrapper>
);

export const IconPrediction = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M3 3v18h18" />
    <path d="m19 9-5 5-4-4-3 3" />
    <path d="M19 9v6" />
    <path d="M19 9h-6" />
  </IconWrapper>
);

export const IconReports = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="16" x2="8" y1="13" y2="13" />
    <line x1="16" x2="8" y1="17" y2="17" />
    <line x1="10" x2="8" y1="9" y2="9" />
  </IconWrapper>
);

export const IconHistory = (props: IconProps) => (
  <IconWrapper {...props}>
    <circle cx="12" cy="12" r="10" />
    <polyline points="12 6 12 12 16 14" />
  </IconWrapper>
);

export const IconNotifications = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
    <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
  </IconWrapper>
);

export const IconSecurity = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <path d="m9 12 2 2 4-4" />
  </IconWrapper>
);

export const IconSettings = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
    <circle cx="12" cy="12" r="3" />
  </IconWrapper>
);

export const IconRiskLow = (props: IconProps) => (
  <IconWrapper {...props}>
    <circle cx="12" cy="12" r="10" />
    <path d="m9 12 2 2 4-4" />
  </IconWrapper>
);

export const IconRiskModerate = (props: IconProps) => (
  <IconWrapper {...props}>
    <circle cx="12" cy="12" r="10" />
    <line x1="12" x2="12" y1="8" y2="12" />
    <line x1="12" x2="12.01" y1="16" y2="16" />
  </IconWrapper>
);

export const IconRiskHigh = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <line x1="12" x2="12" y1="9" y2="13" />
    <line x1="12" x2="12.01" y1="17" y2="17" />
  </IconWrapper>
);

export const IconCheck = (props: IconProps) => (
  <IconWrapper {...props}>
    <path d="M20 6 9 17l-5-5" />
  </IconWrapper>
);

export const IconError = (props: IconProps) => (
  <IconWrapper {...props}>
    <circle cx="12" cy="12" r="10" />
    <line x1="15" x2="9" y1="9" y2="15" />
    <line x1="9" x2="15" y1="9" y2="15" />
  </IconWrapper>
);

export const IconChartBar = (props: IconProps) => (
  <IconWrapper {...props}>
    <rect width="18" height="18" x="3" y="3" rx="2" />
    <path d="M7 17v-4" />
    <path d="M12 17v-8" />
    <path d="M17 17v-6" />
  </IconWrapper>
);

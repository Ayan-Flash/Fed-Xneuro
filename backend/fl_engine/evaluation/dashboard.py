"""
Clinician Dashboard Generator for Fed-XNeuro.

Implements Module 8 from Fed-XNeuro specification:
1. MCI -> AD Progression Risk Score & Category (HIGH / MODERATE / LOW)
2. Clinical Feature Importance (SHAP / path-integrated gradients)
3. 3D MRI Neuroimaging Attribution (Hippocampus heatmap ROI)
4. Longitudinal Progression Timeline (Baseline -> 6M -> 12M -> 24M...)
5. ASCII Terminal, HTML, and JSON report generation
"""

import os
import json
from typing import Dict, Any, Optional


class ClinicianDashboard:
    """
    Renders and exports interpretable clinician-facing diagnostic reports.
    Strict privacy boundary: Generated on the client device using local data.
    """

    @staticmethod
    def render_ascii(report: Dict[str, Any]) -> str:
        """
        Renders ASCII terminal dashboard matching Fed-XNeuro Architecture Section 6 Module 8.
        """
        prob = report.get("risk_probability", 0.0)
        pct = round(prob * 100.0, 1)
        category = report.get("risk_category", "UNKNOWN")
        patient_id = report.get("patient_id", "Unknown Patient")
        mri_attr = report.get("mri_attribution", {})
        hippo_pct = mri_attr.get("hippocampus_importance_pct", 0.0)
        peak_voxel = mri_attr.get("peak_attribution_voxel", [0, 0, 0])
        timeline = report.get("longitudinal_visit_weights", [])

        # Color/Badge styling indicators
        lines = [
            "+---------------------------------------------------------------+",
            "|                     FED-XNEURO DASHBOARD                      |",
            f"| Patient: {patient_id:<52} |",
            "+---------------------------------------------------------------+",
            "|                                                               |",
            "| MCI -> AD Risk Progression Probability                        |",
            f"|                         {pct:>5.1f}%                                |",
            f"| Risk Category: {category:<46} |",
            "|                                                               |",
            "+---------------------------------------------------------------+",
            "| Clinical Feature Importance (SHAP)                            |",
        ]

        # Clinical features
        for item in report.get("clinical_importance", [])[:5]:
            feat = item["feature"]
            feat_pct = item["relative_pct"]
            bar_len = int(feat_pct / 5.0)
            bar = "#" * max(1, bar_len)
            lines.append(f"| {feat:<18} {bar:<25} {feat_pct:>5.1f}% |")

        lines.extend([
            "+---------------------------------------------------------------+",
            "| MRI Neuroimaging Attribution (Integrated Gradients)           |",
            f"| Hippocampus Attribution: {hippo_pct:>5.1f}%                              |",
            f"| Peak Attribution Voxel:  {str(peak_voxel):<36} |",
            "| Medial Temporal Lobe Region: Marked Atrophy Focus             |",
            "+---------------------------------------------------------------+",
            "| Longitudinal Progression Timeline                             |",
            "| Visit Trajectory: Baseline -> 6M -> 12M -> 24M -> 36M -> 48M  |",
        ])

        if timeline:
            weights_str = " | ".join([f"{w:.2f}" for w in timeline])
            lines.append(f"| Temporal Weights: {weights_str:<43} |")

        lines.append("+---------------------------------------------------------------+\n")
        return "\n".join(lines)

    @staticmethod
    def save_json_report(report: Dict[str, Any], output_path: str) -> str:
        """Saves machine-readable JSON report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path

    @staticmethod
    def save_html_report(report: Dict[str, Any], output_path: str) -> str:
        """
        Renders a sleek HTML clinician dashboard report card.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        prob = report.get("risk_probability", 0.0)
        pct = round(prob * 100.0, 1)
        category = report.get("risk_category", "LOW")
        patient_id = report.get("patient_id", "Patient")
        mri_attr = report.get("mri_attribution", {})
        hippo_pct = mri_attr.get("hippocampus_importance_pct", 0.0)
        peak_voxel = mri_attr.get("peak_attribution_voxel", [0, 0, 0])

        color = "#ef4444" if category == "HIGH" else ("#f59e0b" if category == "MODERATE" else "#10b981")

        feat_rows = ""
        for item in report.get("clinical_importance", [])[:6]:
            feat = item["feature"]
            fpct = item["relative_pct"]
            feat_rows += f"""
            <div style="margin-bottom: 8px;">
                <div style="display:flex; justify-content:space-between; font-size:14px; margin-bottom:2px;">
                    <span>{feat}</span>
                    <span style="font-weight:600;">{fpct:.1f}%</span>
                </div>
                <div style="background:#e2e8f0; border-radius:4px; height:8px; overflow:hidden;">
                    <div style="background:#3b82f6; width:{min(100, fpct*2)}%; height:100%;"></div>
                </div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Fed-XNeuro Clinician Dashboard - {patient_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; padding: 24px; }}
  .card {{ max-width: 650px; margin: 0 auto; background: #1e293b; border-radius: 16px; padding: 28px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 13px; text-transform: uppercase; background: {color}; color: #fff; }}
  .risk-val {{ font-size: 48px; font-weight: 800; color: {color}; margin: 12px 0 4px 0; }}
  .sec {{ margin-top: 24px; border-top: 1px solid #334155; padding-top: 16px; }}
  h2 {{ font-size: 16px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 12px; }}
</style>
</head>
<body>
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <h1 style="font-size:20px; margin:0; font-weight:700;">Fed-XNeuro Clinician Dashboard</h1>
    <span class="badge">{category} RISK</span>
  </div>
  <p style="color:#94a3b8; font-size:14px; margin-top:4px;">Decentralized AI Diagnostic Report &bull; {patient_id}</p>

  <div style="text-align:center; padding: 16px 0;">
    <div style="font-size:13px; color:#94a3b8; text-transform:uppercase;">MCI &rarr; AD Conversion Risk</div>
    <div class="risk-val">{pct}%</div>
    <div style="font-size:14px; color:#cbd5e1;">Predicted Progression Probability: <strong>{prob:.4f}</strong></div>
  </div>

  <div class="sec">
    <h2>Clinical Feature Importance (SHAP)</h2>
    {feat_rows}
  </div>

  <div class="sec">
    <h2>MRI Attribution (Integrated Gradients)</h2>
    <div style="background:#0f172a; border-radius:8px; padding:12px; font-size:14px;">
      <div>&bull; Hippocampus Region Attribution: <strong>{hippo_pct:.1f}%</strong> of total gradient magnitude</div>
      <div>&bull; Peak Attribution Voxel Coordinate: <code>{peak_voxel}</code></div>
      <div>&bull; Dominant Biomarker Focus: Medial Temporal Lobe & Ventricle Margin</div>
    </div>
  </div>

  <div class="sec">
    <h2>Longitudinal Timeline</h2>
    <div style="display:flex; justify-content:space-between; font-size:13px; color:#94a3b8; margin-top:8px;">
      <span>Baseline (0M)</span><span>6M</span><span>12M</span><span>24M</span><span>36M</span><span>48M</span>
    </div>
  </div>
</div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path

"""
Local Explainability Module for Fed-XNeuro.

Implements Module 5 & Step 9 from Fed-XNeuro specification:
1. Integrated Gradients for 3D MRI neuroimaging attribution (e.g. Hippocampus heatmaps)
2. SHAP / Path-integrated gradients for Clinical & Cognitive feature importance
3. Clinician-facing explanation report generator executed strictly on-device
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import numpy as np


class IntegratedGradientsMRI:
    """
    Implements Integrated Gradients (Sundararajan et al., 2017)
    for 3D structural MRI scans to produce brain region attribution maps.
    """

    def __init__(self, steps: int = 20) -> None:
        self.steps = max(5, steps)

    def attribute(
        self,
        model: nn.Module,
        mri: torch.Tensor,
        cognitive: torch.Tensor,
        ehr: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes Integrated Gradients attribution map with respect to the 3D MRI input.

        Args:
            model: FedXNeuroModel
            mri: [1, T, 1, D, H, W] or [1, 1, D, H, W]
            cognitive: [1, T, D_cog]
            ehr: [1, T, D_ehr]
            visit_mask: [1, T]
            time_gaps: [1, T]
            baseline: Optional reference tensor (defaults to zero volume)

        Returns:
            attribution_map: Tensor matching mri shape with integrated attribution values
        """
        model.eval()
        device = mri.device
        if baseline is None:
            baseline = torch.zeros_like(mri, device=device)

        mri_diff = mri - baseline
        accumulated_grads = torch.zeros_like(mri, device=device)

        # Scale interpolation steps alpha in [1/steps, 1]
        for step in range(1, self.steps + 1):
            alpha = float(step) / self.steps
            interpolated_mri = baseline + alpha * mri_diff
            interpolated_mri = interpolated_mri.clone().detach().requires_grad_(True)

            out = model(
                mri=interpolated_mri,
                cognitive=cognitive,
                ehr=ehr,
                visit_mask=visit_mask,
                time_gaps=time_gaps,
            )
            prob = out["probabilities"][0, 0]

            model.zero_grad()
            if interpolated_mri.grad is not None:
                interpolated_mri.grad.zero_()

            prob.backward(retain_graph=False)
            if interpolated_mri.grad is not None:
                accumulated_grads += interpolated_mri.grad.detach()

        avg_grads = accumulated_grads / float(self.steps)
        attributions = mri_diff * avg_grads
        return attributions


class ClinicalSHAPAttributor:
    """
    Computes path-integrated feature attributions (SHAP approximation)
    for clinical and cognitive variables (MMSE, CDR-SB, ADAS-Cog, Age, APOE4, etc.).
    """

    DEFAULT_COG_NAMES = ["MMSE", "CDR-SB", "ADAS-Cog11", "ADAS-Cog13", "FAQ"]
    DEFAULT_EHR_NAMES = ["Age", "APOE4-Alleles", "Sex", "Education", "Systolic-BP", "BMI", "Cholesterol"]

    def __init__(
        self,
        steps: int = 20,
        cog_names: Optional[List[str]] = None,
        ehr_names: Optional[List[str]] = None,
    ) -> None:
        self.steps = max(5, steps)
        self.cog_names = cog_names or self.DEFAULT_COG_NAMES
        self.ehr_names = ehr_names or self.DEFAULT_EHR_NAMES

    def attribute(
        self,
        model: nn.Module,
        mri: torch.Tensor,
        cognitive: torch.Tensor,
        ehr: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Computes feature attribution importance for tabular clinical inputs.
        """
        model.eval()
        device = cognitive.device

        cog_baseline = torch.zeros_like(cognitive, device=device)
        ehr_baseline = torch.zeros_like(ehr, device=device)

        cog_diff = cognitive - cog_baseline
        ehr_diff = ehr - ehr_baseline

        accum_cog_grads = torch.zeros_like(cognitive, device=device)
        accum_ehr_grads = torch.zeros_like(ehr, device=device)

        for step in range(1, self.steps + 1):
            alpha = float(step) / self.steps
            interp_cog = (cog_baseline + alpha * cog_diff).clone().detach().requires_grad_(True)
            interp_ehr = (ehr_baseline + alpha * ehr_diff).clone().detach().requires_grad_(True)

            out = model(
                mri=mri,
                cognitive=interp_cog,
                ehr=interp_ehr,
                visit_mask=visit_mask,
                time_gaps=time_gaps,
            )
            prob = out["probabilities"][0, 0]

            model.zero_grad()
            prob.backward(retain_graph=False)

            if interp_cog.grad is not None:
                accum_cog_grads += interp_cog.grad.detach()
            if interp_ehr.grad is not None:
                accum_ehr_grads += interp_ehr.grad.detach()

        cog_attr = (cog_diff * (accum_cog_grads / self.steps)).squeeze(0)  # [T, D_cog]
        ehr_attr = (ehr_diff * (accum_ehr_grads / self.steps)).squeeze(0)  # [T, D_ehr]

        # Aggregate across visits (mean absolute attribution)
        cog_scores = cog_attr.abs().mean(dim=0).cpu().numpy()
        ehr_scores = ehr_attr.abs().mean(dim=0).cpu().numpy()

        feature_scores: Dict[str, float] = {}
        for i, val in enumerate(cog_scores):
            name = self.cog_names[i] if i < len(self.cog_names) else f"Cog_Var_{i}"
            feature_scores[name] = float(val)

        for i, val in enumerate(ehr_scores):
            name = self.ehr_names[i] if i < len(self.ehr_names) else f"EHR_Var_{i}"
            feature_scores[name] = float(val)

        # Normalize to percentages
        total = sum(feature_scores.values()) + 1e-8
        ranking = [
            {
                "feature": k,
                "importance": round(v, 4),
                "relative_pct": round((v / total) * 100.0, 1),
            }
            for k, v in sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
        ]

        return {
            "ranking": ranking,
            "raw_scores": feature_scores,
            "top_feature": ranking[0]["feature"] if ranking else "None",
        }


class LocalExplainabilityEngine:
    """
    On-device explainability orchestrator.
    Generates unified clinical explanation reports for MCI -> AD progression predictions.
    """

    def __init__(self, ig_steps: int = 15, shap_steps: int = 15) -> None:
        self.mri_attributor = IntegratedGradientsMRI(steps=ig_steps)
        self.clinical_attributor = ClinicalSHAPAttributor(steps=shap_steps)

    def explain_patient(
        self,
        model: nn.Module,
        patient_data: Dict[str, torch.Tensor],
        patient_id: str = "P001",
    ) -> Dict[str, Any]:
        """
        Generates full explainability report for a patient.
        Strict privacy boundary: Executed locally, never shared centrally.
        """
        mri = patient_data["mri"]
        cognitive = patient_data["cognitive"]
        ehr = patient_data["ehr"]
        visit_mask = patient_data.get("visit_mask")
        time_gaps = patient_data.get("time_gaps")

        # 1. Forward model prediction
        model.eval()
        with torch.no_grad():
            preds = model(
                mri=mri,
                cognitive=cognitive,
                ehr=ehr,
                visit_mask=visit_mask,
                time_gaps=time_gaps,
            )
            prob = float(preds["probabilities"][0, 0].item())
            visit_attn = preds["visit_attention"][0, :, 0].cpu().tolist()

        category = "HIGH" if prob > 0.7 else ("MODERATE" if prob > 0.3 else "LOW")

        # 2. Clinical Feature Importance (SHAP)
        clinical_report = self.clinical_attributor.attribute(
            model=model,
            mri=mri,
            cognitive=cognitive,
            ehr=ehr,
            visit_mask=visit_mask,
            time_gaps=time_gaps,
        )

        # 3. 3D MRI Attribution (Integrated Gradients)
        mri_attr = self.mri_attributor.attribute(
            model=model,
            mri=mri,
            cognitive=cognitive,
            ehr=ehr,
            visit_mask=visit_mask,
            time_gaps=time_gaps,
        )

        # Compute anatomical ROI statistics (e.g. Hippocampus region)
        # Assuming normalized coordinates with hippocampus located centrally/medially
        attr_np = mri_attr.abs().cpu().numpy()
        total_mri_attr = float(np.sum(attr_np)) + 1e-8

        # Define medial temporal / hippocampal bounding box in relative volume coordinates
        d_len, h_len, w_len = attr_np.shape[-3:]
        d_mid, h_mid, w_mid = d_len // 2, h_len // 2, w_len // 2
        d_span, h_span, w_span = max(1, d_len // 4), max(1, h_len // 4), max(1, w_len // 4)

        hippocampus_roi = attr_np[
            ...,
            d_mid - d_span : d_mid + d_span,
            h_mid - h_span : h_mid + h_span,
            w_mid - w_span : w_mid + w_span,
        ]
        hippo_attr_val = float(np.sum(hippocampus_roi))
        hippo_attr_pct = round((hippo_attr_val / total_mri_attr) * 100.0, 1)

        # Peak voxel coordinate
        peak_idx = np.unravel_index(np.argmax(attr_np), attr_np.shape)
        peak_coords = [int(p) for p in peak_idx[-3:]]

        return {
            "patient_id": patient_id,
            "risk_probability": round(prob, 4),
            "risk_category": category,
            "clinical_importance": clinical_report["ranking"],
            "top_clinical_feature": clinical_report["top_feature"],
            "mri_attribution": {
                "hippocampus_importance_pct": hippo_attr_pct,
                "peak_attribution_voxel": peak_coords,
                "global_attribution_mean": float(np.mean(attr_np)),
            },
            "longitudinal_visit_weights": [round(w, 3) for w in visit_attn],
        }

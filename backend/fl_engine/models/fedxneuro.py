"""
Fed-XNeuro Neural Network Architecture.

Implements:
1. Module 1: 3D ResNet MRI Feature Extractor (3D ResNet18 backbone)
2. Module 2: Multimodal Feature Fusion (MRI + Cognitive + EHR)
3. Module 3: Missing-Visit Imputation Module (Longitudinal Attention with Temporal Encoding)
4. Module 4: Longitudinal Temporal Disease Transformer
5. Risk Prediction Head (MCI -> AD Progression Risk)
"""

import math
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from backend.fl_engine.models.base import BaseModel


# =========================================================================
# Module 1: 3D ResNet MRI Feature Extractor
# =========================================================================

class Conv3DBlock(nn.Module):
    """Basic 3D convolutional building block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet3DEncoder(nn.Module):
    """
    3D ResNet encoder for neuroimaging (structural T1-weighted MRI).
    Extracts spatial neuroanatomical representations (e.g. hippocampal atrophy).
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        embedding_dim: int = 128,
        layers: Tuple[int, int, int, int] = (1, 1, 1, 1),
    ) -> None:
        super().__init__()
        self.in_planes = base_channels
        self.embedding_dim = embedding_dim

        # Initial stem
        self.conv1 = nn.Conv3d(
            in_channels,
            base_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        # 4 residual stages
        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(base_channels * 8, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        layers = [Conv3DBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(Conv3DBlock(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 3D MRI tensor of shape [B, C=1, D, H, W]
               or longitudinal sequence [B, T, C=1, D, H, W]
        Returns:
            Embedding of shape [B, embedding_dim] or [B, T, embedding_dim]
        """
        is_sequence = x.dim() == 6
        if is_sequence:
            b, t, c, d, h, w = x.shape
            x = x.view(b * t, c, d, h, w)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.norm(out)

        if is_sequence:
            out = out.view(b, t, self.embedding_dim)

        return out


# =========================================================================
# Module 2: Multimodal Fusion
# =========================================================================

class MultimodalFusion(nn.Module):
    """
    Fuses neuroimaging (MRI), cognitive assessments (MMSE, CDR-SB, ADAS-Cog, FAQ),
    and electronic health records (EHR: Age, APOE4, vitals) into a unified representation.
    """

    def __init__(
        self,
        mri_dim: int = 128,
        cog_dim: int = 5,
        ehr_dim: int = 7,
        fusion_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_dim = fusion_dim

        proj_dim = fusion_dim // 2
        self.mri_proj = nn.Sequential(
            nn.Linear(mri_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.cog_proj = nn.Sequential(
            nn.Linear(cog_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        self.ehr_proj = nn.Sequential(
            nn.Linear(ehr_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        total_cat_dim = proj_dim * 3
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_cat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

    def forward(
        self,
        mri_feats: torch.Tensor,
        cog_scores: torch.Tensor,
        ehr_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mri_feats: [B, T, mri_dim]
            cog_scores: [B, T, cog_dim]
            ehr_feats: [B, T, ehr_dim]
        Returns:
            Fused representations: [B, T, fusion_dim]
        """
        p_mri = self.mri_proj(mri_feats)
        p_cog = self.cog_proj(cog_scores)
        p_ehr = self.ehr_proj(ehr_feats)

        combined = torch.cat([p_mri, p_cog, p_ehr], dim=-1)
        fused = self.fusion_mlp(combined)
        return fused


# =========================================================================
# Module 3: Missing-Visit Imputation Module
# =========================================================================

class ContinuousTemporalEncoding(nn.Module):
    """
    Computes continuous sinusoidal temporal encodings for irregular visit intervals.
    """

    def __init__(self, dim: int, max_period: float = 100.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer("freqs", freqs)

    def forward(self, time_gaps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_gaps: [B, T] representing time elapsed (in months) from baseline.
        Returns:
            Encodings: [B, T, dim]
        """
        # [B, T, 1] * [1, 1, half_dim] -> [B, T, half_dim]
        args = time_gaps.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        sin_part = torch.sin(args)
        cos_part = torch.cos(args)
        emb = torch.cat([sin_part, cos_part], dim=-1)
        if emb.shape[-1] < self.dim:
            # pad if odd dimension
            pad = torch.zeros(*emb.shape[:-1], self.dim - emb.shape[-1], device=emb.device)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class MissingVisitImputationModule(nn.Module):
    """
    Handles patient dropouts and irregular follow-up schedules.
    Uses longitudinal multi-head attention conditioning on observed visits,
    visit masks, and continuous time intervals to recover missing visit representations.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.missing_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.missing_token, std=0.02)

        self.temporal_encoder = ContinuousTemporalEncoding(dim=dim)
        self.mask_indicator = nn.Embedding(2, dim)  # 0: missing, 1: observed

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.imputer_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        visit_mask: torch.Tensor,
        time_gaps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Fused longitudinal sequence [B, T, dim]
            visit_mask: Binary mask [B, T], 1 = observed, 0 = missing
            time_gaps: Elapsed time since baseline [B, T]
        Returns:
            Reconstructed longitudinal sequence [B, T, dim]
        """
        b, t, d = x.shape

        # Expand missing token across batch and time
        missing_expanded = self.missing_token.expand(b, t, d)

        # Masked input: replace missing visits with learnable missing token
        mask_expanded = visit_mask.unsqueeze(-1).to(x.dtype)
        imputed_x = x * mask_expanded + missing_expanded * (1.0 - mask_expanded)

        # Inject temporal encoding and observed/missing indicator
        t_enc = self.temporal_encoder(time_gaps)
        m_ind = self.mask_indicator(visit_mask.long())

        h = imputed_x + t_enc + m_ind
        recovered = self.imputer_transformer(h)
        return self.norm(recovered)


# =========================================================================
# Module 4: Longitudinal Temporal Transformer & Risk Prediction Head
# =========================================================================

class LongitudinalTransformer(nn.Module):
    """
    Models disease progression patterns across longitudinal visits.
    Captures temporal progression toward Alzheimer's Disease.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dim = dim
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Attentive longitudinal pooling across visits
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Reconstructed longitudinal sequence [B, T, dim]
        Returns:
            patient_disease_state: [B, dim]
            attention_weights: [B, T, 1]
        """
        h = self.transformer(x)
        # Compute visit attention weights
        attn_scores = self.attention_pool(h)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)

        # Weighted temporal sum
        disease_state = torch.sum(h * attn_weights, dim=1)  # [B, dim]
        return self.norm(disease_state), attn_weights


class RiskPredictionHead(nn.Module):
    """
    Computes MCI -> AD progression probability P(MCI -> AD)
    and risk category stratification.
    """

    def __init__(self, in_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, disease_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, 1]
            probabilities: [B, 1] in [0, 1]
        """
        logits = self.classifier(disease_state)
        probs = torch.sigmoid(logits)
        return logits, probs


# =========================================================================
# Unified Fed-XNeuro Model
# =========================================================================

class FedXNeuroModel(BaseModel):
    """
    Complete End-to-End Explainable Multimodal Federated Model.

    Inputs:
    - MRI Sequences (3D T1 MRI volumes)
    - Cognitive Scores (MMSE, CDR-SB, ADAS-Cog, FAQ)
    - Longitudinal EHR Records (Age, APOE4, Demographics, Vitals)
    - Visit Masks & Time Gaps

    Outputs:
    - P(MCI -> AD) Progression Risk Probability
    - Trajectory representation & attention maps for Explainability
    """

    def __init__(
        self,
        mri_channels: int = 1,
        mri_base_channels: int = 16,
        mri_embedding_dim: int = 128,
        cog_dim: int = 5,
        ehr_dim: int = 7,
        fusion_dim: int = 128,
        transformer_heads: int = 4,
        imputer_layers: int = 2,
        temporal_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mri_encoder = ResNet3DEncoder(
            in_channels=mri_channels,
            base_channels=mri_base_channels,
            embedding_dim=mri_embedding_dim,
        )
        self.fusion = MultimodalFusion(
            mri_dim=mri_embedding_dim,
            cog_dim=cog_dim,
            ehr_dim=ehr_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        self.missing_visit_module = MissingVisitImputationModule(
            dim=fusion_dim,
            num_heads=transformer_heads,
            num_layers=imputer_layers,
            dropout=dropout,
        )
        self.temporal_transformer = LongitudinalTransformer(
            dim=fusion_dim,
            num_heads=transformer_heads,
            num_layers=temporal_layers,
            dropout=dropout,
        )
        self.risk_head = RiskPredictionHead(in_dim=fusion_dim, dropout=dropout)

    def forward(
        self,
        mri: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]],
        cognitive: Optional[torch.Tensor] = None,
        ehr: Optional[torch.Tensor] = None,
        visit_mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Accepts either separate tensors or a dictionary/tuple package.
        """
        # Unpack input dictionary or tuple if passed as single argument
        if isinstance(mri, dict):
            cognitive = mri.get("cognitive")
            ehr = mri.get("ehr")
            visit_mask = mri.get("visit_mask")
            time_gaps = mri.get("time_gaps")
            mri = mri.get("mri")
        elif isinstance(mri, (tuple, list)):
            mri_tensor = mri[0]
            cognitive = mri[1] if len(mri) > 1 else None
            ehr = mri[2] if len(mri) > 2 else None
            visit_mask = mri[3] if len(mri) > 3 else None
            time_gaps = mri[4] if len(mri) > 4 else None
            mri = mri_tensor

        b, t = cognitive.shape[0], cognitive.shape[1]
        device = cognitive.device

        # Default masks and time gaps if not provided
        if visit_mask is None:
            visit_mask = torch.ones((b, t), dtype=torch.float32, device=device)
        if time_gaps is None:
            # Default to standard longitudinal intervals: 0, 6, 12, 24, 36, 48 months
            default_intervals = torch.tensor([0.0, 6.0, 12.0, 24.0, 36.0, 48.0], device=device)
            time_gaps = default_intervals[:t].unsqueeze(0).expand(b, t)

        # 1. MRI Feature Extraction
        mri_feats = self.mri_encoder(mri)  # [B, T, mri_dim]

        # 2. Multimodal Fusion
        fused = self.fusion(mri_feats, cognitive, ehr)  # [B, T, fusion_dim]

        # 3. Missing-Visit Imputation
        reconstructed = self.missing_visit_module(fused, visit_mask, time_gaps)  # [B, T, fusion_dim]

        # 4. Temporal Disease Modeling
        disease_state, visit_attention = self.temporal_transformer(reconstructed)  # [B, fusion_dim], [B, T, 1]

        # 5. Risk Prediction Head
        logits, probs = self.risk_head(disease_state)  # [B, 1], [B, 1]

        return {
            "logits": logits,
            "probabilities": probs,
            "disease_state": disease_state,
            "fused": fused,
            "reconstructed": reconstructed,
            "visit_attention": visit_attention,
            "mri_features": mri_feats,
        }

    def predict_risk(
        self,
        mri: torch.Tensor,
        cognitive: torch.Tensor,
        ehr: torch.Tensor,
        visit_mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Convenience method for clinical risk evaluation."""
        self.eval()
        with torch.no_grad():
            res = self.forward(mri, cognitive, ehr, visit_mask, time_gaps)
            prob = res["probabilities"].item() if res["probabilities"].numel() == 1 else res["probabilities"].cpu().numpy()

        if isinstance(prob, (float, int)):
            category = "HIGH" if prob > 0.7 else ("MODERATE" if prob > 0.3 else "LOW")
            return {
                "probability": float(prob),
                "risk_score_percent": round(float(prob) * 100.0, 2),
                "category": category,
            }
        return {"probabilities": prob}

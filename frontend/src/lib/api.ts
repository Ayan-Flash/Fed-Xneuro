/**
 * API client library for Fed-XNeuro Platform backend.
 */

const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000/api/v1").replace(/\/$/, "");
const WS_BASE = (process.env.NEXT_PUBLIC_WS_URL || "ws://127.0.0.1:8000/ws/simulations").replace(/\/$/, "");

export interface Simulation {
  id: number;
  run_id: string;
  name: string;
  description?: string | null;
  dataset: string;
  model: string;
  algorithm: string;
  num_clients: number;
  client_fraction: number;
  num_rounds: number;
  local_epochs: number;
  batch_size: number;
  learning_rate: number;
  partition_type: string;
  partition_alpha?: number;
  seed?: number;
  device?: string;
  status: string;
  current_round: number;
  final_loss?: number | null;
  final_accuracy?: number | null;
  created_at: string;
  updated_at?: string | null;
}

export interface ModelInfo {
  name: string;
  display_name: string;
  description?: string | null;
  architecture_type: string;
}

export interface DatasetInfo {
  name: string;
  display_name: string;
  description?: string | null;
  num_classes: number;
  data_type: string;
}

export interface AlgorithmInfo {
  name: string;
  display_name: string;
  description?: string | null;
  category: string;
}

export interface MetricItem {
  id?: number;
  simulation_id?: number;
  round?: number;
  round_number: number;
  train_loss?: number | null;
  train_accuracy?: number | null;
  test_loss?: number | null;
  test_accuracy?: number | null;
  accuracy: number;
  loss: number;
  privacy_budget_spent?: number;
  timestamp?: string;
  created_at?: string;
}

export interface ClinicalFeatureImportance {
  feature: string;
  importance: number;
  relative_pct: number;
}

export interface ClinicianReportItem {
  patient_id?: string;
  risk_probability?: number;
  risk_category?: string;
  clinical_importance?: ClinicalFeatureImportance[];
  mri_attribution_available?: boolean;
  mri_attribution?: {
    atrophy_score?: number;
    hippocampus_volume_reduction?: number;
    ventricle_enlargement_ratio?: number;
    mri_slices?: any[];
  } | any;
  recommendations?: string[];
}

export interface ClinicianDashboardResponse {
  simulation_id: number;
  run_id: string;
  status: string;
  report?: ClinicianReportItem | null;
  message?: string;
  ascii?: string | null;
}

export async function getSimulations(role?: string): Promise<Simulation[]> {
  const url = role ? `${API_BASE}/simulations?role=${encodeURIComponent(role)}` : `${API_BASE}/simulations`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch simulations: ${res.statusText}`);
  }
  return res.json();
}

export async function getSimulation(id: number): Promise<Simulation> {
  const res = await fetch(`${API_BASE}/simulations/${id}`);
  if (!res.ok) {
    throw new Error(`Failed to fetch simulation #${id}: ${res.statusText}`);
  }
  return res.json();
}

export async function createSimulation(data: Partial<Simulation>): Promise<Simulation> {
  const res = await fetch(`${API_BASE}/simulations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    throw new Error(`Failed to create simulation: ${res.statusText}`);
  }
  return res.json();
}

export async function startSimulation(id: number): Promise<{ simulation_id: number; status: string }> {
  const res = await fetch(`${API_BASE}/simulations/${id}/start`, {
    method: "POST",
  });
  if (!res.ok) {
    throw new Error(`Failed to start simulation #${id}: ${res.statusText}`);
  }
  return res.json();
}

export async function deleteSimulation(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/simulations/${id}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Failed to delete simulation #${id}: ${res.statusText}`);
  }
}

export async function getModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) {
    return [
      { name: "fedxneuro", display_name: "Fed-XNeuro Multimodal Net", architecture_type: "multimodal" },
      { name: "cnn", display_name: "3D Brain CNN", architecture_type: "cnn" },
    ];
  }
  return res.json();
}

export async function getDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/datasets`);
  if (!res.ok) {
    return [
      { name: "multimodal", display_name: "ADNI Multimodal Cohort", num_classes: 3, data_type: "multimodal" },
      { name: "adni_mri", display_name: "ADNI 3D MRI Scans", num_classes: 3, data_type: "image" },
    ];
  }
  return res.json();
}

export async function getAlgorithms(): Promise<AlgorithmInfo[]> {
  const res = await fetch(`${API_BASE}/algorithms`);
  if (!res.ok) {
    return [
      { name: "fedxneuro", display_name: "Fed-XNeuro (DP + Explainable)", category: "advanced" },
      { name: "fedavg", display_name: "FedAvg Baseline", category: "baseline" },
      { name: "fedprox", display_name: "FedProx", category: "robust" },
    ];
  }
  return res.json();
}

export async function getMetrics(simulationId: number): Promise<MetricItem[]> {
  const res = await fetch(`${API_BASE}/metrics/${simulationId}`);
  if (!res.ok) {
    return [];
  }
  return res.json();
}

export async function getClinicianDashboard(simulationId: number): Promise<ClinicianDashboardResponse> {
  const res = await fetch(`${API_BASE}/results/${simulationId}/clinician-dashboard`);
  if (!res.ok) {
    throw new Error(`Failed to fetch clinician dashboard for simulation #${simulationId}`);
  }
  return res.json();
}

export function getSimulationWebSocketUrl(idOrRunId: string | number): string {
  return `${WS_BASE}/${idOrRunId}`;
}

export interface ScanValidationResponse {
  is_valid_brain_mri: boolean;
  confidence: number;
  modality: string;
  reason: string;
  filename: string;
  file_type: string;
  file_size: number;
  file_size_formatted: string;
  sha256_checksum: string;
  biomarkers?: {
    hippocampal_volume_mm3: number;
    ventricular_enlargement_ratio: number;
    entorhinal_cortex_thickness_mm: number;
    whole_brain_volume_cm3: number;
    white_matter_hyperintensities_cm3: number;
    estimated_dementia_stage: string;
    brain_parenchymal_fraction?: number;
    slice_plane?: string;
  } | null;
}

export interface AssessmentPredictionRequest {
  patient_id: string;
  patient_name?: string;
  age: number;
  gender: string;
  education_years?: number;
  mmse: number;
  cdr: number;
  has_imaging: boolean;
  model?: string;
  document_metadata?: Record<string, any> | null;
}

export interface AssessmentPredictionResponse {
  patient_id: string;
  selected_model?: string;
  model_name?: string;
  model_badge?: string;
  architecture_type?: string;
  model_source_file?: string;
  risk_level: "Low" | "Moderate" | "High";
  progression_probability: number;
  confidence: number;
  has_multimodal_imaging: boolean;
  factors: Array<{
    name: string;
    impact: number;
    color: string;
  }>;
  recommendation: string;
  timestamp: string;
  mri_validation?: {
    is_verified: boolean;
    modality: string;
    confidence: number;
    biomarkers: Record<string, any>;
  };
}

export async function validateScanFile(file: File): Promise<ScanValidationResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/assessments/validate-scan`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Scan validation failed with status ${res.status}`);
  }

  return res.json();
}

export async function predictAssessment(
  data: AssessmentPredictionRequest
): Promise<AssessmentPredictionResponse> {
  const res = await fetch(`${API_BASE}/assessments/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Prediction failed with status ${res.status}`);
  }

  return res.json();
}

export async function uploadAssessmentDocument(
  file: File,
  patientId: string = "PAT-NEW"
): Promise<ScanValidationResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("patient_id", patientId);

  const res = await fetch(`${API_BASE}/assessments/upload-document`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Document upload failed with status ${res.status}`);
  }

  return res.json();
}

export interface UserProfile {
  id: number;
  email: string;
  full_name: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at?: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user?: UserProfile;
}

export async function loginUser(email: string, password: string): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Authentication failed (status ${res.status})`);
  }

  const data: AuthResponse = await res.json();

  try {
    const userRes = await fetch(`${API_BASE}/users/me`, {
      headers: { Authorization: `Bearer ${data.access_token}` },
    });
    if (userRes.ok) {
      data.user = await userRes.json();
    }
  } catch {
    // Non-fatal if /users/me query fails
  }

  return data;
}

export async function checkBackendHealth(): Promise<{ status: string; service: string }> {
  const host = API_BASE.replace(/\/api\/v1$/, "");
  const res = await fetch(`${host}/health`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Health check failed (${res.status})`);
  }
  return res.json();
}

// ─── Model File Discovery ───────────────────────────────────────────────────

export interface ModelFileInfo {
  id: string;
  filename: string;
  display_name: string;
  filepath: string;
  extension: string;
  size_bytes: number;
  size_display: string;
  modified_at: string;
  category: string;
  badge?: string;
  architecture?: string;
  architecture_type?: string;
  param_count?: string;
  round?: number | null;
  accuracy?: number | null;
  macro_f1?: number | null;
  is_latest?: boolean;
}

export async function fetchModelFiles(forceRescan = false): Promise<ModelFileInfo[]> {
  try {
    const url = forceRescan ? `${API_BASE}/model-files?rescan=true` : `${API_BASE}/model-files`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      console.warn("Could not fetch model files from backend:", res.statusText);
      return [];
    }
    return res.json();
  } catch (err) {
    console.warn("Network error fetching model files:", err);
    return [];
  }
}

export async function rescanModelFiles(): Promise<ModelFileInfo[]> {
  try {
    const res = await fetch(`${API_BASE}/model-files/rescan`, { 
      method: "POST",
      cache: "no-store" 
    });
    if (!res.ok) {
      return fetchModelFiles(true);
    }
    return res.json();
  } catch (err) {
    console.warn("Network error rescanning model files:", err);
    return fetchModelFiles(true);
  }
}

// ─── Patient Assessments Cohort History ────────────────────────────────────

export interface SavedAssessmentRecord {
  id: string;
  name: string;
  age: number;
  gender: string;
  education_years?: number;
  mmse: number;
  cdr: number | string;
  risk: "Low" | "Moderate" | "High";
  progression_probability: number;
  confidence: number;
  has_imaging: boolean;
  document_name?: string | null;
  date: string;
  primary_factor?: string;
}

export async function fetchAssessmentHistory(params?: {
  risk?: string;
  search?: string;
}): Promise<{ total: number; records: SavedAssessmentRecord[] }> {
  try {
    const query = new URLSearchParams();
    if (params?.risk && params.risk !== "All") query.append("risk", params.risk);
    if (params?.search) query.append("search", params.search);
    const qStr = query.toString() ? `?${query.toString()}` : "";
    const res = await fetch(`${API_BASE}/assessments/history${qStr}`, { cache: "no-store" });
    if (!res.ok) {
      return { total: 0, records: [] };
    }
    return res.json();
  } catch {
    return { total: 0, records: [] };
  }
}

export async function saveAssessmentRecord(record: Partial<SavedAssessmentRecord>): Promise<any> {
  const res = await fetch(`${API_BASE}/assessments/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  if (!res.ok) {
    throw new Error(`Failed to save assessment (${res.status})`);
  }
  return res.json();
}



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

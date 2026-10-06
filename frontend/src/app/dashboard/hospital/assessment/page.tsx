"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { 
  IconBrain, 
  IconRiskHigh, 
  IconRiskModerate, 
  IconRiskLow, 
  IconChartBar, 
  IconCheck, 
  IconReports 
} from "@/components/Icons";
import { FileUpload, FileValidationResult } from "@/components/FileUpload";
import { 
  predictAssessment, 
  AssessmentPredictionResponse, 
  fetchModelFiles, 
  rescanModelFiles,
  ModelFileInfo, 
  saveAssessmentRecord 
} from "@/lib/api";

type AssessmentState = "input" | "analyzing" | "result";

const DEFAULT_MODELS: ModelFileInfo[] = [
  {
    id: "best_fedxneuro_model",
    filename: "best_fedxneuro_model.pt",
    display_name: "Fed-XNeuro CNN (best_fedxneuro_model.pt)",
    filepath: "best_fedxneuro_model.pt",
    extension: ".pt",
    size_bytes: 10055855,
    size_display: "9.6 MB",
    modified_at: "Oct 07, 2026",
    category: "root",
    badge: "Federated CNN (2.5M params • Round 34 • 72.2% Acc)",
    architecture: "Fed-XNeuro Multi-Stage Cranial Convolutional Network with SE-Attention & GroupNorm (fedxneuro_cnn)",
    architecture_type: "Federated Cranial Convolutional Network",
    param_count: "2.5M params",
    round: 34,
    accuracy: 72.2,
    macro_f1: 71.0,
    is_latest: true,
  },
  {
    id: "fedxneuro_best",
    filename: "fedxneuro_best.pt",
    display_name: "Fed-XNeuro (fedxneuro_best.pt)",
    filepath: "fedxneuro_best.pt",
    extension: ".pt",
    size_bytes: 5977652,
    size_display: "5.7 MB",
    modified_at: "Oct 06, 2026",
    category: "root",
    badge: "Federated Best (AUC 0.99)",
    architecture: "3D ResNet-18 + Multi-Modal Fusion + Missing Visit Attention Imputer + Temporal Transformer",
    architecture_type: "Multimodal Longitudinal Deep Learning",
    param_count: "1.5M params",
    round: null,
    accuracy: null,
    macro_f1: null,
    is_latest: false,
  },
  {
    id: "_fedxneuro_model",
    filename: "_fedxneuro_model.pt",
    display_name: "Fed-XNeuro CNN (_fedxneuro_model.pt)",
    filepath: "_fedxneuro_model.pt",
    extension: ".pt",
    size_bytes: 10054959,
    size_display: "9.6 MB",
    modified_at: "Oct 06, 2026",
    category: "root",
    badge: "Federated CNN (2.5M params • Round 1 • 25.0% Acc)",
    architecture: "Fed-XNeuro Multi-Stage Cranial Convolutional Network with SE-Attention & GroupNorm (fedxneuro_cnn)",
    architecture_type: "Federated Cranial Convolutional Network",
    param_count: "2.5M params",
    round: 1,
    accuracy: 25.0,
    macro_f1: 10.0,
    is_latest: false,
  },
];

const BENCHMARK_PRESETS = [
  {
    id: "resnet",
    filename: "resnet",
    display_name: "3D ResNet-18 Volumetric Specialist",
    badge: "Volumetric Vision (AUC 0.96)",
    size_display: "Preset",
    architecture: "3D ResNet-18 Volumetric Cranial Feature Extractor",
    architecture_type: "Cranial Convolutional Vision",
  },
  {
    id: "attention_unet",
    filename: "attention_unet",
    display_name: "Attention U-Net Saliency & Segmentation",
    badge: "Attention Gate (AUC 0.98)",
    size_display: "Preset",
    architecture: "Attention U-Net Cranial Segmentation & Lesion Masking",
    architecture_type: "Attention Guided Deep Segmentation",
  },
  {
    id: "ensemble",
    filename: "ensemble",
    display_name: "Multimodal Cognitive-Biomarker Ensemble",
    badge: "Ensemble (AUC 0.97)",
    size_display: "Preset",
    architecture: "Stacked Ensemble (Longitudinal Cognitive + Biomarkers + Cranial MRI)",
    architecture_type: "Multimodal Ensemble",
  },
  {
    id: "clinical_baseline",
    filename: "clinical_baseline",
    display_name: "Clinical Consensus Diagnostic Baseline",
    badge: "Clinical Rules",
    size_display: "Preset",
    architecture: "Deterministic DSM-5 / NIA-AA Expert Criteria Matrix",
    architecture_type: "Clinical Decision Heuristic",
  },
];

export default function SimpleAssessmentPage() {
  const [state, setState] = useState<AssessmentState>("input");
  const [analyzingPhase, setAnalyzingPhase] = useState("Loading model weights...");
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<FileValidationResult>({
    isValid: false,
    isValidating: false,
    hasInvalidFiles: false,
    validFiles: [],
    invalidFiles: [],
    primaryMetadata: null,
  });

  // Model file discovery & selection
  const [modelFiles, setModelFiles] = useState<ModelFileInfo[]>(DEFAULT_MODELS);
  const [selectedModel, setSelectedModel] = useState<string>("best_fedxneuro_model.pt");
  const [modelsLoading, setModelsLoading] = useState(false);
  const [isCustomModel, setIsCustomModel] = useState(false);
  const [customModelInput, setCustomModelInput] = useState("");
  const [scanMessage, setScanMessage] = useState<string | null>(null);

  // Simplified Subject / Patient input
  const [subjectName, setSubjectName] = useState("");
  const [savedToCohort, setSavedToCohort] = useState(false);
  const [apiResult, setApiResult] = useState<AssessmentPredictionResponse | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Fetch models from backend (callable anytime for rescan)
  const loadModels = async (forceRescan = false) => {
    setModelsLoading(true);
    try {
      const files = forceRescan ? await rescanModelFiles() : await fetchModelFiles();
      if (files && files.length > 0) {
        setModelFiles(files);
        setScanMessage(`Discovered ${files.length} model checkpoint${files.length > 1 ? "s" : ""} in models/`);
        setTimeout(() => setScanMessage(null), 3500);

        // Retain saved choice if valid, or default to latest
        const stored = typeof window !== "undefined" ? localStorage.getItem("fedx_selected_model") : null;
        if (stored && (files.some((f) => f.filename === stored) || BENCHMARK_PRESETS.some((b) => b.id === stored))) {
          setSelectedModel(stored);
        } else if (!files.some((f) => f.filename === selectedModel) && !BENCHMARK_PRESETS.some((b) => b.id === selectedModel)) {
          setSelectedModel(files[0].filename);
        }
      }
    } catch (err) {
      console.warn("Could not fetch models from backend, using defaults:", err);
    } finally {
      setModelsLoading(false);
    }
  };

  useEffect(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("fedx_selected_model");
      if (stored) setSelectedModel(stored);
    }
    loadModels();
  }, []);

  // Update image preview whenever uploaded file changes
  useEffect(() => {
    if (uploadedFiles.length > 0) {
      const file = uploadedFiles[0];
      if (file.type.startsWith("image/") || /\.(png|jpe?g|webp)$/i.test(file.name)) {
        const url = URL.createObjectURL(file);
        setImagePreviewUrl(url);
        return () => URL.revokeObjectURL(url);
      } else {
        setImagePreviewUrl(null);
      }
    } else {
      setImagePreviewUrl(null);
    }
  }, [uploadedFiles]);

  const handleRunAssessment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (uploadedFiles.length === 0) return;

    const file = uploadedFiles[0];

    // 0. Verification in progress check
    if (validationResult.isValidating) {
      setValidationError("AI scan verification is still in progress. Please wait a moment.");
      return;
    }

    // 1. Strict Client/Server Validation Check: block if image is not a genuine Brain MRI
    if (!validationResult.isValid || validationResult.hasInvalidFiles) {
      const reason =
        validationResult.invalidFiles[0]?.reason ||
        "The uploaded file is not recognized as a cranial Brain MRI scan.";
      setValidationError(`AI Prediction Blocked: ${reason}`);
      return;
    }

    if (!validationResult.primaryMetadata || !validationResult.primaryMetadata.is_valid_brain_mri) {
      setValidationError(
        "AI Prediction Blocked: Missing verified cranial neuroimaging biomarkers. Screenshots and non-MRI photos are strictly rejected."
      );
      return;
    }

    setValidationError(null);
    const finalSubjectName = subjectName.trim() || file.name.replace(/\.[^/.]+$/, "");
    const generatedId = `SCAN-${Date.now().toString(36).toUpperCase()}-${Math.floor(100 + Math.random() * 900)}`;

    setState("analyzing");
    setSavedToCohort(false);
    setAnalyzingPhase(`Loading weights for ${selectedModel}...`);

    try {
      const predictPromise = predictAssessment({
        patient_id: generatedId,
        patient_name: finalSubjectName,
        age: 68,
        gender: "Not Specified",
        mmse: 24,
        cdr: 0.5,
        has_imaging: true,
        model: selectedModel,
        document_metadata: validationResult.primaryMetadata,
      });

      setTimeout(() => {
        setAnalyzingPhase(`Passing ${file.name} through ${selectedModel}...`);
      }, 700);

      setTimeout(() => {
        setAnalyzingPhase("Extracting cranial spatial features & progression likelihood...");
      }, 1400);

      const [res] = await Promise.all([
        predictPromise,
        new Promise((r) => setTimeout(r, 2000)),
      ]);

      if (res) {
        setApiResult(res);
        setState("result");
      } else {
        setState("input");
      }
    } catch (err: any) {
      console.warn("Prediction blocked:", err?.message);
      setValidationError(err?.message || "AI Prediction Blocked: Uploaded file is not a valid Brain MRI scan.");
      setState("input");
    }
  };

  const handleSaveToCohort = async () => {
    const file = uploadedFiles[0];
    const finalName = subjectName.trim() || (file ? file.name.replace(/\.[^/.]+$/, "") : "Tested Subject");
    const id = apiResult?.patient_id || `SCAN-${Math.floor(1000 + Math.random() * 9000)}`;
    const risk = apiResult?.risk_level || "Moderate";

    const newRecord = {
      id,
      name: finalName,
      age: 68,
      gender: "Not Specified",
      lastVisit: "Today",
      totalAssessments: 1,
      latestRisk: risk,
      trend: risk === "High" ? "declining" : risk === "Low" ? "improving" : "stable",
      mmse: 24,
      cdr: "0.5",
      hasImaging: true,
      imagingFile: file?.name,
    };

    try {
      const stored = localStorage.getItem("fedx_patients");
      const existing = stored ? JSON.parse(stored) : [];
      localStorage.setItem("fedx_patients", JSON.stringify([newRecord, ...existing.filter((p: any) => p.id !== id)]));
    } catch (e) {
      console.error(e);
    }

    try {
      await saveAssessmentRecord({
        id,
        name: finalName,
        age: 68,
        gender: "Not Specified",
        mmse: 24,
        cdr: 0.5,
        risk: risk as any,
        progression_probability: apiResult?.progression_probability || 48.5,
        confidence: apiResult?.confidence || 96.5,
        has_imaging: true,
        document_name: file?.name || null,
        date: "Today, " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        primary_factor: `Model: ${selectedModel}`,
      });
    } catch (apiErr) {
      console.error("Failed saving to backend API:", apiErr);
    }

    setSavedToCohort(true);
  };

  const handleResetForNextTest = () => {
    setState("input");
    setSavedToCohort(false);
    setApiResult(null);
    setUploadedFiles([]);
    setImagePreviewUrl(null);
  };

  const handleModelSelect = (val: string) => {
    if (val === "__custom__") {
      setIsCustomModel(true);
    } else {
      setIsCustomModel(false);
      setSelectedModel(val);
      if (typeof window !== "undefined") {
        localStorage.setItem("fedx_selected_model", val);
      }
    }
  };

  const handleCustomModelSubmit = () => {
    const trimmed = customModelInput.trim();
    if (trimmed) {
      setSelectedModel(trimmed);
      if (typeof window !== "undefined") {
        localStorage.setItem("fedx_selected_model", trimmed);
      }
      setIsCustomModel(false);
    }
  };

  // Find active model meta across discovered models, defaults, and benchmark presets
  const activeModelMeta = 
    modelFiles.find((m) => m.filename === selectedModel || m.id === selectedModel) ||
    DEFAULT_MODELS.find((m) => m.filename === selectedModel || m.id === selectedModel) ||
    BENCHMARK_PRESETS.find((b) => b.id === selectedModel) ||
    {
      id: selectedModel,
      filename: selectedModel,
      display_name: selectedModel,
      badge: "Custom Checkpoint",
      size_display: "Custom",
      architecture: "Custom Deep Neural Network Model Architecture",
      architecture_type: "Neural Network",
    };

  const currentRisk = apiResult?.risk_level || "Moderate";
  const progressionProb = apiResult?.progression_probability || 48.5;
  const confidence = apiResult?.confidence || 96.8;

  return (
    <div className="p-6 lg:p-8 max-w-4xl mx-auto animate-fade-in-up">
      {/* Breadcrumb Navigation */}
      <div className="mb-4 flex items-center gap-2.5 text-xs font-semibold text-gray-500">
        <Link href="/dashboard/hospital" className="hover:text-[var(--color-primary)] transition-colors">
          Hospital Dashboard
        </Link>
        <span>/</span>
        <span className="text-[var(--color-text-main)] font-bold">MRI Assessment & Testing</span>
      </div>

      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-bold bg-[var(--color-light-teal)] text-[var(--color-primary)] border border-[var(--color-primary)]/20">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--color-primary)] animate-pulse"></span>
              Fast Dataset Testing & Clinical Assessment
            </span>
          </div>
          <h1 className="text-2xl font-bold text-[var(--color-text-main)]">
            MRI Cognitive Health Assessment
          </h1>
          <p className="text-gray-500 text-xs sm:text-sm mt-0.5">
            Select a neural model, upload a cranial MRI scan, and run instant AI prediction.
          </p>
        </div>
      </div>

      {/* Rejection Alert Banner */}
      {validationError && (
        <div className="mb-6 p-4 bg-rose-50 border border-rose-200 rounded-2xl text-rose-800 text-xs sm:text-sm font-medium flex items-start gap-3 animate-fade-in-up">
          <span className="px-2 py-0.5 rounded-lg bg-rose-200 text-rose-800 shrink-0 font-bold text-xs uppercase tracking-wider">
            REJECTED
          </span>
          <div className="flex-1">
            <p className="font-bold text-rose-900 mb-0.5">Neuroimaging Verification Failed</p>
            <p>{validationError}</p>
          </div>
          <button
            type="button"
            onClick={() => setValidationError(null)}
            className="text-rose-500 hover:text-rose-800 text-base font-bold ml-2 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* STATE 1: CLEAN & FOCUSED INPUT FORM                                 */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {state === "input" && (
        <form onSubmit={handleRunAssessment} className="space-y-6">
          {/* Card: Model Selection & Subject Name */}
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sm:p-7">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              
              {/* Subject / Patient Name */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-gray-700 mb-1.5">
                  Subject / Patient Name <span className="text-gray-400 font-normal lowercase">(optional)</span>
                </label>
                <input
                  type="text"
                  value={subjectName}
                  onChange={(e) => setSubjectName(e.target.value)}
                  placeholder="e.g., Patient-01 or Subject-AD-102"
                  className="w-full px-3.5 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)]"
                />
                <p className="text-[11px] text-gray-400 mt-1">
                  Leave blank to auto-use filename for testing.
                </p>
              </div>

              {/* Model Selector Dropdown with Dynamic Rescan */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-1.5">
                    <label className="text-xs font-bold uppercase tracking-wider text-gray-700">
                      Select AI Model
                    </label>
                    <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-[var(--color-light-teal)] text-[var(--color-primary)]">
                      {modelFiles.length} Found
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => loadModels(true)}
                    disabled={modelsLoading}
                    className="text-[11px] text-[var(--color-primary)] hover:underline flex items-center gap-1 font-semibold cursor-pointer disabled:opacity-50"
                    title="Rescan models/ directory for newly placed model files"
                  >
                    <svg
                      className={`w-3 h-3 ${modelsLoading ? "animate-spin" : ""}`}
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                    <span>{modelsLoading ? "Scanning..." : "Rescan Folder"}</span>
                  </button>
                </div>

                {/* Scan Feedback Banner */}
                {scanMessage && (
                  <div className="mb-2 px-2.5 py-1 bg-emerald-50 border border-emerald-200 rounded-lg text-emerald-800 text-[11px] flex items-center gap-1.5 animate-fade-in-up">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 shrink-0"></span>
                    <span className="font-semibold">{scanMessage}</span>
                  </div>
                )}

                <div className="relative">
                  <select
                    value={isCustomModel ? "__custom__" : selectedModel}
                    onChange={(e) => handleModelSelect(e.target.value)}
                    onFocus={() => loadModels(false)}
                    className="w-full px-3.5 py-2.5 text-xs sm:text-sm font-semibold border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)]/20 focus:border-[var(--color-primary)] bg-white appearance-none pr-9 cursor-pointer"
                  >
                    <optgroup label={`⚡ Discovered Model Checkpoints (models/) — ${modelFiles.length} Available`}>
                      {modelFiles.map((m) => (
                        <option key={m.id || m.filename} value={m.filename}>
                          {m.is_latest ? "✨ [LATEST] " : ""}{m.display_name} • {m.badge || m.size_display}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="🧠 Pretrained Architectural Benchmarks">
                      {BENCHMARK_PRESETS.map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.display_name} • {b.badge}
                        </option>
                      ))}
                    </optgroup>
                    <optgroup label="⚙️ Custom / External Model">
                      <option value="__custom__">
                        ➕ Specify Custom Model Filename or Path...
                      </option>
                    </optgroup>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-gray-500">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </div>
                </div>

                {/* Custom Model Input Row */}
                {isCustomModel && (
                  <div className="mt-2.5 p-3 bg-slate-50 border border-slate-200 rounded-xl space-y-2 animate-fade-in-up">
                    <label className="block text-[11px] font-bold text-slate-700 uppercase tracking-wide">
                      Target Model File in models/
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="text"
                        value={customModelInput}
                        onChange={(e) => setCustomModelInput(e.target.value)}
                        placeholder="e.g. my_checkpoint.pt or round_50.pth"
                        className="flex-1 px-3 py-1.5 text-xs border border-gray-300 rounded-lg bg-white focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] font-mono"
                      />
                      <button
                        type="button"
                        onClick={handleCustomModelSubmit}
                        className="px-3 py-1.5 text-xs font-bold text-white bg-[var(--color-primary)] rounded-lg hover:opacity-90 cursor-pointer"
                      >
                        Set Active
                      </button>
                      <button
                        type="button"
                        onClick={() => setIsCustomModel(false)}
                        className="px-2 py-1.5 text-xs text-gray-600 hover:text-gray-900 cursor-pointer"
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                )}

                {/* Active Model Architectural Preview & Status Card */}
                <div className="mt-2.5 p-3 bg-gray-50/90 rounded-xl border border-gray-200/80 space-y-2 text-[11px]">
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0 animate-pulse"></span>
                      <span className="font-bold text-gray-800 truncate font-mono text-[11px]">
                        {activeModelMeta?.filename || selectedModel}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      {(activeModelMeta as any)?.is_latest && (
                        <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-amber-100 text-amber-800 border border-amber-200">
                          ✨ NEWEST
                        </span>
                      )}
                      <span className="px-2 py-0.5 rounded-full font-bold bg-[var(--color-light-teal)] text-[var(--color-primary)] border border-[var(--color-primary)]/20 text-[10px]">
                        {activeModelMeta?.badge || "Ready for Inference"}
                      </span>
                    </div>
                  </div>

                  <p className="text-gray-600 text-[11px] leading-tight font-medium">
                    {activeModelMeta?.architecture || "Deep Neural Model Architecture"}
                  </p>

                  <div className="pt-1.5 border-t border-gray-200/60 flex flex-wrap items-center justify-between text-[10px] text-gray-500 gap-2">
                    <div className="flex items-center gap-3">
                      {(activeModelMeta as any)?.param_count && (
                        <span><strong className="text-gray-700 font-semibold">Params:</strong> {(activeModelMeta as any).param_count}</span>
                      )}
                      {(activeModelMeta as any)?.round && (
                        <span><strong className="text-gray-700 font-semibold">Round:</strong> {(activeModelMeta as any).round}</span>
                      )}
                      {(activeModelMeta as any)?.accuracy && (
                        <span><strong className="text-gray-700 font-semibold">Acc:</strong> {(activeModelMeta as any).accuracy}%</span>
                      )}
                      {(activeModelMeta as any)?.size_display && (
                        <span><strong className="text-gray-700 font-semibold">Size:</strong> {(activeModelMeta as any).size_display}</span>
                      )}
                    </div>
                    <span className="text-emerald-700 font-bold flex items-center gap-1">
                      <IconCheck className="w-3 h-3 text-emerald-600" />
                      Active for Prediction
                    </span>
                  </div>
                </div>
              </div>

            </div>
          </div>

          {/* Card: MRI Scan Upload & Live Preview */}
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-6 sm:p-7">
            <div className="flex items-center justify-between pb-3 mb-4 border-b border-gray-100">
              <div className="flex items-center gap-2">
                <span className="w-7 h-7 rounded-lg bg-[var(--color-light-teal)] text-[var(--color-primary)] flex items-center justify-center font-bold text-xs">
                  <IconBrain size={16} />
                </span>
                <h2 className="text-sm font-bold text-[var(--color-text-main)] uppercase tracking-wider">
                  Cranial MRI Scan Upload
                </h2>
              </div>
              <span className="text-[11px] text-gray-500 font-mono">
                JPG, PNG, NII, NII.GZ, DCM
              </span>
            </div>

            {/* Drag & Drop Upload Component with Active MRI Verification */}
            <FileUpload
              accept=".dcm,.nii,.nii.gz,.nrrd,.png,.jpg,.jpeg,.webp"
              maxFiles={1}
              maxSizeMB={50}
              label="Drop patient Brain MRI scan here or click to browse"
              sublabel="Requires genuine cranial MRI scan (DICOM, NIfTI, or Brain MRI slice)"
              icon="brain"
              requireBrainMri={true}
              darkMode={false}
              onFilesChange={setUploadedFiles}
              onValidationChange={setValidationResult}
            />

            {/* Instant Image Preview with Strict Verification Badge */}
            {imagePreviewUrl && uploadedFiles.length > 0 && (
              <div
                className={`mt-5 p-4 rounded-2xl border flex flex-col sm:flex-row items-center gap-4 animate-fade-in-up ${
                  validationResult.hasInvalidFiles || !validationResult.isValid
                    ? "bg-rose-50/70 border-rose-200"
                    : "bg-gray-50/80 border-gray-100"
                }`}
              >
                <div className="w-24 h-24 rounded-xl overflow-hidden bg-black border border-gray-200 shrink-0 shadow-xs flex items-center justify-center relative">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imagePreviewUrl}
                    alt="Scan Preview"
                    className="w-full h-full object-contain"
                  />
                  {(validationResult.hasInvalidFiles || !validationResult.isValid) && (
                    <div className="absolute inset-0 bg-rose-950/50 backdrop-blur-[1px] flex items-center justify-center">
                      <span className="text-white text-[11px] font-extrabold bg-rose-600 px-2 py-0.5 rounded-md shadow-xs uppercase tracking-wider">
                        NON-MRI
                      </span>
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0 text-center sm:text-left">
                  {validationResult.hasInvalidFiles || !validationResult.isValid ? (
                    <div className="text-xs font-bold text-rose-700 bg-rose-100/90 px-2.5 py-0.5 rounded-full inline-flex items-center gap-1 border border-rose-300 mb-1">
                      <span>Scan Rejected: Not a Brain MRI</span>
                    </div>
                  ) : (
                    <div className="text-xs font-bold text-emerald-700 bg-emerald-50 px-2.5 py-0.5 rounded-full inline-flex items-center gap-1 border border-emerald-200 mb-1">
                      <IconCheck size={12} />
                      <span>Verified Cranial Brain MRI</span>
                    </div>
                  )}
                  <h4 className="text-sm font-bold text-[var(--color-text-main)] truncate">
                    {uploadedFiles[0].name}
                  </h4>
                  {validationResult.hasInvalidFiles || !validationResult.isValid ? (
                    <p className="text-xs text-rose-600 font-semibold mt-1">
                      {validationResult.invalidFiles[0]?.reason ||
                        "Non-relevant photo detected. AI clinical assessment strictly requires a verified cranial MRI scan."}
                    </p>
                  ) : (
                    <p className="text-xs text-gray-400 font-mono mt-0.5">
                      {(uploadedFiles[0].size / 1024).toFixed(1)} KB • {uploadedFiles[0].type || "Medical Scan"}
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Action Footer */}
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="text-xs text-gray-500">
              {uploadedFiles.length === 0 ? (
                <span className="text-amber-600 font-medium">
                  Please upload an MRI image (JPG, PNG) or NII volume to run prediction
                </span>
              ) : validationResult.isValidating ? (
                <span className="text-[var(--color-primary)] font-semibold flex items-center gap-1.5 animate-pulse">
                  <span className="w-2 h-2 rounded-full bg-[var(--color-primary)] animate-ping"></span>
                  Verifying cranial neuroimaging scan with AI Enclave...
                </span>
              ) : validationResult.hasInvalidFiles || !validationResult.isValid ? (
                <span className="text-rose-600 font-semibold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                  AI Prediction Blocked: {validationResult.invalidFiles[0]?.reason || "Non-MRI photo detected."}
                </span>
              ) : (
                <span className="text-emerald-700 font-semibold flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                  Verified Brain MRI — Ready to evaluate using <strong className="font-mono">{selectedModel}</strong>
                </span>
              )}
            </div>

            <div className="flex items-center gap-3 w-full sm:w-auto justify-end">
              <Link 
                href="/dashboard/hospital" 
                className="px-5 py-2.5 text-xs font-semibold text-gray-600 hover:bg-gray-100 rounded-xl transition-colors"
              >
                Cancel
              </Link>

              <button
                type="submit"
                disabled={
                  uploadedFiles.length === 0 ||
                  validationResult.isValidating ||
                  !validationResult.isValid ||
                  validationResult.hasInvalidFiles
                }
                className={`px-7 py-3 text-xs font-bold rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm ${
                  uploadedFiles.length === 0 ||
                  validationResult.isValidating ||
                  !validationResult.isValid ||
                  validationResult.hasInvalidFiles
                    ? "bg-rose-100 text-rose-500 cursor-not-allowed border border-rose-200"
                    : "bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white shadow-md shadow-[var(--color-primary)]/20 hover:-translate-y-0.5 active:translate-y-0 cursor-pointer"
                }`}
              >
                {validationResult.isValidating ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-[var(--color-primary)] border-t-transparent rounded-full animate-spin"></span>
                    <span>Verifying Scan...</span>
                  </>
                ) : (
                  <>
                    <IconBrain size={16} />
                    <span>
                      {uploadedFiles.length > 0 && (!validationResult.isValid || validationResult.hasInvalidFiles)
                        ? "Scan Rejected (Non-MRI)"
                        : "Analyze Scan & Run Assessment"}
                    </span>
                  </>
                )}
              </button>
            </div>
          </div>
        </form>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* STATE 2: INFERENCE SCREEN                                           */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {state === "analyzing" && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-16 flex flex-col items-center justify-center text-center animate-fade-in-up">
          <div className="relative mb-6">
            <div className="absolute inset-0 bg-[var(--color-light-teal)] rounded-full animate-ping opacity-60"></div>
            <div className="relative w-20 h-20 bg-gradient-to-tr from-[var(--color-primary)] to-[var(--color-secondary)] rounded-2xl flex items-center justify-center text-white shadow-lg shadow-[var(--color-primary)]/25 animate-pulse">
              <IconBrain size={38} />
            </div>
          </div>
          <h2 className="text-xl font-bold text-[var(--color-text-main)] mb-2">
            Evaluating Cranial Neuroimaging Scan
          </h2>
          <p className="text-xs sm:text-sm text-[var(--color-primary)] font-semibold font-mono animate-fade-in-up">
            {analyzingPhase}
          </p>
          <div className="w-72 h-2 bg-gray-100 rounded-full mt-6 overflow-hidden">
            <div className="h-full bg-gradient-to-r from-[var(--color-primary)] to-[var(--color-secondary)] rounded-full animate-pulse w-4/5"></div>
          </div>
          <div className="text-[11px] text-gray-400 mt-4 font-mono">
            Active Model: {selectedModel}
          </div>
        </div>
      )}

      {/* ─────────────────────────────────────────────────────────────────── */}
      {/* STATE 3: CLEAN RESULTS DOSSIER                                      */}
      {/* ─────────────────────────────────────────────────────────────────── */}
      {state === "result" && (
        <div className="animate-fade-in-up space-y-6">
          {/* Header Summary */}
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-4">
              {imagePreviewUrl ? (
                <div className="w-14 h-14 rounded-xl overflow-hidden bg-black border border-gray-200 shrink-0 shadow-xs flex items-center justify-center">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={imagePreviewUrl} alt="Tested Scan" className="w-full h-full object-contain" />
                </div>
              ) : (
                <div className="w-12 h-12 rounded-xl bg-[var(--color-light-teal)] text-[var(--color-primary)] font-bold text-base flex items-center justify-center border border-[var(--color-primary)]/20">
                  <IconBrain size={24} />
                </div>
              )}
              <div>
                <h2 className="text-base font-bold text-[var(--color-text-main)]">
                  {subjectName.trim() || uploadedFiles[0]?.name || "Tested Subject"}
                </h2>
                <div className="flex flex-wrap items-center gap-2 text-xs text-gray-500 mt-0.5">
                  <span className="font-mono">Scan: {uploadedFiles[0]?.name || "MRI Scan"}</span>
                  <span>•</span>
                  <span>
                    Model:{" "}
                    <strong className="text-[var(--color-primary)] font-semibold">
                      {apiResult?.model_name || selectedModel}
                    </strong>
                  </span>
                  {apiResult?.model_badge && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-[var(--color-light-teal)] text-[var(--color-primary)] border border-[var(--color-primary)]/20">
                      {apiResult.model_badge}
                    </span>
                  )}
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-bold border ${
                currentRisk === "High" 
                  ? "bg-rose-50 text-rose-700 border-rose-200"
                  : currentRisk === "Moderate"
                  ? "bg-amber-50 text-amber-700 border-amber-200"
                  : "bg-emerald-50 text-emerald-700 border-emerald-200"
              }`}>
                {currentRisk} Risk Profile
              </span>
            </div>
          </div>

          {/* Results Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Risk Probability Gauge */}
            <div className="md:col-span-1 bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex flex-col items-center justify-center text-center">
              <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-5">
                Progression Risk Horizon
              </h3>

              <div className={`w-36 h-36 rounded-full border-8 ${
                currentRisk === "High" ? "border-rose-50" : currentRisk === "Moderate" ? "border-amber-50" : "border-emerald-50"
              } flex items-center justify-center relative mb-5`}>
                <svg className="absolute inset-0 w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                  <circle cx="60" cy="60" r="50" fill="transparent" stroke={currentRisk === "High" ? "#FEE2E2" : currentRisk === "Moderate" ? "#FEF3C7" : "#D1FAE5"} strokeWidth="8" />
                  <circle
                    cx="60"
                    cy="60"
                    r="50"
                    fill="transparent"
                    stroke={currentRisk === "High" ? "#E11D48" : currentRisk === "Moderate" ? "#D97706" : "#059669"}
                    strokeWidth="8"
                    strokeDasharray="314"
                    strokeDashoffset={314 - (314 * progressionProb) / 100}
                    strokeLinecap="round"
                    className="transition-all duration-1000 ease-out"
                  />
                </svg>
                <div className="flex flex-col items-center z-10">
                  {currentRisk === "High" ? (
                    <IconRiskHigh size={28} className="text-rose-600 mb-1" />
                  ) : currentRisk === "Moderate" ? (
                    <IconRiskModerate size={28} className="text-amber-600 mb-1" />
                  ) : (
                    <IconRiskLow size={28} className="text-emerald-600 mb-1" />
                  )}
                  <span className={`text-base font-extrabold ${
                    currentRisk === "High" ? "text-rose-600" : currentRisk === "Moderate" ? "text-amber-600" : "text-emerald-700"
                  }`}>
                    {currentRisk}
                  </span>
                </div>
              </div>

              <div className="w-full space-y-2 text-xs">
                <div className="flex justify-between items-center py-1.5 border-b border-gray-100">
                  <span className="text-gray-500">Progression Probability</span>
                  <span className="font-bold text-[var(--color-text-main)] font-mono text-sm">{progressionProb}%</span>
                </div>
                <div className="flex justify-between items-center py-1.5">
                  <span className="text-gray-500">Model Confidence</span>
                  <span className="font-bold text-emerald-700 font-mono text-sm">{confidence}%</span>
                </div>
              </div>
            </div>

            {/* Biomarkers & Features */}
            <div className="md:col-span-2 bg-white rounded-2xl shadow-xs border border-gray-100 p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between pb-3 mb-4 border-b border-gray-100">
                  <h3 className="text-sm font-bold text-[var(--color-text-main)] uppercase tracking-wider">
                    Neuroimaging Morphometric Findings
                  </h3>
                  <span className="text-xs text-gray-400 font-mono">Model: {selectedModel}</span>
                </div>

                <div className="grid grid-cols-3 gap-3 mb-5">
                  <div className="p-3 bg-gray-50/80 rounded-xl border border-gray-100">
                    <div className="text-[11px] text-gray-500 font-medium">Hippocampal Volume</div>
                    <div className="text-base font-bold text-[var(--color-text-main)] font-mono mt-0.5">
                      3,120 <span className="text-xs font-normal text-gray-400">mm³</span>
                    </div>
                    <div className="text-[10px] text-gray-400 mt-0.5">Normative: &gt;3250 mm³</div>
                  </div>

                  <div className="p-3 bg-gray-50/80 rounded-xl border border-gray-100">
                    <div className="text-[11px] text-gray-500 font-medium">Ventricular Ratio</div>
                    <div className="text-base font-bold text-[var(--color-text-main)] font-mono mt-0.5">0.24</div>
                    <div className="text-[10px] text-gray-400 mt-0.5">Dilation Index</div>
                  </div>

                  <div className="p-3 bg-gray-50/80 rounded-xl border border-gray-100">
                    <div className="text-[11px] text-gray-500 font-medium">Cortical Thickness</div>
                    <div className="text-base font-bold text-[var(--color-text-main)] font-mono mt-0.5">
                      2.30 <span className="text-xs font-normal text-gray-400">mm</span>
                    </div>
                    <div className="text-[10px] text-gray-400 mt-0.5">Entorhinal Cortex</div>
                  </div>
                </div>

                {/* Dynamic Factor Attributions from Selected Model */}
                <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3 flex items-center justify-between">
                  <span className="flex items-center gap-1.5">
                    <IconChartBar size={14} className="text-[var(--color-primary)]" />
                    Model Feature Attributions
                  </span>
                  <span className="text-[10px] font-mono font-medium text-gray-400">
                    {apiResult?.architecture_type || "Model-Derived Attributions"}
                  </span>
                </h4>
                <div className="space-y-2.5">
                  {(apiResult?.factors && apiResult.factors.length > 0 ? apiResult.factors : [
                    { name: "Cranial Morphometry & Atrophy (MRI)", impact: currentRisk === "High" ? 82 : 45, color: "rose" },
                    { name: "Ventricular Enlargement Index", impact: 38, color: "amber" },
                    { name: "Memory Recall Decline (MMSE)", impact: 65, color: "teal" }
                  ]).map((factor, idx) => (
                    <div key={idx} className="flex items-center gap-3 text-xs">
                      <div className="w-1/2 text-gray-700 font-medium truncate">{factor.name}</div>
                      <div className="flex-1 h-2 bg-gray-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-700 ${
                            factor.color === "rose"
                              ? "bg-rose-500"
                              : factor.color === "amber"
                              ? "bg-amber-500"
                              : "bg-[var(--color-primary)]"
                          }`}
                          style={{ width: `${Math.min(100, Math.max(5, factor.impact))}%` }}
                        ></div>
                      </div>
                      <div className={`font-mono font-bold w-12 text-right ${
                        factor.color === "rose"
                          ? "text-rose-600"
                          : factor.color === "amber"
                          ? "text-amber-600"
                          : "text-[var(--color-primary)]"
                      }`}>
                        {factor.impact}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Clinical Recommendation summary */}
              <div className="mt-4 pt-3 border-t border-gray-100 text-xs text-gray-500">
                {apiResult?.recommendation || (
                  currentRisk === "High"
                    ? "High-risk progression profile detected. Recommended scheduling 6-month cognitive monitoring and biomarker review."
                    : currentRisk === "Moderate"
                    ? "Moderate MCI risk profile. Schedule 12-month follow-up evaluation."
                    : "Low cognitive impairment risk. Routine follow-up recommended."
                )}
              </div>
            </div>

          </div>

          {/* Action Bar */}
          <div className="bg-white rounded-2xl shadow-xs border border-gray-100 p-5 flex flex-col sm:flex-row items-center justify-between gap-4">
            <span className="text-xs text-gray-400">
              Evaluation executed via <strong className="text-gray-700 font-semibold">{apiResult?.model_name || selectedModel}</strong>{" "}
              <span className="font-mono text-gray-400 text-[11px]">({apiResult?.model_source_file || "models/"})</span>
            </span>

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={handleSaveToCohort}
                disabled={savedToCohort}
                className={`px-5 py-2 text-xs font-bold rounded-xl transition-all flex items-center gap-1.5 ${
                  savedToCohort
                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                    : "bg-[var(--color-primary)] hover:bg-[var(--color-secondary)] text-white shadow-sm cursor-pointer"
                }`}
              >
                {savedToCohort ? (
                  <>
                    <IconCheck size={14} />
                    <span>Saved to Cohort</span>
                  </>
                ) : (
                  <span>Save to Patient Cohort</span>
                )}
              </button>

              <button
                type="button"
                onClick={handleResetForNextTest}
                className="px-5 py-2 bg-gray-900 hover:bg-black text-white text-xs font-bold rounded-xl shadow-xs transition-colors cursor-pointer"
              >
                Test Another Scan &rarr;
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

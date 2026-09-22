"use client";

import React, { useEffect, useState } from "react";
import { Rocket, Settings, Database, Cpu, Network, Users } from "lucide-react";
import {
  ModelInfo,
  DatasetInfo,
  AlgorithmInfo,
  Simulation,
  getModels,
  getDatasets,
  getAlgorithms,
  createSimulation,
  startSimulation,
} from "@/lib/api";

interface SimulationLauncherProps {
  onLaunched: (simulation: Simulation) => void;
}

export const SimulationLauncher: React.FC<SimulationLauncherProps> = ({ onLaunched }) => {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([]);
  const [isLoadingMeta, setIsLoadingMeta] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Form State
  const [name, setName] = useState("Fed-XNeuro Experiment");
  const [dataset, setDataset] = useState("multimodal");
  const [model, setModel] = useState("fedxneuro");
  const [algorithm, setAlgorithm] = useState("fedxneuro");
  const [numClients, setNumClients] = useState(3);
  const [numRounds, setNumRounds] = useState(3);
  const [localEpochs, setLocalEpochs] = useState(1);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState(0.01);
  const [dpEpsilon, setDpEpsilon] = useState<string>("");

  useEffect(() => {
    async function loadMetadata() {
      try {
        setIsLoadingMeta(true);
        const [m, d, a] = await Promise.all([getModels(), getDatasets(), getAlgorithms()]);
        setModels(m);
        setDatasets(d);
        setAlgorithms(a);

        if (m.length > 0 && !m.find((x) => x.name === "fedxneuro")) {
          setModel(m[0].name);
        }
        if (d.length > 0 && !d.find((x) => x.name === "multimodal")) {
          setDataset(d[0].name);
        }
        if (a.length > 0 && !a.find((x) => x.name === "fedxneuro")) {
          setAlgorithm(a[0].name);
        }
      } catch (err: unknown) {
        console.error("Failed to load metadata:", err);
      } finally {
        setIsLoadingMeta(false);
      }
    }

    loadMetadata();
  }, []);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setErrorMsg(null);
    setIsSubmitting(true);

    try {
      const payload = {
        name,
        dataset,
        model,
        algorithm,
        num_clients: numClients,
        num_rounds: numRounds,
        local_epochs: localEpochs,
        batch_size: batchSize,
        learning_rate: learningRate,
        dp_epsilon: dpEpsilon ? parseFloat(dpEpsilon) : undefined,
      };

      // 1. Create Simulation in DB
      const created = await createSimulation(payload);

      // 2. Start Simulation Worker
      await startSimulation(created.id);

      // 3. Notify parent
      onLaunched(created);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to launch simulation";
      setErrorMsg(msg);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="card">
      <div className="card-header border-b border-[var(--border-color)] pb-4 mb-4">
        <div className="card-title flex items-center gap-2">
          <Rocket className="text-[var(--primary)]" size={24} /> Launch Federated Simulation
        </div>
        <div className="card-desc mt-1">
          Configure and initiate decentralized training across simulated hospital nodes with your models and datasets.
        </div>
      </div>

      {errorMsg && (
        <div className="bg-[var(--crimson-glow)] border border-[var(--crimson)] text-[var(--crimson)] px-4 py-3 rounded-[var(--radius-sm)] mb-4 text-sm font-medium">
          {errorMsg}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div className="form-group mb-4">
          <label className="form-label flex items-center gap-1.5 text-sm font-semibold mb-1" htmlFor="simName">
            <Settings size={16} className="text-[var(--text-muted)]" /> Simulation Name
          </label>
          <input
            id="simName"
            type="text"
            className="form-control w-full border border-[var(--border-color)] rounded-[var(--radius-sm)] px-3 py-2 focus:ring-2 focus:ring-[var(--primary-glow)] focus:border-[var(--primary)] outline-none transition-all"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
            disabled={isSubmitting}
          />
        </div>

        <div className="grid-2 gap-4 mb-4">
          <div className="form-group">
            <label className="form-label flex items-center gap-1.5 text-sm font-semibold mb-1" htmlFor="simDataset">
              <Database size={16} className="text-[var(--text-muted)]" /> Dataset
            </label>
            <select
              id="simDataset"
              className="form-control w-full border border-[var(--border-color)] rounded-[var(--radius-sm)] px-3 py-2 bg-white focus:ring-2 focus:ring-[var(--primary-glow)] focus:border-[var(--primary)] outline-none transition-all"
              value={dataset}
              onChange={(e) => setDataset(e.target.value)}
              disabled={isLoadingMeta || isSubmitting}
            >
              {datasets.length === 0 ? (
                <option value="multimodal">ADNI Multimodal (MRI + Cog + EHR)</option>
              ) : (
                datasets.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.display_name} ({d.data_type})
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label flex items-center gap-1.5 text-sm font-semibold mb-1" htmlFor="simModel">
              <Cpu size={16} className="text-[var(--text-muted)]" /> Neural Architecture
            </label>
            <select
              id="simModel"
              className="form-control w-full border border-[var(--border-color)] rounded-[var(--radius-sm)] px-3 py-2 bg-white focus:ring-2 focus:ring-[var(--primary-glow)] focus:border-[var(--primary)] outline-none transition-all"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isLoadingMeta || isSubmitting}
            >
              {models.length === 0 ? (
                <option value="fedxneuro">Fed-XNeuro (3D ResNet + Transformer)</option>
              ) : (
                models.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.display_name} ({m.architecture_type})
                  </option>
                ))
              )}
            </select>
          </div>
        </div>

        <div className="grid-2 gap-4 mb-4">
          <div className="form-group">
            <label className="form-label flex items-center gap-1.5 text-sm font-semibold mb-1" htmlFor="simAlgorithm">
              <Network size={16} className="text-[var(--text-muted)]" /> FL Aggregation Algorithm
            </label>
            <select
              id="simAlgorithm"
              className="form-control w-full border border-[var(--border-color)] rounded-[var(--radius-sm)] px-3 py-2 bg-white focus:ring-2 focus:ring-[var(--primary-glow)] focus:border-[var(--primary)] outline-none transition-all"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
              disabled={isLoadingMeta || isSubmitting}
            >
              {algorithms.length === 0 ? (
                <option value="fedxneuro">Fed-XNeuro (Sample-Weighted)</option>
              ) : (
                algorithms.map((a) => (
                  <option key={a.name} value={a.name}>
                    {a.display_name}
                  </option>
                ))
              )}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label flex items-center gap-1.5 text-sm font-semibold mb-1" htmlFor="simClients">
              <Users size={16} className="text-[var(--text-muted)]" /> Hospital Clients
            </label>
            <input
              id="simClients"
              type="number"
              className="form-control w-full border border-[var(--border-color)] rounded-[var(--radius-sm)] px-3 py-2 focus:ring-2 focus:ring-[var(--primary-glow)] focus:border-[var(--primary)] outline-none transition-all"
              min={1}
              max={20}
              value={numClients}
              onChange={(e) => setNumClients(parseInt(e.target.value) || 1)}
              required
              disabled={isSubmitting}
            />
          </div>
        </div>



        <button
          type="submit"
          className="btn btn-primary w-full py-3 flex items-center justify-center gap-2 bg-[var(--primary)] hover:bg-[var(--primary-hover)] text-white font-semibold rounded-[var(--radius-md)] transition-all shadow-md disabled:opacity-70"
          disabled={isSubmitting}
        >
          {isSubmitting ? (
             <>
               <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                 <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                 <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
               </svg>
               Starting Simulation...
             </>
          ) : (
            <><Rocket size={18} /> Launch Federated Simulation</>
          )}
        </button>
      </form>
    </div>
  );
};

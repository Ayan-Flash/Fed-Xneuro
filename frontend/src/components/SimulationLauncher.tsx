"use client";

import React, { useEffect, useState } from "react";
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
import { IconRocket, IconGear, IconCheck, IconChevronDown } from "@/components/Icons";

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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showToast, setShowToast] = useState(false);
  const [toastName, setToastName] = useState("");

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

      const created = await createSimulation(payload);
      await startSimulation(created.id);

      setToastName(created.name);
      setShowToast(true);
      setTimeout(() => setShowToast(false), 3500);

      onLaunched(created);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to launch simulation";
      setErrorMsg(msg);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <>
      <div className="card">
        <div className="card-header">
          <div className="card-title">
            <IconRocket size={20} /> Launch Simulation
          </div>
          <div className="card-desc">
            Configure and run decentralized federated training across simulated hospital nodes.
          </div>
        </div>

        {errorMsg && <div className="alert-error">{errorMsg}</div>}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="simName">
              Experiment Name
            </label>
            <input
              id="simName"
              type="text"
              className="form-control"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              disabled={isSubmitting}
            />
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="simDataset">
                Dataset
              </label>
              <select
                id="simDataset"
                className="form-control"
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
              <label className="form-label" htmlFor="simModel">
                Neural Architecture
              </label>
              <select
                id="simModel"
                className="form-control"
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

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="simAlgorithm">
                FL Algorithm
              </label>
              <select
                id="simAlgorithm"
                className="form-control"
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
              <label className="form-label" htmlFor="simClients">
                Hospital Clients
              </label>
              <input
                id="simClients"
                type="number"
                className="form-control"
                min={1}
                max={20}
                value={numClients}
                onChange={(e) => setNumClients(parseInt(e.target.value) || 1)}
                required
                disabled={isSubmitting}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="simRounds">
              Communication Rounds
            </label>
            <input
              id="simRounds"
              type="number"
              className="form-control"
              min={1}
              max={100}
              value={numRounds}
              onChange={(e) => setNumRounds(parseInt(e.target.value) || 1)}
              required
              disabled={isSubmitting}
            />
          </div>

          {/* Advanced Settings Accordion */}
          <div style={{ marginBottom: "1rem" }}>
            <button
              type="button"
              className={`accordion-trigger ${showAdvanced ? "open" : ""}`}
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <IconGear size={15} /> Advanced Training Parameters
              </span>
              <IconChevronDown size={14} className="chevron" />
            </button>
            <div className={`accordion-content ${showAdvanced ? "open" : ""}`}>
              <div className="grid-3">
                <div className="form-group">
                  <label className="form-label" htmlFor="simEpochs">
                    Local Epochs
                  </label>
                  <input
                    id="simEpochs"
                    type="number"
                    className="form-control"
                    min={1}
                    max={20}
                    value={localEpochs}
                    onChange={(e) => setLocalEpochs(parseInt(e.target.value) || 1)}
                    required
                    disabled={isSubmitting}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="simBatch">
                    Batch Size
                  </label>
                  <input
                    id="simBatch"
                    type="number"
                    className="form-control"
                    min={1}
                    max={256}
                    value={batchSize}
                    onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                    required
                    disabled={isSubmitting}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="simLr">
                    Learning Rate
                  </label>
                  <input
                    id="simLr"
                    type="number"
                    step="0.001"
                    className="form-control"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0.01)}
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="simDp">
                  Differential Privacy ε (Optional)
                </label>
                <input
                  id="simDp"
                  type="number"
                  step="0.1"
                  placeholder="e.g. 5.0 — leave empty for no DP"
                  className="form-control"
                  value={dpEpsilon}
                  onChange={(e) => setDpEpsilon(e.target.value)}
                  disabled={isSubmitting}
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            className={`btn btn-primary ${isSubmitting ? "launching" : ""}`}
            style={{ width: "100%", marginTop: "0.25rem" }}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              "Starting Simulation…"
            ) : (
              <><IconRocket size={16} /> Launch Federated Simulation</>
            )}
          </button>
        </form>
      </div>

      {/* Success Toast */}
      {showToast && (
        <div className="toast-container">
          <div className="toast success">
            <IconCheck size={16} color="#a7f3d0" /> Simulation &ldquo;{toastName}&rdquo; launched successfully!
          </div>
        </div>
      )}
    </>
  );
};

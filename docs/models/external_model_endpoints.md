# External Model Endpoints & Training Integration Guide

This guide explains how to train models in a separate directory or standalone script on your machine and connect them directly to the **PS32 Federated Learning Simulator**.

---

## 1. Overview of the Connection Architecture

The simulator provides a dedicated [ModelConnector](file:///f:/PS32/backend/fl_engine/models/connector.py) endpoint layer:

```text
  Your External Directory                      PS32 FL Simulator
 ┌───────────────────────┐                    ┌───────────────────────┐
 │  Custom Training      │                    │   ModelConnector      │
 │  Script / Notebook    │                    │   Endpoint Layer      │
 └──────────┬────────────┘                    └──────────┬────────────┘
            │                                            │
            ▼                                            ▼
   Exports Weights (.pt)    ────────────────────►   Loads into Server
   Or Custom Architecture                           & Broadcasts to
   (.py file)                                       Clients for FL
```

---

## 2. How to Save Weights in Your External Training Directory

In your external training script, train your PyTorch model as usual and save its `state_dict`:

```python
import torch

# After training your model in your own directory:
torch.save(model.state_dict(), "my_trained_model.pt")

# Or with metadata:
torch.save({
    "state_dict": model.state_dict(),
    "epoch": 50,
    "best_acc": 92.4
}, "my_trained_model.pt")
```

The [ModelConnector](file:///f:/PS32/backend/fl_engine/models/connector.py) automatically detects and extracts the weights whether saved as a raw `state_dict`, wrapped in a dictionary (`state_dict` / `model_state_dict`), or saved from `nn.DataParallel`.

---

## 3. How to Connect Your Trained Weights to PS32

### Option A: Using the CLI Runner
You can initialize the federated simulation directly from your trained weights:

```powershell
python backend/run_simulation.py `
    --model cnn `
    --checkpoint "path/to/my_trained_model.pt" `
    --rounds 5
```

The simulator will:
1. Initialize the CNN architecture.
2. Load your pre-trained weights into the global model.
3. Broadcast those weights to the federated clients to continue federated training or fine-tuning.

---

### Option B: Using Custom Architectures from Another Directory

If you wrote a completely custom neural network in your directory (e.g. `../my_research/mci_model.py`):

```python
# ../my_research/mci_model.py
import torch
import torch.nn as nn
from backend.fl_engine.models.base import BaseModel

class MCIProgressionNet(BaseModel):
    def __init__(self, in_features=128, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)
```

You can connect both your custom architecture file and weights directly into PS32:

```powershell
python backend/run_simulation.py `
    --model mci_net `
    --custom-model-path "path/to/mci_model.py" `
    --custom-model-class "MCIProgressionNet" `
    --checkpoint "path/to/my_trained_model.pt"
```

---

## 4. Python API Usage

You can also use the endpoints programmatically in your own scripts:

```python
from backend.fl_engine.core.model_manager import ModelManager
from backend.fl_engine.simulation.config import SimulationConfig
from backend.fl_engine.simulation.engine import SimulationEngine

# 1. Register external architecture
ModelManager.register_external_architecture(
    file_path="C:/path/to/mci_model.py",
    class_name="MCIProgressionNet",
    register_as="mci_net"
)

# 2. Configure simulation with external checkpoint
config = SimulationConfig(
    model="mci_net",
    checkpoint_path="C:/path/to/my_trained_model.pt",
    num_clients=5,
    num_rounds=5
)

# 3. Run simulation
engine = SimulationEngine(config)
state, summary = engine.run()
```

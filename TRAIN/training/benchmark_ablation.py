#!/usr/bin/env python
"""
TRAIN / Training / Section 10 Ablation Benchmark Runner.

Executes comparative benchmarks across all 9 baseline configurations:
1. FedAvg (MRI only)
2. FedAvg (Cognitive only)
3. FedAvg (EHR only)
4. FedAvg (Multimodal early fusion)
5. FedAvg (Fed-XNeuro cross-modal attention)
6. FedProx (Fed-XNeuro, non-IID)
7. SCAFFOLD (Fed-XNeuro, non-IID)
8. FedAdam (Fed-XNeuro)
9. Centralized baseline
"""

import os
import sys

workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from backend.benchmark_fedxneuro import run_full_ablation_study, parse_args


def main() -> None:
    args = parse_args()
    run_full_ablation_study(
        num_rounds=args.rounds,
        num_clients=args.clients,
        local_epochs=args.epochs,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

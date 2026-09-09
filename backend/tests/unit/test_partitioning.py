import numpy as np
import pytest
from backend.fl_engine.partitioning.iid import IIDPartitioner
from backend.fl_engine.partitioning.non_iid import LabelNonIIDPartitioner
from backend.fl_engine.partitioning.dirichlet import DirichletPartitioner
from backend.fl_engine.partitioning.unbalanced import UnbalancedPartitioner
from backend.fl_engine.partitioning import create_partitioner


def test_iid_partitioning_coverage_and_balance(synthetic_targets):
    num_clients = 5
    partitioner = IIDPartitioner(num_clients=num_clients, seed=42)
    partitions = partitioner.partition(synthetic_targets)

    assert len(partitions) == num_clients
    assert set(partitions.keys()) == set(range(num_clients))

    # All samples accounted for and no duplicates
    all_indices = []
    for cid, indices in partitions.items():
        all_indices.extend(indices)
        # Approximate balance (100 samples / 5 clients = 20 each)
        assert len(indices) == 20

    assert len(all_indices) == len(synthetic_targets)
    assert len(set(all_indices)) == len(synthetic_targets)


def test_label_non_iid_class_skew(synthetic_targets):
    num_clients = 5
    partitioner = LabelNonIIDPartitioner(num_clients=num_clients, shards_per_client=2, seed=42)
    partitions = partitioner.partition(synthetic_targets)

    assert len(partitions) == num_clients
    all_indices = []
    for cid, indices in partitions.items():
        all_indices.extend(indices)
        assert len(indices) > 0

    assert len(set(all_indices)) == len(synthetic_targets)

    # Verify class distribution differs across clients
    client_classes = [set(synthetic_targets[indices]) for indices in partitions.values()]
    # At least some clients should have distinct class subsets
    assert len(set(tuple(sorted(c)) for c in client_classes)) > 1


def test_dirichlet_partitioning_validity(synthetic_targets):
    num_clients = 4
    partitioner = DirichletPartitioner(num_clients=num_clients, alpha=0.5, seed=42)
    partitions = partitioner.partition(synthetic_targets)

    assert len(partitions) == num_clients
    all_indices = []
    for cid, indices in partitions.items():
        all_indices.extend(indices)
        assert len(indices) > 0

    assert len(set(all_indices)) == len(synthetic_targets)


def test_unbalanced_partitioning_produces_variance(synthetic_targets):
    num_clients = 5
    partitioner = UnbalancedPartitioner(num_clients=num_clients, imbalance_factor=1.0, seed=42)
    partitions = partitioner.partition(synthetic_targets)

    counts = [len(indices) for indices in partitions.values()]
    assert len(counts) == num_clients
    assert sum(counts) == len(synthetic_targets)
    # Variance should be > 0 because partition is unbalanced
    assert np.var(counts) > 0


def test_create_partitioner_factory():
    p = create_partitioner("iid", num_clients=3)
    assert isinstance(p, IIDPartitioner)

    p_dir = create_partitioner("dirichlet", num_clients=3, alpha=0.1)
    assert isinstance(p_dir, DirichletPartitioner)
    assert p_dir.alpha == 0.1

    with pytest.raises(ValueError):
        create_partitioner("non_existent_type", num_clients=3)

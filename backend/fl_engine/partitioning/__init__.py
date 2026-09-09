from typing import Dict, Any, Type
from backend.fl_engine.partitioning.base import BasePartitioner
from backend.fl_engine.partitioning.iid import IIDPartitioner
from backend.fl_engine.partitioning.non_iid import LabelNonIIDPartitioner
from backend.fl_engine.partitioning.dirichlet import DirichletPartitioner
from backend.fl_engine.partitioning.unbalanced import UnbalancedPartitioner

PARTITIONER_MAP: Dict[str, Type[BasePartitioner]] = {
    "iid": IIDPartitioner,
    "non_iid": LabelNonIIDPartitioner,
    "label_non_iid": LabelNonIIDPartitioner,
    "dirichlet": DirichletPartitioner,
    "unbalanced": UnbalancedPartitioner,
}


def create_partitioner(partition_type: str, num_clients: int, **kwargs: Any) -> BasePartitioner:
    """
    Factory function for dataset partitioners.
    
    Args:
        partition_type: Type of partitioning ('iid', 'non_iid', 'dirichlet', 'unbalanced').
        num_clients: Total number of clients to partition across.
        **kwargs: Additional strategy-specific arguments (e.g. alpha, seed, shards_per_client).

    Returns:
        Instance of BasePartitioner.
    """
    key = partition_type.lower()
    if key not in PARTITIONER_MAP:
        raise ValueError(
            f"Unknown partition type '{partition_type}'. Available: {list(PARTITIONER_MAP.keys())}"
        )
    partitioner_cls = PARTITIONER_MAP[key]
    
    # Filter kwargs to match constructor
    import inspect
    valid_keys = inspect.signature(partitioner_cls.__init__).parameters.keys()
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
    
    return partitioner_cls(num_clients=num_clients, **filtered_kwargs)


__all__ = [
    "BasePartitioner",
    "IIDPartitioner",
    "LabelNonIIDPartitioner",
    "DirichletPartitioner",
    "UnbalancedPartitioner",
    "create_partitioner",
    "PARTITIONER_MAP",
]

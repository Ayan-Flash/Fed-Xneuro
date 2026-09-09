# Dataset Management and Partitioning in PS32

## 1. Supported Datasets

The PS32 simulator supports standard image classification benchmarks, structured to easily integrate multimodal clinical datasets (OASIS, ADNI) in future project phases.

### 1.1 MNIST (Phase 1 Baseline)
- **Modality**: Grayscale 28x28 handwritten digit images (0–9).
- **Size**: 60,000 training images, 10,000 test images.
- **Normalization**: Normalized using standard channel mean $0.1307$ and standard deviation $0.3081$.
- **Storage**: Automatically downloaded by `MNISTDatasetManager` to `data/raw/mnist/`.

### 1.2 CIFAR-10 & Fashion-MNIST (Pre-configured)
- Pre-registered in `DatasetRegistry` for seamless multi-dataset experimentation.

---

## 2. Partitioning Strategies

Real-world federated environments exhibit substantial variance in data volume and label distribution across participating sites (memory clinics, hospitals, mobile devices). PS32 provides four partitioning algorithms:

### 2.1 IID Partitioning (`IIDPartitioner`)
- **Mechanism**: The dataset indices are uniformly shuffled and sliced into $K$ equal-sized subsets.
- **Characteristics**: Each client observes identical, balanced marginal class distributions ($P_k(y) \approx P(y)$).
- **Use Case**: Serves as the control baseline for evaluating federated algorithm convergence.

### 2.2 Label-based Non-IID Partitioning (`LabelNonIIDPartitioner`)
- **Mechanism**:
  1. The training dataset is sorted strictly by class label.
  2. The sorted sequence is partitioned into $S = K \times \text{shards\_per\_client}$ continuous shards.
  3. Each client is allocated a fixed number of shards (default: 2 shards per client).
- **Characteristics**: Creates extreme label skew where individual clients train on only 1 or 2 distinct classes.
- **Use Case**: Simulates specialized memory clinics that diagnose specific disease categories.

### 2.3 Dirichlet Non-IID Partitioning (`DirichletPartitioner`)
- **Mechanism**:
  For each class $c \in \{0, \dots, C-1\}$, the class samples are distributed across $K$ clients according to a Dirichlet distribution:
  $$p_c \sim \text{Dir}(\alpha, \alpha, \dots, \alpha)$$
- **Characteristics**:
  - Controlled by the concentration parameter $\alpha > 0$.
  - **Small $\alpha$ (e.g., $\alpha = 0.1, 0.5$)**: Extreme non-IID heterogeneity; clients hold vastly disparate class frequencies.
  - **Large $\alpha$ (e.g., $\alpha \to \infty$)**: Approaches uniform IID distributions.
- **Use Case**: Highly realistic simulation of multi-institutional clinical distributions.

### 2.4 Unbalanced Partitioning (`UnbalancedPartitioner`)
- **Mechanism**: Assigns differing quantities of total training samples to clients following a log-normal distribution:
  $$w_k \sim \text{LogNormal}(0, \sigma)$$
- **Characteristics**: Reflects institutional size discrepancies (e.g., major research hospitals vs. rural clinics).
